# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPUWorker subclass that records PP stage traces.

vllm-ascend replaces the GPU Worker, so ``gpu_worker.py`` hooks never run
on Ascend. This wrapper times the same blocking recv / compute / send path
used by NPUWorker when ``VLLM_PP_STAGE_TRACE`` is set.
"""

from __future__ import annotations

import copy
from types import NoneType
from typing import TYPE_CHECKING

from vllm_ascend.utils import enable_sp
from vllm_ascend.worker.worker import NPUWorker

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.distributed.pp_hetero import PPHeteroConfig, sync_torch_device, time_call
from vllm.distributed.pp_stage_trace import (
    PPStageTracer,
    layer_range_from_runner,
    maybe_create_pp_stage_tracer,
    tensor_dict_nbytes,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def _all_gather_group():
    return None if enable_sp() else get_tp_group()


def _token_mix(scheduler_output: SchedulerOutput) -> tuple[int, int, int, int]:
    new_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
    num_ctx_tokens = 0
    num_gen_tokens = 0
    for req_id, ntok in scheduler_output.num_scheduled_tokens.items():
        if req_id in new_ids:
            num_ctx_tokens += ntok
        else:
            num_gen_tokens += ntok
    return (
        len(new_ids),
        num_ctx_tokens,
        len(scheduler_output.num_scheduled_tokens) - len(new_ids),
        num_gen_tokens,
    )


class PPAscendWorker(NPUWorker):
    """NPUWorker that writes ``pp_stage_pp*_tp*.jsonl`` while tracing."""

    def init_device(self):
        super().init_device()
        self._pp_stage_tracer = maybe_create_pp_stage_tracer(self.device)
        self._pp_hetero = PPHeteroConfig.from_env()

    def shutdown(self) -> None:
        if tracer := getattr(self, "_pp_stage_tracer", None):
            tracer.close()
            self._pp_stage_tracer = None
        parent = getattr(super(), "shutdown", None)
        if callable(parent):
            parent()

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        tracer = getattr(self, "_pp_stage_tracer", None)
        hetero = getattr(self, "_pp_hetero", None) or PPHeteroConfig()
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if not forward_pass:
            return super().execute_model(scheduler_output)
        if tracer is None and not hetero.enabled:
            return super().execute_model(scheduler_output)
        return self._execute_model_with_pp_trace(scheduler_output, tracer, hetero)

    def _execute_model_with_pp_trace(
        self,
        scheduler_output: SchedulerOutput,
        tracer: PPStageTracer | None,
        hetero: PPHeteroConfig,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        start_layer, end_layer = layer_range_from_runner(self.model_runner)
        n_ctx_req, n_ctx_tok, n_gen_req, n_gen_tok = _token_mix(scheduler_output)
        gather = _all_gather_group()
        pp_rank = get_pp_group().rank_in_group

        def _sync() -> None:
            sync_torch_device(self.device)

        recv_ms: float | None = None
        recv_bytes: int | None = None
        intermediate_tensors: IntermediateTensors | None = None
        if not get_pp_group().is_first_rank:

            def _recv() -> IntermediateTensors:
                return IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=gather)
                )

            if tracer is not None:
                intermediate_tensors, recv_ms = tracer.measure_comm(_recv)
            else:
                intermediate_tensors, recv_ms = time_call(_recv, _sync)
            recv_bytes = tensor_dict_nbytes(
                intermediate_tensors.tensors,
                all_gather_size=1 if gather is None else gather.world_size,
            )
            recv_ms = hetero.stretch_recv(
                pp_rank, recv_ms, payload_bytes=recv_bytes
            )

        def _run_forward():
            return self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )

        if tracer is not None:
            output, compute_ms = tracer.measure_compute(_run_forward)
        else:
            output, compute_ms = time_call(_run_forward, _sync)
        compute_ms = hetero.stretch_compute(pp_rank, compute_ms)

        send_ms: float | None = None
        send_bytes: int | None = None
        if isinstance(output, IntermediateTensors):
            send_bytes = tensor_dict_nbytes(
                output.tensors,
                all_gather_size=1 if gather is None else gather.world_size,
            )

            def _send() -> None:
                get_pp_group().send_tensor_dict(output.tensors, all_gather_group=gather)

            if tracer is not None:
                _, send_ms = tracer.measure_comm(_send)
            else:
                _, send_ms = time_call(_send, _sync)
            send_ms = hetero.stretch_send(
                pp_rank, send_ms, payload_bytes=send_bytes
            )
            if tracer is not None:
                tracer.record(
                    num_tokens=scheduler_output.total_num_scheduled_tokens,
                    num_reqs=len(scheduler_output.num_scheduled_tokens),
                    num_ctx_requests=n_ctx_req,
                    num_ctx_tokens=n_ctx_tok,
                    num_generation_requests=n_gen_req,
                    num_generation_tokens=n_gen_tok,
                    compute_ms=compute_ms,
                    recv_ms=recv_ms,
                    send_ms=send_ms,
                    recv_bytes=recv_bytes,
                    send_bytes=send_bytes,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    send_transfer_ms=hetero.transfer_ms(pp_rank, send_bytes),
                    compute_scale=hetero.compute_scale(pp_rank),
                    comm_scale=hetero.comm_scale(pp_rank),
                )
            kv_connector_output = getattr(output, "kv_connector_output", None)
            if not kv_connector_output:
                return None
            if (
                not kv_connector_output.finished_sending
                and not kv_connector_output.finished_recving
            ):
                return EMPTY_MODEL_RUNNER_OUTPUT
            result = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            result.kv_connector_output = kv_connector_output
            return result

        if tracer is not None:
            tracer.record(
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                num_reqs=len(scheduler_output.num_scheduled_tokens),
                num_ctx_requests=n_ctx_req,
                num_ctx_tokens=n_ctx_tok,
                num_generation_requests=n_gen_req,
                num_generation_tokens=n_gen_tok,
                compute_ms=compute_ms,
                recv_ms=recv_ms,
                send_ms=None,
                recv_bytes=recv_bytes,
                send_bytes=None,
                start_layer=start_layer,
                end_layer=end_layer,
                compute_scale=hetero.compute_scale(pp_rank),
                comm_scale=hetero.comm_scale(pp_rank),
            )
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):
            return output
        raise TypeError(f"Unexpected traced PP output type: {type(output)}")
