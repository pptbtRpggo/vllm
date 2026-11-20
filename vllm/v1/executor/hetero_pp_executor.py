# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit opt-in Ray executor for CPU-staged four-stage Ascend PP.

Ray only carries scheduler/configuration/control and final CPU model results.
NPU activation tensors never cross Ray serialization. The cluster must already
exist; this executor does not launch Ray daemons or alter node environments.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

from vllm.v1.executor.abstract import Executor


class HeteroPPExecutor(Executor):
    uses_ray = True
    supports_pp = True

    @property
    def max_concurrent_batches(self) -> int:
        # EdgeShard measures isolated steps. Serving uses vLLM's PP queue size.
        return 1 if self._profiling_enabled else super().max_concurrent_batches

    def _init_executor(self):
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from vllm.distributed.hetero_pp.actor import PipelineActor
        from vllm.distributed.hetero_pp.config import (
            PipelinePlan,
            profile_actor_environment,
            validate_runtime,
        )
        from vllm.utils.network_utils import get_open_port

        self.ray = ray
        self.workers = []
        self._failed = False
        self._failure_lock = threading.Lock()
        self._shutdown_lock = threading.RLock()
        self._shutting_down = self._shutdown_complete = False
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hetero-pp-rpc"
        )
        self._responses = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hetero-pp-results"
        )
        try:
            raw = os.environ.get("VLLM_HETERO_PP_PLAN")
            if not raw:
                raise ValueError(
                    "VLLM_HETERO_PP_PLAN must contain the pipeline JSON plan"
                )
            layers = self.model_config.hf_text_config.num_hidden_layers
            self.plan = PipelinePlan.from_dict(json.loads(raw), layers)
            self.timeout_s = self.plan.timeout_s
            profile_env = profile_actor_environment()
            self._profiling_enabled = bool(profile_env["VLLM_PP_STAGE_TRACE"])
            verify_overlap = os.environ.get("VLLM_HETERO_PP_VERIFY_OVERLAP") == "1"
            if verify_overlap and self._profiling_enabled:
                raise ValueError("PP overlap verification requires online PP mode")
            profile_env["VLLM_HETERO_PP_VERIFY_OVERLAP"] = "1" if verify_overlap else ""
            os.environ["VLLM_PP_LAYER_PARTITION"] = self.plan.partition
            validate_runtime(self.vllm_config, len(self.plan.stages))
            architectures = self.model_config.hf_config.architectures
            if architectures not in (["LlamaForCausalLM"], ["Qwen2ForCausalLM"]):
                raise ValueError(
                    "first backend admits LlamaForCausalLM/Qwen2ForCausalLM only"
                )
            if not ray.is_initialized():
                ray.init(address="auto")
            nodes = {n["NodeManagerAddress"]: n for n in ray.nodes() if n["Alive"]}
            for ip in self.plan.nodes:
                if ip not in nodes or nodes[ip]["Resources"].get("NPU", 0) < 2:
                    raise ValueError(f"{ip} must be an alive Ray node with two NPUs")
            # rank0 hosts the TCP store; do not silently choose an IP on another
            # machine. Run the engine/head on the first plan node.
            head_node = ray.get_runtime_context().get_node_id()
            if nodes[self.plan.nodes[0]]["NodeID"] != head_node:
                raise ValueError("engine must run on the first pipeline node")
            remote = ray.remote(
                num_cpus=1,
                resources={"NPU": 1},
                runtime_env={"env_vars": profile_env},
                max_restarts=0,
                max_task_retries=0,
            )(PipelineActor)
            for stage in self.plan.stages:
                worker = remote.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=nodes[stage.node_ip]["NodeID"], soft=False
                    )
                ).remote()
                self.workers.append(worker)
            identities = ray.get(
                [w.identity.remote() for w in self.workers], timeout=self.timeout_s
            )
            # Actor allocation order is not device order. Sort each node's two
            # allocations by actual Ray NPU id before assigning PP ranks.
            ordered: list[Any] = []
            for ip in self.plan.nodes:
                owned = [
                    (w, ident)
                    for w, ident in zip(self.workers, identities)
                    if ident["node_id"] == nodes[ip]["NodeID"]
                ]
                if len(owned) != 2 or len({i["device_id"] for _, i in owned}) != 2:
                    raise ValueError(
                        "Ray did not reserve two distinct devices per node"
                    )
                ordered.extend(
                    w for w, _ in sorted(owned, key=lambda item: item[1]["device_id"])
                )
            self.workers = ordered
            rendezvous = f"tcp://{self.plan.nodes[0]}:{get_open_port()}"
            # All initialize RPCs are submitted before waiting (group creation
            # is collective). Same rule applies to every subsequent RPC.
            reports = ray.get(
                [
                    w.initialize.remote(self.vllm_config, rank, self.plan, rendezvous)
                    for rank, w in enumerate(self.workers)
                ],
                timeout=self.timeout_s,
            )
            for key in (
                "vllm",
                "torch",
                "torch_npu",
                "ray",
                "model_config_sha256",
                "sources",
            ):
                if any(r[key] != reports[0][key] for r in reports):
                    raise ValueError(f"pipeline endpoints disagree on {key}")
            if any(r["world_backend"] != "gloo" for r in reports):
                raise ValueError("mixed-SoC WORLD must not use HCCL")
            self.startup_reports = reports
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Heterogeneous PP startup reports: %s", json.dumps(reports))
            logger.info(
                "Heterogeneous PP max in-flight batches=%d", self.max_concurrent_batches
            )
        except BaseException:
            self._kill_owned()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._responses.shutdown(wait=False, cancel_futures=True)
            raise

    def _kill_owned(self):
        import logging
        import traceback

        with self._failure_lock:
            if self._failed:
                return
            self._failed = True
        logging.getLogger(__name__).debug(
            "Owned actor termination thread=%s caller=%s",
            threading.get_ident(),
            "".join(traceback.format_stack(limit=5)),
        )
        for worker in self.workers:
            with suppress(Exception):
                self.ray.kill(worker, no_restart=True)

    def _check_healthy(self):
        if self._failed:
            raise RuntimeError("heterogeneous PP executor has failed or shut down")

    def _post_rpc(self, method, args, kwargs):
        self._check_healthy()
        try:
            return [w.call.remote(method, args, kwargs) for w in self.workers]
        except BaseException:
            self._kill_owned()
            raise

    def _collect_rpc(self, refs, deadline):
        self._check_healthy()
        try:
            return self.ray.get(refs, timeout=max(0, deadline - time.monotonic()))
        except BaseException:
            self._kill_owned()
            raise

    def _rpc(self, method, args, kwargs, timeout):
        # Control RPCs (health, cache setup, etc.) retain their drain barrier.
        deadline = time.monotonic() + (self.timeout_s if timeout is None else timeout)
        try:
            self._responses.submit(self._check_healthy).result(
                timeout=max(0, deadline - time.monotonic())
            )
            return self._collect_rpc(self._post_rpc(method, args, kwargs), deadline)
        except BaseException:
            self._kill_owned()
            raise

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        kwargs = {} if kwargs is None else kwargs
        with self._shutdown_lock:
            self._check_accepting()
            future = self._pool.submit(self._rpc, method, args, kwargs, timeout)
        # Both forms use the same queue: direct synchronous dispatch could
        # overtake an accepted model step and reorder calls between ranks.
        return future if non_block else future.result()

    def _check_accepting(self):
        if self._shutting_down or self._shutdown_complete:
            raise RuntimeError("heterogeneous PP executor is shutting down")

    def _last_output(self, refs, deadline):
        return self._collect_rpc(refs, deadline)[-1]

    def _post_model_rpc(self, method, value):
        # One submitter preserves FIFO on each actor. Waiting for any stage's
        # result here would prevent an idle early stage from starting batch B.
        deadline = time.monotonic() + self.timeout_s
        refs = self._post_rpc(method, (value,), {})
        return self._responses.submit(self._last_output, refs, deadline)

    def _model_rpc(self, method, value, non_block):
        with self._shutdown_lock:
            self._check_accepting()
            submission = self._pool.submit(self._post_model_rpc, method, value)
        # Wait only until the RPCs are posted to the actor queues.
        future = submission.result()
        return future if non_block else future.result()

    def execute_model(self, scheduler_output, non_block=False):
        return self._model_rpc("execute_model", scheduler_output, non_block)

    def sample_tokens(self, grammar_output, non_block=False):
        return self._model_rpc("sample_tokens", grammar_output, non_block)

    def check_health(self):
        self.collective_rpc("check_health")

    def _exit_owned(self, deadline):
        if not self.workers:
            return
        refs = [w.__ray_terminate__.remote() for w in self.workers]
        ready, pending = self.ray.wait(
            refs,
            num_returns=len(refs),
            timeout=max(0, deadline - time.monotonic()),
        )
        if pending:
            raise TimeoutError("heterogeneous PP actors did not exit in time")
        for ref in ready:
            # exit_actor intentionally ends its RPC with ActorDiedError.
            # Unlike ray.kill, it runs Python/CANN exit handlers.
            with suppress(self.ray.exceptions.ActorDiedError):
                self.ray.get(ref, timeout=0)

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutting_down or self._shutdown_complete:
                return
            self._shutting_down = True
        exited = False
        deadline = time.monotonic() + min(self.timeout_s, 20)
        try:
            if not self._failed:
                # Close admission, drain posts, then drain responses before
                # destroying the communication groups.
                for pool in (self._pool, self._responses):
                    pool.submit(lambda: None).result(
                        timeout=max(0, deadline - time.monotonic())
                    )
                self._check_healthy()
                refs = [w.shutdown.remote() for w in self.workers]
                self.ray.get(refs, timeout=max(0, deadline - time.monotonic()))
                self._exit_owned(deadline)
                exited = True
        finally:
            if not exited:
                self._kill_owned()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._responses.shutdown(wait=False, cancel_futures=True)
            with self._shutdown_lock:
                self._failed = True
                self._shutdown_complete = True
