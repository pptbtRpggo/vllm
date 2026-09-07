# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in Ascend worker for tau trace collection, without editing the plugin.

Only the runner invocation is wrapped. PP communication, async output handling,
model loading and graph capture continue to use the installed NPUWorker.
The measurement is host elapsed time with an end-of-call NPU synchronization,
not device-event time. Synchronization can change pipeline overlap.
"""

from functools import wraps

import torch
from vllm_ascend.worker.worker import NPUWorker

from vllm.logger import init_logger
from vllm.v1.core.sched.tau_batch.trace import trace_worker_phase

logger = init_logger(__name__)


class TauAscendWorker(NPUWorker):
    def init_device(self):
        result = super().init_device()
        runner = self.model_runner
        execute = runner.execute_model
        if getattr(execute, "_tau_compute_wrapper", False):
            return result

        @wraps(execute)
        def traced_execute(scheduler_output, *args, **kwargs):
            if getattr(scheduler_output, "tau_fwd_id", None) is None:
                return execute(scheduler_output, *args, **kwargs)
            # Explicit NPU sync: missing APIs and synchronization failures must
            # propagate. Never record a successful compute interval on failure.
            with trace_worker_phase(
                self.vllm_config, scheduler_output, "compute", sync_end=False
            ):
                output = execute(scheduler_output, *args, **kwargs)
                torch.npu.synchronize()
            # Preserve the exact output, including AsyncModelRunnerOutput.
            return output

        traced_execute._tau_compute_wrapper = True
        # Assign to this instance only, leaving the plugin class unmodified.
        runner.execute_model = traced_execute
        logger.info(
            "Tau Ascend compute tracing installed on %s: host runner time "
            "with NPU end synchronization",
            type(runner).__qualname__,
        )
        return result
