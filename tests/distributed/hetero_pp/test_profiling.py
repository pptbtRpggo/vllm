# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU module hooks, trace lifecycle, and staged-completion contracts."""

import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_profiling():
    path = ROOT / "vllm/distributed/hetero_pp/profiling.py"
    if not path.exists():
        return None
    batch = ROOT / "vllm/distributed/pp_batch_shape.py"
    if batch.exists():
        name = "vllm.distributed.pp_batch_shape"
        spec = importlib.util.spec_from_file_location(name, batch)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    spec = importlib.util.spec_from_file_location("profile_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TinyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.start_layer, self.end_layer = 0, 2
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
        self.embed_tokens = torch.nn.Linear(2, 2)
        self.norm = torch.nn.Identity()

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()
        self.logits_processor = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.model(x)


def schedule():
    new = NS(
        req_id="r",
        num_computed_tokens=0,
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
        mm_features=[],
    )
    return NS(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"r": 2},
        scheduled_new_reqs=[new],
        scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
        scheduled_spec_decode_tokens={},
    )


class ProfilingTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_profiling()
        self.assertIsNotNone(self.mod, "staged layer profiling is not implemented")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = TinyModel()
        self.runner = NS(
            model=self.model,
            requests={},
            vllm_config=NS(scheduler_config=NS(async_scheduling=False)),
        )
        self.prof = self.mod.StageProfiler(
            self.directory.name,
            "session",
            self.model,
            torch.device("cpu"),
            0,
            4,
            identity={"profile_device_id": 0},
            timeout_s=3,
        )
        self.addCleanup(self.prof.close)

    def rows(self):
        return [
            json.loads(line) for line in Path(self.prof.path).read_text().splitlines()
        ]

    def test_execute_and_sample_are_measured_once_without_rpc_gap(self):
        step = self.prof.begin(schedule(), self.runner, 0)
        x = self.prof.execute(step, lambda: self.model(torch.ones(1, 2)))
        time.sleep(0.01)  # Deliberate gap is not part of either device interval.
        self.prof.sample(lambda: self.model.logits_processor(x))
        receipt = self.prof.flush()
        row = self.rows()[0]
        self.assertEqual(set(row["layer_compute_ms"]), {"0", "1"})
        self.assertTrue(row["sample_called"])
        self.assertIsNotNone(row["lm_head_ms"])
        self.assertAlmostEqual(
            row["compute_wall_ms"], row["execute_device_ms"] + row["sample_device_ms"]
        )
        modules = sum(row["layer_compute_ms"].values())
        endpoints = sum(
            row[k] or 0 for k in ("embedding_ms", "final_norm_ms", "lm_head_ms")
        )
        self.assertAlmostEqual(
            modules + row["non_layer_compute_ms"], row["compute_wall_ms"]
        )
        self.assertAlmostEqual(
            endpoints + row["runner_overhead_ms"], row["non_layer_compute_ms"]
        )
        self.assertEqual((receipt["record_count"], receipt["pending"]), (1, 0))

    def test_no_sample_is_explicit_zero_and_missing_head_is_none(self):
        step = self.prof.begin(schedule(), self.runner, 0)
        self.prof.execute(step, lambda: self.model(torch.ones(1, 2)))
        self.prof.flush()
        row = self.rows()[0]
        self.assertFalse(row["sample_called"])
        self.assertEqual(row["sample_device_ms"], 0)
        self.assertIsNone(row["lm_head_ms"])

    def test_warmup_and_layer_records_are_snapshotted_before_next_step(self):
        self.prof.is_warmup = True
        first = self.prof.begin(schedule(), self.runner, 0)
        self.prof.execute(first, lambda: self.model(torch.ones(1, 2)))
        self.prof.is_warmup = False
        second = self.prof.begin(schedule(), self.runner, 1)
        self.prof.execute(second, lambda: self.model(torch.ones(1, 2)))
        receipt = self.prof.flush()
        self.assertEqual([r["is_warmup"] for r in self.rows()], [True, False])
        self.assertEqual([r["sequence"] for r in self.rows()], [0, 1])
        self.assertEqual(
            (receipt["warmup_records"], receipt["measured_records"]), (1, 1)
        )

    def test_send_window_finishes_only_after_all_work_waits(self):
        step = self.prof.begin(schedule(), self.runner, 0)
        events = []

        class Work:
            def wait(self):
                events.append("complete")
                return True

        link = self.prof.timed_link(
            step,
            NS(
                recv=lambda seq: {"x": torch.ones(2)},
                send=lambda values, seq: [Work(), Work()],
            ),
        )
        values = link.recv(0)
        works = link.send(values, 0)
        self.assertIsNone(step.send_end_ns)
        works[0].wait()
        self.assertIsNone(step.send_end_ns)
        works[1].wait()
        self.assertGreaterEqual(step.send_end_ns, step.send_start_ns)
        self.assertEqual(step.send_bytes, 8)
        self.assertEqual(events, ["complete", "complete"])
        self.prof.execute(step, lambda: self.model(torch.ones(1, 2)))

    def test_gpu_like_send_waits_for_execute_completion_before_native_send(self):
        import threading

        step = self.prof.begin(schedule(), self.runner, 0)
        self.prof.execute(step, lambda: self.model(torch.ones(1, 2)))
        entered, ready = threading.Event(), threading.Event()
        calls, errors = [], []

        class Completion:
            def synchronize(self):
                entered.set()
                if not ready.wait(2):
                    raise RuntimeError("fixture completion was not released")

        original_end, original_backend = step.execute.end, self.prof.backend
        step.execute.end = Completion()
        self.prof.backend = object()  # Only the event dependency is doubled.

        def native_send(*_):
            calls.append("native_send")
            return []

        link = self.prof.timed_link(step, NS(send=native_send))

        def send():
            try:
                link.send({"x": torch.ones(2)}, 0)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=send)
        thread.start()
        try:
            self.assertTrue(
                entered.wait(0.5), "send never waited for device completion"
            )
            self.assertEqual(calls, [], "native send started while forward was pending")
            ready.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertEqual(calls, ["native_send"])
            self.assertLessEqual(
                step.send_submit_start_ns, step.send_ready_wait_start_ns
            )
            self.assertLessEqual(step.send_ready_wait_end_ns, step.send_start_ns)
        finally:
            ready.set()
            thread.join(2)
            step.execute.end, self.prof.backend = original_end, original_backend

    def test_skipped_decoder_layer_fails_and_hooks_are_removed(self):
        step = self.prof.begin(schedule(), self.runner, 0)
        with self.assertRaisesRegex(ValueError, "every local decoder layer"):
            self.prof.execute(
                step, lambda: self.model.model.layers[0](torch.ones(1, 2))
            )
        self.assertTrue(
            all(
                not m._forward_hooks and not m._forward_pre_hooks
                for m in self.model.modules()
            )
        )
        # A failed measurement never becomes a successful canonical row.
        self.assertFalse(self.rows())
        self.prof.close(raise_errors=False)

    def test_background_write_errors_are_reported_by_flush(self):
        step = self.prof.begin(schedule(), self.runner, 0)
        self.prof.execute(step, lambda: self.model(torch.ones(1, 2)))
        with (
            patch.object(self.prof, "_write_row", side_effect=OSError("disk fixture")),
            self.assertRaisesRegex(RuntimeError, "profile writer"),
        ):
            self.prof.flush()
        # Preserve failure diagnostics, but avoid re-raising the same expected
        # fixture during unittest cleanup.
        self.prof.close(raise_errors=False)

    def test_worker_routes_execute_and_sample_through_real_profiler(self):
        from contextlib import nullcontext

        path = ROOT / "vllm/distributed/hetero_pp/worker.py"
        spec = importlib.util.spec_from_file_location("profile_worker_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if TYPE_CHECKING:
            from vllm.distributed.hetero_pp.worker import StagingWorkerMixin
        else:
            StagingWorkerMixin = module.StagingWorkerMixin

        class Worker(StagingWorkerMixin):
            def _hetero_context(self):
                return nullcontext()

            def _run_staged_step(self, *args):
                return args[5](None)

        worker = Worker()
        worker.rank = 3
        worker.parallel_config = NS(world_size=4)
        worker._hetero_sequence = 0
        worker._hetero_channel = None
        worker._hetero_profiler = self.prof
        self.runner.execute_model = lambda *_: self.model(torch.ones(1, 2))
        self.runner.sample_tokens = lambda _: self.model.logits_processor(
            torch.ones(1, 2)
        )
        worker.model_runner = self.runner
        with patch.dict(
            sys.modules, {"vllm.sequence": NS(IntermediateTensors=lambda x: x)}
        ):
            worker.execute_model(schedule())
            worker.sample_tokens(None)
        self.assertEqual(self.prof.flush()["record_count"], 1)
        self.assertTrue(self.rows()[0]["sample_called"])

    def test_deferred_events_are_not_rerecorded_by_a_later_forward(self):
        import threading

        clock = [0]

        class Event:
            def __init__(self, **_):
                self.value = None

            def record(self, _):
                self.value = clock[0]
                clock[0] += 1

            def query(self):
                return True

            def elapsed_time(self, other):
                return other.value - self.value

        class ClockLayer(torch.nn.Module):
            duration = 2

            def forward(self, x):
                clock[0] += self.duration
                return x

        layer = ClockLayer()
        self.model.model.layers[0] = layer
        # Rebuild the profiler's module map after changing the fixture model.
        self.prof.layers[0] = layer
        self.prof.backend = NS(
            Event=Event, current_stream=lambda _: None, set_device=lambda _: None
        )
        entered, release = threading.Event(), threading.Event()
        resolve = self.prof._resolve

        def delayed(step):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release deferred trace")
            return resolve(step)

        try:
            with patch.object(self.prof, "_resolve", delayed):
                first = self.prof.begin(schedule(), self.runner, 0)
                self.prof.execute(first, lambda: self.model(torch.ones(1, 2)))
                second = self.prof.begin(schedule(), self.runner, 1)
                self.assertTrue(entered.wait(1))
                layer.duration = 8
                self.prof.execute(second, lambda: self.model(torch.ones(1, 2)))
                release.set()
                self.prof.flush()
        finally:
            release.set()
        rows = self.rows()
        # A reused layer event pair would overwrite the first value with 9.
        self.assertEqual([r["layer_compute_ms"]["0"] for r in rows], [3, 9])


class MemoryObservationTests(unittest.TestCase):
    def test_actual_kv_owner_storage_is_counted_once_and_specs_are_preserved(self):
        module = load_profiling()
        describe = getattr(module, "describe_kv_cache", None)
        if describe is None:
            self.fail("actual KV storage observation is missing")
        owner = torch.arange(32, dtype=torch.float16)
        worker = NS(
            device=torch.device("cpu"),
            cache_config=NS(num_gpu_blocks=4),
            model_runner=NS(
                kv_caches=[owner[:16], owner[16:]], kv_cache_config=NS(num_blocks=4)
            ),
            get_kv_cache_spec=lambda: {
                "model.layers.0.attn": NS(
                    page_size_bytes=16,
                    block_size=2,
                    num_kv_heads=2,
                    head_size=2,
                    dtype=torch.float16,
                )
            },
        )
        result = describe(worker)
        self.assertEqual(result["kv_cache_storage_bytes"], 64)
        self.assertEqual(result["kv_cache_storage_owners"], 1)
        self.assertEqual(len(result["kv_cache_tensors"]), 2)
        self.assertEqual(
            result["kv_cache_specs"]["model.layers.0.attn"]["page_size_bytes"], 16
        )
        self.assertEqual(result["num_gpu_blocks"], 4)


class BatchShapeTests(unittest.TestCase):
    def setUp(self):
        load_profiling()
        self.batch = sys.modules["vllm.distributed.pp_batch_shape"]

    def test_cached_prefill_and_decode_use_scheduler_positions(self):
        runner = NS(
            requests={
                "p": NS(prompt_token_ids=[0] * 128, num_computed_tokens=99),
                "d": NS(prompt_token_ids=[0] * 16),
            }
        )
        scheduler = NS(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=NS(
                req_ids=["p", "d"], num_computed_tokens=[64, 1024]
            ),
            num_scheduled_tokens={"p": 32, "d": 1},
        )
        row = self.batch.scheduled_batch_shape(scheduler, runner)
        self.assertEqual(
            row["batch_shape"],
            [
                {"query_tokens": 32, "context_tokens": 64, "prompt_tokens": 32},
                {"query_tokens": 1, "context_tokens": 1024, "prompt_tokens": 0},
            ],
        )
        self.assertEqual((row["num_ctx_tokens"], row["num_generation_tokens"]), (32, 1))

    def test_unknown_state_and_async_are_explicitly_unknown(self):
        scheduler = NS(scheduled_new_reqs=[], num_scheduled_tokens={"missing": 1})
        row = self.batch.scheduled_batch_shape(scheduler, NS())
        self.assertIsNone(row["batch_shape"])
        self.assertEqual(row["num_generation_tokens"], 0)
        self.assertIsNone(
            self.batch.scheduled_batch_shape(
                NS(), NS(vllm_config=NS(scheduler_config=NS(async_scheduling=True)))
            )["batch_shape"]
        )

    def test_new_prompt_embeds_need_no_cached_requests(self):
        scheduler = NS(
            scheduled_new_reqs=[
                NS(
                    req_id="r",
                    prompt_token_ids=None,
                    prompt_embeds=NS(shape=(10, 4)),
                    num_computed_tokens=5,
                )
            ],
            num_scheduled_tokens={"r": 3},
        )
        self.assertEqual(
            self.batch.scheduled_batch_shape(scheduler, NS())["batch_shape"],
            [{"query_tokens": 3, "context_tokens": 5, "prompt_tokens": 3}],
        )


class PlannerMappingTests(unittest.TestCase):
    def test_four_rank_real_traces_keep_global_and_repeated_local_ids_separate(self):
        profiling = load_profiling()
        for short in ("pp_comm_trace", "pp_memory", "pp_partition"):
            name = "vllm.distributed." + short
            if name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    name, ROOT / "vllm/distributed" / (short + ".py")
                )
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
        planner = sys.modules["vllm.distributed.pp_partition"]

        class PartitionedInner(TinyInner):
            def __init__(self, rank):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(8)]
                )
                self.start_layer, self.end_layer = rank * 2, rank * 2 + 2

            def forward(self, x):
                if self.start_layer == 0:
                    x = self.embed_tokens(x)
                for layer in self.layers[self.start_layer : self.end_layer]:
                    x = layer(x)
                return self.norm(x) if self.end_layer == 8 else x

        receipts = []
        with tempfile.TemporaryDirectory() as directory:
            for rank, local_id in enumerate([0, 1, 0, 1]):
                model = TinyModel()
                model.model = PartitionedInner(rank)
                runner = NS(
                    model=model,
                    requests={},
                    vllm_config=NS(scheduler_config=NS(async_scheduling=False)),
                )
                profiler = profiling.StageProfiler(
                    directory,
                    "two-node-session",
                    model,
                    torch.device("cpu"),
                    rank,
                    4,
                    identity={
                        "profile_device_id": rank,
                        "device_id": local_id,
                        "node_ip": "host-a" if rank < 2 else "host-b",
                    },
                )
                try:
                    step = profiler.begin(schedule(), runner, 0)
                    profiler.execute(step, lambda model=model: model(torch.ones(1, 2)))
                    receipts.append(profiler.flush())
                finally:
                    profiler.close()
            # This is the actual unchanged core loader, not a test mapping.
            traces = planner.load_trace_records(directory)
            self.assertEqual(
                [traces[r][0]["profile_device_id"] for r in range(4)], [0, 1, 2, 3]
            )
            self.assertEqual(
                [traces[r][0]["device_id"] for r in range(4)], [0, 1, 2, 3]
            )
            self.assertEqual(
                [traces[r][0]["local_device_id"] for r in range(4)], [0, 1, 0, 1]
            )
            self.assertEqual(
                [x["identity"]["device_id"] for x in receipts], [0, 1, 0, 1]
            )


if __name__ == "__main__":
    unittest.main()
