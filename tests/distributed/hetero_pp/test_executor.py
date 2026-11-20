# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ray boundary tests: finite waits, all-rank error propagation, owned cleanup."""

import importlib.util
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch


class ExecutorTests(unittest.TestCase):
    def load(self):
        p = (
            Path(__file__).resolve().parents[3]
            / "vllm/v1/executor/hetero_pp_executor.py"
        )
        self.assertTrue(p.exists(), "Ray heterogeneous executor is not implemented")
        dep = ModuleType("vllm.v1.executor.abstract")

        class Executor:
            vllm_config: NS

            @property
            def max_concurrent_batches(self):
                return self.vllm_config.max_concurrent_batches

        dep.__dict__["Executor"] = Executor
        with patch.dict(sys.modules, {"vllm.v1.executor.abstract": dep}):
            spec = importlib.util.spec_from_file_location("executor_test", p)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def make(self, error=False):
        mod = self.load()
        events: list[tuple] = []

        class ActorError(Exception):
            pass

        class ActorExit(ActorError):
            pass

        class Ray:
            exceptions = NS(RayActorError=ActorError, ActorDiedError=ActorExit)

            def get(self, refs, timeout):
                if isinstance(refs, tuple) and refs[0] == "exit":
                    events.append(("exit-complete", refs[1]))
                    raise ActorExit("intentional actor exit")
                events.append(("get", len(refs), timeout))
                if error:
                    raise RuntimeError("stage1 failed")
                return [None, None, None, "last-output"]

            def kill(self, actor, no_restart):
                events.append(("kill", actor.rank))

            def wait(self, refs, *, num_returns, timeout):
                events.append(("wait-exit", num_returns, timeout))
                return refs, []

        class Actor:
            def __init__(self, rank):
                self.rank = rank

                def submit(*args):
                    events.append(("submit", rank))
                    return rank

                def terminate():
                    events.append(("exit", rank))
                    return "exit", rank

                self.call = NS(remote=submit)
                self.shutdown = NS(remote=lambda: events.append(("cleanup", rank)))
                self.__ray_terminate__ = NS(remote=terminate)

        obj = object.__new__(mod.HeteroPPExecutor)
        obj.ray = Ray()
        obj.workers = [Actor(r) for r in range(4)]
        obj.vllm_config = NS(max_concurrent_batches=len(obj.workers))
        obj._profiling_enabled = False
        obj.timeout_s = 2
        obj._failed = False
        obj._failure_lock = threading.Lock()
        obj._pool = ThreadPoolExecutor(max_workers=1)
        obj._responses = ThreadPoolExecutor(max_workers=1)
        obj._shutdown_lock = threading.RLock()
        obj._shutting_down = obj._shutdown_complete = False
        self.addCleanup(obj._responses.shutdown)
        self.addCleanup(obj._pool.shutdown)
        return obj, events

    def test_model_output_is_last_stage_not_first(self):
        obj, events = self.make()
        self.assertEqual(obj.execute_model("schedule"), "last-output")
        self.assertEqual(
            events[:4], [("submit", 0), ("submit", 1), ("submit", 2), ("submit", 3)]
        )
        method, count, timeout = events[4]
        self.assertEqual((method, count), ("get", 4))
        self.assertTrue(0 < timeout <= 2)

    def test_middle_error_kills_only_owned_actors_and_refuses_more_work(self):
        obj, events = self.make(error=True)
        with self.assertRaises(RuntimeError):
            obj.execute_model("schedule")
        self.assertEqual(
            [e for e in events if e[0] == "kill"],
            [("kill", 0), ("kill", 1), ("kill", 2), ("kill", 3)],
        )
        before = len(events)
        with self.assertRaises(RuntimeError):
            obj.execute_model("again")
        self.assertEqual(len(events), before)

    def test_nonblocking_call_returns_resolvable_future(self):
        obj, events = self.make()
        future = obj.execute_model("schedule", non_block=True)
        self.assertEqual(future.result(timeout=2), "last-output")

    def test_serving_uses_native_pipeline_depth(self):
        obj, _ = self.make()
        self.assertEqual(obj.max_concurrent_batches, len(obj.workers))

    def test_profiling_keeps_one_batch(self):
        obj, _ = self.make()
        obj._profiling_enabled = True
        self.assertEqual(obj.max_concurrent_batches, 1)

    def test_stage_zero_starts_b_while_stage_one_still_runs_a(self):
        obj, _ = self.make()
        in_a, release_a, started_b, sampled_a = (threading.Event() for _ in range(4))
        stage_calls: list[list[tuple[str, str]]] = [[] for _ in obj.workers]

        def run(rank, method, batch):
            stage_calls[rank].append((method, batch))
            if (rank, method, batch) == (1, "execute_model", "A"):
                in_a.set()
                self.assertTrue(release_a.wait(5))
            if (rank, method, batch) == (0, "execute_model", "B"):
                self.assertTrue(in_a.wait(5))
                started_b.set()
            if (rank, method, batch) == (3, "sample_tokens", "A"):
                self.assertTrue(release_a.wait(5))
                sampled_a.set()
            return batch if method == "sample_tokens" and rank == 3 else None

        actor_pools = [ThreadPoolExecutor(max_workers=1) for _ in obj.workers]
        for worker, pool in zip(obj.workers, actor_pools, strict=True):
            worker.call = NS(
                remote=lambda method, args, kwargs, rank=worker.rank, pool=pool: (
                    pool.submit(run, rank, method, args[0])
                )
            )
        obj.ray.get = lambda refs, timeout: [ref.result(timeout) for ref in refs]
        obj.timeout_s = 5
        try:
            forward_a = obj.execute_model("A", non_block=True)
            sample_a = obj.sample_tokens("A", non_block=True)
            self.assertTrue(in_a.wait(2))
            forward_b = obj.execute_model("B", non_block=True)
            sample_b = obj.sample_tokens("B", non_block=True)
            self.assertTrue(started_b.wait(1), "Stage 0 is still waiting for batch A")
            self.assertFalse(forward_a.done())
            self.assertFalse(sampled_a.is_set())
            release_a.set()
            self.assertIsNone(forward_a.result(2))
            self.assertIsNone(forward_b.result(2))
            self.assertEqual(sample_a.result(2), "A")
            self.assertEqual(sample_b.result(2), "B")
            expected = [
                ("execute_model", "A"),
                ("sample_tokens", "A"),
                ("execute_model", "B"),
                ("sample_tokens", "B"),
            ]
            self.assertEqual(stage_calls, [expected] * 4)
        finally:
            release_a.set()
            for pool in actor_pools:
                pool.shutdown()

    def test_stage_failure_fails_all_posted_batches(self):
        obj, events = self.make()
        collecting, fail = threading.Event(), threading.Event()

        def gather(refs, timeout):
            collecting.set()
            self.assertTrue(fail.wait(2))
            raise RuntimeError("stage1 failed")

        obj.ray.get = gather
        first = obj.execute_model("A", non_block=True)
        try:
            self.assertTrue(collecting.wait(2))
            second = obj.execute_model("B", non_block=True)
            self.assertEqual(events, [("submit", r) for r in range(4)] * 2)
        finally:
            fail.set()
        with self.assertRaisesRegex(RuntimeError, "stage1 failed"):
            first.result(2)
        with self.assertRaisesRegex(RuntimeError, "executor has failed"):
            second.result(2)
        self.assertEqual(
            [e for e in events if e[0] == "kill"], [("kill", rank) for rank in range(4)]
        )

    def test_synchronous_rpc_cannot_overtake_pending_model_rpc(self):
        """All public calls must share one rank-consistent submission queue."""
        import threading

        obj, events = self.make()
        model_waiting = threading.Event()
        release_model = threading.Event()
        health_posted = threading.Event()
        for worker in obj.workers:
            rank = worker.rank

            def submit(method, args, kwargs, rank=rank):
                events.append((method, rank))
                if method == "check_health":
                    health_posted.set()
                return method, rank

            worker.call = NS(remote=submit)

        def gather(refs, timeout):
            if refs[0][0] == "execute_model":
                model_waiting.set()
                if not release_model.wait(2):
                    raise TimeoutError("test did not release model RPC")
            return [None, None, None, "last-output"]

        obj.ray.get = gather
        model = obj.execute_model("schedule", non_block=True)
        self.assertTrue(model_waiting.wait(2))
        with ThreadPoolExecutor(max_workers=1) as client:
            health = client.submit(obj.collective_rpc, "check_health")
            try:
                self.assertFalse(
                    health_posted.wait(0.1), "health RPC overtook pending model work"
                )
            finally:
                release_model.set()
            self.assertEqual(model.result(timeout=2), "last-output")
            self.assertEqual(len(health.result(timeout=2)), 4)
        self.assertEqual(
            events,
            [("execute_model", r) for r in range(4)]
            + [("check_health", r) for r in range(4)],
        )

    def test_reentrant_shutdown_does_not_kill_pending_cleanup(self):
        obj, events = self.make()
        import threading

        obj._shutdown_lock = threading.RLock()
        obj._shutting_down = obj._shutdown_complete = False
        for worker in obj.workers:
            worker.shutdown = NS(remote=lambda: None)
        nested = False

        def gather(refs, timeout):
            nonlocal nested
            if not nested:
                nested = True
                obj.shutdown()
                self.assertFalse(
                    any(e[0] == "kill" for e in events),
                    "nested shutdown killed actors before cleanup returned",
                )
            return [None] * 4

        obj.ray.get = gather
        obj.shutdown()
        obj.shutdown()
        self.assertEqual([e for e in events if e[0] == "kill"], [])
        self.assertEqual(
            [e for e in events if e[0] == "exit"],
            [("exit", 0), ("exit", 1), ("exit", 2), ("exit", 3)],
        )

    def test_successful_shutdown_uses_graceful_exit_after_cleanup(self):
        obj, events = self.make()
        obj.shutdown()
        self.assertEqual(
            [e for e in events if e[0] == "cleanup"],
            [("cleanup", 0), ("cleanup", 1), ("cleanup", 2), ("cleanup", 3)],
        )
        self.assertEqual(
            [e for e in events if e[0] == "exit-complete"],
            [
                ("exit-complete", 0),
                ("exit-complete", 1),
                ("exit-complete", 2),
                ("exit-complete", 3),
            ],
        )
        self.assertLess(events.index(("cleanup", 3)), events.index(("exit", 0)))
        self.assertEqual([e for e in events if e[0] == "kill"], [])
        self.assertTrue(obj._shutdown_complete)
        with self.assertRaises(RuntimeError):
            obj.execute_model("after-shutdown")

    def test_exit_timeout_force_cleans_owned_actors_and_propagates(self):
        obj, events = self.make()
        obj.ray.wait = lambda refs, **kwargs: (refs[:3], refs[3:])
        with self.assertRaises(TimeoutError):
            obj.shutdown()
        self.assertEqual(
            [e for e in events if e[0] == "kill"],
            [("kill", 0), ("kill", 1), ("kill", 2), ("kill", 3)],
        )
        self.assertTrue(obj._shutdown_complete)

    def test_cleanup_failure_is_not_hidden_by_actor_exit(self):
        obj, events = self.make(error=True)
        with self.assertRaisesRegex(RuntimeError, "stage1 failed"):
            obj.shutdown()
        self.assertFalse(any(e[0] == "exit" for e in events))
        self.assertEqual(
            [e for e in events if e[0] == "kill"],
            [("kill", 0), ("kill", 1), ("kill", 2), ("kill", 3)],
        )

    def test_exit_rpc_error_is_not_treated_as_intentional_exit(self):
        obj, events = self.make()
        original = obj.ray.get

        def gather(refs, timeout):
            if isinstance(refs, tuple):
                raise ValueError("bad exit RPC")
            return original(refs, timeout)

        obj.ray.get = gather
        with self.assertRaisesRegex(ValueError, "bad exit RPC"):
            obj.shutdown()
        self.assertEqual(len([e for e in events if e[0] == "kill"]), 4)

    def test_unavailable_actor_is_not_assumed_dead(self):
        obj, events = self.make()
        original = obj.ray.get

        def gather(refs, timeout):
            if isinstance(refs, tuple):
                raise obj.ray.exceptions.RayActorError("temporarily unavailable")
            return original(refs, timeout)

        obj.ray.get = gather
        with self.assertRaisesRegex(Exception, "temporarily unavailable"):
            obj.shutdown()
        self.assertEqual(len([e for e in events if e[0] == "kill"]), 4)

    def test_shutdown_drains_submitted_model_rpc_before_actor_cleanup(self):
        import threading

        obj, events = self.make()
        entered, release, model_done = (threading.Event() for _ in range(3))

        def gather(refs, timeout):
            entered.set()
            self.assertTrue(release.wait(2))
            model_done.set()
            return [None, None, None, "last-output"]

        obj.ray.get = gather

        def cleanup():
            self.assertTrue(
                model_done.is_set(), "actor cleanup raced pending model RPC"
            )
            return None

        for worker in obj.workers:
            worker.shutdown = NS(remote=cleanup)
        future = obj.execute_model("final-empty-step", non_block=True)
        self.assertTrue(entered.wait(2))
        timer = threading.Timer(0.05, release.set)
        timer.start()
        try:
            obj.shutdown()
            self.assertEqual(future.result(timeout=2), "last-output")
        finally:
            release.set()
            timer.join()


if __name__ == "__main__":
    unittest.main()
