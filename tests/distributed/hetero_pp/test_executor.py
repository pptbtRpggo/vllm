"""Ray boundary tests: finite waits, all-rank error propagation, owned cleanup."""
import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch


class ExecutorTests(unittest.TestCase):
    def load(self):
        p = Path(__file__).resolve().parents[3] / 'vllm/v1/executor/hetero_pp_executor.py'
        self.assertTrue(p.exists(), 'Ray heterogeneous executor is not implemented')
        dep = ModuleType('vllm.v1.executor.abstract');dep.Executor = type('Executor', (), {})
        with patch.dict(sys.modules, {'vllm.v1.executor.abstract': dep}):
            spec = importlib.util.spec_from_file_location('executor_test', p)
            mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def make(self, error=False):
        mod = self.load(); events = []
        class Ray:
            def get(self, refs, timeout):
                events.append(('get', len(refs), timeout))
                if error: raise RuntimeError('stage1 failed')
                return [None, None, None, 'last-output']
            def kill(self, actor, no_restart): events.append(('kill', actor.rank))
        class Actor:
            def __init__(self, rank):
                self.rank = rank
                self.call = NS(remote=lambda *a: events.append(('submit', rank)) or rank)
        obj = object.__new__(mod.HeteroPPExecutor)
        obj.ray = Ray();obj.workers = [Actor(r) for r in range(4)]
        obj.timeout_s = 2;obj._failed = False;obj._pool = ThreadPoolExecutor(max_workers=1)
        import threading
        obj._shutdown_lock = threading.RLock()
        obj._shutting_down = obj._shutdown_complete = False
        self.addCleanup(obj._pool.shutdown)
        return obj, events

    def test_model_output_is_last_stage_not_first(self):
        obj, events = self.make()
        self.assertEqual(obj.execute_model('schedule'), 'last-output')
        self.assertEqual(events[:4], [('submit', 0), ('submit', 1), ('submit', 2), ('submit', 3)])
        self.assertEqual(events[4], ('get', 4, 2))

    def test_middle_error_kills_only_owned_actors_and_refuses_more_work(self):
        obj, events = self.make(error=True)
        with self.assertRaises(RuntimeError): obj.execute_model('schedule')
        self.assertEqual([e for e in events if e[0] == 'kill'], [('kill', 0), ('kill', 1), ('kill', 2), ('kill', 3)])
        before = len(events)
        with self.assertRaises(RuntimeError): obj.execute_model('again')
        self.assertEqual(len(events), before)

    def test_nonblocking_call_returns_resolvable_future(self):
        obj, events = self.make()
        future = obj.execute_model('schedule', non_block=True)
        self.assertEqual(future.result(timeout=2), 'last-output')

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
                if method == 'check_health':
                    health_posted.set()
                return method, rank
            worker.call = NS(remote=submit)
        def gather(refs, timeout):
            if refs[0][0] == 'execute_model':
                model_waiting.set()
                if not release_model.wait(2):
                    raise TimeoutError('test did not release model RPC')
            return [None, None, None, 'last-output']
        obj.ray.get = gather
        model = obj.execute_model('schedule', non_block=True)
        self.assertTrue(model_waiting.wait(2))
        with ThreadPoolExecutor(max_workers=1) as client:
            health = client.submit(obj.collective_rpc, 'check_health')
            try:
                self.assertFalse(health_posted.wait(.1),
                                 'health RPC overtook pending model work')
            finally:
                release_model.set()
            self.assertEqual(model.result(timeout=2), 'last-output')
            self.assertEqual(len(health.result(timeout=2)), 4)
        self.assertEqual(events, [('execute_model', r) for r in range(4)] +
                         [('check_health', r) for r in range(4)])

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
                self.assertFalse(any(e[0] == 'kill' for e in events),
                                 'nested shutdown killed actors before cleanup returned')
            return [None] * 4
        obj.ray.get = gather
        obj.shutdown()
        obj.shutdown()
        self.assertEqual([e for e in events if e[0] == 'kill'],
                         [('kill', 0), ('kill', 1), ('kill', 2), ('kill', 3)])

    def test_shutdown_drains_submitted_model_rpc_before_actor_cleanup(self):
        import threading
        obj, events = self.make()
        entered, release, model_done = (threading.Event() for _ in range(3))
        def gather(refs, timeout):
            entered.set()
            self.assertTrue(release.wait(2))
            model_done.set()
            return [None, None, None, 'last-output']
        obj.ray.get = gather
        def cleanup():
            self.assertTrue(model_done.is_set(), 'actor cleanup raced pending model RPC')
            return None
        for worker in obj.workers: worker.shutdown = NS(remote=cleanup)
        future = obj.execute_model('final-empty-step', non_block=True)
        self.assertTrue(entered.wait(2))
        timer = threading.Timer(.05, release.set)
        timer.start()
        try:
            obj.shutdown()
            self.assertEqual(future.result(timeout=2), 'last-output')
        finally:
            release.set();timer.join()


if __name__ == '__main__': unittest.main()
