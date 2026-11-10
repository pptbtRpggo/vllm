# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real localhost coordination with CPU fixtures instead of NPU subprocesses.

Both launcher main functions, peer framing, socket coordination, receipt
validation and completion decisions are real. Device discovery and torchrun
are fixture boundaries. No Torch, CANN, SSH or accelerator code is imported.
"""

import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import uuid
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo.py"


def load_helper() -> Any:
    name = "paired_comm_launcher_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def free_adjacent_ports():
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
            if not 1024 <= port <= 65534:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as coordination:
                    coordination.bind(("127.0.0.1", port + 1))
                    return port
            except OSError:
                continue
    raise RuntimeError("no free adjacent localhost ports for the fixture")


def rank_receipt(rank):
    return {
        "rank": rank,
        "local_rank": rank % 2,
        "passed": True,
        "backend": "gloo",
        "destination_device": "npu",
        "checks": [{"M": m, "all_values_equal": True} for m in (1, 64, 256, 2048)],
        "wire_calls": [
            {
                "operation": operation,
                "peer": peer,
                "tag": 100,
                "device": "cpu",
                "backend": "gloo",
                "bytes": 16,
                "dtype": "torch.uint8",
            }
            for operation, peer in (
                ("irecv", 3 if rank == 0 else rank - 1),
                ("isend", 0 if rank == 3 else rank + 1),
            )
        ],
    }


class PairedLauncherTests(unittest.TestCase):
    def run_pair(self, directory, failed_child=None, source_mismatch=False):
        modules = [load_helper(), load_helper()]
        output = [directory / "head", directory / "worker"]
        port = free_adjacent_ports()
        started: list[int] = []
        connections: list[socket.socket] = []
        connection_lock = threading.Lock()
        results: list[int | None] = [None, None]
        errors: list[tuple[int, BaseException]] = []

        def choose(head, _interfaces, _route, interface=None):
            node = {"head": 0, "worker": 1}[interface]
            return {
                "node_rank": node,
                "node_ip": f"127.0.0.{node + 1}",
                "interface": interface,
                "head_ip": head,
            }

        def discover(args, _output, _deadline):
            node = {"head": 0, "worker": 1}[args.interface]
            soc = "910B" if node == 0 else "310P"
            return (
                sys.executable,
                {},
                {
                    "python": sys.executable,
                    "torch": "2.10.0+cpu",
                    "byteorder": "little",
                    "soc": soc,
                    "devices": [{"logical_id": i, "soc": soc} for i in (0, 1)],
                    "assertions_enabled": True,
                    "local_operations_passed": True,
                },
            )

        def process(command, environment, log, _timeout):
            node = {"head": 0, "worker": 1}[environment["GLOO_SOCKET_IFNAME"]]
            self.assertIn("torch.distributed.run", command)
            self.assertIn("--nnodes=2", command)
            self.assertIn("--nproc-per-node=2", command)
            self.assertIn(f"--node-rank={node}", command)
            destination = Path(command[command.index("--output") + 1])
            self.assertEqual(destination, output[node])
            started.append(node)
            Path(log).write_text(
                "CPU fixture: no actual torchrun or NPU was started.\n"
            )
            # Reproduce the probe's important boundary: passed rank files can
            # already exist when later process cleanup returns an error.
            for rank in range(node * 2, node * 2 + 2):
                (destination / f"cpu_gloo_rank{rank}.json").write_text(
                    json.dumps(rank_receipt(rank))
                )
            return 7 if failed_child == node else 0

        def invoke(node):
            try:
                results[node] = modules[node].main(
                    [
                        "--head",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--interface",
                        "head" if node == 0 else "worker",
                        "--timeout",
                        "30",
                        "--output",
                        str(output[node]),
                    ]
                )
            except BaseException as error:
                errors.append((node, error))

        with ExitStack() as stack:
            # Only the Linux admission guard is bypassed. Coordination uses
            # real TCP, including the unchanged connect_peer implementation.
            stack.enter_context(patch.object(sys, "platform", "linux"))
            for node, module in enumerate(modules):
                stack.enter_context(patch.object(module, "choose_network", choose))
                stack.enter_context(patch.object(module, "ipv4_interfaces", lambda: {}))
                stack.enter_context(patch.object(module, "discover_runtime", discover))
                stack.enter_context(patch.object(module, "run_process", process))
                pins = {name: "a" * 64 for name in module.SOURCE_FILES}
                if source_mismatch and node == 1:
                    pins[module.SOURCE_FILES[0]] = "b" * 64
                stack.enter_context(
                    patch.object(module, "source_pins", lambda _, p=pins: dict(p))
                )
                connect = module.connect_peer

                def bounded_connect(network, address, deadline, original=connect):
                    connection = original(
                        network, address, min(deadline, time.monotonic() + 5)
                    )
                    with connection_lock:
                        connections.append(connection)
                    return connection

                stack.enter_context(
                    patch.object(module, "connect_peer", bounded_connect)
                )

            threads = [
                threading.Thread(target=invoke, args=(node,), daemon=True)
                for node in (0, 1)
            ]
            try:
                for thread in threads:
                    thread.start()
                deadline = time.monotonic() + 10
                for thread in threads:
                    thread.join(timeout=max(0, deadline - time.monotonic()))
            finally:
                # Closing only these fixture sockets also unblocks a broken
                # completion exchange, so a regression leaves no test service.
                with connection_lock:
                    for connection in connections:
                        with suppress(OSError):
                            connection.shutdown(socket.SHUT_RDWR)
                        connection.close()
                for thread in threads:
                    thread.join(timeout=2)
            self.assertTrue(
                all(not thread.is_alive() for thread in threads),
                "paired fixture left a launcher thread running",
            )
            self.assertFalse(errors, errors)
        return results, output, started

    def test_both_endpoints_publish_completion_after_real_tcp_exchange(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, output, started = self.run_pair(Path(tmp))
            self.assertEqual(results, [0, 0])
            self.assertEqual(sorted(started), [0, 1])
            reports = [
                json.loads((path / "complete.json").read_text()) for path in output
            ]
            self.assertEqual(reports[0]["session"], reports[1]["session"])
            for path, report in zip(output, reports):
                self.assertIs(report["passed"], True)
                self.assertEqual(report["scope"], "mixed_910B_310P")
                self.assertEqual(
                    {
                        r["rank"]
                        for r in report["local_receipts"] + report["peer_receipts"]
                    },
                    {0, 1, 2, 3},
                )
                self.assertFalse((path / "failure.json").exists())

    def test_either_child_failure_blocks_both_completions_despite_passed_rank_files(
        self,
    ):
        for failed in (0, 1):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                results, output, started = self.run_pair(Path(tmp), failed_child=failed)
                self.assertTrue(all(code != 0 for code in results))
                self.assertEqual(sorted(started), [0, 1])
                for path in output:
                    self.assertFalse((path / "complete.json").exists())
                    self.assertTrue((path / "failure.json").exists())
                    self.assertEqual(len(list(path.glob("cpu_gloo_rank*.json"))), 2)
                process_exit = json.loads(
                    (output[failed] / "process_exit.json").read_text()
                )
                self.assertEqual(process_exit["returncode"], 7)

    def test_source_mismatch_prevents_both_probe_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, output, started = self.run_pair(Path(tmp), source_mismatch=True)
            self.assertTrue(all(code != 0 for code in results))
            self.assertEqual(started, [])
            for path in output:
                self.assertFalse((path / "complete.json").exists())
                self.assertFalse((path / "launch.json").exists())
                self.assertTrue((path / "failure.json").exists())


if __name__ == "__main__":
    unittest.main()
