# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the standalone TCP meter without vLLM or NPU imports."""

import json
import socket
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/check_tcp_bandwidth.py"
)


class TCPBandwidthTest(unittest.TestCase):
    def run_transfer(self, parallel, warmup):
        # Catches sender-only timing and truncated/unacknowledged transfers.
        self.assertTrue(SCRIPT.exists(), "TCP meter has not been implemented")
        server = subprocess.Popen(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "server",
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
                "--timeout",
                "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert server.stdout is not None
            ready = server.stdout.readline().strip()
            self.assertTrue(ready.startswith("Listening on 127.0.0.1:"), ready)
            port = ready.rsplit(":", 1)[1]
            client = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    str(SCRIPT),
                    "client",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    port,
                    "--seconds",
                    "0.3",
                    "--parallel",
                    str(parallel),
                    "--warmup",
                    str(warmup),
                    "--timeout",
                    "5",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output, error = server.communicate(timeout=10)
            self.assertEqual(client.returncode, 0, client.stderr + error)
            self.assertEqual(server.returncode, 0, error)
            received = json.loads(output)
            self.assertEqual(received, json.loads(client.stdout))
            self.assertGreater(received["received_bytes"], 0)
            self.assertGreater(received["seconds"], 0)
            self.assertEqual(received["parallel"], parallel)
            self.assertEqual(received["warmup_seconds"], warmup)
            self.assertEqual(received["seconds"], 0.3)
            self.assertEqual(len(set(received["receiver_processes"])), parallel)
            self.assertTrue(all(n > 0 for n in received["per_stream_bytes"]))
            self.assertEqual(
                sum(received["per_stream_bytes"]), received["received_bytes"]
            )
            self.assertGreater(
                received["total_received_bytes"], received["received_bytes"]
            )
            self.assertAlmostEqual(
                received["gbps"],
                received["received_bytes"] * 8 / received["seconds"] / 1e9,
                places=6,
            )
        finally:
            if server.poll() is None:
                server.kill()
            server.communicate(timeout=5)

    def test_receiver_acknowledges_all_bytes_and_reports_same_result(self):
        self.run_transfer(parallel=1, warmup=0)

    def test_parallel_receivers_share_one_window_and_exclude_warmup(self):
        self.run_transfer(parallel=2, warmup=0.2)

    def test_four_independent_streams(self):
        self.run_transfer(parallel=4, warmup=0.1)

    def test_early_disconnect_is_not_reported_as_a_successful_measurement(self):
        server = subprocess.Popen(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "server",
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
                "--timeout",
                "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert server.stdout is not None
            port = int(server.stdout.readline().strip().rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", port), 5) as control:
                settings = {"parallel": 1, "warmup": 0, "seconds": 2}
                control.sendall(json.dumps(settings).encode() + b"\n")
                with control.makefile("rb") as reply:
                    self.assertEqual(json.loads(reply.readline()), {"ready": True})
                    with socket.create_connection(("127.0.0.1", port), 5) as data:
                        self.assertEqual(data.recv(1), b"G")
                        data.sendall(b"short transfer")
                        data.shutdown(socket.SHUT_WR)
                    self.assertEqual(reply.readline(), b"")
            output, error = server.communicate(timeout=10)
            self.assertEqual(server.returncode, 1, error)
            self.assertIn("stream ended early", error)
            self.assertNotIn("gbps", output)
        finally:
            if server.poll() is None:
                server.kill()
            server.communicate(timeout=5)

    def test_rejects_invalid_duration_before_connecting(self):
        self.assertTrue(SCRIPT.exists(), "TCP meter has not been implemented")
        for seconds in ("0", "-1", "nan", "inf", "301"):
            with self.subTest(seconds=seconds):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-S",
                        str(SCRIPT),
                        "client",
                        "--host",
                        "127.0.0.1",
                        "--seconds",
                        seconds,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stderr)

    def test_server_wait_is_bounded(self):
        self.assertTrue(SCRIPT.exists(), "TCP meter has not been implemented")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "server",
                "--bind",
                "127.0.0.1",
                "--port",
                "0",
                "--timeout",
                "0.1",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("timed out", result.stderr)
        self.assertNotIn("received_bytes", result.stdout)


if __name__ == "__main__":
    unittest.main()
