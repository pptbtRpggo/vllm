# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-connection, memory-to-memory TCP throughput; standard library only.

Run on A: python check_tcp_bandwidth.py server
Run on B: python check_tcp_bandwidth.py client --host <A_HOST_IP>
Swap roles to test the other direction. The server exits after one test.
This measures host TCP payload throughput, not NPU or full Gloo handoff speed.
"""

import argparse
import json
import math
import socket
import sys
import time

CHUNK_BYTES = 1024 * 1024


def positive_seconds(value: str) -> float:
    """Parse a finite duration between zero (exclusive) and 300 seconds."""
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 300:
        raise argparse.ArgumentTypeError("must be > 0 and <= 300 seconds")
    return number


def receive_test(conn: socket.socket) -> dict:
    """Count received payload bytes and acknowledge EOF with receiver timing."""
    buffer = bytearray(CHUNK_BYTES)
    total = 0
    conn.sendall(b"GO\n")
    start = time.perf_counter()
    while True:
        count = conn.recv_into(buffer)
        if count == 0:
            break
        total += count
    elapsed = time.perf_counter() - start
    if total == 0:
        raise ValueError("no payload received")
    result = {
        "received_bytes": total,
        "seconds": elapsed,
        "gbps": total * 8 / elapsed / 1e9,
        "MB_per_second": total / elapsed / 1e6,
    }
    conn.sendall(json.dumps(result).encode("ascii") + b"\n")
    return result


def serve(args: argparse.Namespace) -> dict:
    """Accept one IPv4 client, measure its transfer, then close the listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(args.timeout)
        listener.bind((args.bind, args.port))
        listener.listen(1)
        host, port = listener.getsockname()
        print(f"Listening on {host}:{port}", flush=True)
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(args.timeout)
            return receive_test(conn)


def send_test(args: argparse.Namespace) -> dict:
    """Send until the deadline; require an exact receiver byte-count receipt."""
    buffer = bytes(CHUNK_BYTES)
    with socket.create_connection((args.host, args.port), args.timeout) as conn:
        conn.settimeout(args.timeout)
        with conn.makefile("rb") as reply:
            if reply.readline(64) != b"GO\n":
                raise ValueError("unexpected server greeting")
            deadline = time.perf_counter() + args.seconds
            total = 0
            while time.perf_counter() < deadline:
                conn.sendall(buffer)
                total += len(buffer)
            conn.shutdown(socket.SHUT_WR)
            result = json.loads(reply.readline(4096))
            if result["received_bytes"] != total:
                raise ValueError("receiver byte count does not match sent bytes")
            return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    server = modes.add_parser("server", help="receive one test, then exit")
    server.add_argument("--bind", default="0.0.0.0", help="local IPv4 address")
    client = modes.add_parser("client", help="send to the server")
    client.add_argument("--host", required=True, help="server host NIC IPv4/IP name")
    client.add_argument("--seconds", type=positive_seconds, default=10.0)
    for mode in (server, client):
        mode.add_argument("--port", type=int, default=5201)
        mode.add_argument(
            "--timeout",
            type=positive_seconds,
            default=120.0,
            help="accept/connect/socket inactivity timeout in seconds (default: 120)",
        )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or (args.mode == "client" and args.port == 0):
        parser.error("port must be 1..65535 (server also accepts 0 for a free port)")
    try:
        result = serve(args) if args.mode == "server" else send_test(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Test failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Test interrupted", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
