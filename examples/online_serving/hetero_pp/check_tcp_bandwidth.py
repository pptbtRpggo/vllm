# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multiprocess, memory-to-memory TCP throughput; standard library only.

Run on A: python check_tcp_bandwidth.py server
Run on B: python check_tcp_bandwidth.py client --host <A_HOST_IP> \
--parallel 4 --warmup 3 --seconds 30
Swap roles to test the other direction. The server exits after one test.
Only the client specifies parallelism, warmup and measurement duration.
On Linux, add --interface to bind sockets to the local network interface.
All receivers use one interval on the server's monotonic clock. Whole recv calls
are counted by completion time (up to 1 MiB boundary granularity per stream).
This measures host TCP payload throughput, not NPU or full Gloo handoff speed.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
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


def nonnegative_seconds(value: str) -> float:
    """Allow zero warmup, otherwise use the same finite-duration bound."""
    return 0.0 if float(value) == 0 else positive_seconds(value)


def read_message(conn: socket.socket) -> dict:
    """Read one bounded JSON line without buffering subsequent socket data."""
    with conn.makefile("rb") as reply:
        return json.loads(reply.readline(65536))


def write_message(conn: socket.socket, message: dict) -> None:
    """Write one JSON control message."""
    conn.sendall(json.dumps(message).encode("ascii") + b"\n")


def receive_test(conn, window, barrier, start_event, timeout) -> dict:
    """Count recv completions in one shared server-clock interval, then drain."""
    # Spawn transfers the descriptor, not the Python socket's timeout state.
    conn.settimeout(timeout)
    buffer = bytearray(CHUNK_BYTES)
    total = measured = 0
    barrier.wait(timeout=timeout)
    if not start_event.wait(timeout):
        raise TimeoutError("receiver start timed out")
    conn.sendall(b"G")
    last_receive = 0.0
    while True:
        count = conn.recv_into(buffer)
        if count == 0:
            break
        total += count
        last_receive = time.perf_counter()
        if window[0] <= last_receive < window[1]:
            measured += count
    if measured == 0 or last_receive < window[1]:
        raise ValueError("stream ended early or received no data in the test window")
    write_message(conn, {"total_received_bytes": total})
    return {"bytes": measured, "total": total, "pid": os.getpid()}


def bind_interface(conn: socket.socket, interface) -> None:
    """Apply Linux SO_BINDTODEVICE before connecting or listening."""
    if interface is None:
        return
    option = getattr(socket, "SO_BINDTODEVICE", None)
    if option is None:
        raise ValueError("--interface requires Linux SO_BINDTODEVICE support")
    name = os.fsencode(interface)
    # Linux IFNAMSIZ is 16 bytes including the terminating NUL.
    if not 0 < len(name) < 16 or b"\0" in name:
        raise ValueError(f"invalid interface name: {interface!r}")
    try:
        conn.setsockopt(socket.SOL_SOCKET, option, name + b"\0")
    except OSError as error:
        raise OSError(
            error.errno, f"cannot bind to interface {interface!r}: {error.strerror}"
        ) from error


def connect(host, port, timeout, interface) -> socket.socket:
    """Connect using the requested interface or the system's default routing."""
    if interface is None:
        return socket.create_connection((host, port), timeout)
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        conn.settimeout(timeout)
        bind_interface(conn, interface)
        conn.connect((host, port))
    except BaseException:
        conn.close()
        raise
    return conn


def send_stream(host, port, duration, timeout, interface) -> dict:
    """Send one stream through warmup, measurement and a short drain margin."""
    buffer = bytes(CHUNK_BYTES)
    with connect(host, port, timeout, interface) as conn:
        conn.settimeout(timeout)
        if conn.recv(1) != b"G":
            raise ValueError("unexpected server greeting")
        deadline = time.perf_counter() + duration + 0.25
        total = 0
        while time.perf_counter() < deadline:
            conn.sendall(buffer)
            total += len(buffer)
        conn.shutdown(socket.SHUT_WR)
        if read_message(conn)["total_received_bytes"] != total:
            raise ValueError("receiver byte count does not match sent bytes")
        return {"total": total}


def worker(results, function, *args) -> None:
    """Report child errors to the parent; let the parent handle Ctrl+C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        results.put(function(*args))
    except Exception as error:
        results.put({"error": f"{type(error).__name__}: {error}"})
    finally:
        if function is receive_test:
            args[0].close()


def collect(processes, results, deadline) -> list:
    """Require a result and successful exit from every child by the deadline."""
    rows = []
    for _ in processes:
        try:
            row = results.get(timeout=max(0, deadline - time.perf_counter()))
        except queue.Empty as error:
            raise TimeoutError("transfer workers timed out") from error
        if "error" in row:
            raise ValueError(row["error"])
        rows.append(row)
    for process in processes:
        process.join(timeout=max(0, deadline - time.perf_counter()))
        if process.exitcode != 0:
            raise RuntimeError("transfer worker did not exit successfully")
    return rows


def stop_workers(processes, results) -> None:
    """Close only this invocation's children and result queue."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
    results.close()


def validate_settings(settings: dict) -> dict:
    """Validate both CLI and remote settings before starting child processes."""
    parallel = settings["parallel"]
    if type(parallel) is not int or not 1 <= parallel <= 32:
        raise ValueError("parallel must be an integer from 1 to 32")
    seconds = positive_seconds(str(settings["seconds"]))
    warmup = nonnegative_seconds(str(settings["warmup"]))
    return {"parallel": parallel, "seconds": seconds, "warmup": warmup}


def serve(args: argparse.Namespace) -> dict:
    """Accept control plus data connections; use one shared receiver window."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(args.timeout)
        bind_interface(listener, args.interface)
        listener.bind((args.bind, args.port))
        listener.listen(32)
        host, port = listener.getsockname()
        print(f"Listening on {host}:{port}", flush=True)
        control, _ = listener.accept()
        with control:
            control.settimeout(args.timeout)
            settings = validate_settings(read_message(control))
            duration = settings["warmup"] + settings["seconds"]
            deadline = time.perf_counter() + duration + args.timeout
            context = mp.get_context("spawn")
            window = context.Array("d", 2, lock=False)
            barrier = context.Barrier(settings["parallel"] + 1)
            start_event = context.Event()
            results = context.Queue()
            processes = []
            try:
                write_message(control, {"ready": True})
                for _ in range(settings["parallel"]):
                    listener.settimeout(max(0.001, deadline - time.perf_counter()))
                    conn, _ = listener.accept()
                    with conn:
                        conn.settimeout(args.timeout)
                        process = context.Process(
                            target=worker,
                            args=(
                                results,
                                receive_test,
                                conn,
                                window,
                                barrier,
                                start_event,
                                args.timeout,
                            ),
                        )
                        process.start()
                        processes.append(process)
                barrier.wait(timeout=args.timeout)
                window[0] = time.perf_counter() + settings["warmup"]
                window[1] = window[0] + settings["seconds"]
                start_event.set()
                rows = collect(processes, results, deadline)
                total = sum(row["bytes"] for row in rows)
                result = {
                    "parallel": settings["parallel"],
                    "warmup_seconds": settings["warmup"],
                    "seconds": settings["seconds"],
                    "received_bytes": total,
                    "total_received_bytes": sum(row["total"] for row in rows),
                    "gbps": total * 8 / settings["seconds"] / 1e9,
                    "MB_per_second": total / settings["seconds"] / 1e6,
                    "per_stream_bytes": [row["bytes"] for row in rows],
                    "receiver_processes": [row["pid"] for row in rows],
                }
                write_message(control, result)
                return result
            finally:
                stop_workers(processes, results)


def send_test(args: argparse.Namespace) -> dict:
    """Start independent sender processes and verify the aggregate receipt."""
    settings = validate_settings(vars(args))
    duration = args.warmup + args.seconds
    context = mp.get_context("spawn")
    results = context.Queue()
    processes = []
    try:
        with connect(args.host, args.port, args.timeout, args.interface) as control:
            control.settimeout(duration + args.timeout)
            write_message(control, settings)
            if read_message(control) != {"ready": True}:
                raise ValueError("unexpected server greeting")
            deadline = time.perf_counter() + duration + args.timeout
            for _ in range(args.parallel):
                process = context.Process(
                    target=worker,
                    args=(
                        results,
                        send_stream,
                        args.host,
                        args.port,
                        duration,
                        args.timeout,
                        args.interface,
                    ),
                )
                process.start()
                processes.append(process)
            rows = collect(processes, results, deadline)
            result = read_message(control)
            if result["total_received_bytes"] != sum(row["total"] for row in rows):
                raise ValueError("aggregate receiver byte count does not match")
            return result
    finally:
        stop_workers(processes, results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    server = modes.add_parser("server", help="receive one test, then exit")
    server.add_argument("--bind", default="0.0.0.0", help="local IPv4 address")
    client = modes.add_parser("client", help="send to the server")
    client.add_argument("--host", required=True, help="server host NIC IPv4/IP name")
    client.add_argument("--seconds", type=positive_seconds, default=10.0)
    client.add_argument("--warmup", type=nonnegative_seconds, default=3.0)
    client.add_argument("--parallel", type=int, default=4)
    for mode in (server, client):
        mode.add_argument(
            "-I", "--interface", help="bind all sockets to this local interface (Linux)"
        )
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
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        argparse.ArgumentTypeError,
    ) as error:
        print(f"Test failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Test interrupted", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
