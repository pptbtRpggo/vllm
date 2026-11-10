# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auto-detect two local Ascend devices and check the existing four-rank probe.

The bash entrypoint needs only a bootstrap Python. Torch and TorchNPU are
imported in bounded subprocesses, using each host's own installed environment.
No packages, system settings, model weights or running services are changed.
"""

import argparse
import hashlib
import ipaddress
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROC_ROOT = Path("/proc")
MARKER = "COMM_CHECK_RUNTIME="
MAX_MESSAGE = 64 * 1024
PROBE = "examples/online_serving/hetero_pp/verify_cpu_gloo.py"
SOURCE_FILES = (
    PROBE,
    "vllm/distributed/hetero_pp/channel.py",
    "vllm/distributed/device_communicators/cpu_staging.py",
    "examples/online_serving/hetero_pp/check_cpu_gloo.py",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def install_signal_handlers():
    def interrupted(number, _frame):
        raise KeyboardInterrupt(f"收到信号 {number}")

    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("超过本轮整体时间上限")
    return seconds


def prepare_environment(repo, environment, devices=None):
    """Keep the local SDK and visibility; ensure probe assertions are enabled."""
    env = dict(environment)
    env.pop("PYTHONOPTIMIZE", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(repo) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    libraries = env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    for directory in (
        "/usr/local/Ascend/driver/lib64/common",
        "/usr/local/Ascend/driver/lib64/driver",
        "/usr/local/Ascend/driver/lib64",
    ):
        if Path(directory).is_dir() and directory not in libraries:
            libraries.append(directory)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in libraries if p)
    if devices is not None:
        values = devices.split(",")
        if len(values) != 2 or not all(v.isascii() and v.isdigit() for v in values):
            raise ValueError("--devices 需要两个不同的设备号，例如 0,1")
        if int(values[0]) == int(values[1]):
            raise ValueError("--devices 的两个设备号不能相同")
        env["ASCEND_RT_VISIBLE_DEVICES"] = devices
    return env


def python_candidates(repo, explicit=None, environment=None):
    """Search bounded, existing locations; an explicit interpreter never falls back."""
    env = os.environ if environment is None else environment
    if explicit:
        path = shutil.which(explicit) or explicit
        if not Path(path).is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Python 不存在或不可执行：{explicit}")
        return [str(Path(path).absolute())]
    choices = []
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if env.get(key):
            choices.append(str(Path(env[key]) / "bin/python"))
    choices.extend([str(Path(repo) / ".venv/bin/python"), sys.executable])
    choices.extend(
        str(Path(root) / "bin/python")
        for root in ("/opt/conda", "/opt/venv", str(Path.home() / "miniconda3"))
    )
    for name in ("python", "python3"):
        location = shutil.which(name, path=env.get("PATH"))
        if location:
            choices.append(location)
    return list(
        dict.fromkeys(
            str(Path(p).absolute())
            for p in choices
            if Path(p).is_file() and os.access(p, os.X_OK)
        )
    )


def ipv4_interfaces():
    """Read interface ownership, including secondary IPs when iproute2 exists."""
    if shutil.which("ip"):
        result = subprocess.run(
            ["ip", "-j", "-4", "address", "show"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            try:
                return {
                    entry["ifname"]: [
                        item["local"]
                        for item in entry.get("addr_info", [])
                        if item.get("family") == "inet"
                    ]
                    for entry in json.loads(result.stdout)
                }
            except (KeyError, TypeError, ValueError):
                pass
    import fcntl

    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            try:
                value = fcntl.ioctl(
                    sock.fileno(), 0x8915, struct.pack("256s", name.encode()[:15])
                )
                found[name] = [socket.inet_ntoa(value[20:24])]
            except OSError:
                continue
    return found


def choose_network(head_ip, interfaces, route_ip, interface=None):
    """Choose head ownership before routing: self-routes must never select lo."""
    head = ipaddress.IPv4Address(head_ip)
    if head.is_loopback or head.is_unspecified or head.is_multicast:
        raise ValueError("--head 必须是一台机器实际网卡上的 IPv4，不能是回环地址")
    owners = [name for name, ips in interfaces.items() if head_ip in ips]
    rank = 0 if owners else 1
    if interface:
        if interface not in interfaces or not interfaces[interface]:
            raise ValueError(f"网卡没有可用的 IPv4：{interface}")
        chosen = interface
        if rank == 0 and head_ip not in interfaces[chosen]:
            raise ValueError("head IP 不属于指定网卡；请核对 --head/--interface")
    elif owners:
        if len(owners) != 1:
            raise ValueError("head IP 对应多个接口，请用 --interface 明确指定")
        chosen = owners[0]
    else:
        routes = [name for name, ips in interfaces.items() if route_ip in ips]
        if len(routes) != 1:
            raise ValueError("无法从到 head 的路由识别网卡，请指定 --interface")
        chosen = routes[0]
    address = (
        head_ip
        if rank == 0
        else route_ip
        if route_ip in interfaces[chosen]
        else interfaces[chosen][0]
    )
    if chosen == "lo" or ipaddress.IPv4Address(address).is_loopback:
        raise ValueError("两机通信不能使用 lo/回环地址")
    return {
        "head_ip": head_ip,
        "node_rank": rank,
        "node_ip": address,
        "interface": chosen,
    }


def source_pins(repo):
    return {
        name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def inspect_runtime(repo):
    """Exercise the probe's local operations before attributing errors to Gloo."""
    if sys.flags.optimize:
        raise RuntimeError("Python 优化关闭了 assert；请移除 -O/PYTHONOPTIMIZE")
    import inspect

    import torch
    import torch.distributed as dist
    import torch_npu

    import vllm
    from vllm.distributed.device_communicators.cpu_staging import CpuStagingTransport
    from vllm.distributed.hetero_pp import channel

    if Path(channel.__file__).resolve() != Path(repo) / (
        "vllm/distributed/hetero_pp/channel.py"
    ):
        raise RuntimeError("导入的 channel 不来自本轮项目，检查 PYTHONPATH/源码构建")
    if Path(inspect.getfile(CpuStagingTransport)).resolve() != Path(repo) / (
        "vllm/distributed/device_communicators/cpu_staging.py"
    ):
        raise RuntimeError("实际导入的 CPU staging transport 不来自本轮项目")
    if not dist.is_gloo_available():
        raise RuntimeError("当前 PyTorch 没有 Gloo 支持")
    count = torch.npu.device_count()
    if count != 2:
        raise RuntimeError(
            f"当前可见 NPU 为 {count} 张，本入口需要两张；"
            "多卡机器用 --devices 指定两张，少于两张时检查容器映射/设备权限"
        )
    devices = []
    for index in range(count):
        torch.npu.set_device(index)
        name = torch.npu.get_device_name(index)
        lower = name.lower()
        family = next(
            (
                soc
                for soc in ("910B", "310P")
                if lower.startswith("ascend" + soc.lower())
            ),
            None,
        )
        suffix = lower[len("ascend") + len(family) :] if family else ""
        if family is None or suffix and not (suffix.isascii() and suffix.isdigit()):
            raise RuntimeError(f"不支持的本机 SoC：{name}")
        soc = family
        base = (torch.arange(128).reshape(8, 16) % 101).half().T
        value = base.to("npu")
        residual = value * 2
        value += 1
        ids = torch.arange(8, dtype=torch.int64, device="npu")
        token = torch.tensor(17, device="npu")
        empty = torch.empty((0, 3), device="npu")
        module = torch.get_device_module("npu")
        event = module.Event()
        event.record(module.current_stream())
        event.synchronize()
        equal = (
            torch.equal(value.cpu(), base + 1)
            and torch.equal(residual.cpu(), base * 2)
            and torch.equal(ids.cpu(), torch.arange(8))
            and token.item() == 17
            and empty.shape == (0, 3)
        )
        value.fill_(-99)
        torch.npu.synchronize()
        if not equal or not torch.equal(value.cpu(), torch.full_like(base, -99)):
            raise RuntimeError(f"本地 NPU 算子/复制预检失败：逻辑设备 {index}")
        devices.append({"logical_id": index, "name": name, "soc": soc})
    families = {device["soc"] for device in devices}
    if len(families) != 1:
        raise RuntimeError("首版入口要求每台机器的两张卡属于同一 SoC 系列")
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "vllm": vllm.__version__,
        "vllm_source": vllm.__file__,
        "byteorder": sys.byteorder,
        "soc": devices[0]["soc"],
        "devices": devices,
        "device_identity_scope": "runtime_logical_devices",
        "assertions_enabled": True,
        "local_operations_passed": True,
    }


def process_table():
    """Read Linux PID ownership and birth times without invoking system ps."""
    result = {}
    directory = PROC_ROOT
    if directory.is_dir():
        for item in directory.iterdir():
            if not item.name.isdigit():
                continue
            try:
                text = (item / "stat").read_text(errors="replace")
                fields = text[text.rfind(")") + 2 :].split()
                result[int(item.name)] = (
                    int(fields[1]),
                    int(fields[2]),
                    int(fields[19]),
                    fields[0],
                )
            except (OSError, ValueError, IndexError):
                continue
    return result


def owned_family(root_pid):
    """Include descendants that start their own sessions, as torchrun workers do."""
    table = process_table()
    family = {root_pid: table[root_pid]} if root_pid in table else {}
    while True:
        found = {
            pid: row
            for pid, row in table.items()
            if pid not in family and row[0] in family
        }
        if not found:
            return family
        family.update(found)


def signal_family(family, number, skip_group=None):
    table = process_table()
    for pid, original in family.items():
        current = table.get(pid)
        if (
            current is not None
            and current[2] == original[2]
            and current[3] not in ("Z", "X")
            and current[1] != skip_group
        ):
            with suppress(ProcessLookupError):
                os.kill(pid, number)


def remember_family(family, root_pid):
    snapshot = owned_family(root_pid)
    expected = family.get(root_pid)
    actual = snapshot.get(root_pid)
    if expected is None or actual is None or actual[2] != expected[2]:
        return
    for pid, row in snapshot.items():
        family.setdefault(pid, row)


def stop_process(process, known=None):
    """Clean up the owned tree, including torchrun's separate worker sessions."""
    family = dict(known or {})
    if known is None and process.returncode is None:
        family = owned_family(process.pid)
    elif family:
        remember_family(family, process.pid)
    table = process_table()
    group_owned = any(
        pid in table
        and table[pid][2] == row[2]
        and table[pid][1] == process.pid
        and table[pid][3] not in ("Z", "X")
        for pid, row in family.items()
    )
    if not PROC_ROOT.is_dir() and process.returncode is None:
        group_owned = True  # CPU-only subprocess checks on non-Linux hosts.
    if group_owned:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    signal_family(family, signal.SIGTERM, skip_group=process.pid)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
    # PID birth-time checks prevent killing a PID reused by an unrelated job.
    signal_family(family, signal.SIGKILL)


def capture(command, env, timeout):
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    family = owned_family(process.pid)
    try:
        text, _ = process.communicate(timeout=timeout)
        return process.returncode, text
    except BaseException:
        stop_process(process, family)
        raise


def run_process(command, env, log, timeout):
    """Keep raw output and preserve exit status; bound the whole child session."""
    started = last_report = time.monotonic()
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        family = owned_family(process.pid)
        try:
            while process.poll() is None:
                remember_family(family, process.pid)
                now = time.monotonic()
                if now - started >= timeout:
                    raise TimeoutError("通信测试超过时间上限；只停止本轮自建进程组")
                if now - last_report >= 15:
                    print(
                        f"通信测试仍在运行，已 {now - started:.0f}s；日志：{log}",
                        flush=True,
                    )
                    last_report = now
                time.sleep(min(0.2, max(0, timeout - (now - started))))
            table = process_table()
            if any(
                pid in table
                and table[pid][2] == row[2]
                and table[pid][3] not in ("Z", "X")
                for pid, row in family.items()
            ):
                raise RuntimeError("主进程已退出但本轮 worker 尚未退出，拒绝报告通过")
            return process.returncode
        except BaseException:
            stop_process(process, family)
            raise


def cann_scripts(environment, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("--cann-env 必须是现有 set_env.sh 文件")
        return [path]
    paths = []
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if environment.get(variable):
            paths.append(Path(environment[variable]) / "set_env.sh")
    paths.extend(
        Path(root) / suffix
        for root in ("/usr/local/Ascend", str(Path.home() / "Ascend"))
        for suffix in (
            "ascend-toolkit/set_env.sh",
            "ascend-toolkit/latest/set_env.sh",
        )
    )
    return list(dict.fromkeys(p.resolve() for p in paths if p.is_file()))[:1]


def load_sdk_environment(script, environment, deadline):
    """Source one existing SDK script in a child; copy only SDK environment keys."""
    code = (
        "import json, os; print(json.dumps({k:v for k,v in os.environ.items() "
        "if k in ('PATH','LD_LIBRARY_PATH','PYTHONPATH','LD_PRELOAD') "
        "or k.startswith(('ASCEND_','ATB_','CANN_','TBE_','TORCH_NPU_'))}))"
    )
    rc, output = capture(
        [
            "/bin/bash",
            "-c",
            'set -e; set +u; source "$1" >&2; exec "$2" -c "$3"',
            "comm-check-sdk",
            str(script),
            sys.executable,
            code,
        ],
        environment,
        min(20, remaining(deadline)),
    )
    if rc:
        raise RuntimeError(f"加载本机 SDK 环境失败：{script}\n{output[-4000:]}")
    try:
        values = json.loads(output.splitlines()[-1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"无法读取 SDK 环境：{script}") from error
    return dict(environment, **values)


def discover_runtime(args, output, deadline):
    original = prepare_environment(ROOT, os.environ, args.devices)
    contexts = [("current", original)]
    scripts = cann_scripts(original, args.cann_env)
    if args.cann_env:
        contexts = []
    errors = []
    for context_index in range(len(contexts) + len(scripts)):
        if context_index < len(contexts):
            label, base_env = contexts[context_index]
        else:
            script = scripts[context_index - len(contexts)]
            label = str(script)
            base_env = load_sdk_environment(script, original, deadline)
        for python in python_candidates(ROOT, args.python, base_env):
            env = prepare_environment(ROOT, base_env, args.devices)
            print(f"检查本机环境：{python}；SDK={label}", flush=True)
            try:
                site_rc, site_text = capture(
                    [
                        python,
                        "-c",
                        "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                    ],
                    env,
                    min(10, remaining(deadline)),
                )
                plugin = Path(site_text.strip()) / "vllm_ascend"
                if site_rc == 0 and plugin.is_dir():
                    env["LD_LIBRARY_PATH"] = (
                        str(plugin) + os.pathsep + env["LD_LIBRARY_PATH"]
                    )
                rc, text = capture(
                    [
                        python,
                        str(Path(__file__).resolve()),
                        "--inspect-runtime",
                        str(ROOT),
                    ],
                    env,
                    min(90, remaining(deadline)),
                )
                atb = Path("/usr/local/Ascend/nnal/atb/set_env.sh")
                if rc and "libatb.so" in text and atb.is_file():
                    env = prepare_environment(
                        ROOT, load_sdk_environment(atb, env, deadline), args.devices
                    )
                    rc, text = capture(
                        [
                            python,
                            str(Path(__file__).resolve()),
                            "--inspect-runtime",
                            str(ROOT),
                        ],
                        env,
                        min(90, remaining(deadline)),
                    )
                    label = label + " + " + str(atb)
            except (OSError, subprocess.TimeoutExpired) as error:
                rc, text = 1, f"{type(error).__name__}: {error}"
            with (output / "environment_attempts.log").open("a") as stream:
                stream.write(f"Python: {python}\nSDK: {label}\nexit: {rc}\n{text}\n")
            records = [
                line[len(MARKER) :]
                for line in text.splitlines()
                if line.startswith(MARKER)
            ]
            if rc == 0 and records:
                info = json.loads(records[-1])
                info["sdk_environment"] = label
                info["visibility"] = env.get(
                    "ASCEND_RT_VISIBLE_DEVICES", "runtime-default"
                )
                return python, env, info
            errors.append(f"{python} ({label}): {text[-1500:]}")
    raise RuntimeError(
        "没有找到能完成本地 NPU 预检的环境。"
        "可用 --python/--cann-env/--devices 明确指定；"
        "不会安装依赖。详细记录见 environment_attempts.log。\n"
        + "\n".join(errors)[-5000:]
    )


def send_message(connection, value):
    data = json.dumps(value, ensure_ascii=False).encode("utf-8")
    if not 0 < len(data) <= MAX_MESSAGE:
        raise ValueError("对端检查消息超过 64 KiB 限制")
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive_message(connection, deadline=None):
    def exact(count):
        pieces = bytearray()
        while len(pieces) < count:
            if deadline is not None:
                connection.settimeout(remaining(deadline))
            chunk = connection.recv(count - len(pieces))
            if not chunk:
                raise ConnectionError("对端在检查完成前断开连接")
            pieces.extend(chunk)
        return bytes(pieces)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("对端检查消息长度不合法")
    value = json.loads(exact(size))
    if not isinstance(value, dict):
        raise ValueError("对端检查消息必须为 JSON object")
    return value


def validate_peer(local, peer):
    if (
        type(peer.get("node_rank")) is not int
        or peer["node_rank"] != 1 - local["node_rank"]
    ):
        raise ValueError("两端 node-rank 不互补，检查 head IP")
    if not peer.get("node_ip") or peer["node_ip"] == local["node_ip"]:
        raise ValueError("两端必须使用不同的本机 IPv4")
    if peer.get("soc") not in ("910B", "310P"):
        raise ValueError("对端没有识别到受支持的 NPU SoC")
    pins = local.get("sources")
    if (
        not isinstance(pins, dict)
        or len(pins) < 3
        or not all(
            isinstance(v, str)
            and len(v) == 64
            and all(char in "0123456789abcdef" for char in v)
            for v in pins.values()
        )
        or peer.get("sources") != pins
    ):
        raise ValueError("两端通信源码 SHA 不一致，请同步同一份项目")
    if peer.get("byteorder") != local["byteorder"]:
        raise ValueError("两端 CPU 字节序不同，当前原始字节传输不能直接混用")
    if (
        not isinstance(peer.get("torch"), str)
        or peer["torch"].split("+", 1)[0] != local["torch"].split("+", 1)[0]
    ):
        raise ValueError("两端 Torch 基础版本不同，请先准备匹配环境")


def connect_peer(network, port, deadline):
    print(f"等待另一台机器运行相同命令；协调端口 {port}……", flush=True)
    if network["node_rank"] == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((network["head_ip"], port))
            listener.listen(1)
            listener.settimeout(remaining(deadline))
            connection, _ = listener.accept()
        return connection
    while True:
        try:
            return socket.create_connection(
                (network["head_ip"], port), timeout=min(3, remaining(deadline))
            )
        except OSError:
            time.sleep(min(0.5, remaining(deadline)))


def validate_receipts(output, node_rank):
    expected = {2 * node_rank, 2 * node_rank + 1}
    files = list(Path(output).glob("cpu_gloo_rank*.json"))
    if {p.name for p in files} != {f"cpu_gloo_rank{rank}.json" for rank in expected}:
        raise ValueError("本轮 rank 回执缺失或混入其他 rank，拒绝报告通过")
    results = []
    for path in sorted(files):
        data = json.loads(path.read_text())
        rank = data.get("rank")
        if (
            type(rank) is not int
            or rank not in expected
            or path.name != f"cpu_gloo_rank{rank}.json"
            or data.get("local_rank") != rank % 2
            or data.get("passed") is not True
            or data.get("backend") != "gloo"
            or data.get("destination_device") != "npu"
        ):
            raise ValueError(f"rank 回执身份或通过状态不正确：{path}")
        checks = data.get("checks", [])
        if (
            not isinstance(checks, list)
            or len(checks) != 4
            or not all(isinstance(item, dict) for item in checks)
            or [item.get("M") for item in checks] != [1, 64, 256, 2048]
            or any(item.get("all_values_equal") is not True for item in checks)
        ):
            raise ValueError(f"四个 tensor 内容检查未全部通过：{path}")
        calls = data.get("wire_calls", [])
        if (
            not isinstance(calls, list)
            or not calls
            or any(
                not isinstance(call, dict)
                or call.get("device") != "cpu"
                or call.get("backend") != "gloo"
                or call.get("operation") not in ("isend", "irecv")
                or type(call.get("peer")) is not int
                or call["peer"]
                != (rank + (1 if call["operation"] == "isend" else -1)) % 4
                or type(call.get("tag")) is not int
                or not 0 <= call["tag"] < 2**31
                or type(call.get("bytes")) is not int
                or call["bytes"] <= 0
                or call.get("dtype") != "torch.uint8"
                for call in calls
            )
        ):
            raise ValueError(f"真实载荷未全部走 CPU/Gloo：{path}")
        results.append(
            {
                "rank": rank,
                "passed": True,
                "checks": 4,
                "wire_calls": len(calls),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return results


def argument_parser():
    parser = argparse.ArgumentParser(
        description="两机各两卡通用检查：自动识别本机环境、设备及 CPU/Gloo 网卡",
    )
    parser.add_argument("--head", help="其中一台机器实际网卡上的 IPv4；两端填同一个")
    parser.add_argument("--python", help="指定本机已有 NPU Python 环境")
    parser.add_argument("--cann-env", help="指定本机 CANN set_env.sh")
    parser.add_argument("--interface", help="指定本机 Gloo 网卡，默认自动识别")
    parser.add_argument("--devices", help="本机两张卡的可见列表，例如 0,1")
    parser.add_argument(
        "--port", type=int, default=19090, help="torchrun 端口；协调另用 port+1"
    )
    parser.add_argument(
        "--timeout", type=float, default=600, help="整轮上限秒数，默认 600"
    )
    parser.add_argument("--output", type=Path, help="本机全新日志目录，默认自动创建")
    parser.add_argument("--inspect-runtime", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.inspect_runtime:
        print(MARKER + json.dumps(inspect_runtime(args.inspect_runtime)), flush=True)
        return 0
    if not args.head:
        parser.error("两端都需要 --head <其中一台机器的本地 IPv4>")
    if not sys.platform.startswith("linux"):
        parser.error("本入口用于 Linux Ascend 环境")
    if not 1024 <= args.port <= 65534 or not 30 <= args.timeout <= 3600:
        parser.error("端口需在 1024..65534，整体 timeout 需在 30..3600 秒")
    deadline = time.monotonic() + args.timeout
    output = args.output or ROOT / "output" / (
        "comm_check_" + time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    )
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    connection = None
    sent_finish = False
    phase = "network_discovery"
    print(f"本轮本机日志目录：{output}", flush=True)
    try:
        args.head = str(ipaddress.IPv4Address(args.head))
        interfaces = ipv4_interfaces()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((args.head, args.port))
            route_ip = route.getsockname()[0]
        network = choose_network(
            args.head,
            interfaces,
            route_ip,
            args.interface or os.environ.get("GLOO_SOCKET_IFNAME"),
        )
        print(
            f"本机 IP={network['node_ip']}，node-rank={network['node_rank']}，"
            f"网卡={network['interface']}",
            flush=True,
        )
        phase = "local_npu_preflight"
        python, env, inventory = discover_runtime(args, output, deadline)
        env["GLOO_SOCKET_IFNAME"] = network["interface"]
        local = dict(
            inventory,
            **network,
            sources=source_pins(ROOT),
            session=uuid.uuid4().hex,
            protocol=1,
        )
        write_json(output / "local_preflight.json", local)
        print(f"本机 {local['soc']} 两卡预检通过；Python={python}", flush=True)
        phase = "peer_preflight"
        connection = connect_peer(network, args.port + 1, deadline)
        connection.settimeout(remaining(deadline))
        send_message(connection, local)
        peer = receive_message(connection, deadline)
        validate_peer(local, peer)
        if peer.get("protocol") != 1:
            raise ValueError("对端检查协议版本不一致")
        session = local["session"] if network["node_rank"] == 0 else peer["session"]
        write_json(output / "peer_preflight.json", peer)
        print(f"对端 {peer['node_ip']} / {peer['soc']}，通信源码一致。", flush=True)
        phase = "communication_probe"
        command = [
            python,
            "-m",
            "torch.distributed.run",
            "--nnodes=2",
            "--nproc-per-node=2",
            f"--node-rank={network['node_rank']}",
            f"--master-addr={args.head}",
            f"--master-port={args.port}",
            "--rdzv-conf=timeout=90",
            "--max-restarts=0",
            str(ROOT / PROBE),
            "--output",
            str(output),
        ]
        write_json(
            output / "launch.json",
            {"command": command, "session": session, "overall_timeout_s": args.timeout},
        )
        print("开始四 rank NPU→CPU→Gloo→CPU→NPU 内容检查……", flush=True)
        returncode = run_process(
            command, env, output / "probe.log", remaining(deadline)
        )
        write_json(output / "process_exit.json", {"returncode": returncode})
        if returncode != 0:
            raise RuntimeError(f"本机 torchrun 退出码 {returncode}；见 probe.log")
        phase = "receipt_validation"
        receipts = validate_receipts(output, network["node_rank"])
        if source_pins(ROOT) != local["sources"]:
            raise RuntimeError("测试期间通信源码发生变化，保留日志但拒绝报告通过")
        connection.settimeout(remaining(deadline))
        send_message(
            connection,
            {
                "phase": "finished",
                "passed": True,
                "session": session,
                "receipts": receipts,
            },
        )
        sent_finish = True
        finish = receive_message(connection, deadline)
        peer_ranks = {2 * peer["node_rank"], 2 * peer["node_rank"] + 1}
        if (
            finish.get("phase") != "finished"
            or finish.get("passed") is not True
            or finish.get("session") != session
            or {item.get("rank") for item in finish.get("receipts", [])} != peer_ranks
        ):
            raise RuntimeError("对端未确认本轮两 rank 正常退出并通过，不能报告整体通过")
        write_json(output / "peer_completion.json", finish)
        result = {
            "passed": True,
            "session": session,
            "scope": "mixed_910B_310P"
            if local["soc"] != peer["soc"]
            else "homogeneous_control",
            "local": local,
            "peer": peer,
            "local_receipts": receipts,
            "peer_receipts": finish["receipts"],
            "bandwidth_measured": False,
            "model_inference_validated": False,
        }
        write_json(output / "complete.json", result)
        print(
            f"两端四 rank 通信功能检查通过：{local['soc']} ↔ {peer['soc']}。\n"
            f"完整回执：{output / 'complete.json'}\n"
            "此结果为内容/完成依赖检查，未测带宽或模型推理。",
            flush=True,
        )
        return 0
    except BaseException as error:
        write_json(
            output / "failure.json",
            {"phase": phase, "error": str(error), "type": type(error).__name__},
        )
        if connection is not None and not sent_finish:
            with suppress(Exception):
                connection.settimeout(2)
                send_message(
                    connection,
                    {"phase": "finished", "passed": False, "error": str(error)},
                )
        print(
            f"检查未通过（{phase}）：{error}\n原始日志保留在：{output}", file=sys.stderr
        )
        return (
            130
            if isinstance(error, KeyboardInterrupt)
            else 124
            if isinstance(error, TimeoutError)
            else 1
        )
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    install_signal_handlers()
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
