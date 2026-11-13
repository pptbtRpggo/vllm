# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start the existing mixed-Ascend PP service from the same CLI on both nodes.

Run inside each already working container. No Docker, package installation,
device remapping or global Ray stop is performed. Explicit profile mode runs
the finite internal collection workload instead of an HTTP service.
"""

import argparse
import ipaddress
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path

from check_cpu_gloo import (
    ROOT,
    capture,
    choose_network,
    discover_runtime,
    install_signal_handlers,
    ipv4_interfaces,
    owned_family,
    prepare_environment,
    remember_family,
    stop_process,
    write_json,
)


def argument_parser():
    parser = argparse.ArgumentParser(
        description="两机各两卡异构 PP：910B 自动启动模型，310P 加入 Ray"
    )
    parser.add_argument("--head", required=True, help="910B 端 IPv4；两边填一样")
    parser.add_argument("--peer", required=True, help="310P 端 IPv4；两边填一样")
    parser.add_argument(
        "--model", required=True, type=Path, help="两端同路径的本地完整模型"
    )
    parser.add_argument(
        "--layers", default="18,18,6,6", help="四 stage 层数，总和须等于模型层数"
    )
    parser.add_argument("--python", help="已有 NPU Python，默认沿用环境探测")
    parser.add_argument("--cann-env", help="可选，本机已有 CANN set_env.sh")
    parser.add_argument(
        "--chat-template",
        help="codellama-instruct 或模板文件路径；省略时使用 tokenizer 自带模板",
    )
    parser.add_argument(
        "--devices", help="可选，当前 runtime 的两个逻辑卡号；默认不更改可见性"
    )
    parser.add_argument("--interface", help="本机 Gloo 网卡，默认自动选择")
    parser.add_argument(
        "--node-ip", help="可选，明确本机 IP；dry-run 时可用于预览任一角色"
    )
    parser.add_argument("--ray-port", type=int, default=16379)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument(
        "--startup-timeout", type=float, default=1200, help="每项启动/等节点上限秒数"
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--block-size", type=int, help="可选，显式 KV block token 数")
    parser.add_argument(
        "--num-gpu-blocks-override", type=int, help="可选，固定 KV block 数"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="连接 Ray 后采集并生成 DP 推荐，不启动 API",
    )
    parser.add_argument(
        "--trace-root", type=Path, help="profiling 两端真正共享的结果根目录"
    )
    parser.add_argument(
        "--profile-timeout",
        type=float,
        default=3600,
        help="profiling 全流程上限秒数，默认 3600",
    )
    parser.add_argument(
        "--output", type=Path, help="全新日志目录；默认 output/hetero_pp_start_*"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只预览，不导入 Torch/Ray 或启动服务"
    )
    return parser


def validate_options(args):
    if args.profile:
        if args.trace_root is None or not args.trace_root.is_absolute():
            raise ValueError("--profile 需要 --trace-root 共享目录的绝对路径")
        if not args.trace_root.is_dir():
            raise ValueError("--trace-root 必须是已有共享目录；不会自动建立共享存储")
    elif args.trace_root is not None:
        raise ValueError("--trace-root 仅用于 --profile")
    if (
        not math.isfinite(args.profile_timeout)
        or not 30 <= args.profile_timeout <= 86400
    ):
        raise ValueError("profile-timeout 需在 30..86400 秒")
    for value in (args.block_size, args.num_gpu_blocks_override):
        if value is not None and value <= 0:
            raise ValueError("KV block 参数必须为正整数")
    for name in ("head", "peer"):
        address = ipaddress.IPv4Address(getattr(args, name))
        if address.is_loopback or address.is_multicast or address.is_unspecified:
            raise ValueError("head/peer 必须是两端互通的非回环 IPv4")
        setattr(args, name, str(address))
    if args.head == args.peer:
        raise ValueError("head 和 peer 不能相同")
    if args.node_ip and args.node_ip not in (args.head, args.peer):
        raise ValueError("本机 IP 必须是 head 或 peer")
    if (
        not 1024 <= args.ray_port <= 65535
        or not 1024 <= args.api_port <= 65535
        or args.ray_port == args.api_port
    ):
        raise ValueError("Ray/API 端口需不同且在 1024..65535")
    if (
        not math.isfinite(args.startup_timeout)
        or not 30 <= args.startup_timeout <= 1800
    ):
        raise ValueError("startup-timeout 需在 30..1800 秒")
    if min(args.max_model_len, args.max_num_batched_tokens, args.max_num_seqs) < 1:
        raise ValueError("上下文、token budget、并发上限必须为正整数")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization 需在 0 和 1 之间")
    args.model = args.model.expanduser().resolve(strict=True)
    config = json.loads((args.model / "config.json").read_text())
    if config.get("architectures") not in (["LlamaForCausalLM"], ["Qwen2ForCausalLM"]):
        raise ValueError("当前入口只支持 LlamaForCausalLM/Qwen2ForCausalLM")
    counts = args.layers.split(",")
    if len(counts) != 4 or not all(
        s.isascii() and s.isdigit() and int(s) > 0 for s in counts
    ):
        raise ValueError("layers 需为四个正整数，例如 18,18,6,6")
    if sum(map(int, counts)) != config.get("num_hidden_layers"):
        raise ValueError("四个 stage 层数之和必须等于模型 num_hidden_layers")


def select_network(head, peer, interfaces, route_ip, interface=None):
    network = choose_network(head, interfaces, route_ip, interface)
    expected = head if network["node_rank"] == 0 else peer
    if network["node_ip"] != expected:
        raise ValueError(
            "自动选择的本机地址不等于 head/peer；请核对 IP 或指定 --interface"
        )
    return network


def resolve_chat_template(value):
    """Resolve an explicit template on the API host without guessing model identity."""
    if value is None:
        return None
    path = (
        Path(__file__).with_name("codellama_instruct.jinja")
        if value == "codellama-instruct"
        else Path(value).expanduser()
    )
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"对话模板文件不存在或为空：{path}")
    return str(path.resolve())


def make_launch(args, node_ip, python, environment):
    validate_options(args)
    role = "head" if node_ip == args.head else "worker"
    plan = {
        "timeout_s": args.startup_timeout,
        "stages": [
            {"node_ip": ip, "soc": soc, "layers": int(count)}
            for ip, soc, count in zip(
                (args.head, args.head, args.peer, args.peer),
                ("910B", "910B", "310P", "310P"),
                args.layers.split(","),
            )
        ],
    }
    updates = {
        "VLLM_HOST_IP": node_ip,
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_HETERO_PP_PLAN": json.dumps(plan),
        "RAY_ADDRESS": f"{args.head}:{args.ray_port}",
    }
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "GLOO_SOCKET_IFNAME"):
        if key in environment:
            updates[key] = environment[key]
    ray = [
        python,
        "-m",
        "ray.scripts.scripts",
        "start",
        "--num-cpus=4",
        "--num-gpus=0",
        '--resources={"NPU":2}',
        "--object-store-memory=268435456",
        "--disable-usage-stats",
        # Ray CLI's default 10002..19999 worker range includes our GCS port.
        # Explicit zeros let workers bind OS-selected ports on both nodes.
        "--min-worker-port=0",
        "--max-worker-port=0",
        f"--node-ip-address={node_ip}",
    ]
    if role == "head":
        ray += ["--head", f"--port={args.ray_port}", "--include-dashboard=false"]
    else:
        ray += [f"--address={args.head}:{args.ray_port}"]
    serve = None
    if role == "head" and not args.profile:
        serve = [
            python,
            "-m",
            "vllm.distributed.hetero_pp.serve",
            "--model",
            str(args.model),
            "--served-model-name",
            "hetero-model",
            "--pipeline-parallel-size",
            "4",
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "float16",
            "--enforce-eager",
            "--no-async-scheduling",
            "--max-model-len",
            str(args.max_model_len),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
        ]
        template = resolve_chat_template(args.chat_template)
        if template:
            serve += ["--chat-template", template]
        if args.block_size is not None:
            serve += ["--block-size", str(args.block_size)]
        if args.num_gpu_blocks_override is not None:
            serve += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    profile_command = None
    profile_environment = {}
    if args.profile and role == "head":
        profile_command = [
            "bash",
            str(Path(__file__).with_name("profile_edgeshard.sh")),
        ]
        profile_environment = {
            "MODEL": str(args.model),
            "TRACE_ROOT": str(args.trace_root),
            "PROFILE_MAX_MODEL_LEN": str(args.max_model_len),
            "PROFILE_MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens),
            "PROFILE_MAX_NUM_SEQS": str(args.max_num_seqs),
            "PROFILE_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "PROFILE_BLOCK_SIZE": str(args.block_size or 128),
            "PROFILE_NUM_GPU_BLOCKS": str(args.num_gpu_blocks_override or 64),
        }
    return dict(
        role=role,
        plan=plan,
        environment=updates,
        ray_command=ray,
        serve_command=serve,
        profile_command=profile_command,
        profile_environment=profile_environment,
    )


def verify_profile_storage(address, head, peer, root, timeout):
    """Round-trip fresh files on both nodes before any model is loaded."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    def check(probe, nonce, ip):
        from pathlib import Path

        path = Path(probe)
        if (path / "head_nonce").read_text() != nonce:
            raise RuntimeError("两端不是同一个共享结果目录")
        answer = f"{ip}:{nonce}"
        (path / ip).write_text(answer)
        return answer

    ray.init(address=address)
    try:
        nodes = {n["NodeManagerAddress"]: n for n in ray.nodes() if n["Alive"]}
        probe = Path(tempfile.mkdtemp(prefix=".edgeshard_shared_check_", dir=root))
        nonce = uuid.uuid4().hex
        (probe / "head_nonce").write_text(nonce)
        tasks = [
            ray.remote(num_cpus=0)(check)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=nodes[ip]["NodeID"], soft=False
                )
            )
            .remote(str(probe), nonce, ip)
            for ip in (head, peer)
        ]
        answers = ray.get(tasks, timeout=float(timeout))
        for ip, answer in zip((head, peer), answers):
            if answer != f"{ip}:{nonce}" or (probe / ip).read_text() != answer:
                raise RuntimeError("无法读回远端写入的共享目录记录")
        shutil.rmtree(probe)  # Only this successful check's newly created files.
        print(json.dumps({"shared_trace_verified": True, "root": root}), flush=True)
    finally:
        ray.shutdown()


def profile_foreground(command, environment, log, timeout):
    """Stream finite profiling output; interrupt only this owned child tree."""
    with log.open("w") as writer, log.open() as reader:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=writer,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        family = owned_family(process.pid)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                remember_family(family, process.pid)
                print(reader.read(), end="", flush=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError("profiling 超过时间上限；仅清理本轮子进程")
                time.sleep(0.2)
            print(reader.read(), end="", flush=True)
            return process.returncode
        except BaseException:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
            stop_process(process, family)
            raise


def serve_foreground(command, environment, log, startup_timeout, api_port):
    """Stream the API log and forward interrupts only to the owned process tree."""
    with log.open("w") as writer, log.open() as reader:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=writer,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        family = owned_family(process.pid)
        deadline = time.monotonic() + startup_timeout
        ready = False
        next_check = 0
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            while process.poll() is None:
                remember_family(family, process.pid)
                print(reader.read(), end="", flush=True)
                if not ready and time.monotonic() >= next_check:
                    try:
                        with http.open(
                            f"http://127.0.0.1:{api_port}/health", timeout=1
                        ) as response:
                            ready = response.status == 200
                    except (OSError, urllib.error.URLError):
                        pass
                    next_check = time.monotonic() + 1
                    if ready:
                        write_json(
                            log.parent / "api_ready.json",
                            {
                                "health_http_status": 200,
                                "model_inference_validated": False,
                            },
                        )
                        print(
                            "API 已就绪；请发送请求核对结果。"
                            "Ctrl+C 停止 API，Ray 会保留。",
                            flush=True,
                        )
                if not ready and time.monotonic() >= deadline:
                    raise TimeoutError("模型 API 启动超时；见 api.log")
                time.sleep(0.2)
            print(reader.read(), end="", flush=True)
            if process.returncode == 0 and not ready:
                raise RuntimeError("API 未就绪即退出；见 api.log")
            return process.returncode
        except BaseException:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGINT)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=30)
            stop_process(process, family)
            raise


def start_services(args, launch, environment, output, python):
    """Join Ray before serving; an existing or failed Ray startup is never ignored."""
    environment = dict(environment, **launch["environment"])
    rc, text = capture(launch["ray_command"], environment, args.startup_timeout)
    (output / "ray_start.log").write_text(text)
    if rc:
        raise RuntimeError(
            f"Ray 启动失败（exit={rc}）；不会停止现有 Ray；见 ray_start.log"
        )
    write_json(
        output / "ray_started.json",
        {
            "role": launch["role"],
            "model_inference_validated": False,
            "ray_left_running": True,
        },
    )
    if launch["role"] == "worker":
        print(
            "310P 已加入 Ray；本端不启动 API/采集，请在 910B 端查看日志。", flush=True
        )
        print("Ray 保持运行；这不是模型推理通过回执。", flush=True)
        return 0
    print("910B Ray 已启动，等待两端各两张 NPU……", flush=True)
    code = """
import json, sys, time
import ray
address, head, peer, seconds = sys.argv[1:]
ray.init(address=address)
try:
    deadline = time.monotonic() + float(seconds)
    while True:
        nodes = {n['NodeManagerAddress']: n for n in ray.nodes() if n['Alive']}
        if all(ip in nodes and nodes[ip]['Resources'].get('NPU', 0) >= 2
               for ip in (head, peer)):
            print(json.dumps({'ready_nodes': [head, peer]}))
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('等待两端 NPU 资源超时')
        time.sleep(1)
finally:
    ray.shutdown()
"""
    command = [
        python,
        "-c",
        code,
        environment["RAY_ADDRESS"],
        args.head,
        args.peer,
        str(max(1, args.startup_timeout - 20)),
    ]
    rc, text = capture(command, environment, args.startup_timeout)
    (output / "cluster_wait.log").write_text(text)
    if rc:
        raise RuntimeError("Ray 集群未就绪；不启动模型，见 cluster_wait.log")
    if args.profile:
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from start_hetero_pp import verify_profile_storage; "
            "verify_profile_storage(*sys.argv[2:])"
        )
        rc, text = capture(
            [
                python,
                "-c",
                code,
                str(Path(__file__).parent),
                environment["RAY_ADDRESS"],
                args.head,
                args.peer,
                str(args.trace_root),
                str(min(args.startup_timeout, 60)),
            ],
            environment,
            min(args.startup_timeout, 90),
        )
        (output / "shared_storage_check.log").write_text(text)
        if rc:
            raise RuntimeError("两端共享结果目录检查失败；见 shared_storage_check.log")
        profile_env = dict(
            environment,
            **launch["profile_environment"],
            PY=python,
            PLAN=str(output / "plan.json"),
        )
        print(
            f"Ray 和共享目录已就绪，开始 profiling；日志 {output / 'profile.log'}",
            flush=True,
        )
        rc = profile_foreground(
            launch["profile_command"],
            profile_env,
            output / "profile.log",
            args.profile_timeout,
        )
        write_json(
            output / "profile_exit.json",
            {
                "returncode": rc,
                "profile_and_planning_completed": rc == 0,
                "measured_performance_improvement": False,
                "trace_off_validation_required": True,
                "ray_left_running": True,
            },
        )
        return rc if rc >= 0 else 128 - rc
    print(
        f"开始加载模型；API http://127.0.0.1:{args.api_port}；"
        f"日志 {output / 'api.log'}",
        flush=True,
    )
    rc = serve_foreground(
        launch["serve_command"],
        environment,
        output / "api.log",
        args.startup_timeout,
        args.api_port,
    )
    write_json(
        output / "server_exit.json",
        {
            "returncode": rc,
            "model_inference_validated": False,
            "ray_left_running": True,
        },
    )
    return rc if rc >= 0 else 128 - rc


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    try:
        validate_options(args)
        if args.dry_run and args.node_ip:
            node_ip = args.node_ip
            network = {"node_ip": node_ip, "interface": args.interface}
        else:
            interfaces = ipv4_interfaces()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect((args.head, args.ray_port))
                route_ip = args.node_ip or route.getsockname()[0]
            network = select_network(
                args.head,
                args.peer,
                interfaces,
                route_ip,
                args.interface or os.environ.get("GLOO_SOCKET_IFNAME"),
            )
            node_ip = network["node_ip"]
            if args.node_ip and node_ip != args.node_ip:
                raise ValueError("--node-ip 不属于本机所选网卡")
        if node_ip == args.head and not args.profile:
            resolve_chat_template(args.chat_template)
        env = prepare_environment(ROOT, os.environ, args.devices)
        if network["interface"]:
            env["GLOO_SOCKET_IFNAME"] = network["interface"]
        if args.dry_run:
            launch = make_launch(args, node_ip, args.python or sys.executable, env)
            print(
                json.dumps(dict(launch, network=network), ensure_ascii=False, indent=2)
            )
            return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output = (
        (
            args.output
            or ROOT
            / "output"
            / (
                "hetero_pp_start_"
                + time.strftime("%Y%m%d_%H%M%S_")
                + uuid.uuid4().hex[:8]
            )
        )
        .expanduser()
        .absolute()
    )
    output.mkdir(parents=True, exist_ok=False)
    print(f"本机日志目录：{output}", flush=True)
    try:
        if node_ip == args.head:
            ports = [(node_ip, args.ray_port)]
            if not args.profile:
                ports.append(("127.0.0.1", args.api_port))
            for address, port in ports:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
                    check.bind((address, port))
        python, env, inventory = discover_runtime(
            args,
            output,
            time.monotonic() + args.startup_timeout,
        )
        expected = "910B" if node_ip == args.head else "310P"
        if inventory["soc"] != expected:
            raise ValueError(
                f"本机应为 {expected}，实际是 {inventory['soc']}；核对 head/peer"
            )
        env["GLOO_SOCKET_IFNAME"] = network["interface"]
        launch = make_launch(args, node_ip, python, env)
        write_json(output / "local_preflight.json", dict(inventory, network=network))
        write_json(output / "plan.json", launch["plan"])
        write_json(output / "launch.json", launch)
        print(
            f"本机 {expected}，角色={launch['role']}，网卡={network['interface']}",
            flush=True,
        )
        return start_services(args, launch, env, output, python)
    except BaseException as error:
        write_json(
            output / "failure.json", {"type": type(error).__name__, "error": str(error)}
        )
        print(
            f"启动未完成：{error}\nRay 若已启动会保留，不清理其他任务。日志：{output}",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    install_signal_handlers()
    sys.exit(main())
