# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Start configurable two-node Ascend PP from the same CLI on both nodes.

Run inside each already working container. No Docker, package installation,
device remapping or global Ray stop is performed. Explicit profile mode runs
the finite internal collection workload instead of an HTTP service.
"""

import argparse
import ipaddress
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from fractions import Fraction
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
        description="两机各两卡 PP：CPU/Gloo + Ray 或原生 HCCL + mp"
    )
    parser.add_argument("--head", required=True, help="主端 IPv4；两边填一样")
    parser.add_argument("--peer", required=True, help="另一个节点 IPv4；两边填一样")
    parser.add_argument(
        "--transport",
        choices=("gloo", "hccl"),
        default="gloo",
        help="默认 gloo；hccl 使用原生双机 mp，目前限两端 910B，不启用 Ray",
    )
    parser.add_argument(
        "--head-soc",
        type=str.lower,
        choices=("910b", "310p"),
        default="910b",
        help="head 两张卡的型号，默认 910b",
    )
    parser.add_argument(
        "--peer-soc",
        type=str.lower,
        choices=("910b", "310p"),
        default=None,
        help="peer 两张卡的型号；gloo 默认 310p，hccl 默认 910b",
    )
    parser.add_argument(
        "--model", required=True, type=Path, help="两端同路径的本地完整模型"
    )
    parser.add_argument(
        "--layers", help="四 stage 层数；原 910B→310P 默认 18,18,6,6，其他组合均分"
    )
    parser.add_argument(
        "--npu-core-fractions",
        type=parse_npu_core_fractions,
        help="按 PP stage 0–3 设置核心比例，例如 1,1,1/2,1/4",
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
    parser.add_argument(
        "--master-port",
        type=int,
        default=29501,
        help="HCCL/mp 主端端口，默认 29501；两边一致",
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument(
        "--startup-timeout", type=float, default=1200, help="每项启动/等节点上限秒数"
    )
    parser.add_argument(
        "--max-model-len", type=int, help="可选，普通服务沿用 vLLM 默认"
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, help="可选，普通服务沿用 vLLM 默认"
    )
    parser.add_argument("--max-num-seqs", type=int, help="可选，普通服务沿用 vLLM 默认")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, help="可选，普通服务沿用 vLLM 默认"
    )
    parser.add_argument("--block-size", type=int, help="可选，显式 KV block token 数")
    parser.add_argument(
        "--num-gpu-blocks-override", type=int, help="可选，固定 KV block 数"
    )
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="HCCL 普通服务可显式开启异步调度，默认同步",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="HCCL 普通服务可显式选择，默认开启",
    )
    parser.add_argument(
        "--model-runner-v1", action="store_true", help="HCCL 使用 V1 runner（默认）"
    )
    parser.add_argument("--seed", type=int, help="HCCL 普通服务的模型随机种子")
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help="HCCL 回测使用，要求本机模型内容与指定 SHA256 一致",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="采集并生成 EdgeShard DP 推荐，不启动 API",
    )
    parser.add_argument(
        "--verify-overlap",
        action="store_true",
        help="Gloo 在线验证；head 自动发请求并检查 Stage 0/1 NPU kernel 重叠",
    )
    parser.add_argument(
        "--trace-root", type=Path, help="可选，profiling 本地结果根目录，无需共享存储"
    )
    parser.add_argument(
        "--profile-timeout",
        type=float,
        default=3600,
        help="profiling 全流程上限秒数，默认 3600",
    )
    parser.add_argument(
        "--profile-compute-timing",
        choices=("event", "host-sync"),
        default="event",
        help="profiling 计算计时；host-sync 为所有 stage 使用逐层同步 CPU 计时",
    )
    parser.add_argument(
        "--output", type=Path, help="全新日志目录；默认 output/hetero_pp_start_*"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只预览，不导入 Torch/Ray 或启动服务"
    )
    return parser


def parse_npu_core_fractions(value):
    try:
        fractions = [Fraction(part.strip()) for part in value.split(",")]
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(
            "--npu-core-fractions 需要四个有效比例，例如 1,1,1/2,1/4"
        ) from None
    if len(fractions) != 4 or any(not 0 < ratio <= 1 for ratio in fractions):
        raise argparse.ArgumentTypeError(
            "--npu-core-fractions 需要四个比例，每项满足 0 < x <= 1"
        )
    return tuple(map(str, fractions))


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_options(args):
    if args.verify_overlap and (args.profile or args.transport != "gloo"):
        raise ValueError("--verify-overlap 仅用于 Gloo 在线验证")
    if (
        args.verify_overlap
        and args.max_model_len is not None
        and args.max_model_len < 392
    ):
        raise ValueError("--verify-overlap 需要至少 392 token 的模型上下文")
    if args.peer_soc is None:
        args.peer_soc = "910b" if args.transport == "hccl" else "310p"
    if args.transport == "hccl" and (args.head_soc, args.peer_soc) != ("910b", "910b"):
        raise ValueError("HCCL 模式当前只支持两端各两张 910B；异构请用 gloo")
    if args.transport != "hccl" and (
        args.async_scheduling is not None
        or args.enable_chunked_prefill is not None
        or args.model_runner_v1
        or args.seed is not None
        or args.expected_checkpoint_sha256 is not None
    ):
        raise ValueError(
            "调度、runner、seed、checkpoint SHA 开关仅供 HCCL 启动入口使用"
        )
    if args.profile and (
        args.async_scheduling is True or args.enable_chunked_prefill is False
    ):
        raise ValueError("profiling 要求关闭异步调度并开启 chunked prefill")
    if args.profile and args.seed not in (None, 7):
        raise ValueError("profiling 固定使用 --seed 7")
    if args.expected_checkpoint_sha256 is not None and not _valid_sha256(
        args.expected_checkpoint_sha256
    ):
        raise ValueError("--expected-checkpoint-sha256 需要 64 位小写十六进制 SHA256")
    if not args.profile and args.profile_compute_timing != "event":
        raise ValueError("--profile-compute-timing 仅用于 --profile")
    if args.profile:
        args.trace_root = (
            args.trace_root or Path.home() / "vllm_hetero_pp_logs/profiles"
        )
        if not args.trace_root.is_absolute():
            raise ValueError("--trace-root 需要本地目录的绝对路径")
        if args.trace_root.exists() and not args.trace_root.is_dir():
            raise ValueError("--trace-root 不是目录")
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
    coordination_port = args.master_port if args.transport == "hccl" else args.ray_port
    if (
        not 1024 <= coordination_port <= 65535
        or not 1024 <= args.api_port <= 65535
        or coordination_port == args.api_port
    ):
        raise ValueError("主端通信/API 端口需不同且在 1024..65535")
    if (
        not math.isfinite(args.startup_timeout)
        or not 30 <= args.startup_timeout <= 1800
    ):
        raise ValueError("startup-timeout 需在 30..1800 秒")
    capacities = (args.max_model_len, args.max_num_batched_tokens, args.max_num_seqs)
    if any(value is not None and value < 1 for value in capacities):
        raise ValueError("上下文、token budget、并发上限必须为正整数")
    if args.gpu_memory_utilization is not None and not (
        0 < args.gpu_memory_utilization < 1
    ):
        raise ValueError("gpu-memory-utilization 需在 0 和 1 之间")
    args.model = args.model.expanduser().resolve(strict=True)
    config = json.loads((args.model / "config.json").read_text())
    if config.get("architectures") not in (["LlamaForCausalLM"], ["Qwen2ForCausalLM"]):
        raise ValueError("当前入口只支持 LlamaForCausalLM/Qwen2ForCausalLM")
    total = config.get("num_hidden_layers")
    if type(total) is not int or total < 4:
        raise ValueError("模型 num_hidden_layers 必须是至少 4 的整数")
    if args.layers is None:
        if (args.head_soc, args.peer_soc) == ("910b", "310p"):
            args.layers = "18,18,6,6"
        else:
            base, remainder = divmod(total, 4)
            args.layers = ",".join(str(base + (rank < remainder)) for rank in range(4))
    counts = args.layers.split(",")
    if len(counts) != 4 or not all(
        s.isascii() and s.isdigit() and int(s) > 0 for s in counts
    ):
        raise ValueError("layers 需为四个正整数，例如 18,18,6,6")
    if sum(map(int, counts)) != total:
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


def make_launch(args, node_ip, python, environment, output=None):
    validate_options(args)
    role = "head" if node_ip == args.head else "worker"
    plan = {
        "timeout_s": args.startup_timeout,
        "stages": [
            {"node_ip": ip, "soc": soc, "layers": int(count)}
            for ip, soc, count in zip(
                (args.head, args.head, args.peer, args.peer),
                (args.head_soc.upper(),) * 2 + (args.peer_soc.upper(),) * 2,
                args.layers.split(","),
            )
        ],
    }
    if args.npu_core_fractions is not None:
        for stage, ratio in zip(plan["stages"], args.npu_core_fractions, strict=True):
            if ratio != "1":
                stage["npu_core_fraction"] = ratio
    if (args.head_soc, args.peer_soc) != ("910b", "310p"):
        plan["validation_mode"] = (
            "910b_only" if args.head_soc == args.peer_soc == "910b" else "explicit"
        )
    if args.transport == "hccl":
        return make_native_launch(args, node_ip, python, environment, plan, role)
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
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
        ]
        # Omitted options must reach vLLM as omitted, so model/platform defaults
        # remain authoritative instead of duplicating them in this launcher.
        options = {
            name: getattr(args, name)
            for name in (
                "max_model_len",
                "max_num_batched_tokens",
                "max_num_seqs",
                "gpu_memory_utilization",
            )
        }
        if args.verify_overlap:
            updates["VLLM_HETERO_PP_VERIFY_OVERLAP"] = "1"
            options.update(max_num_batched_tokens=128, max_num_seqs=4)
            trace_root = (output or args.output or ROOT / "output/preview").absolute()
            serve += [
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": str(trace_root / "overlap_traces"),
                        "torch_profiler_with_stack": False,
                        "ignore_frontend": True,
                    }
                ),
            ]
        for name, value in options.items():
            if value is not None:
                serve += ["--" + name.replace("_", "-"), str(value)]
        template = resolve_chat_template(args.chat_template)
        if template:
            serve += ["--chat-template", template]
        if args.block_size is not None:
            serve += ["--block-size", str(args.block_size)]
        if args.num_gpu_blocks_override is not None:
            serve += ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
    profile_command, profile_environment = profile_head_command(args, role)
    return dict(
        transport="gloo",
        role=role,
        plan=plan,
        environment=updates,
        ray_command=ray,
        serve_command=serve,
        profile_command=profile_command,
        profile_environment=profile_environment,
    )


_PROFILE_OPTIONS = (
    ("max_model_len", 512, "PROFILE_MAX_MODEL_LEN"),
    ("max_num_batched_tokens", 256, "PROFILE_MAX_NUM_BATCHED_TOKENS"),
    ("max_num_seqs", 4, "PROFILE_MAX_NUM_SEQS"),
    ("gpu_memory_utilization", 0.8, "PROFILE_GPU_MEMORY_UTILIZATION"),
    ("block_size", 128, "PROFILE_BLOCK_SIZE"),
    ("num_gpu_blocks_override", 64, "PROFILE_NUM_GPU_BLOCKS"),
)


def profile_head_command(args, role):
    if not args.profile or role != "head":
        return None, {}
    environment = {
        "MODEL": str(args.model),
        "TRACE_ROOT": str(args.trace_root),
        "PROFILE_COMPUTE_TIMING": args.profile_compute_timing,
        "PROFILE_TRANSPORT": args.transport,
    }
    environment.update(
        {
            key: str(getattr(args, name) or default)
            for name, default, key in _PROFILE_OPTIONS
        }
    )
    return ["bash", str(Path(__file__).with_name("profile_edgeshard.sh"))], environment


def make_native_launch(args, node_ip, python, environment, plan, role):
    """Use native MP/HCCL with optional worker-local quotas and profiling."""
    updates = {
        "VLLM_HOST_IP": node_ip,
        "HCCL_IF_IP": node_ip,
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_PP_LAYER_PARTITION": args.layers,
        "VLLM_HETERO_PP_PLAN": json.dumps(plan),
        "HETERO_PP_RUN_MODE": "profile" if args.profile else "serve",
    }
    interface = args.interface or environment.get("GLOO_SOCKET_IFNAME")
    if interface:
        updates.update(GLOO_SOCKET_IFNAME=interface, TP_SOCKET_IFNAME=interface)
    if "ASCEND_RT_VISIBLE_DEVICES" in environment:
        updates["ASCEND_RT_VISIBLE_DEVICES"] = environment["ASCEND_RT_VISIBLE_DEVICES"]
    command = [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--served-model-name",
        "hetero-model",
        "--distributed-executor-backend",
        "mp",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "4",
        "--nnodes",
        "2",
        "--node-rank",
        "0" if role == "head" else "1",
        "--master-addr",
        args.head,
        "--master-port",
        str(args.master_port),
        "--dtype",
        "float16",
        "--enforce-eager",
        "--async-scheduling" if args.async_scheduling else "--no-async-scheduling",
        "--no-enable-chunked-prefill"
        if args.enable_chunked_prefill is False
        else "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--worker-cls",
        "vllm.distributed.hetero_pp.native_worker.Native910BWorker",
    ]
    if not args.profile and args.seed is not None:
        command += ["--seed", str(args.seed)]
    if role == "head" and not args.profile:
        # The peer checks this API for a bounded startup, then remains attached
        # to its own headless worker process. Health is not inference validation.
        command += ["--host", "0.0.0.0", "--port", str(args.api_port)]
        template = resolve_chat_template(args.chat_template)
        if template:
            command += ["--chat-template", template]
    elif role == "worker":
        command.append("--headless")
    for name, default, _ in _PROFILE_OPTIONS:
        value = getattr(args, name)
        if args.profile and value is None:
            value = default
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    profile_command, profile_environment = profile_head_command(args, role)
    if args.profile:
        updates["HETERO_PP_PROFILE_COMPUTE_TIMING"] = args.profile_compute_timing
        if role == "head":
            profile_environment.update(
                PROFILE_MASTER_ADDR=args.head, PROFILE_MASTER_PORT=str(args.master_port)
            )
        else:
            profile_command = command + ["--seed", "7"]
    return dict(
        transport="hccl",
        role=role,
        plan=plan,
        environment=updates,
        unset_environment=[
            "HETERO_PP_RUN_MODE",
            "HETERO_PP_CHECKPOINT_SHA256",
            "HETERO_PP_PROFILE_COMPUTE_TIMING",
            "VLLM_HETERO_PP_PLAN",
            "VLLM_PP_STAGE_TRACE",
            "VLLM_PP_TRACE_SESSION",
            "VLLM_PP_COMPUTE_MODEL",
            "VLLM_PP_COMPUTE_TIMING",
            "VLLM_PP_COMM_TIMING",
            "RAY_ADDRESS",
            "VLLM_USE_V2_MODEL_RUNNER",
        ],
        ray_command=None,
        serve_command=None if args.profile else command,
        profile_command=profile_command,
        profile_environment=profile_environment,
    )


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


def run_overlap_verifier(python, environment, output, port):
    verifier = Path(__file__).with_name("verify_overlap.py")
    receipt = output / "overlap_verification.json"
    print("开始采集 Stage 0/1 的 NPU 批次重叠……", flush=True)
    try:
        result, text = capture(
            [python, str(verifier), "--port", str(port), "--output", str(output)],
            environment,
            600,
        )
    except subprocess.TimeoutExpired:
        result, text = 2, "重叠验证超过 600 秒\n"
    if not receipt.exists():
        result = 2
        write_json(receipt, {"status": "inconclusive", "reason": text.strip()})
    (output / "overlap_verification.log").write_text(text)
    print(text, end="", flush=True)
    return result


def serve_foreground(
    command,
    environment,
    log,
    startup_timeout,
    api_port,
    *,
    health_host="127.0.0.1",
    headless=False,
    verify_overlap=False,
):
    """Supervise a local server/worker until exit, bounding API readiness waits."""
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
                            f"http://{health_host}:{api_port}/health", timeout=1
                        ) as response:
                            ready = response.status == 200
                    except (OSError, urllib.error.URLError):
                        pass
                    next_check = time.monotonic() + 1
                    if ready:
                        write_json(
                            log.parent
                            / ("head_api_ready.json" if headless else "api_ready.json"),
                            {
                                "health_http_status": 200,
                                "model_inference_validated": False,
                            },
                        )
                        print(
                            "head API 已就绪；本端原生 worker 持续运行。"
                            if headless
                            else "API 已就绪；请发请求核对结果。Ctrl+C 停止本端服务。",
                            flush=True,
                        )
                        if verify_overlap:
                            result = run_overlap_verifier(
                                command[0], environment, log.parent, api_port
                            )
                            if process.poll() is not None:
                                raise RuntimeError("重叠验证期间 API 异常退出")
                            os.killpg(process.pid, signal.SIGINT)
                            api_rc = process.wait(timeout=30)
                            print(reader.read(), end="", flush=True)
                            write_json(
                                log.parent / "server_exit.json",
                                {
                                    "returncode": api_rc,
                                    "overlap_verification_returncode": result,
                                    "ray_left_running": True,
                                },
                            )
                            return result
                if not ready and time.monotonic() >= deadline:
                    raise TimeoutError(f"模型 API 启动超时；见 {log.name}")
                time.sleep(0.2)
            print(reader.read(), end="", flush=True)
            if process.returncode == 0 and not ready:
                raise RuntimeError(f"API 未就绪即退出；见 {log.name}")
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
    """Start native MP or join Ray before serving, preserving startup failures."""
    environment = dict(environment)
    environment.pop("VLLM_HETERO_PP_VERIFY_OVERLAP", None)
    for name in launch.get("unset_environment", []):
        environment.pop(name, None)
    environment = dict(environment, **launch["environment"])
    if launch["transport"] == "hccl":
        headless = launch["role"] == "worker"
        if args.profile:
            profile_env = dict(environment, **launch["profile_environment"])
            profile_env.update(PY=python, PLAN=str(output / "plan.json"))
            log = output / "profile.log"
            print(f"启动 HCCL EdgeShard，角色={launch['role']}；日志 {log}", flush=True)
            rc = profile_foreground(
                launch["profile_command"], profile_env, log, args.profile_timeout
            )
            write_json(
                output / "profile_exit.json",
                {
                    "returncode": rc,
                    "transport": "hccl",
                    "role": launch["role"],
                    "profile_and_planning_completed": rc == 0 and not headless,
                },
            )
            return rc if rc >= 0 else 128 - rc
        log = output / ("worker.log" if headless else "api.log")
        print(f"启动原生 HCCL/mp，角色={launch['role']}；日志 {log}", flush=True)
        rc = serve_foreground(
            launch["serve_command"],
            environment,
            log,
            args.startup_timeout,
            args.api_port,
            health_host=args.head if headless else "127.0.0.1",
            headless=headless,
        )
        write_json(
            output / "server_exit.json",
            {
                "returncode": rc,
                "transport": "hccl",
                "role": launch["role"],
                "model_inference_validated": False,
                "ray_started": False,
            },
        )
        return rc if rc >= 0 else 128 - rc
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
            "peer 已加入 Ray；本端不启动 API/采集，请在 head 端查看日志。", flush=True
        )
        print("Ray 保持运行；这不是模型推理通过回执。", flush=True)
        return 0
    print("head Ray 已启动，等待两端各两张 NPU……", flush=True)
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
        profile_env = dict(
            environment,
            **launch["profile_environment"],
            PY=python,
            PLAN=str(output / "plan.json"),
        )
        print(
            "Ray 已就绪，开始本地采集并自动回收 peer 记录；"
            f"日志 {output / 'profile.log'}",
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
        **({"verify_overlap": True} if args.verify_overlap else {}),
    )
    if not args.verify_overlap:
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
                port = args.master_port if args.transport == "hccl" else args.ray_port
                route.connect((args.head, port))
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
            port = args.master_port if args.transport == "hccl" else args.ray_port
            ports = [(node_ip, port)]
            if not args.profile:
                host = "0.0.0.0" if args.transport == "hccl" else "127.0.0.1"
                ports.append((host, args.api_port))
            for address, port in ports:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
                    check.bind((address, port))
        python, env, inventory = discover_runtime(
            args,
            output,
            time.monotonic() + args.startup_timeout,
        )
        expected = (args.head_soc if node_ip == args.head else args.peer_soc).upper()
        if inventory["soc"] != expected:
            raise ValueError(
                f"本机应为 {expected}，实际是 {inventory['soc']}；"
                "核对 IP 和 --head-soc/--peer-soc"
            )
        env["GLOO_SOCKET_IFNAME"] = network["interface"]
        launch = make_launch(args, node_ip, python, env, output)
        write_json(output / "local_preflight.json", dict(inventory, network=network))
        write_json(output / "plan.json", launch["plan"])
        if args.transport == "hccl":
            print("校验本机模型文件内容，用于两端启动比对……", flush=True)
            helper = ROOT / "vllm/distributed/hetero_pp/checkpoint_identity.py"
            rc, result = capture(
                [python, "-S", str(helper), str(args.model)], env, args.startup_timeout
            )
            digest = result.strip()
            if rc or not _valid_sha256(digest):
                raise ValueError(f"模型文件校验失败：{result.strip()}")
            expected_digest = args.expected_checkpoint_sha256
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(
                    "本机 checkpoint 与回测引用的模型内容不同："
                    f"实际 {digest}，要求 {expected_digest}"
                )
            launch["environment"]["HETERO_PP_CHECKPOINT_SHA256"] = digest
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
            f"启动未完成：{error}\n仅管理本轮子进程，已有 Ray 不变。日志：{output}",
            file=sys.stderr,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    install_signal_handlers()
    sys.exit(main())
