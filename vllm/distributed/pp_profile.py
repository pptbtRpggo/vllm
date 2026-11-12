# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the opt-in CPU/Gloo PP backend, then plan a fixed-order layer split.

Only the four-stage layer-measured backend is admitted. Collection writes a
bound manifest; offline planning refuses unbound old traces. A DP result is a
candidate under measured-cost and supplied-memory assumptions, not a measured
throughput improvement or an unconditional memory-safety guarantee.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import shlex
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm.distributed.pp_memory import PPMemoryProfile
from vllm.distributed.pp_partition import PPPartitionPlan, format_plan
from vllm.logger import init_logger

logger = init_logger(__name__)
_ROOT = Path(__file__).resolve().parents[2]
_TRACE_ENV = (
    "VLLM_PP_STAGE_TRACE",
    "VLLM_PP_TRACE_SESSION",
    "VLLM_PP_COMPUTE_MODEL",
    "VLLM_PP_COMM_TIMING",
    "VLLM_HETERO_PP_PLAN",
)
_TRANSPORT = "cpu_gloo"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ProfileWorkload:
    prompts: list[dict[str, list[int]]]
    max_tokens: int
    num_iters: int = 1
    num_iters_warmup: int = 0
    ignore_eos: bool = True
    temperature: float = 0.0
    warmup_prompts: list[dict[str, list[int]]] | None = None
    iteration_prompts: list[list[dict[str, list[int]]]] | None = None


def build_profile_workload(
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int = 0,
    vocab_size: int = 32000,
    num_iters: int = 1,
    num_iters_warmup: int = 0,
):
    """Create deterministic, disjoint synthetic warmup and measured prompts."""
    for name, value in (
        ("num_prompts", num_prompts),
        ("input_len", input_len),
        ("output_len", output_len),
        ("num_iters", num_iters),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(vocab_size) is not int or vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if type(num_iters_warmup) is not int or num_iters_warmup < 0:
        raise ValueError("num_iters_warmup must be nonnegative")
    rng, previous = random.Random(seed), set()

    def fresh():
        prompts = []
        for _ in range(num_prompts):
            for _attempt in range(1000):
                ids = tuple(rng.randrange(1, vocab_size) for _ in range(input_len))
                if ids not in previous:
                    break
            else:
                raise ValueError("not enough distinct synthetic prompts")
            previous.add(ids)
            prompts.append({"prompt_token_ids": list(ids)})
        return prompts

    measured = [fresh() for _ in range(num_iters)]
    return ProfileWorkload(
        measured[0],
        output_len,
        num_iters,
        num_iters_warmup,
        warmup_prompts=fresh() if num_iters_warmup else None,
        iteration_prompts=measured,
    )


def prepare_trace_dir(
    dump_dir: str | Path | None = None, comm_timing="shared-clock"
) -> Path:
    """Reserve a new/empty trace directory; never delete prior evidence."""
    if comm_timing not in ("shared-clock", "ack"):
        raise ValueError("unknown communication timing")
    if dump_dir is None:
        path = Path(tempfile.mkdtemp(prefix="vllm_staged_pp_profile_"))
    else:
        path = Path(dump_dir).expanduser().resolve()
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(
                "trace directory must be new or empty; preserve old evidence"
            )
        path.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_PP_STAGE_TRACE"] = str(path)
    os.environ["VLLM_PP_TRACE_SESSION"] = uuid.uuid4().hex
    os.environ["VLLM_PP_COMPUTE_MODEL"] = "layer-measured"
    os.environ["VLLM_PP_COMM_TIMING"] = comm_timing
    return path


def run_traced_generate(llm: Any, workload: ProfileWorkload) -> None:
    """Run distinct warmup and measured batches through barriered worker RPCs."""
    from vllm.sampling_params import SamplingParams

    measured = workload.iteration_prompts or [workload.prompts] * workload.num_iters
    if len(measured) != workload.num_iters or any(not rows for rows in measured):
        raise ValueError("provide one nonempty prompt batch per measured iteration")
    keys = {tuple(p["prompt_token_ids"]) for rows in measured for p in rows}
    if workload.num_iters_warmup and (
        not workload.warmup_prompts
        or any(tuple(p["prompt_token_ids"]) in keys for p in workload.warmup_prompts)
    ):
        raise ValueError("warmup requires separate, disjoint warmup_prompts")
    params = SamplingParams(
        temperature=workload.temperature,
        ignore_eos=workload.ignore_eos,
        max_tokens=workload.max_tokens,
    )
    for i in range(workload.num_iters_warmup + workload.num_iters):
        warmup = i < workload.num_iters_warmup
        # The staged executor drains previously admitted RPCs before these
        # worker control calls. generate() returning alone is not the barrier.
        result = llm.collective_rpc("set_pp_profile_warmup", args=(warmup,))
        if not isinstance(result, list) or len(result) != 4:
            raise ValueError("warmup RPC must return all four worker receipts")
        prompts = (
            workload.warmup_prompts
            if warmup
            else measured[i - workload.num_iters_warmup]
        )
        llm.generate(prompts, sampling_params=params, use_tqdm=False)


def shutdown_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(getattr(engine, "engine_core", None), "shutdown", None)
    if callable(shutdown):
        shutdown()


def _load_topology(value: str | Path | dict | None) -> dict:
    if value is None:
        raise ValueError("--hetero-plan is required for the CPU/Gloo profile backend")
    raw = (
        copy.deepcopy(value)
        if isinstance(value, dict)
        else json.loads(Path(value).read_text())
    )
    from vllm.distributed.hetero_pp.config import PipelinePlan

    total = sum(s.get("layers", 0) for s in raw.get("stages", []))
    PipelinePlan.from_dict(raw, total)
    return raw


def validate_profile_args(args: Any) -> None:
    """Reject unsupported requests before constructing a real engine."""
    if not getattr(args, "hetero_plan", None):
        raise ValueError("--hetero-plan is required")
    if getattr(args, "comm_timing", None) not in (None, "shared-clock", "ack"):
        raise ValueError("unknown communication timing")
    if getattr(args, "collect_only", False) and getattr(args, "skip_run", False):
        raise ValueError("--collect-only cannot be combined with --skip-run")
    if getattr(args, "skip_run", False) and not getattr(args, "trace_dir", None):
        raise ValueError("--trace-dir is required with --skip-run")
    if getattr(args, "pipeline_parallel_size", 4) != 4:
        raise ValueError("profiling requires PP=4 with the fixed topology")
    if getattr(args, "tensor_parallel_size", 1) != 1:
        raise ValueError("profiling requires TP=1")
    if getattr(args, "compute_model", "layer-measured") != "layer-measured":
        raise ValueError("only layer-measured is supported")
    if not getattr(args, "skip_run", False):
        if not getattr(args, "enforce_eager", False):
            raise ValueError("profiling requires --enforce-eager")
        if getattr(args, "async_scheduling", None) is not False:
            raise ValueError("profiling requires synchronous --no-async-scheduling")


def _configure_engine(engine_args):
    from vllm.v1.executor.hetero_pp_executor import HeteroPPExecutor

    if engine_args is None:
        raise ValueError("engine_args are required for live profiling")
    args = copy.copy(engine_args)
    # The generated command deliberately supports a small dense-model scope.
    # Semantic extras must not silently disappear when a candidate is served.
    allowed: dict[str, tuple[Any, ...]] = {
        "model_weights": (None, ""),
        "tokenizer": (None,),
        "tokenizer_revision": (None,),
        "tokenizer_mode": (None, "auto"),
        "skip_tokenizer_init": (None, False),
        "enable_prompt_embeds": (None, False),
        "hf_config_path": (None,),
        "hf_overrides": (None, {}),
        "trust_remote_code": (None, False),
        "load_format": (None, "auto"),
        "config_format": (None, "auto"),
        "model_impl": (None, "auto"),
        "model_loader_extra_config": (None, {}),
        "rope_scaling": (None,),
        "rope_theta": (None,),
        "attention_backend": (None,),
        "disable_sliding_window": (None, False),
        "disable_cascade_attn": (None, False, True),
    }
    for name, values in allowed.items():
        if getattr(args, name, None) not in values:
            raise ValueError(f"{name} is not replayed by the minimal profile entry")
    attention = getattr(args, "attention_config", None)
    if (
        attention is not None
        and attention != {}
        and (
            not dataclasses.is_dataclass(attention)
            or isinstance(attention, type)
            or _jsonable(attention) != _jsonable(type(attention)())
        )
    ):
        raise ValueError("custom attention_config is not replayed by this entry")
    if args.pipeline_parallel_size != 4 or args.tensor_parallel_size != 1:
        raise ValueError("live profiling requires PP=4 and TP=1")
    if not args.enforce_eager or args.async_scheduling is not False:
        raise ValueError("live profiling requires eager synchronous execution")
    backend = getattr(args, "distributed_executor_backend", None)
    if backend not in (None, "auto", HeteroPPExecutor):
        raise ValueError(
            "omit --distributed-executor-backend; the profile entry selects its class"
        )
    args.distributed_executor_backend = HeteroPPExecutor
    return args


def _default_llm_factory(engine_args):
    from vllm import LLM

    return LLM.from_engine_args(engine_args)


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, type) or callable(value):
        return (
            getattr(value, "__module__", "")
            + "."
            + getattr(value, "__qualname__", type(value).__name__)
        )
    if dataclasses.is_dataclass(value):
        return {
            f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, SimpleNamespace):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    return str(value)


def _engine_snapshot(config):
    from vllm.distributed.hetero_pp.profile_memory import serving_config_from_engine

    result = serving_config_from_engine(config)
    result.update(
        num_layers=config.model_config.get_total_num_hidden_layers(),
        pipeline_parallel_size=config.parallel_config.pipeline_parallel_size,
        tensor_parallel_size=config.parallel_config.tensor_parallel_size,
        enforce_eager=config.model_config.enforce_eager,
        disable_cascade_attn=config.model_config.disable_cascade_attn,
        async_scheduling=config.scheduler_config.async_scheduling,
        enable_chunked_prefill=getattr(
            config.scheduler_config, "enable_chunked_prefill", None
        ),
        enable_prefix_caching=getattr(
            config.cache_config, "enable_prefix_caching", None
        ),
        num_gpu_blocks_override=getattr(
            config.cache_config, "num_gpu_blocks_override", None
        ),
        kv_cache_memory_bytes=getattr(
            config.cache_config, "kv_cache_memory_bytes", None
        ),
        use_v2_model_runner=getattr(config, "use_v2_model_runner", None),
        seed=getattr(config.model_config, "seed", None),
        served_model_name=getattr(config.model_config, "served_model_name", None),
        additional_config=_jsonable(getattr(config, "additional_config", {})),
    )
    return result


def _snapshot_config(snapshot):
    """Rebuild only the typed fields read by the memory scope validator."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            **{
                k: snapshot[k]
                for k in ("model", "revision", "dtype", "quantization", "max_model_len")
            },
            get_total_num_hidden_layers=lambda: snapshot["num_layers"],
        ),
        cache_config=SimpleNamespace(
            cache_dtype=snapshot["kv_cache_dtype"],
            **{
                k: snapshot[k]
                for k in (
                    "block_size",
                    "gpu_memory_utilization",
                    "num_gpu_blocks_override",
                    "kv_cache_memory_bytes",
                )
            },
        ),
        scheduler_config=SimpleNamespace(
            **{k: snapshot[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
        ),
        parallel_config=SimpleNamespace(
            **{
                k: snapshot[k]
                for k in ("pipeline_parallel_size", "tensor_parallel_size")
            }
        ),
    )


def _source_hashes():
    paths = list((_ROOT / "vllm/distributed/hetero_pp").glob("*.py"))
    paths += [p for p in (_ROOT / "vllm/distributed").glob("pp_*.py")]
    paths += [
        _ROOT / rel
        for rel in (
            "vllm/distributed/device_communicators/cpu_staging.py",
            "vllm/v1/executor/hetero_pp_executor.py",
            "vllm/entrypoints/cli/pp_profile.py",
        )
    ]
    return {str(p.relative_to(_ROOT)): _sha(p) for p in sorted(set(paths))}


def _validate_startup(reports, topology, config):
    if not isinstance(reports, list) or len(reports) != 4:
        raise ValueError("profile manifest requires all four actual startup reports")
    model_sha = hashlib.sha256(
        config.model_config.hf_config.to_json_string().encode()
    ).hexdigest()
    end = 0
    sources = _source_hashes()
    for rank, report in enumerate(reports):
        stage = topology["stages"][rank]
        if (
            report.get("rank") != rank
            or report.get("node_ip") != stage["node_ip"]
            or report.get("soc") != stage["soc"]
            or report.get("layers")
            not in ([end, end + stage["layers"]], (end, end + stage["layers"]))
            or report.get("world_backend") != "gloo"
            or report.get("model_config_sha256") != model_sha
        ):
            raise ValueError(
                f"rank {rank} startup does not match profile topology/model/Gloo"
            )
        for name, digest in report.get("sources", {}).items():
            relative = (
                "vllm/distributed/device_communicators/"
                if name == "cpu_staging.py"
                else "vllm/distributed/hetero_pp/"
            ) + name
            if sources.get(relative) != digest:
                raise ValueError(f"rank {rank} source SHA does not match {relative}")
        if not report.get("sources"):
            raise ValueError("startup report lacks actual adapter source hashes")
        end += stage["layers"]
    for key in ("vllm", "torch", "torch_npu", "ray", "model_config_sha256", "sources"):
        if any(r.get(key) != reports[0].get(key) for r in reports) or not reports[
            0
        ].get(key):
            raise ValueError(f"workers disagree on actual {key}")
    if end != config.model_config.get_total_num_hidden_layers():
        raise ValueError("topology does not cover model layers")
    for start in (0, 2):
        ids = [
            r.get("identity", {}).get("device_id") for r in reports[start : start + 2]
        ]
        if None in ids or len(set(ids)) != 2:
            raise ValueError("profile requires two distinct actual devices per node")


def _validate_flush(
    directory, receipts, session, topology, *, live=False, comm_timing=None
):
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise ValueError("flush must return exactly four rank receipts")
    ordered = sorted(receipts, key=lambda r: r.get("rank", r.get("pp_rank", -1)))
    timing = comm_timing or ordered[0].get("comm_timing", "shared-clock")
    if timing not in ("shared-clock", "ack"):
        raise ValueError("unknown communication timing")
    raw_files, end = {}, 0
    for rank, receipt in enumerate(ordered):
        if receipt.get("comm_timing", "shared-clock") != timing:
            raise ValueError("rank receipt communication timing mismatch")
        name = f"pp_stage_pp{rank}_tp0.jsonl"
        path = directory / name
        remote_path = Path(receipt.get("trace_path", receipt.get("trace_file", "")))
        if (
            receipt.get("rank", receipt.get("pp_rank")) != rank
            or receipt.get("enabled") is not True
            or receipt.get("complete") is not True
            or receipt.get("pending") != 0
            or receipt.get("errors") != []
            or receipt.get("transport") != _TRANSPORT
            or receipt.get("trace_session", receipt.get("session")) != session
            or remote_path.name != name
        ):
            raise ValueError(f"incomplete or mismatched rank {rank} flush receipt")
        if live and remote_path.resolve() != path.resolve():
            raise ValueError(
                "collect each rank trace into the declared shared trace directory"
            )
        if not path.is_file() or _sha(path) != receipt.get("trace_sha256"):
            raise ValueError(f"raw trace SHA changed or missing: {name}")
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if (
            not rows
            or len(rows) != receipt.get("record_count")
            or sum(bool(r.get("is_warmup")) for r in rows)
            != receipt.get("warmup_records")
            or sum(not r.get("is_warmup", False) for r in rows)
            != receipt.get("measured_records")
            or receipt.get("measured_records", 0) < 1
        ):
            raise ValueError(f"rank {rank} trace count differs from flush")
        stop = end + topology["stages"][rank]["layers"]
        for row in rows:
            if row.get("comm_timing", "shared-clock") != timing:
                raise ValueError("rank raw communication timing mismatch")
            if (
                row.get("pp_rank") != rank
                or row.get("tp_rank") != 0
                or row.get("pp_size") != 4
                or row.get("tp_size") != 1
                or row.get("trace_session") != session
                or row.get("transport") != _TRANSPORT
                or row.get("compute_model") != "layer-measured"
                or row.get("device_id") != rank
                or row.get("profile_device_id", rank) != rank
                or (row.get("start_layer"), row.get("end_layer")) != (end, stop)
            ):
                raise ValueError(f"rank {rank} raw identity/config/transport mismatch")
        raw_files[name] = _sha(path)
        end = stop
    if timing == "ack":
        from vllm.distributed.pp_link_profile import measured_transfer_ms
        from vllm.distributed.pp_partition import load_trace_records

        paired = load_trace_records(directory)
        for rank in range(3):
            for row in paired[rank]:
                measured_transfer_ms(row)
    return ordered, raw_files


def _load_bound_manifest(directory, topology):
    path = directory / "profile_manifest.json"
    if not path.is_file():
        raise ValueError(
            "trusted CPU/Gloo profile_manifest.json is required; "
            "old traces are not admitted"
        )
    manifest = json.loads(path.read_text())
    if (
        manifest.get("version") != 1
        or manifest.get("complete") is not True
        or manifest.get("transport") != _TRANSPORT
        or manifest.get("topology_sha256") != _digest(topology)
        or manifest.get("hetero_plan") != topology
        or manifest.get("engine_config_sha256")
        != _digest(manifest.get("engine_config"))
        or manifest.get("source_manifest_sha256")
        != _digest(manifest.get("source_hashes"))
    ):
        raise ValueError("profile manifest topology/config/transport binding mismatch")
    for relative, digest in manifest["source_hashes"].items():
        file = (_ROOT / relative).resolve()
        if not file.is_relative_to(_ROOT) or not file.is_file() or _sha(file) != digest:
            raise ValueError(f"profile source SHA changed: {relative}")
    _, raw = _validate_flush(
        directory,
        manifest["flush_receipts"],
        manifest["trace_session"],
        topology,
        comm_timing=manifest.get("comm_timing", "shared-clock"),
    )
    if raw != manifest.get("raw_files"):
        raise ValueError("raw trace hash manifest changed")
    for name, key in (
        ("memory_observations.json", "memory_observations_sha256"),
        ("pp_memory_profile.json", "memory_profile_sha256"),
        ("memory_evidence.json", "memory_evidence_sha256"),
        ("engine_config.json", "engine_config_full_sha256"),
        ("model_config.json", "model_config_sha256"),
        ("flush_receipts.json", "flush_receipts_sha256"),
        ("engine_args.json", "engine_args_sha256"),
    ):
        if not (directory / name).is_file() or _sha(directory / name) != manifest.get(
            key
        ):
            raise ValueError(f"profile evidence SHA changed: {name}")
    return manifest


def _collect(*, directory, topology, args, llm_factory, workload, memory):
    llm = None
    supplied_memory = memory is not None
    try:
        llm = llm_factory() if llm_factory else _default_llm_factory(args)
        config = llm.llm_engine.vllm_config
        snapshot = _engine_snapshot(config)
        if memory is not None:
            memory.validate_engine_config(config)
        run_traced_generate(llm, workload)
        receipts = llm.collective_rpc("flush_pp_profile")
        receipts, raw_files = _validate_flush(
            directory,
            receipts,
            os.environ["VLLM_PP_TRACE_SESSION"],
            topology,
            live=True,
            comm_timing=os.environ.get("VLLM_PP_COMM_TIMING", "shared-clock"),
        )
        # Worker provenance crosses the same RPC boundary in both in-process
        # and multiprocess EngineCore mode. The driver need not own Executor.
        reports = [receipt.get("identity") for receipt in receipts]
        _validate_startup(reports, topology, config)
        observations = llm.collective_rpc("get_pp_memory_observation")
        if (
            not isinstance(observations, list)
            or len(observations) != 4
            or sorted(r.get("pp_rank", -1) for r in observations) != list(range(4))
        ):
            raise ValueError("memory observation RPC must cover all four ranks")
        if memory is None:
            from vllm.distributed.hetero_pp.profile_memory import (
                build_fixed_block_memory_profile,
            )

            memory = build_fixed_block_memory_profile(observations, config)
        memory.validate_engine_config(config)
        _write(directory / "memory_observations.json", observations)
        _write(directory / "engine_config.json", _jsonable(config))
        _write(directory / "engine_args.json", _jsonable(args))
        (directory / "model_config.json").write_text(
            config.model_config.hf_config.to_json_string()
        )
        _write(directory / "pp_memory_profile.json", memory.to_dict())
        _write(directory / "flush_receipts.json", receipts)
        from vllm.distributed.hetero_pp.profile_memory import (
            memory_envelope_assumptions,
        )

        evidence = {
            "raw_observations": observations,
            "engine_config_scope": snapshot,
            "memory_profile": memory.to_dict(),
            "basis": (
                "explicit_supplied_bounds"
                if supplied_memory
                else "derived_fixed_block_envelope"
            ),
            "assumptions": (
                "User supplied bounds; worker observations are evidence, "
                "not a new safety guarantee."
                if supplied_memory
                else memory_envelope_assumptions()
            ),
        }
        _write(directory / "memory_evidence.json", evidence)
        sources = _source_hashes()
        manifest = dict(
            version=1,
            complete=True,
            transport=_TRANSPORT,
            comm_timing=os.environ.get("VLLM_PP_COMM_TIMING", "shared-clock"),
            trace_session=os.environ["VLLM_PP_TRACE_SESSION"],
            hetero_plan=topology,
            topology_sha256=_digest(topology),
            engine_config=snapshot,
            engine_config_sha256=_digest(snapshot),
            engine_config_full_sha256=_sha(directory / "engine_config.json"),
            source_hashes=sources,
            source_manifest_sha256=_digest(sources),
            model_config_sha256=reports[0]["model_config_sha256"],
            startup_reports=reports,
            actual_runtimes=[
                {k: r[k] for k in ("rank", "vllm", "torch", "torch_npu", "ray")}
                for r in reports
            ],
            flush_receipts=receipts,
            raw_files=raw_files,
            memory_observations_sha256=_sha(directory / "memory_observations.json"),
            memory_profile_sha256=_sha(directory / "pp_memory_profile.json"),
            memory_evidence_sha256=_sha(directory / "memory_evidence.json"),
            flush_receipts_sha256=_sha(directory / "flush_receipts.json"),
            engine_args_sha256=_sha(directory / "engine_args.json"),
            workload=_jsonable(workload),
            sampling_boundary=(
                "execute_model and sample_tokens are separate measured intervals; "
                "the intervening RPC gap is excluded"
            ),
            throughput_scope=(
                "stage bottleneck candidate only; single-inflight executor "
                "throughput is not optimized or measured by this DP"
            ),
        )
    finally:
        if llm is not None:
            shutdown_llm(llm)
    # A complete manifest is written only after flush, memory and teardown succeed.
    _write(directory / "profile_manifest.json", manifest)
    return manifest, memory


def write_profile_result(plan, dump_dir, output_json=None):
    path = (
        Path(output_json) if output_json else Path(dump_dir) / "pp_partition_plan.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, plan.to_dict())
    return path


def profile_pp_partition(
    *,
    dump_dir=None,
    hetero_plan=None,
    skip_run=False,
    engine_args=None,
    llm_factory: Callable | None = None,
    workload=None,
    num_prompts=16,
    input_len=256,
    output_len=32,
    num_iters=3,
    num_iters_warmup=1,
    seed=0,
    vocab_size=32000,
    objective="latency",
    workload_kind="all",
    warmup_steps=0,
    num_layers=None,
    min_pp_size=4,
    max_pp_size=4,
    memory_profile=None,
    allow_unchecked_memory=False,
    output_json=None,
    compute_model="layer-measured",
    fit_trace_dirs: Sequence[str | Path] = (),
    collect_only=False,
    device_selection=False,
    layer_aggregation="microbatch",
    comm_timing=None,
):
    """Collect/validate bound staged traces and optionally run fixed-order DP."""
    topology = _load_topology(hetero_plan)
    if comm_timing not in (None, "shared-clock", "ack"):
        raise ValueError("unknown communication timing")
    if collect_only and skip_run:
        raise ValueError("--collect-only cannot be combined with --skip-run")
    if (
        compute_model != "layer-measured"
        or device_selection
        or fit_trace_dirs
        or min_pp_size != 4
        or max_pp_size != 4
        or warmup_steps != 0
    ):
        raise ValueError(
            "profile backend requires layer-measured, fixed PP4 order, "
            "no extra fit traces and warmup_steps=0"
        )
    if allow_unchecked_memory:
        raise ValueError("unchecked memory is not a release path for this backend")
    if objective not in ("latency", "throughput"):
        raise ValueError("unknown objective")
    memory = (
        memory_profile
        if isinstance(memory_profile, PPMemoryProfile)
        else PPMemoryProfile.from_file(memory_profile)
        if memory_profile
        else None
    )
    if not skip_run:
        args = _configure_engine(engine_args)
        if memory is None and getattr(args, "num_gpu_blocks_override", None) is None:
            raise ValueError(
                "automatic KV allocation needs an explicit --memory-profile; "
                "otherwise set --num-gpu-blocks-override for the fixed-block envelope"
            )
        previous = {key: os.environ.get(key) for key in _TRACE_ENV}
        try:
            directory = prepare_trace_dir(dump_dir, comm_timing or "shared-clock")
            os.environ["VLLM_HETERO_PP_PLAN"] = json.dumps(topology)
            workload = workload or build_profile_workload(
                num_prompts=num_prompts,
                input_len=input_len,
                output_len=output_len,
                seed=seed,
                vocab_size=vocab_size,
                num_iters=num_iters,
                num_iters_warmup=num_iters_warmup,
            )
            manifest, memory = _collect(
                directory=directory,
                topology=topology,
                args=args,
                llm_factory=llm_factory,
                workload=workload,
                memory=memory,
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    else:
        if dump_dir is None:
            raise ValueError("--trace-dir is required with --skip-run")
        directory = Path(dump_dir).expanduser().resolve()
        manifest = _load_bound_manifest(directory, topology)
        if comm_timing is not None and comm_timing != manifest.get(
            "comm_timing", "shared-clock"
        ):
            raise ValueError(
                "requested communication timing differs from bound profile"
            )
        memory = memory or PPMemoryProfile.from_file(
            directory / "pp_memory_profile.json"
        )
    if collect_only:
        return None
    memory.validate_engine_config(_snapshot_config(manifest["engine_config"]))
    if num_layers is not None and num_layers != manifest["engine_config"]["num_layers"]:
        raise ValueError("num_layers does not match the bound model")
    from vllm.distributed.pp_partition import plan_from_trace_dir

    plan = plan_from_trace_dir(
        directory,
        expected_pp_size=4,
        objective=objective,
        workload=workload_kind,
        warmup_steps=0,
        num_layers=manifest["engine_config"]["num_layers"],
        min_pp_size=4,
        max_pp_size=4,
        memory_profile=memory,
        compute_model="layer-measured",
        device_selection=False,
        layer_aggregation=layer_aggregation,
    )
    write_profile_result(plan, directory, output_json)
    recommendation = copy.deepcopy(topology)
    for stage, count in zip(recommendation["stages"], plan.partitions):
        stage["layers"] = count
    _write(directory / "recommended_hetero_plan.json", recommendation)
    _write(
        directory / "recommendation_receipt.json",
        {
            "profile_manifest_sha256": _sha(directory / "profile_manifest.json"),
            "recommended_hetero_plan_sha256": _sha(
                directory / "recommended_hetero_plan.json"
            ),
            "plan_sha256": _sha(
                Path(output_json)
                if output_json
                else directory / "pp_partition_plan.json"
            ),
            "plan_content_sha256": _digest(plan.to_dict()),
            "objective": objective,
            "trace_off_validation_required": True,
            "measured_performance_improvement": False,
        },
    )
    return plan


def format_serve_command(plan: PPPartitionPlan, dump_dir=None) -> str:
    """Generate the dedicated trace-off entrypoint from the bound runtime scope."""
    if dump_dir is None:
        return (
            "Use the bound profile directory to generate the trace-off "
            "hetero serving command."
        )
    directory = Path(dump_dir).resolve()
    raw = json.loads((directory / "recommended_hetero_plan.json").read_text())
    manifest = _load_bound_manifest(
        directory,
        json.loads((directory / "profile_manifest.json").read_text())["hetero_plan"],
    )
    expected = copy.deepcopy(manifest["hetero_plan"])
    for stage, count in zip(expected["stages"], plan.partitions):
        stage["layers"] = count
    receipt = json.loads((directory / "recommendation_receipt.json").read_text())
    if (
        raw != expected
        or receipt.get("recommended_hetero_plan_sha256")
        != _sha(directory / "recommended_hetero_plan.json")
        or receipt.get("profile_manifest_sha256")
        != _sha(directory / "profile_manifest.json")
        or receipt.get("plan_content_sha256") != _digest(plan.to_dict())
    ):
        raise ValueError(
            "recommended hetero topology/plan SHA does not match DP output"
        )
    scope = manifest["engine_config"]
    command = [
        sys.executable,
        "-m",
        "vllm.distributed.hetero_pp.serve",
        "--model",
        scope["model"],
        "--pipeline-parallel-size",
        "4",
        "--tensor-parallel-size",
        "1",
        "--enforce-eager",
        "--no-async-scheduling",
        "--dtype",
        scope["dtype"],
    ]
    for key in (
        "revision",
        "quantization",
        "kv_cache_dtype",
        "block_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "num_gpu_blocks_override",
        "kv_cache_memory_bytes",
        "seed",
    ):
        if scope.get(key) is not None:
            command += ["--" + key.replace("_", "-"), str(scope[key])]
    for key in (
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "disable_cascade_attn",
    ):
        if scope.get(key) is not None:
            command.append("--" + ("" if scope[key] else "no-") + key.replace("_", "-"))
    served = scope.get("served_model_name")
    if served:
        command += ["--served-model-name"]
        command += [served] if isinstance(served, str) else served
    prefix = (
        "env -u VLLM_PP_STAGE_TRACE -u VLLM_PP_TRACE_SESSION -u VLLM_PP_COMPUTE_MODEL "
    )
    prefix += (
        "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_HETERO_PP_PLAN="
        + shlex.quote(json.dumps(raw))
        + " "
    )
    return (
        "Trace-off candidate; validate full token IDs and actual memory "
        "after re-serve:\n" + prefix + shlex.join(command)
    )


def add_cli_args(parser):
    from vllm.engine.arg_utils import EngineArgs

    parser.add_argument("model_tag", nargs="?")
    parser.add_argument(
        "--hetero-plan", type=str, help="Required CPU/Gloo topology JSON."
    )
    parser.add_argument("--trace-dir", type=str)
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--comm-timing",
        choices=("shared-clock", "ack"),
        default=None,
        help=(
            "Live default: shared-clock; use ack across physical hosts. "
            "Offline default: infer bound mode."
        ),
    )
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--objective",
        choices=("latency", "throughput"),
        default="latency",
        help=(
            "throughput means a stage-bottleneck candidate, "
            "not measured single-inflight throughput."
        ),
    )
    parser.add_argument(
        "--workload",
        dest="workload_kind",
        choices=("all", "prefill", "decode", "mixed"),
        default="all",
    )
    parser.add_argument(
        "--compute-model", choices=("layer-measured",), default="layer-measured"
    )
    parser.add_argument(
        "--layer-aggregation",
        choices=("microbatch", "phase-balanced"),
        default="microbatch",
    )
    parser.add_argument("--memory-profile", type=str)
    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        pipeline_parallel_size=4, tensor_parallel_size=1, enable_prefix_caching=False
    )
    return parser


def run_from_cli_args(args):
    validate_profile_args(args)
    if args.model_tag is not None:
        args.model = args.model_tag
    engine_args = None
    if not args.skip_run:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs.from_cli_args(args)
    return profile_pp_partition(
        dump_dir=args.trace_dir,
        hetero_plan=args.hetero_plan,
        skip_run=args.skip_run,
        engine_args=engine_args,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        num_iters=args.num_iters,
        num_iters_warmup=args.num_iters_warmup,
        seed=getattr(args, "seed", 0) or 0,
        vocab_size=args.vocab_size,
        objective=args.objective,
        workload_kind=args.workload_kind,
        memory_profile=args.memory_profile,
        output_json=args.output_json,
        compute_model=args.compute_model,
        layer_aggregation=args.layer_aggregation,
        collect_only=args.collect_only,
        comm_timing=getattr(args, "comm_timing", None),
    )


def main(argv=None):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = add_cli_args(FlexibleArgumentParser(description=__doc__))
    args = parser.parse_args(argv)
    plan = run_from_cli_args(args)
    if plan is not None:
        print(format_plan(plan))
        print(format_serve_command(plan, args.trace_dir))


if __name__ == "__main__":
    main()
