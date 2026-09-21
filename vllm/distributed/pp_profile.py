# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end PP stage profiling: dummy load → trace → DP split.

``vllm pp-profile`` (or :func:`profile_pp_partition`) does four things:

1. Build a synthetic token workload (no dataset required).
2. Launch the engine with ``VLLM_PP_STAGE_TRACE`` set and feed that workload.
3. Fit per-rank layer compute / hop communication from the JSONL traces.
4. Run the EdgeShard-style DP and write ``VLLM_PP_LAYER_PARTITION``.

Each step is a public function so unit tests can exercise it without a GPU.
"""

from __future__ import annotations

import json
import os
import random
import shlex
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from vllm.distributed.pp_memory import PPMemoryProfile, resolve_memory_profile
from vllm.distributed.pp_partition import (
    ComputeModel,
    Objective,
    PPPartitionPlan,
    Workload,
    add_cost_model_args,
    format_plan,
    load_trace_records,
    plan_from_trace_dir,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

PromptTokens = dict[str, list[int]]
LLMFactory = Callable[[], Any]


@dataclass(frozen=True)
class ProfileWorkload:
    """Synthetic generate() inputs used while tracing a PP engine."""

    prompts: list[PromptTokens]
    max_tokens: int
    num_iters: int = 1
    num_iters_warmup: int = 0
    ignore_eos: bool = True
    temperature: float = 1.0


def build_profile_workload(
    *,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int = 0,
    vocab_size: int = 32000,
    num_iters: int = 1,
    num_iters_warmup: int = 0,
) -> ProfileWorkload:
    """Build dummy ``prompt_token_ids`` plus generate lengths.

    Args:
        num_prompts: Concurrent prompts in each generate() call.
        input_len: Tokens per prompt (prefill width).
        output_len: Tokens to generate (decode length).
        seed: RNG seed so two calls with the same args match.
        vocab_size: Exclusive upper bound for sampled token ids.
        num_iters: Traced generate() calls after warmup.
        num_iters_warmup: Extra generate() calls before the traced iters.

    Returns:
        A workload the engine runner can feed unchanged.

    Raises:
        ValueError: If any size is not a positive integer, or vocab_size < 2.
    """
    if num_prompts < 1:
        raise ValueError(f"num_prompts must be >= 1, got {num_prompts}")
    if input_len < 1:
        raise ValueError(f"input_len must be >= 1, got {input_len}")
    if output_len < 1:
        raise ValueError(f"output_len must be >= 1, got {output_len}")
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}")
    if num_iters_warmup < 0:
        raise ValueError(f"num_iters_warmup must be >= 0, got {num_iters_warmup}")

    rng = random.Random(seed)
    prompts: list[PromptTokens] = []
    for _ in range(num_prompts):
        token_ids = [rng.randrange(1, vocab_size) for _ in range(input_len)]
        prompts.append({"prompt_token_ids": token_ids})
    return ProfileWorkload(
        prompts=prompts,
        max_tokens=output_len,
        num_iters=num_iters,
        num_iters_warmup=num_iters_warmup,
    )


def prepare_trace_dir(dump_dir: str | Path | None = None) -> Path:
    """Create the trace directory and export ``VLLM_PP_STAGE_TRACE``.

    Must run *before* the engine is constructed so workers inherit the env.

    Args:
        dump_dir: Destination. ``None`` reuses the env var, else a temp dir.

    Returns:
        Absolute path of the trace directory.
    """
    if dump_dir is None:
        existing = os.environ.get("VLLM_PP_STAGE_TRACE")
        dump_dir = existing or tempfile.mkdtemp(prefix="vllm_pp_profile_")
    path = Path(dump_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_PP_STAGE_TRACE"] = str(path)
    os.environ["VLLM_PP_TRACE_SESSION"] = uuid.uuid4().hex
    return path


def require_complete_traces(
    dump_dir: str | Path,
    pp_size: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Load traces and fail if any expected PP rank is missing or empty.

    Args:
        dump_dir: Directory of ``pp_stage_pp*_tp*.jsonl`` files.
        pp_size: If set, require exactly this many contiguous ranks from 0.

    Returns:
        Records grouped by PP rank.

    Raises:
        FileNotFoundError: Missing files or missing ranks.
        ValueError: A rank file exists but has no records.
    """
    records = load_trace_records(dump_dir)
    if not records:
        raise ValueError(f"empty traces in {dump_dir}")
    ranks = sorted(records)
    if ranks != list(range(len(ranks))):
        raise FileNotFoundError(
            f"PP ranks must be contiguous from 0, found {ranks} in {dump_dir}"
        )
    if pp_size is not None and len(ranks) != pp_size:
        raise FileNotFoundError(
            f"expected traces for {pp_size} PP ranks, found {ranks} in {dump_dir}"
        )
    empty = [rank for rank, recs in records.items() if not recs]
    if empty:
        raise ValueError(f"empty traces for PP ranks {empty} in {dump_dir}")
    return records


def run_traced_generate(llm: Any, workload: ProfileWorkload) -> None:
    """Feed ``workload`` into ``llm.generate`` (warmup then measured iters).

    Args:
        llm: Object exposing ``generate(prompts, sampling_params, use_tqdm=)``.
        workload: Dummy prompts and decode length.
    """
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        temperature=workload.temperature,
        ignore_eos=workload.ignore_eos,
        max_tokens=workload.max_tokens,
    )
    total = workload.num_iters_warmup + workload.num_iters
    for i in range(total):
        if callable(rpc := getattr(llm, "collective_rpc", None)):
            rpc("set_pp_profile_warmup", args=(i < workload.num_iters_warmup,))
        logger.info(
            "PP profile generate %d/%d (%s)",
            i + 1,
            total,
            "warmup" if i < workload.num_iters_warmup else "traced",
        )
        llm.generate(
            workload.prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )


def shutdown_llm(llm: Any) -> None:
    """Flush PP tracers by shutting the engine core if it exposes shutdown."""
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        shutdown()


def write_profile_result(
    plan: PPPartitionPlan,
    dump_dir: str | Path,
    output_json: str | Path | None = None,
) -> Path:
    """Write ``plan`` as JSON next to the traces (or to ``output_json``).

    Args:
        plan: DP result.
        dump_dir: Default parent of ``pp_partition_plan.json``.
        output_json: Optional explicit path.

    Returns:
        Path of the written JSON file.
    """
    path = (
        Path(output_json)
        if output_json is not None
        else Path(dump_dir) / "pp_partition_plan.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.to_dict()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if plan.memory_profile is not None:
        memory_path = Path(dump_dir) / "pp_memory_profile.json"
        memory_path.write_text(
            json.dumps(plan.memory_profile.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    logger.info("Wrote PP partition plan to %s", path)
    return path


def format_serve_command(plan: PPPartitionPlan) -> str:
    """Shell snippet with the chosen split and declared serving configuration."""
    if plan.memory_profile is None:
        return (
            "Timing-only candidate: memory feasibility was not checked. "
            "Supply --memory-profile before generating a serve command."
        )
    lines = ["Re-serve with:"]
    exports = [f"VLLM_PP_LAYER_PARTITION={plan.env_value}"]
    for item in exports:
        name, value = item.split("=", 1)
        lines.append(f"  {name}={shlex.quote(value)} \\")
    scope = plan.memory_profile.serving_config
    command = [
        "vllm",
        "serve",
        scope["model"],
        "--pipeline-parallel-size",
        str(plan.pp_size),
        "--tensor-parallel-size",
        str(plan.memory_profile.tp_size),
    ]
    for key, value in scope.items():
        if key != "model" and value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    lines.append("    " + shlex.join(command))
    lines.append(
        "Use the same device budgets and runtime settings as the memory profile."
    )
    return "\n".join(lines)


def clear_trace_files(dump_dir: str | Path) -> None:
    """Remove leftover ``pp_stage_pp*_tp*.jsonl`` so a live run cannot reuse them."""
    dump_dir = Path(dump_dir)
    for pattern in ("pp_stage_pp*_tp*.jsonl", "pp_link_pp*_tp*.jsonl"):
        for path in dump_dir.glob(pattern):
            path.unlink()


def _ensure_ascend_pp_worker(engine_args: Any) -> None:
    """Use the tracing NPUWorker subclass when vllm-ascend is installed."""
    try:
        import vllm_ascend  # noqa: F401
    except ImportError:
        return
    current = getattr(engine_args, "worker_cls", None)
    if current in (None, "", "auto"):
        engine_args.worker_cls = "vllm.v1.worker.pp_ascend_worker.PPAscendWorker"


def _default_llm_factory(engine_args: Any) -> Any:
    from vllm import LLM

    _ensure_ascend_pp_worker(engine_args)
    if hasattr(LLM, "from_engine_args"):
        return LLM.from_engine_args(engine_args)
    return LLM(**vars(engine_args))


def _pipeline_parallel_size(engine_args: Any | None) -> int | None:
    if engine_args is None:
        return None
    return int(engine_args.pipeline_parallel_size)


def profile_pp_partition(
    *,
    dump_dir: str | Path | None = None,
    skip_run: bool = False,
    engine_args: Any | None = None,
    llm_factory: LLMFactory | None = None,
    workload: ProfileWorkload | None = None,
    num_prompts: int = 16,
    input_len: int = 256,
    output_len: int = 32,
    num_iters: int = 3,
    num_iters_warmup: int = 1,
    seed: int = 0,
    vocab_size: int = 32000,
    objective: Objective = "throughput",
    workload_kind: Workload = "all",
    warmup_steps: int = 5,
    num_layers: int | None = None,
    min_pp_size: int = 1,
    max_pp_size: int | None = None,
    overlap_comm: bool = False,
    memory_profile: PPMemoryProfile | str | Path | None = None,
    allow_unchecked_memory: bool = False,
    output_json: str | Path | None = None,
    compute_model: ComputeModel = "shape-affine",
    fit_trace_dirs: Sequence[str | Path] = (),
    collect_only: bool = False,
    comm_source: str = "serving",
) -> PPPartitionPlan | None:
    """Run the profile pipeline and return the DP layer split.

    Costs come from measured traces. Mock slowdown, if enabled, executes
    on workers during collection; offline planning never rescales traces.

    Args:
        dump_dir: Trace directory. Created if missing.
        skip_run: If True, only fit+DP existing traces (no engine).
        engine_args: ``EngineArgs`` used when ``llm_factory`` is omitted.
        llm_factory: Injectable engine constructor for tests.
        workload: Pre-built dummy prompts; built from the size knobs if None.
        num_prompts: Dummy concurrent prompts when ``workload`` is None.
        input_len: Dummy prefill length when ``workload`` is None.
        output_len: Dummy decode length when ``workload`` is None.
        num_iters: Traced generate() calls.
        num_iters_warmup: Extra generate() calls before traced iters.
        seed: Dummy-token RNG seed.
        vocab_size: Dummy-token exclusive upper bound.
        objective: ``throughput`` or ``latency`` (see pp_partition).
        workload_kind: Which traced steps to fit.
        warmup_steps: Leading steps dropped when fitting costs.
        num_layers: Optional hidden-layer count override.
        min_pp_size: Smallest PP size the DP may choose.
        max_pp_size: Largest PP size the DP may choose.
        overlap_comm: Throughput DP uses max(compute, comm) when True.
        memory_profile: Explicit target serving memory bounds, or sidecar JSON.
        allow_unchecked_memory: Permit timing-only analysis without bounds.
        output_json: Optional plan JSON path.
        compute_model: Measured multi-shard fixed/per-layer cost fit.
        fit_trace_dirs: Extra shard profiles; dump_dir sets the workload weights.
        collect_only: Save traces without fitting (returns None).

    Returns:
        The chosen partition (written under ``dump_dir``), or None for collect-only.

    Raises:
        ValueError: PP size is 1 for a live run, or workload knobs are invalid.
        FileNotFoundError: Traces are missing after the run.
    """
    if collect_only and skip_run:
        raise ValueError("--collect-only cannot be combined with --skip-run")
    pp_size = _pipeline_parallel_size(engine_args)
    memory_profile = resolve_memory_profile(
        memory_profile,
        dump_dir,
        allow_unchecked_memory=allow_unchecked_memory or collect_only,
    )
    planner_hetero: str | None
    if not skip_run:
        if llm_factory is None:
            if engine_args is None:
                raise ValueError(
                    "engine_args or llm_factory is required unless --skip-run"
                )
            if pp_size is None or pp_size <= 1:
                raise ValueError(
                    "pipeline_parallel_size must be > 1 to profile PP stages "
                    f"(got {pp_size})"
                )
            llm_factory = partial(_default_llm_factory, engine_args)
        elif pp_size is not None and pp_size <= 1:
            raise ValueError(
                "pipeline_parallel_size must be > 1 to profile PP stages "
                f"(got {pp_size})"
            )
        dump_dir = prepare_trace_dir(dump_dir)
        clear_trace_files(dump_dir)
        if workload is None:
            workload = build_profile_workload(
                num_prompts=num_prompts,
                input_len=input_len,
                output_len=output_len,
                seed=seed,
                vocab_size=vocab_size,
                num_iters=num_iters,
                num_iters_warmup=num_iters_warmup,
            )
        llm = llm_factory()
        try:
            if memory_profile is not None:
                config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
                if config is None:
                    raise ValueError(
                        "cannot validate memory profile: missing engine config"
                    )
                memory_profile.validate_engine_config(config)
            run_traced_generate(llm, workload)
            if comm_source == "replay":
                llm.collective_rpc("profile_pp_links", args=())
        finally:
            shutdown_llm(llm)
        require_complete_traces(dump_dir, pp_size=pp_size)
        # Traces already include worker stretch; do not scale again.
        planner_hetero = ""
    else:
        if dump_dir is None:
            raise ValueError("--trace-dir is required with --skip-run")
        dump_dir = Path(dump_dir)
        require_complete_traces(dump_dir, pp_size=pp_size)
        planner_hetero = None

    if collect_only:
        if memory_profile is not None:
            (Path(dump_dir) / "pp_memory_profile.json").write_text(
                json.dumps(memory_profile.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        logger.info("Collected PP traces in %s; no partition was fitted", dump_dir)
        return None

    plan = plan_from_trace_dir(
        dump_dir,
        objective=objective,
        workload=workload_kind,
        warmup_steps=warmup_steps,
        num_layers=num_layers,
        min_pp_size=min_pp_size,
        max_pp_size=max_pp_size,
        overlap_comm=overlap_comm,
        memory_profile=memory_profile,
        allow_unchecked_memory=allow_unchecked_memory,
        hetero=planner_hetero,
        compute_model=compute_model,
        fit_trace_dirs=fit_trace_dirs,
        comm_source=comm_source,
    )
    write_profile_result(plan, dump_dir, output_json=output_json)
    logger.info("PP profile result:\n%s", format_plan(plan))
    return plan


def add_cli_args(parser: Any) -> Any:
    """Register ``vllm pp-profile`` flags, including EngineArgs."""
    from vllm.engine.arg_utils import EngineArgs

    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="Model to load (optional if --model is set).",
    )
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=None,
        help="Directory for pp_stage_pp*_tp*.jsonl (default: a temp dir).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Plan JSON path (default: <trace-dir>/pp_partition_plan.json).",
    )
    parser.add_argument(
        "--objective",
        choices=("latency", "throughput"),
        default="throughput",
        help="latency = sequential sum; throughput = pipeline bottleneck.",
    )
    parser.add_argument(
        "--workload",
        dest="workload_kind",
        choices=("all", "decode", "prefill", "mixed"),
        default="all",
        help="Which traced steps to fit costs from.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument(
        "--min-pp-size",
        type=int,
        default=1,
        help="Smallest PP size the DP may choose.",
    )
    parser.add_argument(
        "--max-pp-size",
        type=int,
        default=None,
        help="Largest PP size the DP may choose (default: traced ranks).",
    )
    add_cost_model_args(parser)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect a shard profile without fitting a partition; preserves traces.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not launch the engine; only fit+DP traces in --trace-dir.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Exclusive upper bound for dummy prompt token ids.",
    )
    parser = EngineArgs.add_cli_args(parser)
    # Prefix cache would skip prefill compute on later iters.
    parser.set_defaults(enable_prefix_caching=False)
    return parser


def run_from_cli_args(args: Any) -> PPPartitionPlan | None:
    """Dispatch :func:`profile_pp_partition` from parsed CLI args."""
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag

    engine_args = None
    if not args.skip_run:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs.from_cli_args(args)

    return profile_pp_partition(
        dump_dir=args.trace_dir,
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
        warmup_steps=args.warmup_steps,
        num_layers=args.num_layers,
        min_pp_size=args.min_pp_size,
        max_pp_size=args.max_pp_size,
        overlap_comm=args.overlap_comm,
        memory_profile=args.memory_profile,
        allow_unchecked_memory=args.allow_unchecked_memory,
        output_json=args.output_json,
        compute_model=args.compute_model,
        fit_trace_dirs=args.fit_trace_dirs,
        comm_source=args.comm_source,
        collect_only=args.collect_only,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_args(parser)
    args = parser.parse_args(list(argv) if argv is not None else None)
    plan = run_from_cli_args(args)
    if plan is not None:
        print(format_plan(plan))
        print()
        print(format_serve_command(plan))


if __name__ == "__main__":
    main()
