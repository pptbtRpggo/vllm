# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Controller contracts; the engine double supplies genuine raw trace files."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_profile():
    path = ROOT / "vllm/distributed/pp_profile.py"
    if not path.exists():
        return None
    import sys

    spec = importlib.util.spec_from_file_location("control_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def topology():
    return {
        "validation_mode": "910b_only",
        "timeout_s": 30,
        "stages": [
            {"node_ip": "10.0.0.1", "soc": "910B", "layers": 2},
            {"node_ip": "10.0.0.1", "soc": "910B", "layers": 2},
            {"node_ip": "10.0.0.2", "soc": "910B", "layers": 2},
            {"node_ip": "10.0.0.2", "soc": "910B", "layers": 2},
        ],
    }


def engine_args():
    return NS(
        pipeline_parallel_size=4,
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        distributed_executor_backend=None,
        num_gpu_blocks_override=None,
        kv_cache_memory_bytes=None,
    )


def config():
    model = NS(
        model="fixture-model",
        revision=None,
        dtype="float16",
        quantization=None,
        max_model_len=128,
        enforce_eager=True,
        disable_cascade_attn=False,
        hf_config=NS(to_json_string=lambda: '{"num_hidden_layers":8}'),
        get_total_num_hidden_layers=lambda: 8,
    )
    return NS(
        model_config=model,
        parallel_config=NS(pipeline_parallel_size=4, tensor_parallel_size=1),
        cache_config=NS(
            cache_dtype="auto",
            block_size=16,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=False,
            num_gpu_blocks_override=None,
            kv_cache_memory_bytes=None,
        ),
        scheduler_config=NS(
            max_num_seqs=2,
            max_num_batched_tokens=32,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
        use_v2_model_runner=False,
    )


def memory_profile():
    from vllm.distributed.pp_memory import DeviceMemory, PPMemoryProfile

    scope = dict(
        model="fixture-model",
        revision=None,
        dtype="float16",
        quantization=None,
        kv_cache_dtype="auto",
        block_size=16,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.9,
    )
    devices = tuple(
        DeviceMemory(rank, 0, 1_000_000, (1,) * 8, (1,) * 8, 0, 0, 0, 0, 0, 0, 1, 1)
        for rank in range(4)
    )
    return PPMemoryProfile(8, 4, 1, scope, devices)


class FixtureEngine:
    """Only device execution is replaced; controller file/RPC checks stay real."""

    def __init__(
        self,
        *,
        bad_sha=False,
        missing_rank=False,
        bad_device=False,
        expose_executor=True,
    ):
        self.calls = []
        self.bad_sha, self.missing_rank = bad_sha, missing_rank
        self.bad_device = bad_device
        self.closed = False
        self.rows: list[list[dict[str, Any]]] = [[] for _ in range(4)]
        self.warmup = False
        cfg = config()
        model_sha = hashlib.sha256(
            cfg.model_config.hf_config.to_json_string().encode()
        ).hexdigest()
        sources = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "vllm/distributed/hetero_pp").glob("*.py")
        }
        sources["cpu_staging.py"] = hashlib.sha256(
            (ROOT / "vllm/distributed/device_communicators/cpu_staging.py").read_bytes()
        ).hexdigest()
        reports = [
            dict(
                rank=r,
                soc="910B",
                node_ip=topology()["stages"][r]["node_ip"],
                layers=[r * 2, r * 2 + 2],
                identity={"node_id": "node" + str(r // 2), "device_id": r % 2},
                vllm="0.23.0",
                torch="2.10.0",
                torch_npu="2.10.0.post4",
                ray="2.48.0",
                sources=sources,
                model_config_sha256=model_sha,
                world_backend="gloo",
            )
            for r in range(4)
        ]
        self.llm_engine = NS(
            vllm_config=cfg,
            model_executor=NS(startup_reports=reports),
            engine_core=NS(shutdown=self.shutdown),
        )
        self.worker_reports = reports
        if not expose_executor:
            del self.llm_engine.model_executor

    def shutdown(self):
        self.closed = True

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append(("generate", self.warmup, prompts))
        step = len(self.rows[0])
        for rank in range(4):
            self.rows[rank].append(
                dict(
                    step=step,
                    sequence=step,
                    pp_rank=rank,
                    pp_size=4,
                    tp_rank=0,
                    tp_size=1,
                    trace_session=os.environ["VLLM_PP_TRACE_SESSION"],
                    transport="cpu_gloo",
                    device_id=rank % 2 if self.bad_device else rank,
                    is_warmup=self.warmup,
                    clock_domain="fixture-clock",
                    batch_id="batch-" + str(step),
                    start_layer=rank * 2,
                    end_layer=rank * 2 + 2,
                    num_tokens=2,
                    num_reqs=1,
                    batch_shape=[
                        dict(query_tokens=2, context_tokens=0, prompt_tokens=2)
                    ],
                    num_ctx_requests=1,
                    num_ctx_tokens=2,
                    num_generation_requests=0,
                    num_generation_tokens=0,
                    compute_model="layer-measured",
                    layer_compute_ms={str(rank * 2): 1.0, str(rank * 2 + 1): 1.0},
                    compute_ms=3.0,
                    compute_wall_ms=3.0,
                    non_layer_compute_ms=1.0,
                    runner_overhead_ms=0.75 if rank == 0 else 0.5 if rank == 3 else 1.0,
                    embedding_ms=0.25 if rank == 0 else None,
                    final_norm_ms=0.25 if rank == 3 else None,
                    lm_head_ms=0.25 if rank == 3 else None,
                    compute_scale=1.0,
                    comm_scale=1.0,
                    recv_bytes=None if rank == 0 else 8,
                    send_bytes=None if rank == 3 else 8,
                    send_start_ns=(
                        None if rank == 3 else step * 1_000_000 + rank * 10_000 + 1
                    ),
                    send_end_ns=(
                        None if rank == 3 else step * 1_000_000 + rank * 10_000 + 1001
                    ),
                    recv_start_ns=(
                        None
                        if rank == 0
                        else step * 1_000_000 + (rank - 1) * 10_000 + 1
                    ),
                    recv_end_ns=(
                        None
                        if rank == 0
                        else step * 1_000_000 + (rank - 1) * 10_000 + 1001
                    ),
                )
            )
        return []

    def collective_rpc(self, method, args=(), **kwargs):
        self.calls.append(("rpc", method))
        if method == "set_pp_profile_warmup":
            self.warmup = args[0]
            return [None] * 4
        if method == "get_pp_memory_observation":
            return [dict(pp_rank=r, tp_rank=0, observation="fixture") for r in range(4)]
        if method != "flush_pp_profile":
            raise AssertionError(method)
        receipts = []
        directory = Path(os.environ["VLLM_PP_STAGE_TRACE"])
        for rank, rows in enumerate(self.rows):
            path = directory / f"pp_stage_pp{rank}_tp0.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            receipts.append(
                dict(
                    enabled=True,
                    complete=True,
                    rank=rank,
                    pp_rank=rank,
                    trace_path=str(path),
                    trace_file=str(path),
                    trace_sha256=(
                        "bad"
                        if self.bad_sha
                        else hashlib.sha256(path.read_bytes()).hexdigest()
                    ),
                    record_count=len(rows),
                    rows=len(rows),
                    warmup_records=sum(x["is_warmup"] for x in rows),
                    measured_records=sum(not x["is_warmup"] for x in rows),
                    pending=0,
                    errors=[],
                    session=os.environ["VLLM_PP_TRACE_SESSION"],
                    trace_session=os.environ["VLLM_PP_TRACE_SESSION"],
                    transport="cpu_gloo",
                    clock_domain="fixture-clock",
                    identity=self.worker_reports[rank],
                )
            )
        return receipts[:-1] if self.missing_rank else receipts


class ProfileControllerTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_profile()
        self.assertIsNotNone(self.mod, "v0.23 profiling controller is not implemented")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "trace"

    def collect(self, engine):
        return self.mod.profile_pp_partition(
            dump_dir=self.directory,
            hetero_plan=topology(),
            engine_args=engine_args(),
            llm_factory=lambda: engine,
            memory_profile=memory_profile(),
            collect_only=True,
            num_prompts=1,
            input_len=2,
            output_len=2,
            num_iters=1,
            num_iters_warmup=1,
        )

    def test_nonempty_trace_directory_is_preserved(self):
        self.directory.mkdir()
        receipt = self.directory / "successful_receipt.json"
        receipt.write_text("original")
        with self.assertRaisesRegex(ValueError, "empty|new"):
            self.mod.prepare_trace_dir(self.directory)
        self.assertEqual(receipt.read_text(), "original")

    def test_missing_plan_is_rejected_before_engine_creation(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "hetero.plan"):
            self.mod.profile_pp_partition(
                engine_args=engine_args(),
                llm_factory=lambda: calls.append(True),
                memory_profile=memory_profile(),
            )
        self.assertFalse(calls)

    def test_unreplayed_custom_model_and_compute_flags_are_rejected(self):
        for key, value in (
            ("hf_overrides", {"num_hidden_layers": 9}),
            ("tokenizer", "/alternate/tokenizer"),
            ("attention_config", {"backend": "CUSTOM"}),
            ("load_format", "dummy"),
        ):
            args = engine_args()
            setattr(args, key, value)
            with self.subTest(flag=key), self.assertRaisesRegex(ValueError, key):
                self.mod._configure_engine(args)

    def test_core_default_cascade_flag_is_admitted(self):
        args = engine_args()
        args.disable_cascade_attn = True
        selected = self.mod._configure_engine(args)
        self.assertTrue(selected.disable_cascade_attn)
        self.assertIsNotNone(selected.distributed_executor_backend)

    def test_generated_command_replays_effective_cascade_bool(self):
        for effective in (True, False):
            with (
                self.subTest(effective=effective),
                tempfile.TemporaryDirectory() as tmp,
            ):
                self.directory = Path(tmp) / "trace"
                engine = FixtureEngine(expose_executor=False)
                engine.llm_engine.vllm_config.model_config.disable_cascade_attn = (
                    effective
                )
                self.collect(engine)
                plan = self.mod.profile_pp_partition(
                    dump_dir=self.directory, hetero_plan=topology(), skip_run=True
                )
                manifest = json.loads(
                    (self.directory / "profile_manifest.json").read_text()
                )
                self.assertIs(
                    manifest["engine_config"]["disable_cascade_attn"], effective
                )
                command = self.mod.format_serve_command(plan, self.directory)
                words = shlex.split(command.splitlines()[-1])
                parser = argparse.ArgumentParser(add_help=False)
                parser.add_argument(
                    "--disable-cascade-attn",
                    action=argparse.BooleanOptionalAction,
                    default=None,
                )
                actual, _ = parser.parse_known_args(words[words.index("-m") + 2 :])
                self.assertIs(actual.disable_cascade_attn, effective)

    def test_workload_uses_disjoint_warmup_and_measured_tokens(self):
        workload = self.mod.build_profile_workload(
            num_prompts=3,
            input_len=4,
            output_len=2,
            num_iters=2,
            num_iters_warmup=1,
            seed=5,
        )
        warm = {tuple(x["prompt_token_ids"]) for x in workload.warmup_prompts}
        measured = [
            tuple(x["prompt_token_ids"])
            for rows in workload.iteration_prompts
            for x in rows
        ]
        self.assertEqual(len(set(measured)), 6)
        self.assertFalse(warm.intersection(measured))

    def test_collection_saves_trusted_flush_memory_and_bound_manifest(self):
        engine = FixtureEngine()
        self.assertIsNone(self.collect(engine))
        manifest = json.loads((self.directory / "profile_manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["transport"], "cpu_gloo")
        self.assertEqual(len(manifest["flush_receipts"]), 4)
        self.assertTrue(engine.closed)
        memory = json.loads((self.directory / "memory_observations.json").read_text())
        self.assertEqual([r["pp_rank"] for r in memory], [0, 1, 2, 3])
        self.assertEqual(
            [c[1] for c in engine.calls if c[0] == "generate"], [True, False]
        )
        self.assertTrue((self.directory / "pp_memory_profile.json").exists())

    def test_collection_uses_worker_provenance_with_multiprocess_core(self):
        engine = FixtureEngine(expose_executor=False)
        self.collect(engine)
        manifest = json.loads((self.directory / "profile_manifest.json").read_text())
        self.assertEqual([r["rank"] for r in manifest["startup_reports"]], [0, 1, 2, 3])

    def test_collection_rejects_incomplete_or_corrupt_rank_receipts(self):
        for engine in (
            FixtureEngine(missing_rank=True),
            FixtureEngine(bad_sha=True),
            FixtureEngine(bad_device=True),
        ):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as tmp:
                self.directory = Path(tmp) / "trace"
                with self.assertRaises(ValueError):
                    self.collect(engine)
                self.assertFalse((self.directory / "profile_manifest.json").exists())
                self.assertTrue(engine.closed)

    def test_offline_rejects_unbound_old_trace(self):
        self.directory.mkdir()
        (self.directory / "pp_stage_pp0_tp0.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.mod.profile_pp_partition(
                dump_dir=self.directory,
                hetero_plan=topology(),
                skip_run=True,
                memory_profile=memory_profile(),
            )

    def test_offline_plan_reuses_raw_and_rejects_tampering(self):
        self.collect(FixtureEngine())
        plan = self.mod.profile_pp_partition(
            dump_dir=self.directory, hetero_plan=topology(), skip_run=True
        )
        self.assertEqual(len(plan.partitions), 4)
        self.assertEqual(sum(plan.partitions), 8)
        recommended = json.loads(
            (self.directory / "recommended_hetero_plan.json").read_text()
        )
        self.assertEqual(
            [s["node_ip"] for s in recommended["stages"]],
            [s["node_ip"] for s in topology()["stages"]],
        )
        raw = self.directory / "pp_stage_pp1_tp0.jsonl"
        raw.write_text(raw.read_text() + "{}\n")
        with self.assertRaisesRegex(ValueError, "SHA|hash|changed"):
            self.mod.profile_pp_partition(
                dump_dir=self.directory, hetero_plan=topology(), skip_run=True
            )

    def test_offline_rejects_changed_full_runtime_configuration(self):
        self.collect(FixtureEngine())
        runtime = self.directory / "engine_config.json"
        runtime.write_text('{"cache_config":{"enable_prefix_caching":true}}')
        with self.assertRaisesRegex(ValueError, "SHA|hash|changed"):
            self.mod.profile_pp_partition(
                dump_dir=self.directory, hetero_plan=topology(), skip_run=True
            )

    def test_offline_rejects_changed_checkpoint_configuration(self):
        self.collect(FixtureEngine())
        (self.directory / "model_config.json").write_text('{"num_hidden_layers":9}')
        with self.assertRaisesRegex(ValueError, "SHA|hash|changed"):
            self.mod.profile_pp_partition(
                dump_dir=self.directory, hetero_plan=topology(), skip_run=True
            )

    def test_offline_rejects_changed_original_engine_arguments(self):
        self.collect(FixtureEngine())
        (self.directory / "engine_args.json").write_text('{"hf_overrides":{"x":1}}')
        with self.assertRaisesRegex(ValueError, "SHA|hash|changed"):
            self.mod.profile_pp_partition(
                dump_dir=self.directory, hetero_plan=topology(), skip_run=True
            )

    def test_generated_command_rejects_changed_node_placement(self):
        self.collect(FixtureEngine())
        plan = self.mod.profile_pp_partition(
            dump_dir=self.directory, hetero_plan=topology(), skip_run=True
        )
        path = self.directory / "recommended_hetero_plan.json"
        changed = json.loads(path.read_text())
        changed["stages"][2]["node_ip"] = "10.0.0.9"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "recommend|topology|SHA"):
            self.mod.format_serve_command(plan, self.directory)

    def test_explicit_fixed_block_bounds_are_labelled_supplied(self):
        from dataclasses import replace

        engine = FixtureEngine()
        engine.llm_engine.vllm_config.cache_config.num_gpu_blocks_override = 64
        args = engine_args()
        args.num_gpu_blocks_override = 64
        memory = replace(
            memory_profile(),
            cache_policy=dict(
                mode="fixed_blocks",
                num_gpu_blocks=64,
                block_size=16,
                kv_cache_dtype="auto",
            ),
        )
        self.mod.profile_pp_partition(
            dump_dir=self.directory,
            hetero_plan=topology(),
            engine_args=args,
            llm_factory=lambda: engine,
            memory_profile=memory,
            collect_only=True,
            num_prompts=1,
            input_len=2,
            output_len=2,
            num_iters=1,
            num_iters_warmup=0,
        )
        evidence = json.loads((self.directory / "memory_evidence.json").read_text())
        self.assertEqual(evidence["basis"], "explicit_supplied_bounds")


if __name__ == "__main__":
    unittest.main()
