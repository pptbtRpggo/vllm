"""Serve CodeLlama-34B P90 TTFT scenarios after target-total link calibration."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pp_sla_experiment import Experiment

SCENARIOS = {
    "100g_c16": dict(base=16, raised=24, calibration=(640, 704), formal=(704, 832), project_slo_ms=100, target_gbps=100),
    "100g_c32": dict(base=32, raised=48, calibration=(832, 896), formal=(896, 1088), project_slo_ms=180, target_gbps=100),
    "100g_c64": dict(base=64, raised=96, calibration=(1088, 1152), formal=(1152, 1472), project_slo_ms=500, target_gbps=100),
    "25g_c16": dict(base=16, raised=24, calibration=(1472, 1536), formal=(1536, 1664), project_slo_ms=500, target_gbps=25),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--phase", choices=("uniform", "dp"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-calibration", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18762)
    args = parser.parse_args()
    scenario = SCENARIOS[args.scenario]
    native = json.loads(args.native_calibration.read_text())
    args.kv_blocks = 1024
    args.network_mode = "target_total"
    args.cross_bandwidth_gbps = scenario["target_gbps"]
    # The real 25/100 Gbps network latency is unknown; emulate bandwidth only.
    args.cross_extra_latency_ms = 0
    args.native_bandwidth_gbps = native["native_bandwidth_gbps"]
    args.native_latency_ms = native["native_latency_ms"]
    root = args.root.resolve() / args.scenario
    args.output = str(root / args.phase)
    exp = Experiment(args)
    evaluation = exp.data["evaluation"]
    calibration = evaluation[slice(*scenario["calibration"])]
    formal = evaluation[slice(*scenario["formal"])]
    assert len(calibration) == 64 and len(formal) >= scenario["raised"]
    assert not {x["prompt_sha256"] for x in calibration} & {
        x["prompt_sha256"] for x in formal
    }
    if args.phase == "uniform":
        parts = [12, 12, 12, 12]
    else:
        parts = json.loads((root / "profile" / "plans.json").read_text())[
            "latency"
        ]
        if not (root / "slo.json").exists():
            raise ValueError("uniform phase must freeze SLO before DP phase")
    exp.root.mkdir(parents=True, exist_ok=False)
    (exp.root / "experiment.json").write_text(
        json.dumps(
            dict(
                scenario=args.scenario,
                phase=args.phase,
                partition=parts,
                network=exp.network,
                native_calibration=native,
                compute_slowdown=[1, 1, 2, 4],
                calibration_indices=scenario["calibration"],
                formal_indices=scenario["formal"],
                project_slo_ms=scenario["project_slo_ms"],
                warmup_output_tokens=16,
                SLO_rule="ceil(0.8 * uniform_base_calibration_P90 / 100) * 100",
            ),
            indent=2,
        ) + "\n"
    )
    with exp.server(args.phase, parts):
        if args.phase == "uniform":
            base = scenario["base"]
            exp.load("warmup_calibration.json", "warmup", max(16, base), base, 16)
            exp.data["evaluation"] = calibration
            result = exp.load("calibration.json", "evaluation", 64, base)
            p90 = result["summary"]["p90_ttft_ms"]
            slo = math.ceil(0.8 * p90 / 100) * 100
            (root / "slo.json").write_text(
                json.dumps(
                    dict(
                        calibration_p90_ms=p90,
                        SLO_ms=slo,
                        project_slo_ms=scenario["project_slo_ms"],
                        rule="ceil(0.8 * P90 / 100) * 100",
                    ),
                    indent=2,
                ) + "\n"
            )
            print("FROZEN_SLO", args.scenario, slo, flush=True)
        exp.data["evaluation"] = formal
        for concurrency in (scenario["base"], scenario["raised"]):
            exp.load(
                f"warmup_c{concurrency}.json",
                "warmup",
                max(16, concurrency),
                concurrency,
                16,
            )
            exp.load(
                f"formal_c{concurrency}.json",
                "evaluation",
                len(formal),
                concurrency,
            )


if __name__ == "__main__":
    main()
