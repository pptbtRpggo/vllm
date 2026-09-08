# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the two Tau launch configurations without importing vLLM/NPU code."""

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

SERVE_KEYS = {
    "MODEL",
    "ASCEND_RT_VISIBLE_DEVICES",
    "PP",
    "TP",
    "HOST",
    "PORT",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_MICROBATCHES",
    "MIN_WAITING",
    "GPU_MEM",
    "TRACE_ENABLED",
    "TRACE",
    "RUN_DIR",
}
BENCH_KEYS = {
    "RUN_DIR",
    "MODE",
    "REQUEST_RATE",
    "BURSTINESS",
    "IGNORE_EOS",
    "SEED",
    "SLO_SEED",
    "DATASET",
    "DOWNLOAD",
    "BASE_URL",
    "RESULT_DIR",
    "WARMUP_REQUESTS",
    "READY_TIMEOUT",
}
MODE_KEYS = {"NUM_PROMPTS", "OUTPUT_LEN", "CONCURRENCY"}
PATH_KEYS = {"RUN_DIR", "TRACE", "DATASET", "RESULT_DIR"}


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label}: missing keys {sorted(keys - actual)}; "
            f"unknown keys {sorted(actual - keys, key=str)}"
        )


def scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"Expected a scalar configuration value, got {value!r}")
    return str(value)


def load(kind, path, overrides, environ):
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    config = yaml.safe_load(content)
    keys = SERVE_KEYS if kind == "serve" else BENCH_KEYS | {"MODES", "SLO"}
    exact_keys(config, keys, str(path))
    values = {k: v for k, v in config.items() if k not in ("MODES", "SLO")}
    allowed = SERVE_KEYS if kind == "serve" else BENCH_KEYS | MODE_KEYS | {"SLO_CONFIG"}
    if set(overrides) - allowed:
        raise ValueError("Unknown command-line configuration override")
    inline_slo = ""
    if kind == "bench":
        exact_keys(config["MODES"], {"smoke", "collect"}, "MODES")
        for name, preset in config["MODES"].items():
            exact_keys(preset, MODE_KEYS, f"MODES.{name}")
        mode = overrides.get("MODE") or environ.get("MODE") or values["MODE"]
        if not isinstance(mode, str) or mode not in config["MODES"]:
            raise ValueError("MODE must be smoke or collect")
        values.update(config["MODES"][mode])
        exact_keys(config["SLO"], {"enabled", "profiles", "ratios"}, "SLO")
        if not isinstance(config["SLO"]["enabled"], bool):
            raise ValueError("SLO.enabled must be true or false")
        if config["SLO"]["enabled"]:
            inline_slo = json.dumps(
                {k: v for k, v in config["SLO"].items() if k != "enabled"},
                allow_nan=False,
            )
        values["SLO_CONFIG"] = ""  # Legacy CLI/environment override only.
    for key, value in values.items():
        value = scalar(value)
        if value and (
            key in PATH_KEYS or (key == "MODEL" and value.startswith((".", "~")))
        ):
            candidate = Path(value).expanduser()
            value = str((path.parent / candidate).resolve())
        values[key] = value
    # Environment/CLI paths retain their existing current-directory semantics.
    values.update({k: environ[k] for k in allowed if environ.get(k)})
    if kind == "serve" and not values["RUN_DIR"] and environ.get("RUN"):
        values["RUN_DIR"] = environ["RUN"]
    values.update(overrides)
    if kind == "bench":
        values["SLO_SEED"] = values["SLO_SEED"] or values["SEED"]
        values["SLO_INLINE"] = "" if values["SLO_CONFIG"] else inline_slo
    values["TAU_LAUNCH_CONFIG"] = json.dumps(
        {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "environment_overrides": sorted(
                k for k in allowed if environ.get(k) and k not in overrides
            ),
            "cli_overrides": sorted(overrides),
        },
        allow_nan=False,
    )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("serve", "bench"))
    parser.add_argument("path")
    parser.add_argument("--set", nargs=2, action="append", default=[])
    args = parser.parse_args()
    values = load(args.kind, args.path, dict(args.set), os.environ)
    # Keys are validated against a fixed schema; values are shell-quoted data.
    print("\n".join(f"export {k}={shlex.quote(v)}" for k, v in values.items()))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
