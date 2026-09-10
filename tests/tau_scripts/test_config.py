# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "tau_config", ROOT / "tools/tau_config.py"
)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


def test_precision_is_optional_and_shared_by_schedulers(tmp_path):
    data = yaml.safe_load((ROOT / "configs/serve.yaml").read_text())
    del data["DTYPE"]
    path = tmp_path / "legacy_serve.yaml"
    path.write_text(yaml.safe_dump(data))
    assert config.load("serve", path, {}, {})["DTYPE"] == ""
    for scheduler in ("tau", "default"):
        values = config.load(
            "serve", path, {"SCHEDULER": scheduler}, {"DTYPE": "float16"}
        )
        assert values["DTYPE"] == "float16"


def test_precedence_selects_mode_before_workload_defaults():
    values = config.load(
        "bench",
        ROOT / "configs/bench.yaml",
        {"MODE": "collect", "CONCURRENCY": "5", "SEED": "9"},
        {"MODE": "smoke", "CONCURRENCY": "7", "OUTPUT_LEN": "20"},
    )
    assert values["NUM_PROMPTS"] == "1000"
    assert values["CONCURRENCY"] == "5"
    assert values["OUTPUT_LEN"] == "20"
    assert values["SLO_SEED"] == "9"
    source = json.loads(values["TAU_LAUNCH_CONFIG"])
    assert source["environment_overrides"] == ["OUTPUT_LEN"]


def test_config_paths_and_external_slo_override(tmp_path):
    data = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    data.update(DATASET="./data.json", RUN_DIR="../run")
    data["SLO"]["enabled"] = True
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(data))
    values = config.load("bench", path, {}, {})
    assert values["DATASET"] == str(tmp_path / "data.json")
    assert values["RUN_DIR"] == str(tmp_path.parent / "run")
    assert json.loads(values["SLO_INLINE"])["profiles"] == data["SLO"]["profiles"]
    values = config.load(
        "bench", path, {"SLO_CONFIG": "old.json"}, {"DATASET": "cwd.json"}
    )
    assert values["SLO_INLINE"] == ""
    assert values["SLO_CONFIG"] == "old.json"
    assert values["DATASET"] == "cwd.json"


def test_bench_yaml_accepts_any_group_count_and_sampler_specs(tmp_path):
    data = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    data["SLO"]["enabled"] = True
    data["SLO"]["profiles"]["sampled"] = {
        "ttft_slo_ms": {
            "distribution": "normal",
            "mean": 2000,
            "std": 300,
            "min": 1000,
            "max": 3000,
        },
        "tpot_slo_ms": {"distribution": "uniform", "min": 80, "max": 120},
    }
    data["SLO"]["ratios"] = dict(tight=0.2, loose=0.3, sampled=0.5)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(data))
    values = config.load("bench", path, {}, {})
    assert json.loads(values["SLO_INLINE"]) == {
        k: v for k, v in data["SLO"].items() if k != "enabled"
    }


@pytest.mark.parametrize("fault", ["unknown", "missing", "mode", "slo_bool"])
def test_invalid_configuration_is_rejected(tmp_path, fault):
    data = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    if fault == "unknown":
        data["CONCURRRENCY"] = 3
    elif fault == "missing":
        del data["DATASET"]
    elif fault == "mode":
        data["MODE"] = "typo"
    else:
        data["SLO"]["enabled"] = "false"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        config.load("bench", path, {}, {})
