# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU subprocess coverage for launcher environment selection.

The SDK is a temporary shell fixture and candidate interpreters are executable
wrappers around this test's Python. No Torch, NPU, network or installed SDK is
used, and the parent process environment must remain unchanged.
"""

import importlib.util
import json
import os
import shlex
import sys
import tempfile
import textwrap
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "examples/online_serving/hetero_pp/check_cpu_gloo.py"


def load_helper() -> Any:
    name = "comm_launcher_environment_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def executable(path: Path, contents: str) -> Path:
    path.write_text(contents)
    path.chmod(0o700)
    return path


class LauncherEnvironmentTests(unittest.TestCase):
    helper: Any

    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper()

    def test_explicit_python_is_the_only_candidate_and_missing_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            python = executable(directory / "chosen-python", "#!/bin/sh\nexit 0\n")
            environment = {"VIRTUAL_ENV": str(directory / "other-environment")}
            actual = self.helper.python_candidates(
                directory, explicit=str(python), environment=environment
            )
            self.assertEqual(actual, [str(python.absolute())])
            with self.assertRaisesRegex(ValueError, "不存在或不可执行"):
                self.helper.python_candidates(
                    directory,
                    explicit=str(directory / "missing-python"),
                    environment=environment,
                )

    def test_prepare_environment_preserves_visibility_and_does_not_mutate_input(self):
        environment = {
            "ASCEND_RT_VISIBLE_DEVICES": "8,9",
            "ASCEND_HOME_PATH": "/existing/toolkit",
            "LD_LIBRARY_PATH": "/existing/sdk/lib64",
            "PYTHONPATH": "/existing/python",
            "PYTHONOPTIMIZE": "2",
        }
        before = dict(environment)
        actual = self.helper.prepare_environment(ROOT, environment)
        self.assertEqual(actual["ASCEND_RT_VISIBLE_DEVICES"], "8,9")
        self.assertEqual(actual["ASCEND_HOME_PATH"], "/existing/toolkit")
        self.assertIn(
            "/existing/sdk/lib64", actual["LD_LIBRARY_PATH"].split(os.pathsep)
        )
        self.assertEqual(
            actual["PYTHONPATH"].split(os.pathsep)[:2],
            [str(ROOT), "/existing/python"],
        )
        self.assertNotIn("PYTHONOPTIMIZE", actual)
        self.assertEqual(actual["PYTHONUNBUFFERED"], "1")
        self.assertEqual(environment, before)

    def test_explicit_cann_script_does_not_select_an_environment_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            default = directory / "default-sdk"
            default.mkdir()
            (default / "set_env.sh").write_text("export ASCEND_HOME_PATH=default\n")
            selected = directory / "selected-set-env.sh"
            selected.write_text("export ASCEND_HOME_PATH=selected\n")
            actual = self.helper.cann_scripts(
                {"ASCEND_HOME_PATH": str(default)}, explicit=str(selected)
            )
            self.assertEqual(actual, [selected.resolve()])

    def test_real_sdk_source_captures_only_sdk_keys_and_isolates_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "set-env.sh"
            script.write_text(
                textwrap.dedent(
                    """\
                    export ASCEND_HOME_PATH=/fixture/toolkit
                    export ATB_HOME_PATH=/fixture/atb
                    export CANN_FIXTURE=toolkit
                    export TBE_FIXTURE=operators
                    export TORCH_NPU_FIXTURE=runtime
                    export PYTHONPATH=/fixture/python
                    export LD_LIBRARY_PATH=/fixture/lib64
                    export PATH=/fixture/bin:$PATH
                    export NON_SDK_CHILD=must-not-leak
                    echo 'SDK fixture diagnostic'
                    """
                )
            )
            environment = dict(os.environ, UNCHANGED_ENV="original")
            before = dict(environment)
            parent_before = dict(os.environ)
            actual = self.helper.load_sdk_environment(
                script, environment, time.monotonic() + 30
            )
            for key, expected in {
                "ASCEND_HOME_PATH": "/fixture/toolkit",
                "ATB_HOME_PATH": "/fixture/atb",
                "CANN_FIXTURE": "toolkit",
                "TBE_FIXTURE": "operators",
                "TORCH_NPU_FIXTURE": "runtime",
                "PYTHONPATH": "/fixture/python",
                "LD_LIBRARY_PATH": "/fixture/lib64",
                "UNCHANGED_ENV": "original",
            }.items():
                self.assertEqual(actual[key], expected)
            self.assertEqual(actual["PATH"].split(os.pathsep)[0], "/fixture/bin")
            self.assertNotIn("NON_SDK_CHILD", actual)
            self.assertEqual(environment, before)
            self.assertEqual(dict(os.environ), parent_before)

    def test_runtime_discovery_runs_bad_then_good_candidate_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = directory / "interpreter-fixture.py"
            fixture.write_text(
                textwrap.dedent(
                    """\
                    import json
                    import os
                    import sys
                    from pathlib import Path

                    mode, arguments = sys.argv[1], sys.argv[2:]
                    if arguments[0] == "-c":
                        print(os.environ["FIXTURE_PURELIB"])
                        raise SystemExit(0)
                    assert arguments[1] == "--inspect-runtime", arguments
                    observation = {
                        "candidate": mode,
                        "optimized": "PYTHONOPTIMIZE" in os.environ,
                        "visibility": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                        "pythonpath": os.environ.get("PYTHONPATH"),
                    }
                    with Path(os.environ["FIXTURE_CALLS"]).open("a") as stream:
                        stream.write(json.dumps(observation) + "\\n")
                    if mode == "bad":
                        print("fixture interpreter lacks the required runtime")
                        raise SystemExit(7)
                    print("COMM_CHECK_RUNTIME=" + json.dumps(observation))
                    """
                )
            )
            candidates = [
                executable(
                    directory / f"{mode}-python",
                    "#!/bin/sh\nexec "
                    + " ".join(
                        shlex.quote(argument)
                        for argument in (sys.executable, str(fixture), mode)
                    )
                    + ' "$@"\n',
                )
                for mode in ("bad", "good")
            ]
            calls = directory / "calls.jsonl"
            args = SimpleNamespace(devices=None, cann_env=None, python=None)
            with (
                patch.dict(
                    os.environ,
                    {
                        "PYTHONOPTIMIZE": "2",
                        "ASCEND_RT_VISIBLE_DEVICES": "8,9",
                        "FIXTURE_PURELIB": str(directory / "empty-site-packages"),
                        "FIXTURE_CALLS": str(calls),
                    },
                ),
                patch.object(
                    self.helper,
                    "python_candidates",
                    return_value=[str(candidate) for candidate in candidates],
                ),
                patch.object(self.helper, "cann_scripts", return_value=[]),
            ):
                python, environment, inventory = self.helper.discover_runtime(
                    args, directory, time.monotonic() + 30
                )
                self.assertEqual(os.environ["PYTHONOPTIMIZE"], "2")
            self.assertEqual(python, str(candidates[1]))
            self.assertEqual(inventory["candidate"], "good")
            self.assertEqual(inventory["sdk_environment"], "current")
            self.assertEqual(inventory["visibility"], "8,9")
            self.assertNotIn("PYTHONOPTIMIZE", environment)
            observations = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(
                [item["candidate"] for item in observations], ["bad", "good"]
            )
            self.assertTrue(all(item["optimized"] is False for item in observations))
            self.assertTrue(all(item["visibility"] == "8,9" for item in observations))
            self.assertTrue(
                all(
                    item["pythonpath"].split(os.pathsep)[0] == str(ROOT)
                    for item in observations
                )
            )
            log = (directory / "environment_attempts.log").read_text()
            self.assertIn(str(candidates[0]), log)
            self.assertIn("exit: 7", log)
            self.assertIn("fixture interpreter lacks the required runtime", log)
            self.assertIn(str(candidates[1]), log)
            self.assertIn("exit: 0", log)
            self.assertIn(self.helper.MARKER, log)


if __name__ == "__main__":
    unittest.main()
