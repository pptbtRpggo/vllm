# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CodeLlama-Instruct prompt bytes, without importing Torch or loading weights."""

from pathlib import Path

import jinja2.sandbox
import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "examples/online_serving/hetero_pp/codellama_instruct.jinja"
)


def render(messages):
    def reject(message):
        raise jinja2.TemplateError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = reject
    return env.from_string(SOURCE.read_text()).render(
        messages=messages,
        bos_token="<s>",
        eos_token="</s>",
        add_generation_prompt=True,
    )


def test_single_user_has_exact_instruction_delimiters_without_trailing_whitespace():
    assert render([dict(role="user", content=" Hello ")]) == "<s>[INST] Hello [/INST]"


def test_system_and_history_preserve_bos_eos_and_turn_boundaries():
    messages = [
        dict(role="system", content="Be brief."),
        dict(role="user", content="Hello"),
        dict(role="assistant", content=" Hi "),
        dict(role="user", content="Next"),
    ]
    assert render(messages) == (
        "<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nHello [/INST]"
        " Hi </s><s>[INST] Next [/INST]"
    )


def test_consecutive_user_messages_are_rejected():
    with pytest.raises(jinja2.TemplateError, match="alternate"):
        render([dict(role="user", content="A"), dict(role="user", content="B")])
