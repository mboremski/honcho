from datetime import datetime
from unittest.mock import patch

import pytest

from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)


def test_minimal_deriver_prompt_includes_custom_instructions_when_present() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        current_time=datetime(2025, 1, 1),
        custom_instructions="Prefer concrete timeline facts.",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "Prefer concrete timeline facts." in prompt


def test_minimal_deriver_prompt_omits_custom_instructions_when_absent() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        current_time=datetime(2025, 1, 1),
        custom_instructions=None,
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_minimal_deriver_prompt_includes_temporal_context() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="2017-03-08 13:56:00 alice: I work in Erkelenz",
        current_time=datetime(2026, 5, 30),
    )

    assert "TEMPORAL CONTEXT:" in prompt
    assert "Today's date is 2026-05-30." in prompt


def test_minimal_deriver_prompt_guards_against_fabrication() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="2025-10-31 15:48:00 alice: Hi",
        current_time=datetime(2026, 5, 30),
    )

    # Anti-contamination guardrails are present...
    assert "DO NOT FABRICATE" in prompt
    assert "return no observations rather than inventing any" in prompt
    # ...and the old few-shot example values are no longer copyable literals.
    assert "NYC" not in prompt
    assert "June 21st" not in prompt


def test_estimate_deriver_prompt_tokens_increases_with_custom_instructions() -> None:
    base_tokens = estimate_minimal_deriver_prompt_tokens()
    custom_tokens = estimate_deriver_prompt_tokens(
        "Prefer explicit facts with absolute dates and keep the subject precise."
    )

    assert custom_tokens > base_tokens


def test_estimate_deriver_prompt_tokens_propagates_token_estimation_errors() -> None:
    estimate_minimal_deriver_prompt_tokens.cache_clear()

    with patch(
        "src.deriver.prompts.estimate_tokens",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens(None)

        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens("Prefer concrete facts.")
