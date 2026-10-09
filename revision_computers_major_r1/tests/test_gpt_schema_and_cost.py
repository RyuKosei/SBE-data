from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gpt_common import cache_identity, estimated_cost, judge_schema, normalize_judgment  # noqa: E402


def valid() -> dict:
    return {
        "covered": [True, False],
        "added_key_information": False,
        "task_result_changed_by_addition": False,
        "factual_contradiction": False,
        "style_B_intensity": 50,
        "naturalness": 4,
        "unjudgeable": False,
    }


def test_bool_is_not_accepted_as_numeric_score() -> None:
    value = valid()
    value["style_B_intensity"] = True
    with pytest.raises(ValueError):
        normalize_judgment(value, 2)


def test_missing_checkpoint_is_data_error() -> None:
    with pytest.raises(ZeroDivisionError):
        normalize_judgment({**valid(), "covered": []}, 0)


def test_cost_uses_cached_and_uncached_rates() -> None:
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 250_000, "output_tokens": 100_000}
    price = {"input": 2.0, "cached_input": 0.2, "output": 10.0}
    assert estimated_cost(usage, price) == pytest.approx(2.55)


def test_identical_text_from_distinct_output_identities_is_not_collapsed() -> None:
    schema = judge_schema(2)
    first = cache_identity("gpt-6-sol", "same prompt", schema, "same text", "sample-a")
    second = cache_identity("gpt-6-sol", "same prompt", schema, "same text", "sample-b")
    assert first["evaluation_cache_key"] != second["evaluation_cache_key"]
