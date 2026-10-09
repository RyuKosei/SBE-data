from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from common import sha256_text


JUDGE_SYSTEM_PROMPT = (
    "你是严格的中文文本评估器。独立判断内容保持与风格强度，只输出一个合法 JSON 对象，不写 Markdown 或解释。"
    "风格强度只看表达方式，不因文本更长就自动判定更接近风格B。"
)


def judge_prompt(row: dict[str, Any]) -> str:
    checks = row.get("content_checks") or []
    numbered = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(checks))
    return f"""请评价候选文本。核心内容检查点必须逐项判断；同义表达算覆盖。新增关键信息指原要求没有、且会影响任务结论或事实状态的信息。事实矛盾指与核心语义直接冲突。若确实无法判断，unjudgeable=true。

核心语义：{row['input_text']}
原子内容检查点：
{numbered}

风格A：{row['style_A']}
风格B：{row['style_B']}

候选文本：{row['generated_text']}

返回字段必须恰好采用以下结构：
{{"covered":[true,false],"added_key_information":false,"task_result_changed_by_addition":false,"factual_contradiction":false,"style_B_intensity":50,"naturalness":4,"unjudgeable":false}}
covered 的布尔值数量必须等于检查点数量；style_B_intensity 是 0 到 100；naturalness 是 1 到 5。"""


def judge_schema(check_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "covered": {"type": "array", "items": {"type": "boolean"}, "minItems": check_count, "maxItems": check_count},
            "added_key_information": {"type": "boolean"},
            "task_result_changed_by_addition": {"type": "boolean"},
            "factual_contradiction": {"type": "boolean"},
            "style_B_intensity": {"type": "integer", "minimum": 0, "maximum": 100},
            "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
            "unjudgeable": {"type": "boolean"},
        },
        "required": [
            "covered",
            "added_key_information",
            "task_result_changed_by_addition",
            "factual_contradiction",
            "style_B_intensity",
            "naturalness",
            "unjudgeable",
        ],
        "additionalProperties": False,
    }


def normalize_judgment(value: Any, check_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judgment must be an object")
    required = set(judge_schema(check_count)["required"])
    if set(value) != required:
        raise ValueError("judgment fields do not match the locked schema")
    covered = value["covered"]
    if not isinstance(covered, list) or len(covered) != check_count or not all(isinstance(item, bool) for item in covered):
        raise ValueError("covered must contain exactly one boolean per checkpoint")
    for field in ("added_key_information", "task_result_changed_by_addition", "factual_contradiction", "unjudgeable"):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")
    intensity = value["style_B_intensity"]
    naturalness = value["naturalness"]
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= float(intensity) <= 100:
        raise ValueError("style_B_intensity must be numeric in [0,100] and not boolean")
    if isinstance(naturalness, bool) or not isinstance(naturalness, (int, float)) or not 1 <= float(naturalness) <= 5:
        raise ValueError("naturalness must be numeric in [1,5] and not boolean")
    coverage = sum(covered) / check_count
    content_pass = (
        coverage >= 0.90
        and not value["factual_contradiction"]
        and not value["task_result_changed_by_addition"]
        and not value["unjudgeable"]
    )
    return {
        **value,
        "style_B_intensity": float(intensity),
        "naturalness": float(naturalness),
        "content_coverage_rate": coverage,
        "content_preservation_pass": content_pass,
    }


def usage_fields(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"input_tokens": 0, "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "cache_write_tokens": int(getattr(input_details, "cache_write_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def load_prices(path: Path, model: str) -> dict[str, float]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if model not in config["models"]:
        raise ValueError(f"missing price for model: {model}")
    return {key: float(value) for key, value in config["models"][model].items()}


def estimated_cost(usage: dict[str, int], price: dict[str, float]) -> float:
    cached = min(usage.get("cached_input_tokens", 0), usage.get("input_tokens", 0))
    uncached = usage.get("input_tokens", 0) - cached
    return (
        uncached * price["input"]
        + cached * price.get("cached_input", price["input"])
        + usage.get("output_tokens", 0) * price["output"]
    ) / 1_000_000


def cache_identity(model: str, prompt: str, schema: dict[str, Any], text: str, item_identity: str) -> dict[str, str]:
    prompt_hash = sha256_text(JUDGE_SYSTEM_PROMPT + "\n" + prompt)
    schema_hash = sha256_text(json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    text_hash = sha256_text(text)
    cache_key = sha256_text("::".join([item_identity, model, prompt_hash, schema_hash, text_hash]))
    return {"prompt_sha256": prompt_hash, "schema_sha256": schema_hash, "output_text_sha256": text_hash, "evaluation_cache_key": cache_key, "cache_key_version": "item_identity_v2"}
