from __future__ import annotations

import json
from typing import Any

from common import sha256_text
from gpt_common import JUDGE_SYSTEM_PROMPT, judge_prompt, judge_schema, normalize_judgment


EVALUATOR_ID = "qwen36_35b_a3b_content_style_judge_v1"


def schema_hash(check_count: int) -> str:
    return sha256_text(json.dumps(judge_schema(check_count), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def cache_identity(row: dict[str, Any]) -> dict[str, str]:
    prompt = judge_prompt(row)
    prompt_hash = sha256_text(prompt)
    schema_digest = schema_hash(len(row["content_checks"]))
    text_hash = sha256_text(row["generated_text"])
    cache_key = sha256_text("::".join([row["evaluation_item_id"], EVALUATOR_ID, prompt_hash, schema_digest, text_hash]))
    return {
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_digest,
        "output_text_sha256": text_hash,
        "evaluation_cache_key": cache_key,
        "cache_key_version": "item_identity_v2",
    }


def parse_qwen_json(text: str, check_count: int) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    return normalize_judgment(value, check_count)
