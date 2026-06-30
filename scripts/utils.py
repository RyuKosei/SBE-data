from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "::".join(str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?is)^.*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*$", "", text)
    text = re.sub(r"(?is)</?think>", "", text)
    return text.strip()


def load_existing_ids(path: str | Path, key: str) -> set[str]:
    return {str(row[key]) for row in read_jsonl(path) if key in row}


def render_explicit_ratio_prompt(scenario: dict[str, Any], alpha: float) -> str:
    min_chars, max_chars = scenario.get("length_range_zh_chars", [80, 140])
    a_percent = int(round((1.0 - alpha) * 100))
    b_percent = int(round(alpha * 100))
    return (
        "/no_think\n"
        "严禁输出 Thinking Process、Analysis、推理过程、步骤说明、标题或 Markdown 列表。\n"
        "你只能输出最终中文正文，第一句话就必须是正文内容。\n\n"
        "你将根据一个固定核心语义生成中文短文本。\n\n"
        f"核心语义：\n{scenario['content']}\n\n"
        f"风格A：\n{scenario['style_a']}\n\n"
        f"风格B：\n{scenario['style_b']}\n\n"
        "目标风格比例：\n"
        f"{a_percent}% 风格A + {b_percent}% 风格B。\n\n"
        "硬性要求：\n"
        "1. 必须保留核心语义，不添加与核心语义冲突的新事实。\n"
        f"2. 正文必须至少 {min_chars} 个汉字，最多 {max_chars} 个汉字；少于 {min_chars} 个汉字不合格。\n"
        "3. 只输出正文，不要解释比例，不要列标题。\n"
        "4. 不要输出任何英文分析，不要输出 Thinking Process。"
    )


def render_scalar_judge_prompt(generation: dict[str, Any]) -> str:
    checks = generation.get("content_checks", [])
    checks_text = "、".join(checks) if isinstance(checks, list) else str(checks)
    return (
        "/no_think\n"
        "你是文本风格评估器。请判断给定文本中“风格B”相对于“风格A”的占比。\n\n"
        f"核心语义：\n{generation['content']}\n\n"
        f"核心语义检查点：\n{checks_text}\n\n"
        f"风格A：\n{generation['style_a']}\n\n"
        f"风格B：\n{generation['style_b']}\n\n"
        f"待评估文本：\n{generation['output']}\n\n"
        "请只输出合法 JSON，不要输出解释性段落：\n"
        "{\n"
        '  "style_b_ratio": 0.0到1.0之间的小数,\n'
        '  "content_preservation": 1到5的整数,\n'
        '  "style_mixture_naturalness": 1到5的整数,\n'
        '  "reason": "一句话说明"\n'
        "}"
    )


def render_pairwise_judge_prompt(left: dict[str, Any], right: dict[str, Any]) -> str:
    return (
        "/no_think\n"
        "你是文本风格比较器。给定风格A和风格B，请判断哪段文本更接近风格B。\n\n"
        f"核心语义：\n{left['content']}\n\n"
        f"风格A：\n{left['style_a']}\n\n"
        f"风格B：\n{left['style_b']}\n\n"
        f"文本1：\n{left['output']}\n\n"
        f"文本2：\n{right['output']}\n\n"
        "请只输出合法 JSON：\n"
        "{\n"
        '  "winner": "text_1" 或 "text_2" 或 "tie",\n'
        '  "confidence": 1到5的整数\n'
        "}"
    )


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    scalar = _parse_loose_scalar_judge(text)
    if scalar is not None:
        return scalar, None
    pairwise = _parse_loose_pairwise_judge(text)
    if pairwise is not None:
        return pairwise, None
    return None, last_error if "last_error" in locals() else "no json object found"


def _parse_loose_scalar_judge(text: str) -> dict[str, Any] | None:
    ratio = re.search(r'"style_b_ratio"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    content = re.search(r'"content_preservation"\s*:\s*([1-5])', text)
    naturalness = re.search(r'"style_mixture_naturalness"\s*:\s*([1-5])', text)
    if not (ratio and content and naturalness):
        return None
    reason = re.search(r'"reason"\s*:\s*"(.*)"\s*\}?$', text, flags=re.DOTALL)
    return {
        "style_b_ratio": float(ratio.group(1)),
        "content_preservation": int(content.group(1)),
        "style_mixture_naturalness": int(naturalness.group(1)),
        "reason": reason.group(1).strip() if reason else "",
    }


def _parse_loose_pairwise_judge(text: str) -> dict[str, Any] | None:
    winner = re.search(r'"winner"\s*:\s*"(text_1|text_2|tie)"', text)
    confidence = re.search(r'"confidence"\s*:\s*([1-5])', text)
    if not winner:
        return None
    parsed: dict[str, Any] = {"winner": winner.group(1)}
    if confidence:
        parsed["confidence"] = int(confidence.group(1))
    return parsed


async def gather_limited(limit: int, coros: Iterable[Any]) -> list[Any]:
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(coro) for coro in coros))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model-name", default=None, help="Served model name. Defaults to --model or --judge-model.")
