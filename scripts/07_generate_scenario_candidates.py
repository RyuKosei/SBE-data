from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from utils import ROOT, add_common_model_args, gather_limited, read_yaml, write_jsonl


DOMAIN_HINTS = {
    "emotion_apology": [
        "consumer",
        "public_service",
        "education",
        "travel",
        "technology",
        "daily_life",
        "family",
        "workplace",
        "community",
        "housing",
    ],
    "expertise_explain": [
        "science",
        "education",
        "daily_life",
        "technology",
        "health_non_diagnostic",
        "environment",
        "family",
        "public_service",
        "travel",
        "workplace",
    ],
    "objectivity_cat": [
        "daily_life",
        "science",
        "environment",
        "family",
        "travel",
        "education",
        "public_service",
        "technology",
        "community",
        "consumer",
    ],
    "formality_request": [
        "workplace",
        "daily_life",
        "consumer",
        "community",
        "education",
        "technology",
        "public_service",
        "family",
        "travel",
        "housing",
    ],
    "politeness_refusal": [
        "workplace",
        "family",
        "daily_life",
        "education",
        "consumer",
        "community",
        "travel",
        "technology",
        "public_service",
        "health_non_diagnostic",
    ],
    "concision_detail": [
        "technology",
        "education",
        "science",
        "workplace",
        "daily_life",
        "public_service",
        "consumer",
        "travel",
        "family",
        "environment",
    ],
    "humor_neutral": [
        "daily_life",
        "consumer",
        "technology",
        "education",
        "workplace",
        "travel",
        "family",
        "community",
        "housing",
        "public_service",
    ],
    "empathy_clinical": [
        "education",
        "workplace",
        "family",
        "daily_life",
        "health_non_diagnostic",
        "travel",
        "community",
        "consumer",
        "technology",
        "public_service",
    ],
}


def parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass
    return []


def render_prompt(axis: dict[str, Any], count: int, batch_idx: int) -> str:
    domains = DOMAIN_HINTS.get(axis["axis_id"], [])
    rotated_domains = domains[(batch_idx - 1) % len(domains) :] + domains[: (batch_idx - 1) % len(domains)] if domains else []
    domain_text = "、".join(rotated_domains[: max(count, 6)])
    axis_extra = ""
    if axis["axis_id"] == "emotion_apology":
        axis_extra = "emotion_apology 不要只写电商售后；同一批最多 2 条售后/物流。"
    elif axis["axis_id"] == "formality_request":
        axis_extra = "formality_request 不要连续生成会议确认；要混合私人、工作、公共通知和服务沟通。"
    elif axis["axis_id"] == "empathy_clinical":
        axis_extra = "empathy_clinical 避免自伤、绝望、严重疾病诊断、处方建议；可以写焦虑、压力、失眠、担忧，但保持非危机场景。"
    elif axis["axis_id"] == "expertise_explain":
        axis_extra = "expertise_explain 必须是可用儿童比喻和博士后术语两种方式自然解释的问题；避免需要争议事实或最新新闻。"
    elif axis["axis_id"] == "objectivity_cat":
        axis_extra = "objectivity_cat 必须是可客观观察也可抒情描写的对象或场景；避免抽象道德判断。"
    elif axis["axis_id"] == "politeness_refusal":
        axis_extra = "politeness_refusal 必须是可以直白拒绝也可以委婉拒绝的请求；不要涉及歧视、违法或医疗建议。"
    elif axis["axis_id"] == "concision_detail":
        axis_extra = "concision_detail 必须可自然写成极简摘要或详尽说明；避免本身信息过少导致无法展开。"
    elif axis["axis_id"] == "humor_neutral":
        axis_extra = "humor_neutral 必须允许轻微幽默吐槽但不损害核心语义；避免攻击个人或群体。"
    return (
        "/no_think\n"
        "你在构造一个中文风格比例控制 benchmark 的候选场景。请只输出合法 JSON 数组，不要 Markdown，不要解释。\n\n"
        f"风格轴 ID：{axis['axis_id']}\n"
        f"风格A：{axis['style_a']}\n"
        f"风格B：{axis['style_b']}\n"
        f"场景类型：{axis.get('scenario_instruction', '')}\n\n"
        f"请生成 {count} 条彼此明显不同的候选场景。每条 JSON 必须包含：\n"
        "- content: 一个固定核心语义，中文，12-80 个汉字，不要包含具体人名、品牌、政治、医疗诊断、违法、自伤、威胁或事实争议。\n"
        "- content_checks: 2-4 个短检查点，必须能验证核心语义是否被保留。\n"
        "- domain_tag: 必须从这些英文标签中选一个：daily_life, education, science, workplace, public_service, consumer, family, health_non_diagnostic, travel, technology, community, housing, environment。\n"
        "- risk_note: 一句话说明事实争议或安全风险；如果没有风险，写“普通低风险场景”。\n\n"
        "要求：\n"
        "1. content 只能描述核心语义，不要直接写成风格A或风格B。\n"
        "2. 同一批不要换词复述同一个模板。\n"
        "3. A/B 两种风格都必须能自然表达该 content。\n"
        f"4. 本批优先覆盖这些 domain_tag：{domain_text}。\n"
        f"5. {axis_extra}\n"
        "6. 输出格式示例：\n"
        "[{\"content\":\"请对方明天下午三点前确认会议时间是否合适。\",\"content_checks\":[\"明天下午三点前\",\"确认会议时间\"],\"domain_tag\":\"workplace\",\"risk_note\":\"普通低风险场景。\"}]\n"
        f"批次编号：{batch_idx}"
    )


async def call_openai(args: argparse.Namespace, prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    response = await client.chat.completions.create(
        model=args.model_name or args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.request_timeout_seconds,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None,
    )
    return response.choices[0].message.content or ""


async def run_axis_batch(args: argparse.Namespace, axis: dict[str, Any], batch_idx: int, count: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    prompt = render_prompt(axis, count, batch_idx)
    try:
        if args.dry_run:
            raw = json.dumps(
                [
                    {
                        "content": f"{axis['axis_id']} 候选场景 {batch_idx}-{idx}，需要保留两个核心信息。",
                        "content_checks": ["核心信息一", "核心信息二"],
                        "domain_tag": (DOMAIN_HINTS.get(axis["axis_id"], ["daily_life"])[idx - 1] if idx <= len(DOMAIN_HINTS.get(axis["axis_id"], [])) else "daily_life"),
                        "risk_note": "普通低风险场景。",
                    }
                    for idx in range(1, count + 1)
                ],
                ensure_ascii=False,
            )
        else:
            raw = await call_openai(args, prompt)
        parsed = parse_json_array(raw)
        rows = []
        for idx, item in enumerate(parsed, 1):
            rows.append(
                {
                    "scenario_id": f"{axis['axis_id']}_candidate_{batch_idx:02d}_{idx:02d}",
                    "axis_id": axis["axis_id"],
                    "content": item.get("content"),
                    "style_a": axis["style_a"],
                    "style_b": axis["style_b"],
                    "attribute_name": axis.get("attribute_name"),
                    "content_checks": item.get("content_checks", []),
                    "length_range_zh_chars": axis["default_length_zh_chars"],
                    "domain_tag": item.get("domain_tag", "daily_life"),
                    "risk_note": item.get("risk_note", item.get("safety_note", "普通低风险场景。")),
                    "candidate_batch": batch_idx,
                }
            )
        failure = None if rows else {"axis_id": axis["axis_id"], "batch_idx": batch_idx, "error": "no_json_array", "raw_output": raw}
        return rows, failure
    except Exception as exc:
        return [], {"axis_id": axis["axis_id"], "batch_idx": batch_idx, "error": repr(exc)}


async def async_main(args: argparse.Namespace) -> None:
    config = read_yaml(args.config)
    axes = config["style_axes"]
    if args.axes:
        wanted = set(args.axes)
        axes = [axis for axis in axes if axis["axis_id"] in wanted]

    jobs = []
    for axis in axes:
        remaining = args.per_axis
        batch_idx = 1
        while remaining > 0:
            count = min(args.batch_size, remaining)
            jobs.append(run_axis_batch(args, axis, batch_idx, count))
            remaining -= count
            batch_idx += 1

    results = await gather_limited(args.concurrency, jobs)
    rows = [row for batch, _failure in results for row in batch]
    failures = [failure for _batch, failure in results if failure]
    write_jsonl(args.out, rows)
    if args.failure_out:
        write_jsonl(args.failure_out, failures)
    print(f"Wrote {len(rows)} candidates to {Path(args.out)}; failures={len(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs/style_axes.yaml")
    parser.add_argument("--out", default=ROOT / "data/scenarios/candidate_scenarios.jsonl")
    parser.add_argument("--failure-out", default=ROOT / "results/logs/scenario_candidate_failures.jsonl")
    parser.add_argument("--axes", nargs="*", default=None)
    parser.add_argument("--per-axis", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=int, default=180)
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    add_common_model_args(parser)
    args = parser.parse_args()
    if not args.dry_run and not args.base_url:
        raise SystemExit("Real candidate generation requires --base-url, or use --dry-run.")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
