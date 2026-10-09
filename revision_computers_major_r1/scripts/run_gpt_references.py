from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from common import append_jsonl, atomic_write_json, read_jsonl, redact, require_credential, require_environment_value, sha256_text, stable_id, utc_now
from gpt_common import estimated_cost, load_prices, usage_fields


SYSTEM = "你是中文短文本生成器。只输出最终中文正文，不写标题、Markdown、解释、分析或推理过程。"


def reference_prompt(scenario: dict[str, Any], endpoint: str) -> str:
    minimum, maximum = scenario["length_range_zh_chars"]
    endpoint_style = scenario["style_a"] if endpoint == "A" else scenario["style_b"]
    numbered = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(scenario["content_checks"]))
    return f"""请围绕固定核心语义生成一条中文短文本，并完整采用指定端点风格。

核心语义：{scenario['content']}
必须覆盖的原子内容检查点：
{numbered}

风格A：{scenario['style_a']}
风格B：{scenario['style_b']}
本次指定端点：风格{endpoint}（{endpoint_style}）

约束：
1. 完整保留核心语义和检查点，不添加冲突事实。
2. 正文必须在 {minimum} 至 {maximum} 个汉字之间。
3. 只输出正文，不解释风格、比例或生成过程。"""


def build_jobs(config: dict[str, Any], scenarios: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for endpoint in ("A", "B"):
            prompt = reference_prompt(scenario, endpoint)
            jobs.append({
                "reference_id": stable_id("major_r1_reference", config["experiment_id"], model, scenario["scenario_id"], endpoint),
                "scenario_id": scenario["scenario_id"],
                "style_axis": scenario["axis_id"],
                "endpoint": endpoint,
                "style_A": scenario["style_a"],
                "style_B": scenario["style_b"],
                "input_text": scenario["content"],
                "content_checks": scenario["content_checks"],
                "length_range_zh_chars": scenario["length_range_zh_chars"],
                "prompt": prompt,
                "prompt_sha256": sha256_text(SYSTEM + "\n" + prompt),
                "requested_model": model,
            })
    return jobs


async def one(client: AsyncOpenAI, job: dict[str, Any], model: str, retries: int, semaphore: asyncio.Semaphore, credential: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await client.responses.create(
                    model=model,
                    instructions=SYSTEM,
                    input=job["prompt"],
                    reasoning={"effort": "none"},
                    max_output_tokens=256,
                )
                text = (response.output_text or "").strip()
                status = getattr(response, "status", None)
                if status == "completed" and text:
                    return {
                        **{key: value for key, value in job.items() if key != "prompt"},
                        "reference_text": text,
                        "reference_text_sha256": sha256_text(text),
                        "response_model": getattr(response, "model", None),
                        "response_id_sha256": sha256_text(str(response.id)) if getattr(response, "id", None) else None,
                        "reference_source_model": model,
                        "reference_status": "auto_generated",
                        "human_verified": False,
                        "approval_status": None,
                        "reasoning_effort_requested": "none",
                        "temperature_requested": None,
                        "snapshot_frozen": False,
                        **usage_fields(response.usage),
                        "retry_count": attempt - 1,
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "timestamp": utc_now(),
                    }, None
                last_error = f"non-completed or empty response: status={status}"
            except Exception as exc:
                last_error = redact(repr(exc), [credential])
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 8) + random.random())
        return None, {**{key: value for key, value in job.items() if key != "prompt"}, "status": "failed", "error": last_error, "retry_count": retries, "timestamp": utc_now()}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    service = config["gpt"]
    credential = require_credential(service["credential_env"])
    employee_id = require_environment_value(service["emp_id_env"])
    scenarios = read_jsonl(config["scenario_file"])
    jobs = build_jobs(config, scenarios, args.model)
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = {row["reference_id"] for row in existing_rows}
    todo = [job for job in jobs if job["reference_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    price = load_prices(args.pricing, args.model)
    cumulative_usage = {key: sum(int(row.get(key) or 0) for row in existing_rows) for key in usage_fields(None)}
    client = AsyncOpenAI(
        base_url=service["base_url"],
        api_key=credential,
        default_headers={"empId": employee_id, "iai-tag": "sbe-computers-major-r1-gpt-reference"},
        timeout=90.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency or int(service["concurrency_start"]))
    successes = failures = 0
    chunk_size = max(1, semaphore._value * 2)
    for start in range(0, len(todo), chunk_size):
        if estimated_cost(cumulative_usage, price) >= args.budget_usd:
            break
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(*(one(client, job, args.model, int(service["retry_attempts"]), semaphore, credential) for job in chunk))
        good = [row for row, _ in results if row is not None]
        bad = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, good)
        append_jsonl(args.failure_log, bad)
        successes += len(good); failures += len(bad)
        for row in good:
            for key in cumulative_usage:
                cumulative_usage[key] += int(row.get(key) or 0)
        progress = {"updated_at": utc_now(), "model": args.model, "planned": len(jobs), "completed_total": len(existing) + successes, "failures_this_run": failures, "usage": cumulative_usage, "estimated_cost_usd": estimated_cost(cumulative_usage, price), "budget_usd": args.budget_usd, "account_balance_available": False}
        atomic_write_json(args.progress, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    await client.close()
    return {"planned": len(jobs), "existing": len(existing), "successes": successes, "failures": failures, "estimated_cost_usd": estimated_cost(cumulative_usage, price)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-usd", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output = args.output_root / "data/gpt_fixed_references.jsonl"
    args.failure_log = args.output_root / "logs/gpt_reference_failures.jsonl"
    args.progress = args.output_root / "manifests/gpt_reference_usage_progress.json"
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
