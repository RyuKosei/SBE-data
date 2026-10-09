from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from common import atomic_write_json, read_jsonl, redact, require_credential, require_environment_value, sha256_text, utc_now
from gpt_common import (
    JUDGE_SYSTEM_PROMPT,
    cache_identity,
    estimated_cost,
    judge_prompt,
    judge_schema,
    load_prices,
    normalize_judgment,
    usage_fields,
)


async def one(client: AsyncOpenAI, row: dict[str, Any], model: str, retries: int, max_output_tokens: int, semaphore: asyncio.Semaphore, credential: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prompt = judge_prompt(row)
    schema = judge_schema(len(row["content_checks"]))
    identity = cache_identity(model, prompt, schema, row["generated_text"], row["evaluation_item_id"])
    async with semaphore:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await client.responses.create(
                    model=model,
                    instructions=JUDGE_SYSTEM_PROMPT,
                    input=prompt,
                    reasoning={"effort": "none"},
                    max_output_tokens=max_output_tokens,
                    text={"format": {"type": "json_schema", "name": "sbe_judgment", "schema": schema, "strict": True}},
                )
                if getattr(response, "status", None) != "completed" or not response.output_text:
                    raise ValueError(f"incomplete or empty response: {getattr(response, 'status', None)}")
                parsed = normalize_judgment(json.loads(response.output_text), len(row["content_checks"]))
                return {
                    "evaluation_item_id": row["evaluation_item_id"],
                    "run_id": row["run_id"],
                    "evaluator_id": f"{model}_content_style_judge_major_r1",
                    "requested_model": model,
                    "response_model": getattr(response, "model", None),
                    "response_id_sha256": sha256_text(str(response.id)) if getattr(response, "id", None) else None,
                    "snapshot_frozen": False,
                    **identity,
                    **parsed,
                    **usage_fields(response.usage),
                    "reasoning_effort_requested": "none",
                    "temperature_requested": None,
                    "retry_count": attempt - 1,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "timestamp": utc_now(),
                }, None
            except Exception as exc:
                last_error = redact(repr(exc), [credential])
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
        return None, {"evaluation_item_id": row["evaluation_item_id"], "run_id": row["run_id"], **identity, "status": "failed", "error": last_error, "retry_count": retries, "timestamp": utc_now()}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    service = config["gpt"]
    credential = require_credential(service["credential_env"])
    employee_id = require_environment_value(service["emp_id_env"])
    items = read_jsonl(args.index)
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    cached = {row["evaluation_cache_key"] for row in existing_rows}
    terminal_failures = read_jsonl(args.failure_log) if args.failure_log.exists() else []
    terminal = {row["evaluation_cache_key"] for row in terminal_failures}
    prepared: list[dict[str, Any]] = []
    for row in items:
        prompt = judge_prompt(row)
        schema = judge_schema(len(row["content_checks"]))
        identity = cache_identity(args.model, prompt, schema, row["generated_text"], row["evaluation_item_id"])
        if identity["evaluation_cache_key"] not in cached and identity["evaluation_cache_key"] not in terminal:
            prepared.append(row)
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(prepared)
    if args.limit is not None:
        prepared = prepared[: args.limit]
    price = load_prices(args.pricing, args.model)
    cumulative_usage = {key: sum(int(row.get(key) or 0) for row in existing_rows) for key in usage_fields(None)}
    client = AsyncOpenAI(
        base_url=service["base_url"],
        api_key=credential,
        default_headers={"empId": employee_id, "iai-tag": "sbe-computers-major-r1-gpt-judge"},
        timeout=90.0,
        max_retries=0,
    )
    concurrency = args.concurrency or int(service["concurrency_start"])
    semaphore = asyncio.Semaphore(concurrency)
    successes = failures = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        with args.output.open("a", encoding="utf-8") as output_handle, args.failure_log.open("a", encoding="utf-8") as failure_handle:
            next_index = 0
            active: set[asyncio.Task] = set()

            def launch_one(row: dict[str, Any]) -> asyncio.Task:
                return asyncio.create_task(
                    one(
                        client,
                        row,
                        args.model,
                        int(service["retry_attempts"]),
                        int(service["max_output_tokens_judge"]),
                        semaphore,
                        credential,
                    )
                )

            while next_index < len(prepared) and len(active) < concurrency and estimated_cost(cumulative_usage, price) < args.budget_usd:
                active.add(launch_one(prepared[next_index]))
                next_index += 1

            while active:
                done, pending = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                active = set(pending)
                for future in done:
                    row, failure = await future
                    if row is not None:
                        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        successes += 1
                        for key in cumulative_usage:
                            cumulative_usage[key] += int(row.get(key) or 0)
                    if failure is not None:
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_handle.flush()
                        failures += 1
                    completed += 1
                    if completed % 128 == 0 or completed == len(prepared):
                        os.fsync(output_handle.fileno())
                        os.fsync(failure_handle.fileno())
                        progress = {
                            "updated_at": utc_now(),
                            "model": args.model,
                            "input_items": len(items),
                            "existing_at_start": len(existing_rows),
                            "attempted_this_run": completed,
                            "successes_this_run": successes,
                            "failures_this_run": failures,
                            "usage": cumulative_usage,
                            "estimated_cost_usd": estimated_cost(cumulative_usage, price),
                            "budget_usd": args.budget_usd,
                            "account_balance_available": False,
                            "account_balance_note": "No balance or quota headers were returned by the tested relay Responses endpoint.",
                        }
                        atomic_write_json(args.progress, progress)
                        print(json.dumps(progress, ensure_ascii=False), flush=True)

                while next_index < len(prepared) and len(active) < concurrency and estimated_cost(cumulative_usage, price) < args.budget_usd:
                    active.add(launch_one(prepared[next_index]))
                    next_index += 1

            if completed % 128:
                progress = {
                    "updated_at": utc_now(),
                    "model": args.model,
                    "input_items": len(items),
                    "existing_at_start": len(existing_rows),
                    "attempted_this_run": completed,
                    "successes_this_run": successes,
                    "failures_this_run": failures,
                    "usage": cumulative_usage,
                    "estimated_cost_usd": estimated_cost(cumulative_usage, price),
                    "budget_usd": args.budget_usd,
                    "account_balance_available": False,
                    "account_balance_note": "No balance or quota headers were returned by the tested relay Responses endpoint.",
                }
                atomic_write_json(args.progress, progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            os.fsync(output_handle.fileno())
            os.fsync(failure_handle.fileno())
    finally:
        await client.close()
    return {"input_items": len(items), "existing": len(existing_rows), "terminal_failures": len(terminal), "todo_at_start": len(prepared), "successes": successes, "failures": failures, "estimated_cost_usd": estimated_cost(cumulative_usage, price)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget-usd", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=20260916)
    args = parser.parse_args()
    args.index = args.index or args.output_root / "data/evaluation_index.jsonl"
    args.output = args.output or args.output_root / "metrics/gpt_judge_evaluations.jsonl"
    args.failure_log = args.failure_log or args.output_root / "logs/gpt_judge_failures.jsonl"
    args.progress = args.progress or args.output_root / "manifests/gpt_judge_usage_progress.json"
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
