from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from common import append_jsonl, atomic_write_json, read_jsonl, utc_now
from gpt_common import JUDGE_SYSTEM_PROMPT, judge_prompt
from qwen_judge import EVALUATOR_ID, cache_identity, parse_qwen_json


async def one(client: AsyncOpenAI, row: dict[str, Any], args: argparse.Namespace, semaphore: asyncio.Semaphore) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    prompt = judge_prompt(row)
    identity = cache_identity(row)
    async with semaphore:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, args.retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.served_model_name,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=256,
                    timeout=args.timeout,
                    extra_body={"seed": 20260917, "chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = response.choices[0].message.content or ""
                parsed = parse_qwen_json(raw, len(row["content_checks"]))
                usage = response.usage
                return {
                    "evaluation_item_id": row["evaluation_item_id"],
                    "run_id": row["run_id"],
                    "source_run_id": row["original_source_run_id"],
                    "scenario_id": row["scenario_id"],
                    "style_axis": row["style_axis"],
                    "model_id": row["model_id"],
                    "seed": row.get("seed"),
                    "replicate_id": row.get("replicate_id"),
                    "target_ratio": row["target_ratio"],
                    "evaluator_id": EVALUATOR_ID,
                    "evaluator_checkpoint": "/home/data_cpfs/zicheng/models/Qwen3.6-35B-A3B",
                    "schema_version": "content_style_v1",
                    "temperature": 0.0,
                    "evaluator_seed": 20260917,
                    **identity,
                    "content_check_count": len(row["content_checks"]),
                    "content_covered_count": sum(parsed["covered"]),
                    **parsed,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "retry_count": attempt - 1,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "timestamp": utc_now(),
                    "cache_reuse_status": "new_evaluation",
                }, None
            except Exception as exc:
                last_error = repr(exc)
                if attempt < args.retries:
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
        return None, {"evaluation_item_id": row["evaluation_item_id"], "run_id": row["run_id"], **identity, "status": "failed", "error": last_error, "retry_count": args.retries, "timestamp": utc_now()}


async def run(args: argparse.Namespace) -> dict[str, int]:
    items = read_jsonl(args.index)
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    cached = {(row["evaluation_item_id"], row["evaluation_cache_key"]) for row in existing_rows}
    terminal_failures = read_jsonl(args.failure_log) if args.failure_log.exists() else []
    terminal = {(row["evaluation_item_id"], row["evaluation_cache_key"]) for row in terminal_failures}
    todo = [
        row for row in items
        if (row["evaluation_item_id"], cache_identity(row)["evaluation_cache_key"]) not in cached
        and (row["evaluation_item_id"], cache_identity(row)["evaluation_cache_key"]) not in terminal
    ]
    random.Random(args.shuffle_seed).shuffle(todo)
    if args.limit is not None:
        todo = todo[: args.limit]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    successes = failures = 0
    chunk_size = max(1, args.concurrency * 4)
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(*(one(client, row, args, semaphore) for row in chunk))
        good = [row for row, _ in results if row is not None]
        bad = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, good)
        append_jsonl(args.failure_log, bad)
        successes += len(good); failures += len(bad)
        progress = {"updated_at": utc_now(), "index_rows": len(items), "existing_at_start": len(existing_rows), "attempted_this_run": min(start + len(chunk), len(todo)), "successes_this_run": successes, "failures_this_run": failures}
        atomic_write_json(args.progress, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    await client.close()
    return {"index_rows": len(items), "existing": len(existing_rows), "terminal_failures": len(terminal), "todo_at_start": len(todo), "successes": successes, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", default="qwen36_35b_a3b")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--shuffle-seed", type=int, default=20260916)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.index = args.index or args.output_root / "data/evaluation_index.jsonl"
    args.output = args.output or args.output_root / "metrics/qwen_judge_evaluations.jsonl"
    args.failure_log = args.failure_log or args.output_root / "logs/qwen_judge_failures.jsonl"
    args.progress = args.progress or args.output_root / "manifests/qwen_judge_progress.json"
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
