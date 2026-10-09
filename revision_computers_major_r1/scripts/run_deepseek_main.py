from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from openai import AsyncOpenAI

from common import (
    atomic_write_json,
    read_jsonl,
    redact,
    require_credential,
    require_environment_value,
    sha256_text,
    stable_id,
    utc_now,
)
from prompts import GENERATION_SYSTEM_PROMPT, render_external_prompt


def build_jobs(config: dict[str, Any], scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    experiment = config["experiment_id"]
    model_id = config["main"]["external_model_id"]
    template = config["main"]["prompt_template_id"]
    for scenario in scenarios:
        for alpha in config["main"]["target_ratios"]:
            for replicate in config["main"]["external_replicates"]:
                original, sent = render_external_prompt(scenario, float(alpha), template)
                jobs.append(
                    {
                        "run_id": stable_id("major_r1_run", experiment, model_id, scenario["scenario_id"], alpha, replicate, template),
                        "experiment_id": experiment,
                        "scenario_id": scenario["scenario_id"],
                        "style_axis": scenario["axis_id"],
                        "style_A": scenario["style_a"],
                        "style_B": scenario["style_b"],
                        "target_ratio": float(alpha),
                        "model_id": model_id,
                        "provider": config["deepseek"]["provider"],
                        "requested_model": config["deepseek"]["requested_model"],
                        "replicate_id": int(replicate),
                        "seed": None,
                        "actual_seed": None,
                        "seed_supported": False,
                        "prompt_template_id": template,
                        "original_qwen_prompt_sha256": sha256_text(original),
                        "sent_prompt_sha256": sha256_text(sent),
                        "wrapper_difference": "removed_leading_qwen_no_think_control_line",
                        "sent_prompt": sent,
                        "input_text": scenario["content"],
                        "content_checks": scenario.get("content_checks", []),
                        "length_range_zh_chars": scenario.get("length_range_zh_chars"),
                        "endpoint_mode": "self_generated",
                    }
                )
    return jobs


def refusal_reason(response: Any, content: str) -> str | None:
    choice = response.choices[0]
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        return "message_refusal"
    if choice.finish_reason in {"content_filter", "safety"}:
        return str(choice.finish_reason)
    if not content.strip():
        return "empty_content"
    if choice.finish_reason in {"length", "max_tokens"}:
        return "truncated"
    return None


async def generate_one(
    client: AsyncOpenAI,
    job: dict[str, Any],
    service: dict[str, Any],
    semaphore: asyncio.Semaphore,
    credential: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, int(service["retry_attempts"]) + 1):
            request_id = str(uuid4())
            try:
                kwargs: dict[str, Any] = {
                    "model": service["requested_model"],
                    "messages": [
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": job["sent_prompt"]},
                    ],
                    "temperature": float(service["temperature"]),
                    "max_tokens": int(service["max_tokens"]),
                    "timeout": float(service["request_timeout_seconds"]),
                    "extra_headers": {"requestId": request_id},
                }
                if service.get("top_p") is not None:
                    kwargs["top_p"] = float(service["top_p"])
                response = await client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                refusal = refusal_reason(response, content)
                usage = response.usage
                reasoning = getattr(choice.message, "reasoning_content", None)
                status = "ok" if refusal is None else ("truncated" if refusal == "truncated" else "refusal_or_empty")
                row = {
                    **{key: value for key, value in job.items() if key != "sent_prompt"},
                    "generated_text": content,
                    "generated_text_sha256": sha256_text(content),
                    "status": status,
                    "failure_category": refusal,
                    "response_model": getattr(response, "model", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "snapshot_frozen": False,
                    "reasoning_present": bool(reasoning),
                    "reasoning_content_persisted": False,
                    "temperature_requested": float(service["temperature"]),
                    "temperature_effective": None,
                    "top_p_requested": service.get("top_p"),
                    "top_p_effective": None,
                    "max_tokens_requested": int(service["max_tokens"]),
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                    "retry_count": attempt - 1,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "request_id_sha256": sha256_text(request_id),
                    "timestamp": utc_now(),
                }
                # A completed response is an experimental outcome.  Refusal,
                # empty content, and truncation-like content failures are
                # recorded once and are never regenerated based on quality.
                # Only transport/relay exceptions enter the retry path below.
                return row, None
            except Exception as exc:
                last_error = redact(repr(exc), [credential])
                if attempt < int(service["retry_attempts"]):
                    delay = min(2 ** attempt, 16) + random.random()
                    await asyncio.sleep(delay)
                    continue
                break
        return None, {
            **{key: value for key, value in job.items() if key != "sent_prompt"},
            "status": "transport_failure",
            "failure_category": "transport_or_relay_error",
            "error": last_error,
            "retry_count": int(service["retry_attempts"]),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "timestamp": utc_now(),
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    credential = require_credential(config["deepseek"]["credential_env"])
    scenarios = read_jsonl(config["scenario_file"])
    jobs = build_jobs(config, scenarios)
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = {row["run_id"] for row in existing_rows}
    todo = [job for job in jobs if job["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    service = config["deepseek"]
    employee_id = require_environment_value(service["emp_id_env"])
    client = AsyncOpenAI(
        base_url=service["base_url"],
        api_key=credential,
        default_headers={"empId": employee_id, "iai-tag": service["iai_tag"]},
        timeout=120.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency or int(service["concurrency_start"]))
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    successes = failures = content_failures = 0
    chunk_size = max(1, semaphore._value * 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        with args.output.open("a", encoding="utf-8") as output_handle, args.failure_log.open("a", encoding="utf-8") as failure_handle:
            for start in range(0, len(todo), chunk_size):
                chunk = todo[start : start + chunk_size]
                tasks = [asyncio.create_task(generate_one(client, job, service, semaphore, credential)) for job in chunk]
                for future in asyncio.as_completed(tasks):
                    row, failure = await future
                    if row is not None:
                        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        successes += int(row["status"] == "ok")
                        content_failures += int(row["status"] != "ok")
                        for key in totals:
                            totals[key] += int(row.get(key) or 0)
                    if failure is not None:
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                        failure_handle.flush()
                        failures += 1
                    completed += 1
                    if completed % 32 == 0 or completed == len(todo):
                        os.fsync(output_handle.fileno())
                        os.fsync(failure_handle.fileno())
                        progress = {
                            "updated_at": utc_now(),
                            "planned": len(jobs),
                            "existing_at_start": len(existing),
                            "attempted_this_run": completed,
                            "successes_this_run": successes,
                            "content_failures_this_run": content_failures,
                            "transport_failures_this_run": failures,
                            "usage_this_run": totals,
                            "account_balance_available": False,
                            "account_balance_note": "The tested relay API did not expose a balance endpoint or quota headers.",
                        }
                        atomic_write_json(args.usage_manifest, progress)
                        print(json.dumps(progress, ensure_ascii=False), flush=True)
                if args.max_total_tokens is not None and totals["total_tokens"] >= args.max_total_tokens:
                    break
            os.fsync(output_handle.fileno())
            os.fsync(failure_handle.fileno())
    finally:
        await client.close()
    return {"planned": len(jobs), "todo_at_start": len(todo), "successes": successes, "content_failures": content_failures, "transport_failures": failures, **totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    args = parser.parse_args()
    args.output = args.output_root / "data/deepseek_main_outputs.jsonl"
    args.failure_log = args.output_root / "logs/deepseek_main_failures.jsonl"
    args.usage_manifest = args.output_root / "manifests/deepseek_usage_progress.json"
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
