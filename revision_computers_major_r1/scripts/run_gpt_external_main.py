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

from common import atomic_write_json, read_jsonl, redact, require_credential, require_environment_value, sha256_text, stable_id, utc_now
from gpt_common import estimated_cost, load_prices, usage_fields
from prompts import GENERATION_SYSTEM_PROMPT, render_external_prompt


def build_jobs(config: dict[str, Any], scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    service = config["external_generator"]
    experiment = config["experiment_id"]
    model_id = config["main"]["external_model_id"]
    template = config["main"]["prompt_template_id"]
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for alpha in config["main"]["target_ratios"]:
            for replicate in config["main"]["external_replicates"]:
                original, sent = render_external_prompt(scenario, float(alpha), template)
                jobs.append({
                    "run_id": stable_id("major_r1_run", experiment, model_id, scenario["scenario_id"], alpha, replicate, template),
                    "experiment_id": experiment,
                    "scenario_id": scenario["scenario_id"],
                    "style_axis": scenario["axis_id"],
                    "style_A": scenario["style_a"],
                    "style_B": scenario["style_b"],
                    "target_ratio": float(alpha),
                    "model_id": model_id,
                    "provider": service["provider"],
                    "requested_model": service["requested_model"],
                    "replicate_id": int(replicate),
                    "seed": None,
                    "actual_seed": None,
                    "seed_supported": False,
                    "prompt_template_id": template,
                    "original_qwen_prompt_sha256": sha256_text(original),
                    "sent_prompt_sha256": sha256_text(sent),
                    "system_prompt_sha256": sha256_text(GENERATION_SYSTEM_PROMPT),
                    "wrapper_difference": "removed_leading_qwen_no_think_control_line",
                    "sent_prompt": sent,
                    "input_text": scenario["content"],
                    "content_checks": scenario.get("content_checks", []),
                    "length_range_zh_chars": scenario.get("length_range_zh_chars"),
                    "endpoint_mode": "self_generated",
                })
    return jobs


def refusal_present(response: Any) -> bool:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                return True
    return False


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
            try:
                response = await client.responses.create(
                    model=service["requested_model"],
                    instructions=GENERATION_SYSTEM_PROMPT,
                    input=job["sent_prompt"],
                    reasoning={"effort": service["reasoning_effort"]},
                    temperature=float(service["temperature"]),
                    max_output_tokens=int(service["max_output_tokens"]),
                )
                text = (response.output_text or "").strip()
                response_status = getattr(response, "status", None)
                incomplete = getattr(response, "incomplete_details", None)
                incomplete_reason = getattr(incomplete, "reason", None) if incomplete else None
                refused = refusal_present(response)
                if response_status == "completed" and text:
                    status = "ok"
                    failure_category = None
                elif text and incomplete_reason == "max_output_tokens":
                    status = "truncated"
                    failure_category = "max_output_tokens"
                elif text:
                    status = "incomplete_with_text"
                    failure_category = incomplete_reason or str(response_status)
                else:
                    status = "refusal_or_empty"
                    failure_category = "refusal" if refused else (incomplete_reason or "empty_content")
                usage = usage_fields(response.usage)
                return {
                    **{key: value for key, value in job.items() if key != "sent_prompt"},
                    "generated_text": text,
                    "generated_text_sha256": sha256_text(text),
                    "status": status,
                    "failure_category": failure_category,
                    "response_status": response_status,
                    "response_model": getattr(response, "model", None),
                    "response_id_sha256": sha256_text(str(response.id)) if getattr(response, "id", None) else None,
                    "snapshot_frozen": False,
                    "reasoning_effort_requested": service["reasoning_effort"],
                    "reasoning_content_persisted": False,
                    "temperature_requested": float(service["temperature"]),
                    "temperature_effective": float(service["temperature"]),
                    "top_p_requested": None,
                    "top_p_effective": None,
                    "max_output_tokens_requested": int(service["max_output_tokens"]),
                    **usage,
                    "retry_count": attempt - 1,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "timestamp": utc_now(),
                }, None
            except Exception as exc:
                last_error = redact(repr(exc), [credential])
                if attempt < int(service["retry_attempts"]):
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
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
    service = config["external_generator"]
    credential = require_credential(service["credential_env"])
    employee_id = require_environment_value(service["emp_id_env"])
    jobs = build_jobs(config, read_jsonl(config["scenario_file"]))
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = {row["run_id"] for row in existing_rows}
    todo = [job for job in jobs if job["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    prices = load_prices(args.pricing, service["requested_model"])
    usage = {key: sum(int(row.get(key) or 0) for row in existing_rows) for key in usage_fields(None)}
    client = AsyncOpenAI(
        base_url=service["base_url"],
        api_key=credential,
        default_headers={"empId": employee_id, "iai-tag": service["iai_tag"]},
        timeout=float(service["request_timeout_seconds"]),
        max_retries=0,
    )
    concurrency = args.concurrency or int(service["concurrency_start"])
    semaphore = asyncio.Semaphore(concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)
    successes = content_failures = transport_failures = completed = next_index = 0

    def task(row: dict[str, Any]) -> asyncio.Task:
        return asyncio.create_task(generate_one(client, row, service, semaphore, credential))

    try:
        with args.output.open("a", encoding="utf-8") as output_handle, args.failure_log.open("a", encoding="utf-8") as failure_handle:
            active: set[asyncio.Task] = set()
            while next_index < len(todo) and len(active) < concurrency and estimated_cost(usage, prices) < args.budget_usd:
                active.add(task(todo[next_index])); next_index += 1
            while active:
                done, pending = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                active = set(pending)
                for future in done:
                    row, failure = await future
                    if row is not None:
                        output_handle.write(json.dumps(row, ensure_ascii=False) + "\n"); output_handle.flush()
                        successes += int(row["status"] in {"ok", "truncated"})
                        content_failures += int(row["status"] not in {"ok", "truncated"})
                        for key in usage:
                            usage[key] += int(row.get(key) or 0)
                    if failure is not None:
                        failure_handle.write(json.dumps(failure, ensure_ascii=False) + "\n"); failure_handle.flush()
                        transport_failures += 1
                    completed += 1
                    if completed % 32 == 0 or completed == len(todo):
                        os.fsync(output_handle.fileno()); os.fsync(failure_handle.fileno())
                        progress = {
                            "updated_at": utc_now(), "planned": len(jobs), "existing_at_start": len(existing),
                            "attempted_this_run": completed, "successes_this_run": successes,
                            "content_failures_this_run": content_failures, "transport_failures_this_run": transport_failures,
                            "usage": usage, "estimated_cost_usd": estimated_cost(usage, prices), "budget_usd": args.budget_usd,
                            "account_balance_available": False,
                        }
                        atomic_write_json(args.progress, progress); print(json.dumps(progress, ensure_ascii=False), flush=True)
                while next_index < len(todo) and len(active) < concurrency and estimated_cost(usage, prices) < args.budget_usd:
                    active.add(task(todo[next_index])); next_index += 1
            os.fsync(output_handle.fileno()); os.fsync(failure_handle.fileno())
    finally:
        await client.close()
    return {
        "planned": len(jobs), "todo_at_start": len(todo), "successes": successes,
        "content_failures": content_failures, "transport_failures": transport_failures,
        "usage": usage, "estimated_cost_usd": estimated_cost(usage, prices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-usd", type=float, default=10.0)
    args = parser.parse_args()
    args.output = args.output or args.output_root / "data/external_main_outputs.jsonl"
    args.failure_log = args.failure_log or args.output_root / "logs/external_main_failures.jsonl"
    args.progress = args.progress or args.output_root / "manifests/external_main_usage_progress.json"
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
