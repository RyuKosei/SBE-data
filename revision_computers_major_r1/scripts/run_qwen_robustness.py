from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from common import append_jsonl, atomic_write_json, read_jsonl, sha256_text, stable_id, utc_now
from prompts import GENERATION_SYSTEM_PROMPT, render_qwen_prompt


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"(?is)^.*?</think>", "", cleaned)
    return re.sub(r"(?is)</?think>", "", cleaned).strip()


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["model_id"]: row for row in csv.DictReader(handle)}


def build_jobs(config: dict[str, Any], scenarios: list[dict[str, Any]], model: dict[str, str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for alpha in config["robustness"]["interior_ratios"]:
            for seed in config["robustness"]["seeds"]:
                for condition in config["robustness"]["conditions"]:
                    prompt_id = condition["prompt_template_id"]
                    temperature = float(condition["temperature"])
                    run_id = stable_id("major_r1_robust", config["experiment_id"], model["model_id"], scenario["scenario_id"], alpha, seed, prompt_id, temperature)
                    prompt = render_qwen_prompt(scenario, float(alpha), prompt_id)
                    jobs.append({
                        "run_id": run_id,
                        "source_run_id": None,
                        "scenario_id": scenario["scenario_id"],
                        "style_axis": scenario["axis_id"],
                        "style_A": scenario["style_a"],
                        "style_B": scenario["style_b"],
                        "target_ratio": float(alpha),
                        "model_id": model["model_id"],
                        "family_version": model["family_version"],
                        "architecture": model["architecture"],
                        "seed": int(seed),
                        "prompt_template_id": prompt_id,
                        "robustness_condition_id": condition["condition_id"],
                        "temperature": temperature,
                        "top_p": float(config["robustness"]["top_p"]),
                        "max_new_tokens": int(config["robustness"]["max_new_tokens"]),
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "input_text": scenario["content"],
                        "content_checks": scenario.get("content_checks", []),
                        "length_range_zh_chars": scenario.get("length_range_zh_chars"),
                        "endpoint_mode": "fixed_reference_analysis",
                        "robustness_source": "major_r1_generation",
                    })
    return jobs


async def one(client: AsyncOpenAI, job: dict[str, Any], args: argparse.Namespace, config: dict[str, Any], semaphore: asyncio.Semaphore) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, int(config["robustness"]["retry_attempts"]) + 1):
            try:
                response = await client.chat.completions.create(
                    model=args.served_model_name,
                    messages=[{"role": "system", "content": GENERATION_SYSTEM_PROMPT}, {"role": "user", "content": job["prompt"]}],
                    temperature=float(job["temperature"]),
                    top_p=float(job["top_p"]),
                    max_tokens=int(job["max_new_tokens"]),
                    timeout=float(config["robustness"]["request_timeout_seconds"]),
                    extra_body={"seed": int(job["seed"]), "chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = response.choices[0].message.content or ""
                generated = strip_thinking(raw)
                usage = response.usage
                if not generated:
                    return None, {**job, "status": "empty_content", "retry_count": attempt - 1, "timestamp": utc_now()}
                return {
                    **job,
                    "generated_text": generated,
                    "generated_text_sha256": sha256_text(generated),
                    "raw_response_sha256": sha256_text(raw),
                    "thinking_block_removed": generated != raw.strip(),
                    "thinking": "disabled",
                    "seed_passed_to_server": True,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "finish_reason": response.choices[0].finish_reason,
                    "retry_count": attempt - 1,
                    "timestamp": utc_now(),
                    "status": "ok",
                }, None
            except Exception as exc:
                last_error = repr(exc)
                if attempt < int(config["robustness"]["retry_attempts"]):
                    await asyncio.sleep(min(2 ** attempt, 8) + random.random())
        return None, {**job, "status": "transport_failure", "error": last_error, "retry_count": int(config["robustness"]["retry_attempts"]), "timestamp": utc_now()}


async def run(args: argparse.Namespace) -> dict[str, int]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    legacy = Path(config["legacy_root"])
    registry = load_registry(legacy / "config/model_registry.csv")
    if args.model_id not in registry:
        raise ValueError(f"unknown model: {args.model_id}")
    scenarios = read_jsonl(config["scenario_file"])
    jobs = build_jobs(config, scenarios, registry[args.model_id])
    existing_rows = read_jsonl(args.output) if args.output.exists() else []
    existing = {row["run_id"] for row in existing_rows}
    todo = [job for job in jobs if job["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    semaphore = asyncio.Semaphore(args.concurrency)
    successes = failures = 0
    chunk_size = max(args.concurrency * 4, 1)
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(*(one(client, job, args, config, semaphore) for job in chunk))
        good = [row for row, _ in results if row is not None]
        bad = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, good)
        append_jsonl(args.failure_log, bad)
        successes += len(good)
        failures += len(bad)
        progress = {"model_id": args.model_id, "planned": len(jobs), "existing_at_start": len(existing), "attempted": min(start + len(chunk), len(todo)), "successes": successes, "failures": failures, "updated_at": utc_now()}
        atomic_write_json(args.progress, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    await client.close()
    return {"planned": len(jobs), "existing_at_start": len(existing), "todo_at_start": len(todo), "successes": successes, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output = args.output_root / "data/fragments/robustness" / f"{args.model_id}.jsonl"
    args.failure_log = args.output_root / "logs" / f"robustness_failures_{args.model_id}.jsonl"
    args.progress = args.output_root / "manifests" / f"robustness_progress_{args.model_id}.json"
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))


if __name__ == "__main__":
    main()
