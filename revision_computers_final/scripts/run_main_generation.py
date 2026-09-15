"""Resumable three-seed generation through an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        REVISION_ROOT,
        atomic_write_json,
        finish_run_manifest,
        git_commit,
        new_run_manifest,
        read_jsonl_tolerant,
        sha256_file,
        utc_now,
    )
else:
    from .common import (
        REVISION_ROOT,
        atomic_write_json,
        finish_run_manifest,
        git_commit,
        new_run_manifest,
        read_jsonl_tolerant,
        sha256_file,
        utc_now,
    )


SYSTEM_PROMPT = (
    "你是中文短文本生成器。不要展示推理、分析或中间步骤；任何情况下都只输出最终中文正文。"
)


def stable_run_id(*parts: Any) -> str:
    import hashlib

    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]
    return f"revision_run_{digest}"


def ratio_text(alpha: float) -> str:
    known = {
        0.0: "alpha = 0（100% 风格A + 0% 风格B）",
        1 / 6: "alpha = 1/6（约 83.33% 风格A + 16.67% 风格B）",
        2 / 6: "alpha = 2/6（约 66.67% 风格A + 33.33% 风格B）",
        3 / 6: "alpha = 3/6（50% 风格A + 50% 风格B）",
        4 / 6: "alpha = 4/6（约 33.33% 风格A + 66.67% 风格B）",
        5 / 6: "alpha = 5/6（约 16.67% 风格A + 83.33% 风格B）",
        1.0: "alpha = 1（0% 风格A + 100% 风格B）",
    }
    for value, label in known.items():
        if math.isclose(alpha, value, abs_tol=1e-12):
            return label
    raise ValueError(f"unsupported target ratio {alpha}")


def render_prompt(scenario: dict[str, Any], alpha: float, template_id: str) -> str:
    minimum, maximum = scenario.get("length_range_zh_chars", [80, 140])
    if template_id == "computers_explicit_ratio_v1":
        return (
            "/no_think\n"
            "严禁输出 Thinking Process、Analysis、推理过程、步骤说明、标题或 Markdown 列表。\n"
            "你只能输出最终中文正文，第一句话就必须是正文内容。\n\n"
            "你将根据一个固定核心语义生成中文短文本。\n\n"
            f"核心语义：\n{scenario['content']}\n\n"
            f"风格A：\n{scenario['style_a']}\n\n"
            f"风格B：\n{scenario['style_b']}\n\n"
            f"目标风格比例：\n{ratio_text(alpha)}。\n\n"
            "硬性要求：\n"
            "1. 必须保留核心语义，不添加与核心语义冲突的新事实。\n"
            f"2. 正文必须至少 {minimum} 个汉字，最多 {maximum} 个汉字；少于 {minimum} 个汉字不合格。\n"
            "3. 只输出正文，不要解释比例，不要列标题。\n"
            "4. 不要输出任何英文分析，不要输出 Thinking Process。"
        )
    if template_id == "computers_explicit_ratio_v2":
        return (
            "/no_think\n只提交最终中文正文，不展示任何构思、推理、分析、标题或列表。\n\n"
            "请忠实表达下列固定内容，并将语言风格调到指定的连续位置。\n\n"
            f"固定内容：\n{scenario['content']}\n\n"
            f"连续风格区间的A端：\n{scenario['style_a']}\n\n"
            f"连续风格区间的B端：\n{scenario['style_b']}\n\n"
            f"所需B端强度：\n{ratio_text(alpha)}。\n\n"
            "写作约束：\n"
            "- 完整保留固定内容，不引入冲突事实。\n"
            f"- 正文必须在 {minimum} 至 {maximum} 个汉字之间。\n"
            "- 不解释比例，不使用标题或 Markdown，只输出正文。"
        )
    raise ValueError(f"unknown prompt template: {template_id}")


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"(?is)^.*?</think>", "", cleaned)
    return re.sub(r"(?is)</?think>", "", cleaned).strip()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["model_id"]: row for row in rows}


def build_jobs(
    config: dict[str, Any],
    scenarios: list[dict[str, Any]],
    model: dict[str, str],
    template_id: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for scenario in scenarios:
        for alpha in config["target_ratios"]:
            for seed in config["seeds"]:
                run_id = stable_run_id(
                    config["experiment_id"], model["model_id"], scenario["scenario_id"], alpha, seed, template_id
                )
                jobs.append(
                    {
                        "run_id": run_id,
                        "scenario_id": scenario["scenario_id"],
                        "style_axis": scenario["axis_id"],
                        "style_A": scenario["style_a"],
                        "style_B": scenario["style_b"],
                        "target_ratio": float(alpha),
                        "model_id": model["model_id"],
                        "family_version": model["family_version"],
                        "architecture": model["architecture"],
                        "seed": int(seed),
                        "prompt_template_id": template_id,
                        "prompt": render_prompt(scenario, float(alpha), template_id),
                        "input_text": scenario["content"],
                        "content_checks": scenario.get("content_checks", []),
                        "length_range_zh_chars": scenario.get("length_range_zh_chars"),
                        "endpoint_mode": "self_generated",
                    }
                )
    for job in jobs:
        import hashlib

        job["prompt_sha256"] = hashlib.sha256(job["prompt"].encode("utf-8")).hexdigest()
    return jobs


async def generate_one(
    client: AsyncOpenAI,
    job: dict[str, Any],
    model_name: str,
    generation: dict[str, Any],
    semaphore: asyncio.Semaphore,
    dry_run: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(1, int(generation["retry_attempts"]) + 1):
            attempt_started = time.perf_counter()
            try:
                if dry_run:
                    generated = f"[DRY_RUN seed={job['seed']}] {job['input_text']}"
                    prompt_tokens = 0
                    completion_tokens = 0
                    finish_reason = "dry_run"
                else:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": job["prompt"]},
                        ],
                        temperature=float(generation["temperature"]),
                        top_p=float(generation["top_p"]),
                        max_tokens=int(generation["max_new_tokens"]),
                        timeout=float(generation["request_timeout_seconds"]),
                        extra_body={
                            "seed": int(job["seed"]),
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    generated = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason
                    usage = response.usage
                    prompt_tokens = int(usage.prompt_tokens) if usage else -1
                    completion_tokens = int(usage.completion_tokens) if usage else -1
                latency = time.perf_counter() - attempt_started
                row = {
                    **job,
                    "generated_text": strip_thinking(generated),
                    "temperature": float(generation["temperature"]),
                    "top_p": float(generation["top_p"]),
                    "max_new_tokens": int(generation["max_new_tokens"]),
                    "thinking": "disabled",
                    "seed_passed_to_server": not dry_run,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "latency_seconds": round(latency, 6),
                    "finish_reason": finish_reason,
                    "retry_count": attempt - 1,
                    "timestamp": utc_now(),
                    "git_commit": git_commit(),
                }
                return row, None
            except Exception as exc:  # recorded verbatim in the failure JSONL
                last_error = exc
                if attempt < int(generation["retry_attempts"]):
                    await asyncio.sleep(min(2**attempt, 8))
        return None, {
            **job,
            "error": repr(last_error),
            "retry_count": int(generation["retry_attempts"]),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "timestamp": utc_now(),
        }


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


async def run(args: argparse.Namespace) -> tuple[int, int, int]:
    config = load_config(args.config)
    registry = load_registry(Path(config["model_registry"]))
    if args.model_id not in registry or registry[args.model_id]["included"].lower() != "true":
        raise ValueError(f"model is missing or excluded from registry: {args.model_id}")
    model = registry[args.model_id]
    scenarios_result = read_jsonl_tolerant(config["scenario_file"])
    if scenarios_result.bad_lines:
        raise ValueError("scenario source contains malformed JSONL")
    scenarios = scenarios_result.rows
    if args.scenario_id:
        wanted = set(args.scenario_id)
        scenarios = [row for row in scenarios if row["scenario_id"] in wanted]
    jobs = build_jobs(config, scenarios, model, args.prompt_template_id)
    existing_result = read_jsonl_tolerant(args.output) if args.output.exists() else None
    existing = {row["run_id"] for row in existing_result.rows} if existing_result else set()
    todo = [job for job in jobs if job["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    succeeded = 0
    failed = 0
    chunk_size = max(args.concurrency * 4, 1)
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(
            *[
                generate_one(
                    client,
                    job,
                    args.served_model_name,
                    config["generation"],
                    semaphore,
                    args.dry_run,
                )
                for job in chunk
            ]
        )
        success_rows = [row for row, _ in results if row is not None]
        failure_rows = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, success_rows)
        append_jsonl(args.failure_log, failure_rows)
        succeeded += len(success_rows)
        failed += len(failure_rows)
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "planned": len(jobs),
                    "existing": len(existing),
                    "attempted": min(start + len(chunk), len(todo)),
                    "succeeded": succeeded,
                    "failed": failed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return len(jobs), succeeded, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REVISION_ROOT / "config/main_experiment.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=False)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--prompt-template-id", default="computers_explicit_ratio_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--scenario-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.base_url and not args.dry_run:
        parser.error("--base-url is required unless --dry-run is used")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    output_root = REVISION_ROOT / "data/fragments/main"
    args.output = args.output or output_root / f"{args.model_id}.jsonl"
    args.failure_log = args.failure_log or REVISION_ROOT / "logs" / f"generation_failures_{args.model_id}.jsonl"
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    registry = load_registry(Path(config["model_registry"]))
    model = registry.get(args.model_id, {})
    run_id = f"main_generation_{args.model_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path = REVISION_ROOT / "manifests" / f"run_manifest_{run_id}.json"
    manifest = new_run_manifest(
        run_id=run_id,
        command=sys.argv,
        config_paths=[args.config, config["model_registry"]],
        data_paths=[config["scenario_file"]],
        checkpoint_name=model.get("source"),
        checkpoint_path=model.get("checkpoint_path"),
        seeds=config["seeds"],
        output_paths=[args.output, args.failure_log],
    )
    manifest["config_hash"] = sha256_file(args.config)
    atomic_write_json(manifest_path, manifest)
    try:
        planned, succeeded, failed = asyncio.run(run(args))
    except BaseException as exc:
        finish_run_manifest(manifest_path, manifest, status="failed", error=repr(exc))
        raise
    status = "success" if failed == 0 else "partial_failure"
    finish_run_manifest(
        manifest_path,
        manifest,
        status=status,
        counts={"planned": planned, "new_successes": succeeded, "new_failures": failed},
    )


if __name__ == "__main__":
    main()
