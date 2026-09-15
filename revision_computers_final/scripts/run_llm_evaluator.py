"""Run the fixed scalar-style and atomic-content evaluator over main outputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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
        new_run_manifest,
        read_jsonl_tolerant,
        utc_now,
    )
    from run_main_generation import append_jsonl  # type: ignore
else:
    from .common import (
        REVISION_ROOT,
        atomic_write_json,
        finish_run_manifest,
        new_run_manifest,
        read_jsonl_tolerant,
        utc_now,
    )
    from .run_main_generation import append_jsonl


SYSTEM_PROMPT = """你是严格的中文文本评估器。独立判断内容保持与风格强度，只输出一个合法 JSON 对象，不写 Markdown 或解释。风格强度只看表达方式，不因文本更长就自动判定更接近风格B。"""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prompt_for(row: dict[str, Any]) -> str:
    checks = row.get("content_checks") or []
    numbered = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(checks))
    return f"""请评价候选文本。核心内容检查点必须逐项判断；同义表达算覆盖。新增关键信息指原要求没有、且会影响任务结论或事实状态的信息。事实矛盾指与核心语义直接冲突。若确实无法判断，unjudgeable=true。

核心语义：{row['input_text']}
原子内容检查点：
{numbered}

风格A：{row['style_A']}
风格B：{row['style_B']}

候选文本：{row['generated_text']}

返回字段必须恰好采用以下结构：
{{"covered":[true,false],"added_key_information":false,"task_result_changed_by_addition":false,"factual_contradiction":false,"style_B_intensity":50,"naturalness":4,"unjudgeable":false}}
covered 的布尔值数量必须等于检查点数量；style_B_intensity 是 0 到 100；naturalness 是 1 到 5。"""


def parse_object(text: str) -> dict[str, Any]:
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
    if not isinstance(value, dict):
        raise ValueError("evaluator response is not an object")
    return value


def normalize(value: dict[str, Any], check_count: int) -> dict[str, Any]:
    # The evaluator occasionally returns this single transparent spelling typo;
    # retain the raw failure log and accept the semantically unambiguous alias.
    if "factual_contradiction" not in value:
        aliases = [key for key in value if key.startswith("factual_contrad") and isinstance(value[key], bool)]
        if len(aliases) == 1:
            value = {**value, "factual_contradiction": value[aliases[0]]}
    covered = value.get("covered")
    if not isinstance(covered, list) or len(covered) != check_count:
        raise ValueError(f"covered must contain exactly {check_count} entries")
    if not all(isinstance(item, bool) for item in covered):
        raise ValueError("covered entries must be booleans")
    if not isinstance(value.get("unjudgeable"), bool):
        raise ValueError("unjudgeable must be boolean")
    booleans = ("added_key_information", "task_result_changed_by_addition", "factual_contradiction")
    for key in booleans:
        if not isinstance(value.get(key), bool):
            if value["unjudgeable"] and key not in value:
                value = {**value, key: None}
            else:
                raise ValueError(f"{key} must be boolean")
    intensity = value.get("style_B_intensity")
    naturalness = value.get("naturalness")
    if not isinstance(intensity, (int, float)) or not 0 <= float(intensity) <= 100:
        raise ValueError("style_B_intensity outside [0,100]")
    if not isinstance(naturalness, (int, float)) or not 1 <= float(naturalness) <= 5:
        raise ValueError("naturalness outside [1,5]")
    return {
        "covered": covered,
        **{key: value[key] for key in booleans},
        "unjudgeable": value["unjudgeable"],
        "style_B_intensity": float(intensity),
        "naturalness": float(naturalness),
    }


async def evaluate_one(
    client: AsyncOpenAI,
    row: dict[str, Any],
    config: dict[str, Any],
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    async with semaphore:
        prompt = prompt_for(row)
        started = time.perf_counter()
        last_error: Exception | None = None
        raw = ""
        for attempt in range(1, 4):
            try:
                response = await client.chat.completions.create(
                    model=config["served_model_name"],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    max_tokens=int(config["max_tokens"]),
                    timeout=180,
                    response_format={"type": "json_object"},
                    extra_body={
                        "seed": int(config["seed"]),
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                raw = response.choices[0].message.content or ""
                result = normalize(parse_object(raw), len(row.get("content_checks") or []))
                coverage = sum(result["covered"]) / max(len(result["covered"]), 1)
                rules = config["content_preservation_pass"]
                content_pass = (
                    coverage >= float(rules["minimum_coverage"])
                    and result["factual_contradiction"] is False
                    and result["task_result_changed_by_addition"] is False
                    and not result["unjudgeable"]
                )
                usage = response.usage
                return {
                    "run_id": row["run_id"],
                    "scenario_id": row["scenario_id"],
                    "style_axis": row["style_axis"],
                    "model_id": row["model_id"],
                    "seed": int(row["seed"]),
                    "target_ratio": float(row["target_ratio"]),
                    "evaluator_id": config["evaluator_id"],
                    "evaluator_checkpoint": config["checkpoint_path"],
                    "schema_version": config["output_schema_version"],
                    "temperature": float(config["temperature"]),
                    "evaluator_seed": int(config["seed"]),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "content_check_count": len(result["covered"]),
                    "content_covered_count": int(sum(result["covered"])),
                    "content_coverage_rate": coverage,
                    "covered": result["covered"],
                    "added_key_information": result["added_key_information"],
                    "task_result_changed_by_addition": result["task_result_changed_by_addition"],
                    "factual_contradiction": result["factual_contradiction"],
                    "content_preservation_pass": content_pass,
                    "style_B_intensity": result["style_B_intensity"],
                    "naturalness": result["naturalness"],
                    "unjudgeable": result["unjudgeable"],
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "prompt_tokens": int(usage.prompt_tokens) if usage else -1,
                    "output_tokens": int(usage.completion_tokens) if usage else -1,
                    "retry_count": attempt - 1,
                    "timestamp": utc_now(),
                }, None
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    await asyncio.sleep(min(2**attempt, 8))
        return None, {
            "run_id": row["run_id"],
            "scenario_id": row["scenario_id"],
            "model_id": row["model_id"],
            "error": repr(last_error),
            "raw_response_preview": raw[:500],
            "base_url": base_url,
            "timestamp": utc_now(),
        }


async def run(args: argparse.Namespace) -> tuple[int, int, int, int]:
    config = load_yaml(args.config)
    source = read_jsonl_tolerant(args.input)
    if source.bad_lines:
        raise ValueError("main output contains malformed JSONL")
    existing_source = read_jsonl_tolerant(args.output) if args.output.exists() else None
    existing = {row["run_id"] for row in existing_source.rows} if existing_source else set()
    todo = [row for row in source.rows if row["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    successes = failures = 0
    chunk_size = args.concurrency * 4
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(
            *[evaluate_one(client, row, config, args.base_url, semaphore) for row in chunk]
        )
        good = [row for row, _ in results if row is not None]
        bad = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, good)
        append_jsonl(args.failure_log, bad)
        successes += len(good)
        failures += len(bad)
        print(
            json.dumps(
                {
                    "planned": len(source.rows),
                    "existing": len(existing),
                    "attempted": min(start + len(chunk), len(todo)),
                    "successes": successes,
                    "failures": failures,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return len(source.rows), len(existing), successes, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REVISION_ROOT / "config/evaluator.yaml")
    parser.add_argument("--input", type=Path, default=REVISION_ROOT / "data/qwen_family_outputs.jsonl")
    parser.add_argument("--output", type=Path, default=REVISION_ROOT / "metrics/llm_evaluations.jsonl")
    parser.add_argument("--failure-log", type=Path, default=REVISION_ROOT / "logs/llm_evaluator_failures.jsonl")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_id = f"llm_evaluator_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path = REVISION_ROOT / "manifests" / f"run_manifest_{run_id}.json"
    manifest = new_run_manifest(
        run_id=run_id,
        command=sys.argv,
        config_paths=[args.config],
        data_paths=[args.input],
        checkpoint_name=config["model_id"],
        checkpoint_path=config["checkpoint_path"],
        seeds=[int(config["seed"])],
        output_paths=[args.output, args.failure_log],
    )
    atomic_write_json(manifest_path, manifest)
    try:
        planned, existing, successes, failures = asyncio.run(run(args))
    except BaseException as exc:
        finish_run_manifest(manifest_path, manifest, status="failed", error=repr(exc))
        raise
    finish_run_manifest(
        manifest_path,
        manifest,
        status="success" if failures == 0 else "partial_failure",
        counts={"planned": planned, "existing": existing, "new_successes": successes, "new_failures": failures},
    )


if __name__ == "__main__":
    main()
