"""Generate only the three non-main prompt/temperature robustness cells."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_json, finish_run_manifest, new_run_manifest, read_jsonl_tolerant  # type: ignore
    from run_main_generation import append_jsonl, generate_one, load_registry, render_prompt, stable_run_id  # type: ignore
else:
    from .common import REVISION_ROOT, atomic_write_json, finish_run_manifest, new_run_manifest, read_jsonl_tolerant
    from .run_main_generation import append_jsonl, generate_one, load_registry, render_prompt, stable_run_id


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def selected_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["scenario_id"] for row in csv.DictReader(handle)}


def build_jobs(
    robust: dict[str, Any], scenarios: list[dict[str, Any]], model: dict[str, str]
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    generated_conditions = [condition for condition in robust["conditions"] if condition["source"] == "generate"]
    for condition in generated_conditions:
        template = condition["prompt_template_id"]
        temperature = float(condition["temperature"])
        for scenario in scenarios:
            for alpha in robust["interior_ratios"]:
                for seed in robust["generation_seeds"]:
                    condition_id = f"{template}_T{temperature:g}"
                    jobs.append(
                        {
                            "run_id": stable_run_id(robust["experiment_id"], model["model_id"], scenario["scenario_id"], alpha, seed, condition_id),
                            "scenario_id": scenario["scenario_id"],
                            "style_axis": scenario["axis_id"],
                            "style_A": scenario["style_a"],
                            "style_B": scenario["style_b"],
                            "target_ratio": float(alpha),
                            "model_id": model["model_id"],
                            "family_version": model["family_version"],
                            "architecture": model["architecture"],
                            "seed": int(seed),
                            "prompt_template_id": template,
                            "robustness_condition_id": condition_id,
                            "temperature": temperature,
                            "top_p": float(robust["generation"]["top_p"]),
                            "max_new_tokens": int(robust["generation"]["max_new_tokens"]),
                            "prompt": render_prompt(scenario, float(alpha), template),
                            "input_text": scenario["content"],
                            "content_checks": scenario.get("content_checks", []),
                            "length_range_zh_chars": scenario.get("length_range_zh_chars"),
                            "endpoint_mode": robust["analysis_anchor_mode"],
                        }
                    )
    return jobs


async def run(args: argparse.Namespace) -> tuple[int, int, int]:
    from openai import AsyncOpenAI

    robust = load_yaml(args.config)
    main = load_yaml(args.main_config)
    registry = load_registry(Path(main["model_registry"]))
    model = registry[args.model_id]
    wanted = selected_ids(REVISION_ROOT / "config/robustness_scenarios.csv")
    source = read_jsonl_tolerant(main["scenario_file"])
    scenarios = [row for row in source.rows if row["scenario_id"] in wanted]
    if len(scenarios) != 40:
        raise ValueError("robustness selection must contain 40 scenarios")
    jobs = build_jobs(robust, scenarios, model)
    existing_source = read_jsonl_tolerant(args.output) if args.output.exists() else None
    existing = {row["run_id"] for row in existing_source.rows} if existing_source else set()
    todo = [job for job in jobs if job["run_id"] not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    generation = {
        **robust["generation"],
        "temperature": 0.0,
        "request_timeout_seconds": 180,
        "retry_attempts": 3,
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    successes = failures = 0
    chunk_size = args.concurrency * 4
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        results = await asyncio.gather(
            *[generate_one(client, job, args.served_model_name, generation, semaphore, False) for job in chunk]
        )
        good = [row for row, _ in results if row is not None]
        bad = [failure for _, failure in results if failure is not None]
        append_jsonl(args.output, good)
        append_jsonl(args.failure_log, bad)
        successes += len(good)
        failures += len(bad)
        print({"model": args.model_id, "planned": len(jobs), "attempted": min(start + len(chunk), len(todo)), "successes": successes, "failures": failures}, flush=True)
    return len(jobs), successes, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REVISION_ROOT / "config/robustness_experiment.yaml")
    parser.add_argument("--main-config", type=Path, default=REVISION_ROOT / "config/main_experiment.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failure-log", type=Path)
    args = parser.parse_args()
    args.output = args.output or REVISION_ROOT / "data/fragments/robustness" / f"{args.model_id}.jsonl"
    args.failure_log = args.failure_log or REVISION_ROOT / "logs" / f"robustness_failures_{args.model_id}.jsonl"
    robust = load_yaml(args.config)
    main_config = load_yaml(args.main_config)
    model = load_registry(Path(main_config["model_registry"]))[args.model_id]
    run_id = f"robustness_{args.model_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest_path = REVISION_ROOT / "manifests" / f"run_manifest_{run_id}.json"
    manifest = new_run_manifest(
        run_id=run_id,
        command=sys.argv,
        config_paths=[args.config, args.main_config, REVISION_ROOT / "config/robustness_scenarios.csv"],
        data_paths=[main_config["scenario_file"]],
        checkpoint_name=model["source"],
        checkpoint_path=model["checkpoint_path"],
        seeds=robust["generation_seeds"],
        output_paths=[args.output, args.failure_log],
    )
    atomic_write_json(manifest_path, manifest)
    try:
        planned, succeeded, failed = asyncio.run(run(args))
    except BaseException as exc:
        finish_run_manifest(manifest_path, manifest, status="failed", error=repr(exc))
        raise
    finish_run_manifest(
        manifest_path,
        manifest,
        status="success" if failed == 0 else "partial_failure",
        counts={"planned": planned, "new_successes": succeeded, "new_failures": failed},
    )


if __name__ == "__main__":
    main()
