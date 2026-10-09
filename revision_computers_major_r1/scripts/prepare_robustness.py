from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import yaml

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_text, stable_id, utc_now
from prompts import render_qwen_prompt


def key(model: str, scenario: str, alpha: float, seed: int, prompt: str, temperature: float) -> tuple[Any, ...]:
    return model, scenario, float(alpha), int(seed), prompt, float(temperature)


def normalize_reused(row: dict[str, Any], *, condition_id: str, source: str, experiment_id: str) -> dict[str, Any]:
    robustness_id = stable_id(
        "major_r1_robust",
        experiment_id,
        row["model_id"],
        row["scenario_id"],
        row["target_ratio"],
        row["seed"],
        row["prompt_template_id"],
        row["temperature"],
    )
    return {
        **row,
        "source_run_id": row["run_id"],
        "run_id": robustness_id,
        "robustness_condition_id": condition_id,
        "robustness_source": source,
        "prompt_sha256": sha256_text(row["prompt"]),
        "major_r1_imported_at": utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    legacy = Path(config["legacy_root"])
    scenarios = read_jsonl(config["scenario_file"])
    by_scenario = {row["scenario_id"]: row for row in scenarios}
    main_rows = read_jsonl(legacy / "data/qwen_family_outputs.jsonl")
    old_robustness = read_jsonl(legacy / "data/robustness_outputs.jsonl")
    main_by_key = {
        key(row["model_id"], row["scenario_id"], row["target_ratio"], row["seed"], row["prompt_template_id"], row["temperature"]): row
        for row in main_rows
        if row["target_ratio"] not in (0.0, 1.0)
    }
    old_by_key = {
        key(row["model_id"], row["scenario_id"], row["target_ratio"], row["seed"], row["prompt_template_id"], row["temperature"]): row
        for row in old_robustness
    }
    summaries: dict[str, Any] = {}
    for model in config["robustness"]["models"]:
        path = args.output_root / "data/fragments/robustness" / f"{model}.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        existing_by_id = {row["run_id"]: row for row in existing}
        reused: list[dict[str, Any]] = []
        expected_ids: set[str] = set()
        prompt_mismatches: list[dict[str, Any]] = []
        source_counts = collections.Counter()
        for scenario in scenarios:
            for alpha in config["robustness"]["interior_ratios"]:
                for seed in config["robustness"]["seeds"]:
                    for condition in config["robustness"]["conditions"]:
                        prompt_id = condition["prompt_template_id"]
                        temperature = float(condition["temperature"])
                        condition_id = condition["condition_id"]
                        lookup = key(model, scenario["scenario_id"], alpha, seed, prompt_id, temperature)
                        expected_prompt = render_qwen_prompt(scenario, float(alpha), prompt_id)
                        row = main_by_key.get(lookup) if condition_id == "p1_high" else old_by_key.get(lookup)
                        run_id = stable_id("major_r1_robust", config["experiment_id"], *lookup)
                        expected_ids.add(run_id)
                        if row is None:
                            continue
                        if row.get("prompt") != expected_prompt:
                            prompt_mismatches.append({"run_id": row.get("run_id"), "condition_id": condition_id})
                            continue
                        source = "legacy_main_reuse" if condition_id == "p1_high" else "legacy_robustness_reuse"
                        reused.append(normalize_reused(row, condition_id=condition_id, source=source, experiment_id=config["experiment_id"]))
                        source_counts[source] += 1
        if prompt_mismatches:
            raise ValueError(f"{model}: legacy prompt mismatches: {len(prompt_mismatches)}")
        merged = dict(existing_by_id)
        for row in reused:
            merged.setdefault(row["run_id"], row)
        unexpected = set(merged) - expected_ids
        if unexpected:
            raise ValueError(f"{model}: existing output contains {len(unexpected)} unexpected keys")
        ordered = sorted(merged.values(), key=lambda row: (row["style_axis"], row["scenario_id"], row["prompt_template_id"], float(row["temperature"]), float(row["target_ratio"]), int(row["seed"])))
        atomic_write_jsonl(path, ordered)
        summaries[model] = {
            "expected": len(expected_ids),
            "reused_main": source_counts["legacy_main_reuse"],
            "reused_old_robustness": source_counts["legacy_robustness_reuse"],
            "existing_or_reused": len(ordered),
            "missing": len(expected_ids) - len(ordered),
        }
    atomic_write_json(args.output_root / "manifests/robustness_prepare.json", {"generated_at": utc_now(), "models": summaries})
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
