from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import yaml

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, stable_id, utc_now


def robustness_expected(config: dict) -> dict[str, dict]:
    scenarios = read_jsonl(config["scenario_file"])
    output: dict[str, dict] = {}
    for model in config["robustness"]["models"]:
        for scenario in scenarios:
            for alpha in config["robustness"]["interior_ratios"]:
                for seed in config["robustness"]["seeds"]:
                    for condition in config["robustness"]["conditions"]:
                        run_id = stable_id("major_r1_robust", config["experiment_id"], model, scenario["scenario_id"], alpha, seed, condition["prompt_template_id"], float(condition["temperature"]))
                        output[run_id] = {
                            "run_id": run_id,
                            "model_id": model,
                            "scenario_id": scenario["scenario_id"],
                            "style_axis": scenario["axis_id"],
                            "target_ratio": float(alpha),
                            "seed": int(seed),
                            "prompt_template_id": condition["prompt_template_id"],
                            "temperature": float(condition["temperature"]),
                            "robustness_condition_id": condition["condition_id"],
                        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_rows = []
    for model in config["robustness"]["models"]:
        all_rows.extend(read_jsonl(args.output_root / "data/fragments/robustness" / f"{model}.jsonl"))
    ids = [row["run_id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate robustness run IDs: {len(ids)-len(set(ids))}")
    expected_rows = robustness_expected(config)
    expected = set(expected_rows)
    unexpected = set(ids) - expected
    if unexpected:
        raise ValueError(f"unexpected robustness keys: {len(unexpected)}")
    all_rows.sort(key=lambda row: (row["model_id"], row["style_axis"], row["scenario_id"], row["robustness_condition_id"], float(row["target_ratio"]), int(row["seed"])))
    atomic_write_jsonl(args.output_root / "data/robustness_outputs.jsonl", all_rows)
    failures = []
    for path in sorted((args.output_root / "logs").glob("robustness_failures_*.jsonl")):
        failures.extend(read_jsonl(path))
    failure_by_id = collections.Counter(row["run_id"] for row in failures)
    latest_failure = {row["run_id"]: row for row in failures}
    missing = sorted(expected - set(ids))
    success_by_id = {row["run_id"]: row for row in all_rows}
    condition_records = []
    for run_id, key in expected_rows.items():
        if run_id in success_by_id:
            row = success_by_id[run_id]
            condition_records.append({
                **key,
                "status": "ok",
                "robustness_source": row.get("robustness_source"),
                "generated_text_sha256": row.get("generated_text_sha256"),
                "failure_reason": None,
            })
        elif run_id in latest_failure:
            row = latest_failure[run_id]
            condition_records.append({
                **key,
                "status": row.get("status", "generation_failure"),
                "robustness_source": "major_r1_generation",
                "generated_text_sha256": None,
                "failure_reason": row.get("error") or row.get("status"),
            })
        else:
            condition_records.append({
                **key,
                "status": "missing_unattempted",
                "robustness_source": None,
                "generated_text_sha256": None,
                "failure_reason": "no successful output or recorded failure",
            })
    condition_records.sort(key=lambda row: (row["model_id"], row["style_axis"], row["scenario_id"], row["robustness_condition_id"], row["target_ratio"], row["seed"]))
    atomic_write_jsonl(args.output_root / "data/robustness_condition_records.jsonl", condition_records)
    status = {
        "generated_at": utc_now(),
        "expected_records": len(expected),
        "valid_output_records": len(all_rows),
        "missing_records": len(missing),
        "missing_with_recorded_failure": sum(run_id in failure_by_id for run_id in missing),
        "condition_status_counts": dict(collections.Counter(row["status"] for row in condition_records)),
        "duplicate_ids": 0,
        "by_model": dict(collections.Counter(row["model_id"] for row in all_rows)),
        "by_condition": dict(collections.Counter(row["robustness_condition_id"] for row in all_rows)),
        "by_source": dict(collections.Counter(row["robustness_source"] for row in all_rows)),
        "missing_preview": missing[:100],
    }
    atomic_write_json(args.output_root / "manifests/robustness_coverage.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
