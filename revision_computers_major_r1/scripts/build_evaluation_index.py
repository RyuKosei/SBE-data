from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import yaml

from common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_text, stable_id, utc_now


def item(row: dict[str, Any], source_type: str, identity: str) -> dict[str, Any]:
    return {
        "evaluation_item_id": stable_id("major_r1_eval_item", source_type, identity),
        "source_type": source_type,
        "source_identity": identity,
        "run_id": row["run_id"],
        "original_source_run_id": row.get("source_run_id") or row["run_id"],
        "scenario_id": row["scenario_id"],
        "style_axis": row["style_axis"],
        "model_id": row["model_id"],
        "target_ratio": float(row["target_ratio"]),
        "seed": row.get("seed"),
        "replicate_id": row.get("replicate_id"),
        "prompt_template_id": row["prompt_template_id"],
        "temperature": row.get("temperature", row.get("temperature_requested")),
        "robustness_condition_id": row.get("robustness_condition_id"),
        "input_text": row["input_text"],
        "content_checks": row["content_checks"],
        "style_A": row["style_A"],
        "style_B": row["style_B"],
        "generated_text": row["generated_text"],
        "generated_text_sha256": sha256_text(row["generated_text"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    legacy = Path(config["legacy_root"])
    rows: list[dict[str, Any]] = []
    external_failures_excluded = 0
    for row in read_jsonl(legacy / "data/qwen_family_outputs.jsonl"):
        rows.append(item(row, "main_qwen", row["run_id"]))
    external_path = args.output_root / "data/external_main_outputs.jsonl"
    if external_path.exists():
        for row in read_jsonl(external_path):
            if row.get("status") in {"ok", "truncated"} and row.get("generated_text"):
                rows.append(item(row, "main_external", row["run_id"]))
            else:
                external_failures_excluded += 1
    for model in config["robustness"]["models"]:
        path = args.output_root / "data/fragments/robustness" / f"{model}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            if row.get("robustness_condition_id") == "p1_high":
                continue
            if row.get("status") not in (None, "ok") or not row.get("generated_text"):
                continue
            rows.append(item(row, "robustness_qwen", row["run_id"]))
    ids = [row["evaluation_item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evaluation_item_id")
    rows.sort(key=lambda row: row["evaluation_item_id"])
    atomic_write_jsonl(args.output_root / "data/evaluation_index.jsonl", rows)
    counts = collections.Counter(row["source_type"] for row in rows)
    robustness_failures_excluded = 0
    condition_path = args.output_root / "data/robustness_condition_records.jsonl"
    if condition_path.exists():
        robustness_failures_excluded = sum(
            row.get("robustness_condition_id") != "p1_high" and row.get("status") != "ok"
            for row in read_jsonl(condition_path)
        )
    explicitly_failed_and_excluded = external_failures_excluded + robustness_failures_excluded
    status = {
        "generated_at": utc_now(),
        "total": len(rows),
        "counts": dict(counts),
        "expected_complete_total": 112320,
        "usable_index_strictly_complete": len(rows) == 112320,
        "explicitly_failed_and_excluded": explicitly_failed_and_excluded,
        "external_failures_excluded": external_failures_excluded,
        "robustness_failures_excluded": robustness_failures_excluded,
        "all_planned_identities_accounted_for": len(rows) + explicitly_failed_and_excluded == 112320,
        "complete": len(rows) == 112320,
    }
    atomic_write_json(args.output_root / "manifests/evaluation_index_status.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
