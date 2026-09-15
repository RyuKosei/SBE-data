"""Validate per-model fragments and atomically build the required main JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant


REQUIRED_FIELDS = {
    "scenario_id",
    "style_axis",
    "style_A",
    "style_B",
    "target_ratio",
    "model_id",
    "family_version",
    "architecture",
    "seed",
    "prompt_template_id",
    "temperature",
    "top_p",
    "max_new_tokens",
    "endpoint_mode",
    "generated_text",
    "input_text",
    "timestamp",
    "git_commit",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REVISION_ROOT / "config/main_experiment.yaml")
    parser.add_argument("--fragment-dir", type=Path, default=REVISION_ROOT / "data/fragments/main")
    parser.add_argument("--out", type=Path, default=REVISION_ROOT / "data/qwen_family_outputs.jsonl")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(config["model_registry"]).open("r", encoding="utf-8", newline="") as handle:
        registry = {row["model_id"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for model_id in config["models"]:
        if registry[model_id]["included"].lower() != "true":
            continue
        path = args.fragment_dir / f"{model_id}.jsonl"
        if not path.exists():
            problems.append(f"missing fragment: {path}")
            continue
        result = read_jsonl_tolerant(path)
        if result.bad_lines:
            problems.append(f"{path}: {len(result.bad_lines)} malformed rows")
        rows.extend(result.rows)

    keys = [
        (row.get("model_id"), row.get("scenario_id"), float(row.get("target_ratio", -1)), row.get("seed"), row.get("prompt_template_id"))
        for row in rows
    ]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        problems.append(f"duplicate unique keys: {len(duplicates)}")
    missing_fields = sum(not REQUIRED_FIELDS.issubset(row) for row in rows)
    empty_outputs = sum(not str(row.get("generated_text", "")).strip() for row in rows)
    if missing_fields:
        problems.append(f"rows missing required fields: {missing_fields}")
    if empty_outputs:
        problems.append(f"empty generated_text rows: {empty_outputs}")
    expected = len(config["models"]) * 320 * len(config["target_ratios"]) * len(config["seeds"])
    if len(rows) != expected:
        problems.append(f"row count {len(rows)} != expected {expected}")
    if problems and not args.allow_incomplete:
        raise SystemExit("; ".join(problems))
    rows.sort(key=lambda row: (row["model_id"], row["style_axis"], row["scenario_id"], row["target_ratio"], row["seed"]))
    atomic_write_jsonl(args.out, rows)
    print(json.dumps({"rows": len(rows), "expected": expected, "problems": problems, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
