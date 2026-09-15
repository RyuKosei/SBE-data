"""Combine the reused P1/T0.7 cell and three newly generated robustness cells."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, atomic_write_jsonl, read_jsonl_tolerant


REPRESENTATIVE_MODELS = {"qwen35_4b", "qwen35_9b", "qwen36_27b", "qwen36_35b_a3b"}
INTERIOR = {round(value, 10) for value in (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)}


def main() -> None:
    with (REVISION_ROOT / "config/robustness_scenarios.csv").open(encoding="utf-8", newline="") as handle:
        scenario_ids = {row["scenario_id"] for row in csv.DictReader(handle)}
    main_rows = read_jsonl_tolerant(REVISION_ROOT / "data/qwen_family_outputs.jsonl").rows
    reused = []
    for row in main_rows:
        if (
            row["scenario_id"] in scenario_ids
            and row["model_id"] in REPRESENTATIVE_MODELS
            and round(float(row["target_ratio"]), 10) in INTERIOR
        ):
            reused.append(
                {
                    **row,
                    "robustness_condition_id": "computers_explicit_ratio_v1_T0.7",
                    "robustness_source": "reused_main",
                }
            )
    generated = []
    for path in sorted((REVISION_ROOT / "data/fragments/robustness").glob("*.jsonl")):
        generated.extend({**row, "robustness_source": "new_generation"} for row in read_jsonl_tolerant(path).rows)
    rows = reused + generated
    keys = {(row["run_id"], row["robustness_condition_id"]) for row in rows}
    counts = Counter((row["model_id"], row["robustness_condition_id"]) for row in rows)
    problems = []
    if len(reused) != 2400:
        problems.append(f"reused={len(reused)} expected=2400")
    if len(generated) != 7200:
        problems.append(f"generated={len(generated)} expected=7200")
    if len(keys) != len(rows):
        problems.append("duplicate run/condition keys")
    for model in sorted(REPRESENTATIVE_MODELS):
        for condition in (
            "computers_explicit_ratio_v1_T0.7",
            "computers_explicit_ratio_v1_T0.2",
            "computers_explicit_ratio_v2_T0.7",
            "computers_explicit_ratio_v2_T0.2",
        ):
            if counts[(model, condition)] != 600:
                problems.append(f"{model}/{condition}={counts[(model, condition)]} expected=600")
    if problems:
        raise SystemExit("; ".join(problems))
    output = REVISION_ROOT / "data/robustness_outputs.jsonl"
    atomic_write_jsonl(output, sorted(rows, key=lambda row: (row["model_id"], row["robustness_condition_id"], row["scenario_id"], row["target_ratio"], row["seed"])))
    print(json.dumps({"rows": len(rows), "reused": len(reused), "new": len(generated), "conditions": 4, "problems": problems}))


if __name__ == "__main__":
    main()
