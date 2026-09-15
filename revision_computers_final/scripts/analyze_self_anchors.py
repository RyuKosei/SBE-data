"""Describe self-generated endpoint quality and local-coordinate differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=REVISION_ROOT / "metrics/llm_evaluations.jsonl")
    args = parser.parse_args()
    sample = pd.read_parquet(REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    endpoints = sample[~sample.is_interior].copy()
    evaluation = pd.DataFrame(read_jsonl_tolerant(args.evaluation).rows)
    if evaluation.run_id.nunique() != 33600:
        raise SystemExit(f"evaluator incomplete: {evaluation.run_id.nunique()} / 33600")
    endpoints = endpoints.merge(
        evaluation[["run_id", "content_coverage_rate", "content_preservation_pass", "style_B_intensity", "factual_contradiction", "task_result_changed_by_addition"]],
        on="run_id",
        validate="many_to_one",
    )
    pair_keys = ["scenario_id", "style_axis", "model_id", "encoder_id", "seed"]
    records = []
    for values, group in endpoints.groupby(pair_keys, sort=True):
        a = group[np.isclose(group.target_ratio, 0)].iloc[0]
        b = group[np.isclose(group.target_ratio, 1)].iloc[0]
        records.append(
            {
                **dict(zip(pair_keys, values)),
                "endpoint_separation": a.endpoint_separation,
                "endpoint_length_a": a.character_length,
                "endpoint_length_b": b.character_length,
                "endpoint_length_difference_b_minus_a": b.character_length - a.character_length,
                "endpoint_length_absolute_difference": abs(b.character_length - a.character_length),
                "endpoint_content_coverage_mean": (a.content_coverage_rate + b.content_coverage_rate) / 2,
                "endpoint_content_pass_rate": (float(a.content_preservation_pass) + float(b.content_preservation_pass)) / 2,
                "endpoint_contradiction_any": bool(a.factual_contradiction or b.factual_contradiction),
                "endpoint_task_changing_addition_any": bool(a.task_result_changed_by_addition or b.task_result_changed_by_addition),
                "judge_intensity_a": a.style_B_intensity,
                "judge_intensity_b": b.style_B_intensity,
                "judge_endpoint_intensity_separation": b.style_B_intensity - a.style_B_intensity,
                "judge_endpoint_absolute_error_mean": (abs(a.style_B_intensity / 100) + abs(b.style_B_intensity / 100 - 1)) / 2,
            }
        )
    pairs = pd.DataFrame(records)
    pairs.to_parquet(REVISION_ROOT / "metrics/self_anchor_pairs.parquet", index=False)
    numeric = [
        "endpoint_separation", "endpoint_length_difference_b_minus_a", "endpoint_length_absolute_difference",
        "endpoint_content_coverage_mean", "endpoint_content_pass_rate", "endpoint_contradiction_any",
        "endpoint_task_changing_addition_any", "judge_intensity_a", "judge_intensity_b",
        "judge_endpoint_intensity_separation", "judge_endpoint_absolute_error_mean",
    ]
    summary_records = []
    for (model, encoder), group in pairs.groupby(["model_id", "encoder_id"], sort=True):
        for metric in numeric:
            values = group[metric].astype(float)
            summary_records.append(
                {
                    "model_id": model,
                    "encoder_id": encoder,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": values.median(),
                    "p10": values.quantile(0.10),
                    "p90": values.quantile(0.90),
                    "pair_count": len(group),
                }
            )
    summary = pd.DataFrame(summary_records)
    summary.to_csv(REVISION_ROOT / "tables/table03_self_anchor_diagnostics.csv", index=False)
    print(json.dumps({"endpoint_pairs": len(pairs), "summary_rows": len(summary)}))


if __name__ == "__main__":
    main()
