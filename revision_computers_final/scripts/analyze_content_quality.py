"""Aggregate fixed LLM content judgments and deterministic quality metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


SUMMARY_METRICS = [
    "content_coverage_rate",
    "added_key_information",
    "task_result_changed_by_addition",
    "factual_contradiction",
    "content_preservation_pass",
    "naturalness",
    "style_scalar_absolute_error",
    "repetition_rate",
    "distinct_2",
    "distinct_3",
    "truncated",
    "format_anomaly",
]


def bootstrap_summary(frame: pd.DataFrame, metrics: list[str], iterations: int, seed: int) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    records = []
    for model_id, model_frame in frame.groupby("model_id", sort=True):
        scenario = model_frame.groupby(["scenario_id", "style_axis"], as_index=False)[metrics].mean()
        arrays = [axis_frame[metrics].to_numpy(float) for _, axis_frame in scenario.groupby("style_axis", sort=True)]
        distribution = np.zeros((iterations, len(metrics)), dtype=float)
        total = 0
        for values in arrays:
            positions = random.integers(0, len(values), size=(iterations, len(values)))
            distribution += values[positions].sum(axis=1)
            total += len(values)
        distribution /= total
        observed = scenario[metrics].mean()
        for column_index, metric in enumerate(metrics):
            records.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "estimate": float(observed[metric]),
                    "ci_lower": float(np.nanpercentile(distribution[:, column_index], 2.5)),
                    "ci_upper": float(np.nanpercentile(distribution[:, column_index], 97.5)),
                    "scenario_count": scenario.scenario_id.nunique(),
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=REVISION_ROOT / "metrics/llm_evaluations.jsonl")
    parser.add_argument("--quality", type=Path, default=REVISION_ROOT / "metrics/text_quality_metrics.parquet")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    parsed = read_jsonl_tolerant(args.evaluation)
    if parsed.bad_lines:
        raise ValueError("LLM evaluation file has malformed rows")
    evaluations = pd.DataFrame(parsed.rows)
    if evaluations.run_id.nunique() != 33600:
        raise SystemExit(f"evaluator incomplete: {evaluations.run_id.nunique()} / 33600")
    quality = pd.read_parquet(args.quality)
    frame = evaluations.merge(quality, on=["run_id", "scenario_id", "style_axis", "model_id", "seed", "target_ratio"], validate="one_to_one")
    for column in (
        "added_key_information",
        "task_result_changed_by_addition",
        "factual_contradiction",
        "content_preservation_pass",
        "truncated",
        "format_anomaly",
    ):
        frame[column] = frame[column].astype(float)
    frame["style_scalar_ratio"] = frame["style_B_intensity"] / 100.0
    frame["style_scalar_absolute_error"] = (frame["style_scalar_ratio"] - frame["target_ratio"]).abs()
    frame.to_parquet(REVISION_ROOT / "metrics/content_quality_sample_level.parquet", index=False)

    bootstrap = bootstrap_summary(frame, SUMMARY_METRICS, args.iterations, args.seed)
    bootstrap.to_csv(REVISION_ROOT / "statistics/content_quality_bootstrap.csv", index=False)

    scalar_records = []
    cells = frame.groupby(["scenario_id", "style_axis", "model_id", "target_ratio"], as_index=False).agg(
        style_scalar_ratio=("style_scalar_ratio", "mean"),
        style_scalar_absolute_error=("style_scalar_absolute_error", "mean"),
    )
    for (model_id, style_axis), group in cells.groupby(["model_id", "style_axis"], sort=True):
        design = np.column_stack([np.ones(len(group)), group.target_ratio])
        intercept, slope = np.linalg.lstsq(design, group.style_scalar_ratio, rcond=None)[0]
        scalar_records.append(
            {
                "model_id": model_id,
                "style_axis": style_axis,
                "cell_count": len(group),
                "scalar_judge_mae": float(group.style_scalar_absolute_error.mean()),
                "spearman": float(spearmanr(group.target_ratio, group.style_scalar_ratio).statistic),
                "pearson": float(pearsonr(group.target_ratio, group.style_scalar_ratio).statistic),
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
            }
        )
    pd.DataFrame(scalar_records).to_csv(REVISION_ROOT / "metrics/llm_scalar_judge_summary.csv", index=False)

    semantic_paths = sorted((REVISION_ROOT / "metrics/semantic_similarity").glob("*.parquet"))
    semantic_summary = []
    for path in semantic_paths:
        semantic = pd.read_parquet(path)
        for (encoder_id, model_id, style_axis), group in semantic.groupby(["encoder_id", "model_id", "style_axis"], sort=True):
            semantic_summary.append(
                {
                    "encoder_id": encoder_id,
                    "model_id": model_id,
                    "style_axis": style_axis,
                    "mean_source_semantic_similarity": float(group.source_semantic_cosine_similarity.mean()),
                    "median_source_semantic_similarity": float(group.source_semantic_cosine_similarity.median()),
                    "sample_count": len(group),
                }
            )
    pd.DataFrame(semantic_summary).to_csv(REVISION_ROOT / "metrics/semantic_similarity_summary.csv", index=False)

    focus_axes = {"concision_detail", "humor_neutral", "empathy_clinical", "expertise_explain"}
    focus = frame[frame.style_axis.isin(focus_axes)].groupby(["style_axis", "model_id"], as_index=False).agg(
        content_coverage_rate=("content_coverage_rate", "mean"),
        content_pass_rate=("content_preservation_pass", "mean"),
        new_key_information_rate=("added_key_information", "mean"),
        contradiction_rate=("factual_contradiction", "mean"),
        mean_character_length=("character_length", "mean"),
        mean_naturalness=("naturalness", "mean"),
        scalar_judge_mae=("style_scalar_absolute_error", "mean"),
    )
    focus.to_csv(REVISION_ROOT / "metrics/content_style_coupling_focus_axes.csv", index=False)

    human_results = REVISION_ROOT / "human_eval/human_results.csv"
    human_status = "external_pending"
    validation_path = REVISION_ROOT / "statistics/automatic_checker_human_validation.csv"
    if human_results.exists():
        human = pd.read_csv(human_results)
        score_columns = [
            "style_B_intensity_0_100",
            "content_coverage_1_5",
            "naturalness_1_5",
            "adds_key_information_yes_no",
            "contradicts_source_yes_no",
            "unable_to_judge_yes_no",
        ]
        if score_columns and human[score_columns].notna().any().any():
            human_status = "ratings_present_run_analyze_human_eval"
    pd.DataFrame(
        [{"status": human_status, "required_sample_count": 320, "note": "Accuracy, precision, recall, F1, kappa, and confusion matrices require real ratings."}]
    ).to_csv(validation_path, index=False)
    print(json.dumps({"evaluations": len(frame), "bootstrap_rows": len(bootstrap), "semantic_encoders": len(semantic_paths), "human_validation": human_status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
