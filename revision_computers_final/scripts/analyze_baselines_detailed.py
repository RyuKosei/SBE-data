"""Compare all ratio estimators on the same scenario-level statistical unit."""

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
    from aggregate_metrics import add_cv_baselines  # type: ignore
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .aggregate_metrics import add_cv_baselines
    from .common import REVISION_ROOT, read_jsonl_tolerant


STATISTICS = ["mae", "median_absolute_error", "spearman", "pearson", "calibration_slope", "calibration_intercept"]


def scenario_statistics(
    frame: pd.DataFrame, estimate_column: str, absolute_error_column: str | None = None
) -> dict[str, float]:
    alpha = frame.target_ratio.to_numpy(float)
    estimate = frame[estimate_column].to_numpy(float)
    error = (
        frame[absolute_error_column].to_numpy(float)
        if absolute_error_column is not None
        else np.abs(estimate - alpha)
    )
    if np.std(estimate) <= 1e-12:
        spearman = pearson = np.nan
    else:
        spearman = float(spearmanr(alpha, estimate).statistic)
        pearson = float(pearsonr(alpha, estimate).statistic)
    design = np.column_stack([np.ones(len(alpha)), alpha])
    intercept, slope = np.linalg.lstsq(design, estimate, rcond=None)[0]
    return {
        "mae": float(error.mean()),
        "median_absolute_error": float(np.median(error)),
        "spearman": spearman,
        "pearson": pearson,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


def bootstrap(group: pd.DataFrame, metric: str, iterations: int, random: np.random.Generator) -> tuple[float, float]:
    arrays = [axis[metric].to_numpy(float) for _, axis in group.groupby("style_axis", sort=True)]
    if all(np.isnan(values).all() for values in arrays):
        return np.nan, np.nan
    distribution = np.zeros(iterations)
    counts = np.zeros(iterations)
    for values in arrays:
        positions = random.integers(0, len(values), size=(iterations, len(values)))
        selected = values[positions]
        distribution += np.nansum(selected, axis=1)
        counts += np.isfinite(selected).sum(axis=1)
    distribution = np.divide(distribution, counts, out=np.full(iterations, np.nan), where=counts > 0)
    return float(np.nanpercentile(distribution, 2.5)), float(np.nanpercentile(distribution, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    sample = pd.read_parquet(REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    # Fit grouped-CV baselines without leaking a scenario across folds. For each
    # target-ratio cell, average coordinates and absolute errors separately over
    # seeds, matching the primary aggregation used by aggregate_metrics.py.
    cells = add_cv_baselines(sample.copy(), fold_count=5, seed=args.seed)
    cells = cells[cells.is_interior]
    methods = {
        "raw_projection": "projection_raw",
        "endpoint_distance_ratio": "endpoint_distance_ratio_estimate",
        "length_only_grouped_cv": "length_only_cv_estimate",
        "isotonic_grouped_cv": "isotonic_cv_estimate",
        "constant_0.5": "constant_half_estimate",
    }
    records = []
    group_keys = ["scenario_id", "style_axis", "model_id", "encoder_id"]
    cell_keys = group_keys + ["target_ratio"]
    for method, estimate in methods.items():
        method_rows = cells[cell_keys + [estimate]].copy()
        method_rows["absolute_error"] = (method_rows[estimate] - method_rows.target_ratio).abs()
        method_cells = method_rows.groupby(cell_keys, as_index=False).agg(
            estimate=(estimate, "mean"),
            absolute_error=("absolute_error", "mean"),
        )
        for values, group in method_cells.groupby(group_keys, sort=True):
            records.append(
                {
                    **dict(zip(group_keys, values)),
                    "method": method,
                    **scenario_statistics(group, "estimate", "absolute_error"),
                }
            )

    evaluations = pd.DataFrame(read_jsonl_tolerant(REVISION_ROOT / "metrics/llm_evaluations.jsonl").rows)
    evaluations = evaluations[(evaluations.target_ratio > 0) & (evaluations.target_ratio < 1)].copy()
    evaluations["llm_scalar_estimate"] = evaluations.style_B_intensity / 100
    evaluations["absolute_error"] = (evaluations.llm_scalar_estimate - evaluations.target_ratio).abs()
    llm_cells = evaluations.groupby(
        ["scenario_id", "style_axis", "model_id", "target_ratio"], as_index=False
    ).agg(estimate=("llm_scalar_estimate", "mean"), absolute_error=("absolute_error", "mean"))
    for values, group in llm_cells.groupby(["scenario_id", "style_axis", "model_id"], sort=True):
        records.append(
            {
                "scenario_id": values[0],
                "style_axis": values[1],
                "model_id": values[2],
                "encoder_id": "fixed_qwen36_35b_scalar_judge",
                "method": "llm_scalar_judge",
                **scenario_statistics(group, "estimate", "absolute_error"),
            }
        )
    scenario = pd.DataFrame(records)
    scenario.to_parquet(REVISION_ROOT / "metrics/baseline_scenario_level.parquet", index=False)
    random = np.random.default_rng(args.seed)
    summary_records = []
    for values, group in scenario.groupby(["model_id", "encoder_id", "method"], sort=True):
        for statistic in STATISTICS:
            lower, upper = bootstrap(group, statistic, args.iterations, random)
            method = values[2]
            if method == "llm_scalar_judge":
                incremental = "one fixed-judge request per generated text"
                mean_latency = float(evaluations.latency_seconds.mean())
                total_output_tokens = int(evaluations.output_tokens.sum())
            elif method in {"raw_projection", "endpoint_distance_ratio", "isotonic_grouped_cv"}:
                incremental = "sentence-encoder inference; no LLM request"
                mean_latency = np.nan
                total_output_tokens = 0
            else:
                incremental = "deterministic local computation"
                mean_latency = np.nan
                total_output_tokens = 0
            summary_records.append(
                {
                    "model_id": values[0],
                    "encoder_id": values[1],
                    "method": method,
                    "statistic": statistic,
                    "estimate": float(group[statistic].mean()),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "scenario_count": group.scenario_id.nunique(),
                    "incremental_cost": incremental,
                    "mean_llm_judge_latency_seconds": mean_latency,
                    "total_llm_judge_output_tokens_all_models": total_output_tokens,
                    "human_style_correlation": np.nan,
                    "human_style_mae": np.nan,
                    "human_validation_status": "external_pending",
                }
            )
    summary = pd.DataFrame(summary_records)
    summary.to_csv(REVISION_ROOT / "tables/table06_baselines_detailed.csv", index=False)
    print(json.dumps({"scenario_rows": len(scenario), "summary_rows": len(summary), "methods": scenario.method.nunique()}))


if __name__ == "__main__":
    main()
