"""Combine metric fragments, add grouped-CV baselines, and bootstrap scenarios."""

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
    from baselines import grouped_isotonic_cv, grouped_length_only_cv  # type: ignore
    from common import REVISION_ROOT  # type: ignore
else:
    from .baselines import grouped_isotonic_cv, grouped_length_only_cv
    from .common import REVISION_ROOT


SCENARIO_METRICS = [
    "interior_calibration_error",
    "median_absolute_error",
    "normalized_off_axis_drift",
    "interpolation_distance_mean",
    "interpolation_distance_rms",
    "out_of_range_rate",
    "nearest_neighbor_mean_rank",
    "nearest_neighbor_mean_rank_normalized",
    "spearman_monotonicity",
    "pearson_correlation",
    "calibration_slope",
    "calibration_intercept",
    "constant_half_mae",
    "endpoint_distance_ratio_mae",
    "length_only_cv_mae",
    "isotonic_cv_mae",
    "endpoint_separation",
]


def safe_correlation(function: object, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(function(x, y).statistic)  # type: ignore[operator]


def add_cv_baselines(cells: pd.DataFrame, fold_count: int, seed: int) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, group in cells.groupby(["model_id", "encoder_id", "anchor_mode"], sort=True):
        group = group.copy()
        length_prediction, length_fold = grouped_length_only_cv(
            group["length_coordinate"],
            group["target_ratio"],
            group["scenario_id"],
            group["style_axis"],
            fold_count=fold_count,
            seed=seed,
        )
        isotonic_prediction, isotonic_fold = grouped_isotonic_cv(
            group["projection_raw"],
            group["target_ratio"],
            group["scenario_id"],
            group["style_axis"],
            fold_count=fold_count,
            seed=seed,
        )
        if not np.array_equal(length_fold, isotonic_fold):
            raise AssertionError("CV baseline fold assignments differ")
        group["cv_fold"] = length_fold
        group["length_only_cv_estimate"] = length_prediction
        group["isotonic_cv_estimate"] = isotonic_prediction
        output.append(group)
    return pd.concat(output, ignore_index=True)


def make_scenario_level(cells: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    keys = ["scenario_id", "style_axis", "model_id", "family_version", "architecture", "encoder_id", "anchor_mode"]
    for values, group in cells.groupby(keys, sort=True):
        interior = group[group["is_interior"]].sort_values("target_ratio")
        alpha = interior["target_ratio"].to_numpy(float)
        coordinate = interior["projection_raw"].to_numpy(float)
        design = np.column_stack([np.ones(len(alpha)), alpha])
        intercept, slope = np.linalg.lstsq(design, coordinate, rcond=None)[0]
        record = dict(zip(keys, values))
        record.update(
            {
                "interior_point_count": int(len(interior)),
                "interior_calibration_error": float(interior["projection_error_raw"].mean()),
                "median_absolute_error": float(interior["projection_error_raw"].median()),
                "normalized_off_axis_drift": float(interior["off_axis_drift_normalized"].mean()),
                "interpolation_distance_mean": float(interior["interpolation_distance_normalized"].mean()),
                "interpolation_distance_rms": float(np.sqrt(interior["interpolation_distance_squared"].mean())),
                "out_of_range_rate": float(interior["out_of_range"].mean()),
                "nearest_neighbor_mean_rank": float(interior["nearest_neighbor_rank"].mean()),
                "nearest_neighbor_mean_rank_normalized": float(interior["nearest_neighbor_rank_normalized"].mean()),
                "spearman_monotonicity": safe_correlation(spearmanr, alpha, coordinate),
                "pearson_correlation": safe_correlation(pearsonr, alpha, coordinate),
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
                "constant_half_mae": float(np.abs(interior["constant_half_estimate"] - alpha).mean()),
                "endpoint_distance_ratio_mae": float(np.abs(interior["endpoint_distance_ratio_estimate"] - alpha).mean()),
                "length_only_cv_mae": float(np.abs(interior["length_only_cv_estimate"] - alpha).mean()),
                "isotonic_cv_mae": float(np.abs(interior["isotonic_cv_estimate"] - alpha).mean()),
                "endpoint_separation": float(group["endpoint_separation"].mean()),
                "mean_character_length": float(interior["character_length"].mean()),
                "mean_output_tokens": float(interior["server_output_tokens"].mean()),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_group(
    group: pd.DataFrame, metrics: list[str], iterations: int, seed: int
) -> list[dict[str, object]]:
    random = np.random.default_rng(seed)
    arrays = {
        axis: axis_group.reset_index(drop=True)
        for axis, axis_group in group.groupby("style_axis", sort=True)
    }
    sampled_positions = {
        axis: random.integers(0, len(frame), size=(iterations, len(frame)))
        for axis, frame in arrays.items()
    }
    records: list[dict[str, object]] = []
    for metric in metrics:
        observed_values = group[metric].to_numpy(float)
        observed = float(np.nanmean(observed_values))
        samples: list[np.ndarray] = []
        for axis, frame in arrays.items():
            values = frame[metric].to_numpy(float)
            samples.append(values[sampled_positions[axis]])
        distribution = np.nanmean(np.concatenate(samples, axis=1), axis=1)
        records.append(
            {
                "metric": metric,
                "estimate": observed,
                "ci_lower": float(np.nanpercentile(distribution, 2.5)),
                "ci_upper": float(np.nanpercentile(distribution, 97.5)),
                "bootstrap_iterations": iterations,
                "bootstrap_unit": "scenario_id",
                "stratified_by": "style_axis",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment-root", type=Path, default=REVISION_ROOT / "metrics/fragments")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--sample-out", type=Path, default=REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    parser.add_argument("--scenario-out", type=Path, default=REVISION_ROOT / "metrics/scenario_level_metrics.parquet")
    parser.add_argument("--model-out", type=Path, default=REVISION_ROOT / "metrics/model_level_summary.csv")
    parser.add_argument("--bootstrap-out", type=Path, default=REVISION_ROOT / "statistics/bootstrap_results.csv")
    args = parser.parse_args()
    sample_paths = sorted(args.fragment_root.glob("*/*.sample.parquet"))
    if not sample_paths:
        raise SystemExit("no sample metric fragments found")
    samples = pd.concat([pd.read_parquet(path) for path in sample_paths], ignore_index=True)
    unique_pairs = samples[["model_id", "encoder_id"]].drop_duplicates()
    if len(unique_pairs) != 15 and not args.allow_incomplete:
        raise SystemExit(f"found {len(unique_pairs)} model/encoder pairs; expected 15")
    samples.to_parquet(args.sample_out, index=False)

    cell_keys = [
        "scenario_id", "style_axis", "model_id", "family_version", "architecture",
        "encoder_id", "anchor_mode", "target_ratio", "is_interior",
    ]
    numeric_columns = [
        "projection_raw", "projection_clipped", "projection_error_raw",
        "projection_error_clipped", "off_axis_drift_normalized",
        "interpolation_distance_normalized", "out_of_range", "nearest_neighbor_rank",
        "nearest_neighbor_rank_normalized", "endpoint_separation",
        "constant_half_estimate", "endpoint_distance_ratio_estimate",
        "character_length", "chinese_character_count", "server_output_tokens",
        "length_coordinate",
    ]
    samples["interpolation_distance_squared"] = np.square(samples["interpolation_distance_normalized"])
    numeric_columns.append("interpolation_distance_squared")
    cells = samples.groupby(cell_keys, as_index=False)[numeric_columns].mean()
    cells = add_cv_baselines(cells, fold_count=5, seed=args.seed)
    scenario = make_scenario_level(cells)
    args.scenario_out.parent.mkdir(parents=True, exist_ok=True)
    scenario.to_parquet(args.scenario_out, index=False)

    bootstrap_records: list[dict[str, object]] = []
    group_keys = ["model_id", "family_version", "architecture", "encoder_id", "anchor_mode"]
    for values, group in scenario.groupby(group_keys, sort=True):
        records = bootstrap_group(group, SCENARIO_METRICS, args.bootstrap_iterations, args.seed)
        for record in records:
            record.update(dict(zip(group_keys, values)))
            record["scenario_count"] = int(group["scenario_id"].nunique())
            bootstrap_records.append(record)
    bootstrap = pd.DataFrame(bootstrap_records)
    args.bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(args.bootstrap_out, index=False)
    index = group_keys
    model_summary = bootstrap.pivot(index=index, columns="metric", values=["estimate", "ci_lower", "ci_upper"])
    model_summary.columns = [f"{metric}_{statistic}" for statistic, metric in model_summary.columns]
    model_summary.reset_index().to_csv(args.model_out, index=False)
    print(
        json.dumps(
            {
                "sample_rows": len(samples),
                "scenario_rows": len(scenario),
                "model_encoder_pairs": len(unique_pairs),
                "bootstrap_rows": len(bootstrap),
            }
        )
    )


if __name__ == "__main__":
    main()
