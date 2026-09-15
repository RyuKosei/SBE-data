"""Run the fixed endpoint, clipping, encoder, trajectory, axis, and separation ablations."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT  # type: ignore
else:
    from .common import REVISION_ROOT


def bootstrap_mean(values_by_axis: list[np.ndarray], iterations: int, random: np.random.Generator) -> tuple[float, float]:
    total = sum(len(values) for values in values_by_axis)
    distribution = np.zeros(iterations)
    for values in values_by_axis:
        positions = random.integers(0, len(values), size=(iterations, len(values)))
        distribution += values[positions].sum(axis=1)
    distribution /= total
    return float(np.nanpercentile(distribution, 2.5)), float(np.nanpercentile(distribution, 97.5))


def summarize(frame: pd.DataFrame, value: str, group_columns: list[str], iterations: int, seed: int) -> pd.DataFrame:
    records = []
    random = np.random.default_rng(seed)
    for values, group in frame.groupby(group_columns, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        arrays = [axis_group[value].to_numpy(float) for _, axis_group in group.groupby("style_axis", sort=True)]
        lower, upper = bootstrap_mean(arrays, iterations, random)
        records.append(
            {
                **dict(zip(group_columns, values)),
                "estimate": float(group[value].mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "scenario_rows": len(group),
                "metric": value,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    sample = pd.read_parquet(REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    scenario = pd.read_parquet(REVISION_ROOT / "metrics/scenario_level_metrics.parquet")
    tables = REVISION_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    keys = ["scenario_id", "style_axis", "model_id", "encoder_id", "target_ratio", "is_interior"]
    cells = sample.groupby(keys, as_index=False).agg(
        projection_error_raw=("projection_error_raw", "mean"),
        projection_error_clipped=("projection_error_clipped", "mean"),
    )
    endpoint_rows = []
    for values, group in cells.groupby(["scenario_id", "style_axis", "model_id", "encoder_id"], sort=True):
        interior = group[group.is_interior]
        endpoint_rows.append(
            {
                **dict(zip(["scenario_id", "style_axis", "model_id", "encoder_id"], values)),
                "mae_7_point_raw": group.projection_error_raw.mean(),
                "mae_5_point_raw": interior.projection_error_raw.mean(),
                "absolute_underestimate": interior.projection_error_raw.mean() - group.projection_error_raw.mean(),
                "relative_underestimate": 1 - group.projection_error_raw.mean() / interior.projection_error_raw.mean(),
                "mae_5_point_clipped": interior.projection_error_clipped.mean(),
                "raw_minus_clipped": interior.projection_error_raw.mean() - interior.projection_error_clipped.mean(),
            }
        )
    endpoint = pd.DataFrame(endpoint_rows)
    summaries = []
    for metric in ("mae_7_point_raw", "mae_5_point_raw", "absolute_underestimate", "relative_underestimate", "mae_5_point_clipped", "raw_minus_clipped"):
        summaries.append(summarize(endpoint, metric, ["model_id", "encoder_id"], args.iterations, args.seed))
    pd.concat(summaries, ignore_index=True).to_csv(tables / "ablation_5point_7point_raw_clipped.csv", index=False)

    baseline = scenario.melt(
        id_vars=["scenario_id", "style_axis", "model_id", "encoder_id"],
        value_vars=["interior_calibration_error", "endpoint_distance_ratio_mae", "length_only_cv_mae", "isotonic_cv_mae", "constant_half_mae"],
        var_name="method",
        value_name="mae",
    )
    baseline_summary = summarize(baseline.rename(columns={"method": "estimation_method"}), "mae", ["model_id", "encoder_id", "estimation_method"], args.iterations, args.seed)
    baseline_summary.to_csv(tables / "ablation_projection_distance_isotonic_length_constant.csv", index=False)

    axis_rows = []
    for label, frame in (("all_axes", scenario), ("exclude_concision_detail", scenario[scenario.style_axis != "concision_detail"])):
        summarized = summarize(frame, "interior_calibration_error", ["model_id", "encoder_id"], args.iterations, args.seed)
        summarized["axis_set"] = label
        axis_rows.append(summarized)
    pd.concat(axis_rows, ignore_index=True).to_csv(tables / "ablation_all_axes_vs_remove_detail.csv", index=False)

    separation_rows = []
    for encoder, encoder_frame in scenario.groupby("encoder_id", sort=True):
        thresholds = {
            "full": -np.inf,
            "exclude_lowest_10pct": np.nanpercentile(encoder_frame.endpoint_separation, 10),
            "exclude_lowest_20pct": np.nanpercentile(encoder_frame.endpoint_separation, 20),
        }
        for label, threshold in thresholds.items():
            retained = encoder_frame[encoder_frame.endpoint_separation > threshold] if np.isfinite(threshold) else encoder_frame
            summarized = summarize(retained, "interior_calibration_error", ["model_id"], args.iterations, args.seed)
            summarized["encoder_id"] = encoder
            summarized["separation_filter"] = label
            summarized["threshold"] = threshold
            separation_rows.append(summarized)
    pd.concat(separation_rows, ignore_index=True).to_csv(tables / "ablation_low_endpoint_separation.csv", index=False)

    trajectory_paths = sorted((REVISION_ROOT / "metrics/fragments").glob("*/*.loso.parquet"))
    loso = pd.concat([pd.read_parquet(path) for path in trajectory_paths], ignore_index=True)
    loso_scenario = loso.groupby(["scenario_id", "style_axis", "model_id", "encoder_id"], as_index=False).agg(
        endpoint_line_error=("endpoint_line_error_mean", "mean"),
        endpoint_line_along_error=("endpoint_line_along_error_mean", "mean"),
        empirical_centroid_error=("empirical_centroid_error_mean", "mean"),
        empirical_centroid_along_error=("empirical_centroid_along_error_mean", "mean"),
        piecewise_trajectory_error=("piecewise_along_error_mean", "mean"),
        endpoint_line_accuracy=("endpoint_line_nearest_ratio_accuracy", "mean"),
        empirical_centroid_accuracy=("empirical_centroid_nearest_ratio_accuracy", "mean"),
        piecewise_trajectory_accuracy=("piecewise_nearest_ratio_accuracy", "mean"),
        endpoint_line_off_trajectory=("endpoint_line_off_trajectory_mean", "mean"),
        empirical_centroid_off_trajectory=("empirical_centroid_off_trajectory_mean", "mean"),
        piecewise_off_trajectory=("piecewise_off_trajectory_mean", "mean"),
    )
    trajectory_summary = []
    for metric in [column for column in loso_scenario if column not in {"scenario_id", "style_axis", "model_id", "encoder_id"}]:
        trajectory_summary.append(summarize(loso_scenario, metric, ["model_id", "encoder_id"], args.iterations, args.seed))
    pd.concat(trajectory_summary, ignore_index=True).to_csv(tables / "ablation_endpoint_line_centroid_piecewise.csv", index=False)

    model_summary = pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv")
    rank_rows = []
    encoders = sorted(model_summary.encoder_id.unique())
    for first, second in itertools.combinations(encoders, 2):
        for metric in ("interior_calibration_error_estimate", "normalized_off_axis_drift_estimate", "spearman_monotonicity_estimate"):
            pivot = model_summary.pivot(index="model_id", columns="encoder_id", values=metric)
            rank_rows.append(
                {
                    "encoder_first": first,
                    "encoder_second": second,
                    "metric": metric,
                    "model_count": len(pivot),
                    "spearman_rank_correlation": spearmanr(pivot[first], pivot[second]).statistic,
                    "kendall_rank_correlation": kendalltau(pivot[first], pivot[second]).statistic,
                }
            )
    pd.DataFrame(rank_rows).to_csv(tables / "three_encoder_ranking_consistency.csv", index=False)
    two = model_summary[model_summary.encoder_id.isin(["bge_m3", "text2vec_base_chinese"])].groupby("model_id", as_index=False).interior_calibration_error_estimate.mean().rename(columns={"interior_calibration_error_estimate": "two_encoder_mean_ICE"})
    three = model_summary.groupby("model_id", as_index=False).interior_calibration_error_estimate.mean().rename(columns={"interior_calibration_error_estimate": "three_encoder_mean_ICE"})
    two_three = two.merge(three, on="model_id", validate="one_to_one")
    two_three["three_minus_two"] = two_three.three_encoder_mean_ICE - two_three.two_encoder_mean_ICE
    two_three["two_encoder_rank"] = two_three.two_encoder_mean_ICE.rank(method="min")
    two_three["three_encoder_rank"] = two_three.three_encoder_mean_ICE.rank(method="min")
    two_three.to_csv(tables / "ablation_two_vs_three_encoders.csv", index=False)

    shared = pd.read_csv(REVISION_ROOT / "data/shared_anchors_final.csv")
    shared_status = "complete" if "approval_status" in shared and (shared.approval_status == "approved").all() else "external_pending"
    pd.DataFrame([{"status": shared_status, "required_scenarios": 80, "note": "Self/shared metrics and ranking flips require approved human-independent reference endpoints."}]).to_csv(tables / "ablation_self_vs_shared_anchor.csv", index=False)
    print(json.dumps({"endpoint_rows": len(endpoint), "baseline_rows": len(baseline_summary), "shared_anchor_status": shared_status}))


if __name__ == "__main__":
    main()
