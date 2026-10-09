from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from analysis_common import holm_adjust, paired_sign_permutation_pvalue, stratified_cluster_mean_ci
from common import atomic_write_json, read_jsonl, utc_now


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260916
METRICS = {
    "ice": "interior_calibration_error",
    "drift": "normalized_off_axis_drift_mean",
    "interpolation_distance": "interpolation_distance_mean",
    "out_of_range_rate": "out_of_range_rate",
    "nearest_target_rank": "nearest_target_rank_normalized_mean",
}


def ci(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": result["mean"],
        f"{prefix}_ci_low": result["ci_low"],
        f"{prefix}_ci_high": result["ci_high"],
        f"{prefix}_valid_scenarios": result["valid_scenarios"],
    }


def load_trajectories(root: Path) -> pd.DataFrame:
    paths = sorted((root / "metrics/geometry").glob("*/main_trajectory.parquet"))
    if len(paths) != 3:
        raise ValueError(f"expected three encoder trajectory files, found {len(paths)}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def summarize_geometry(trajectories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = trajectories.groupby(["scenario_id", "style_axis", "model_id", "encoder_id", "anchor_protocol"], as_index=False).agg(
        **{name: (column, "mean") for name, column in METRICS.items()},
        endpoint_separation=("endpoint_separation", "mean"),
        endpoint_unstable=("endpoint_unstable", "max"),
        trajectory_count=("replicate", "size"),
        complete_seven_point_trajectory_count=("complete_seven_point_trajectory", "sum"),
        valid_interior_point_count=("valid_interior_points", "sum"),
    )
    records = []
    for key, group in scenario.groupby(["encoder_id", "anchor_protocol", "model_id"], sort=True):
        record: dict[str, Any] = {
            "encoder_id": key[0],
            "anchor_protocol": key[1],
            "model_id": key[2],
            "full_scenario_denominator": int(group.scenario_id.nunique()),
            "trajectory_count": int(group.trajectory_count.sum()),
            "complete_seven_point_trajectory_count": int(group.complete_seven_point_trajectory_count.sum()),
            "valid_interior_point_count": int(group.valid_interior_point_count.sum()),
            "expected_interior_point_count": int(group.trajectory_count.sum() * 5),
            "collapsed_scenario_count": int(group.endpoint_unstable.sum()),
        }
        for metric in METRICS:
            record.update(ci(stratified_cluster_mean_ci(group, metric, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED), metric))
        record.update(ci(stratified_cluster_mean_ci(group, "endpoint_separation", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED), "endpoint_separation"))
        records.append(record)
    summary = pd.DataFrame(records)
    summary["ice_rank_within_encoder_protocol"] = summary.groupby(["encoder_id", "anchor_protocol"]).ice_mean.rank(method="average", ascending=True)
    return scenario, summary


def protocol_comparisons(scenario: pd.DataFrame) -> pd.DataFrame:
    records = []
    for encoder_id in sorted(scenario.encoder_id.unique()):
        encoder = scenario[scenario.encoder_id == encoder_id]
        for model_id in sorted(encoder.model_id.unique()):
            model = encoder[encoder.model_id == model_id]
            for metric in METRICS:
                pivot = model.pivot(index=["scenario_id", "style_axis"], columns="anchor_protocol", values=metric)
                for protocol in ("external_fixed_reference", "fixed_pooled_qwen"):
                    if protocol not in pivot or "self_anchored" not in pivot:
                        continue
                    differences = (pivot[protocol] - pivot["self_anchored"]).dropna()
                    frame = differences.rename("difference").reset_index()
                    result = stratified_cluster_mean_ci(frame, "difference", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
                    sd = float(differences.std(ddof=1))
                    confirmatory = encoder_id == "bge_m3" and metric == "ice" and protocol == "external_fixed_reference"
                    records.append({
                        "correction_family": "primary_bge_external_ice_vs_self_by_model" if confirmatory else "sensitivity_no_confirmatory_test",
                        "encoder_id": encoder_id,
                        "model_id": model_id,
                        "metric": metric,
                        "comparison": f"{protocol}_minus_self_anchored",
                        "mean_difference": result["mean"],
                        "ci_low": result["ci_low"],
                        "ci_high": result["ci_high"],
                        "paired_scenarios": result["valid_scenarios"],
                        "paired_effect_size_dz": result["mean"] / sd if sd > 0 else np.nan,
                        "permutation_p_raw": paired_sign_permutation_pvalue(differences.to_numpy(), iterations=100_000, seed=BOOTSTRAP_SEED) if confirmatory else np.nan,
                    })
    output = pd.DataFrame(records)
    output["permutation_p_holm"] = np.nan
    for family, indices in output.groupby("correction_family").groups.items():
        if family == "sensitivity_no_confirmatory_test":
            continue
        output.loc[indices, "permutation_p_holm"] = holm_adjust(output.loc[indices, "permutation_p_raw"])
    return output


def rank_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for encoder_id, encoder in summary.groupby("encoder_id", sort=True):
        pivot = encoder.pivot(index="model_id", columns="anchor_protocol", values="ice_mean")
        for protocol in ("external_fixed_reference", "fixed_pooled_qwen"):
            paired = pivot[["self_anchored", protocol]].dropna()
            flips = 0
            models = paired.index.tolist()
            for index, first in enumerate(models):
                for second in models[index + 1:]:
                    flips += int(np.sign(paired.loc[first, "self_anchored"] - paired.loc[second, "self_anchored"]) != np.sign(paired.loc[first, protocol] - paired.loc[second, protocol]))
            records.append({
                "encoder_id": encoder_id,
                "comparison": f"{protocol}_vs_self_anchored",
                "models": len(paired),
                "spearman": spearmanr(paired.self_anchored, paired[protocol]).statistic if len(paired) >= 3 else np.nan,
                "kendall": kendalltau(paired.self_anchored, paired[protocol]).statistic if len(paired) >= 3 else np.nan,
                "pairwise_rank_flips": flips,
                "pairwise_model_pairs": len(models) * (len(models) - 1) // 2,
            })
    return pd.DataFrame(records)


def external_content_pass_subset(scenario: pd.DataFrame, root: Path) -> pd.DataFrame:
    quality = pd.read_parquet(root / "metrics/reference_quality_scenario.parquet")[["scenario_id", "both_content_pass"]]
    external = scenario[scenario.anchor_protocol == "external_fixed_reference"].merge(quality, on="scenario_id", how="left", validate="many_to_one")
    records = []
    for key, group in external.groupby(["encoder_id", "model_id"], sort=True):
        matched = group[group.both_content_pass.fillna(False)]
        for subset_label, subset in [("all_computable_reference_pairs", group), ("both_references_content_pass", matched)]:
            result = stratified_cluster_mean_ci(subset, "ice", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
            records.append({
                "encoder_id": key[0],
                "model_id": key[1],
                "subset": subset_label,
                "ice_mean": result["mean"],
                "ice_ci_low": result["ci_low"],
                "ice_ci_high": result["ci_high"],
                "valid_scenarios": result["valid_scenarios"],
                "full_reference_pair_denominator": int(group.scenario_id.nunique()),
            })
    return pd.DataFrame(records)


def judge_scenario(root: Path) -> pd.DataFrame:
    sample_path = root / "metrics/dual_judge_sample.parquet"
    if sample_path.exists():
        data = pd.read_parquet(sample_path)
    else:
        index = pd.DataFrame(read_jsonl(root / "data/evaluation_index.jsonl"))
        qwen = pd.DataFrame(read_jsonl(root / "metrics/qwen_judge_evaluations.jsonl"))
        gpt = pd.DataFrame(read_jsonl(root / "metrics/gpt_judge_evaluations.jsonl"))
        data = index.merge(qwen[["evaluation_item_id", "style_B_intensity", "naturalness", "content_preservation_pass"]].rename(columns=lambda c: c if c == "evaluation_item_id" else f"qwen_{c}"), on="evaluation_item_id", how="left")
        data = data.merge(gpt[["evaluation_item_id", "style_B_intensity", "naturalness", "content_preservation_pass"]].rename(columns=lambda c: c if c == "evaluation_item_id" else f"gpt_{c}"), on="evaluation_item_id", how="left")
    data = data[data.source_type.isin(["main_qwen", "main_external"]) & (data.target_ratio > 0) & (data.target_ratio < 1)].copy()
    for judge in ("qwen", "gpt"):
        data[f"{judge}_style_mae"] = (data[f"{judge}_style_B_intensity"] / 100.0 - data.target_ratio).abs()
    return data.groupby(["scenario_id", "style_axis", "model_id"], as_index=False).agg(
        qwen_style_mae=("qwen_style_mae", "mean"),
        qwen_naturalness=("qwen_naturalness", "mean"),
        qwen_content_pass=("qwen_content_preservation_pass", "mean"),
        gpt_style_mae=("gpt_style_mae", "mean"),
        gpt_naturalness=("gpt_naturalness", "mean"),
        gpt_content_pass=("gpt_content_preservation_pass", "mean"),
    )


def cross_family_table(scenario: pd.DataFrame, root: Path) -> pd.DataFrame:
    geometry = scenario[(scenario.encoder_id == "bge_m3") & scenario.anchor_protocol.isin(["self_anchored", "external_fixed_reference"])].copy()
    judges = judge_scenario(root)
    merged = geometry.merge(judges, on=["scenario_id", "style_axis", "model_id"], how="left", validate="many_to_one")
    records = []
    metrics = ["ice", "drift", "interpolation_distance", "out_of_range_rate", "qwen_style_mae", "qwen_naturalness", "qwen_content_pass", "gpt_style_mae", "gpt_naturalness", "gpt_content_pass"]
    for key, group in merged.groupby(["model_id", "anchor_protocol"], sort=True):
        record: dict[str, Any] = {"model_id": key[0], "anchor_protocol": key[1], "scenario_denominator": int(group.scenario_id.nunique())}
        for metric in metrics:
            record.update(ci(stratified_cluster_mean_ci(group, metric, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED), metric))
        records.append(record)
    return pd.DataFrame(records)


def make_alpha_curves(root: Path) -> None:
    geometry = pd.read_parquet(root / "metrics/geometry/bge_m3/main_sample.parquet")
    geometry = geometry[geometry.anchor_protocol == "external_fixed_reference"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for model_id, group in geometry.groupby("model_id", sort=True):
        scenario = group.groupby(["scenario_id", "style_axis", "target_ratio"], as_index=False).projection_raw.mean()
        points = []
        for alpha, alpha_group in scenario.groupby("target_ratio", sort=True):
            result = stratified_cluster_mean_ci(alpha_group, "projection_raw", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
            points.append({"target_ratio": alpha, **result})
        curve = pd.DataFrame(points)
        line = ax.plot(curve.target_ratio, curve["mean"], marker="o", label=model_id)[0]
        ax.fill_between(curve.target_ratio, curve.ci_low, curve.ci_high, color=line.get_color(), alpha=0.10)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="ideal")
    ax.set(xlabel="Requested B-style ratio (alpha)", ylabel="External-reference coordinate", xlim=(0, 1), ylim=None)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "figures/figure_external_coordinate_by_alpha.png", dpi=300)
    fig.savefig(root / "figures/figure_external_coordinate_by_alpha.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("tables", "statistics", "figures"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.output_root)
    scenario, summary = summarize_geometry(trajectories)
    scenario.to_parquet(args.output_root / "metrics/geometry_scenario_level.parquet", index=False)
    summary.to_csv(args.output_root / "tables/table_anchor_encoder_sensitivity.csv", index=False)
    (args.output_root / "tables/table_anchor_encoder_sensitivity.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    comparisons = protocol_comparisons(scenario)
    comparisons.to_csv(args.output_root / "statistics/anchor_protocol_paired_comparisons.csv", index=False)
    rank = rank_sensitivity(summary)
    rank.to_csv(args.output_root / "statistics/anchor_protocol_rank_sensitivity.csv", index=False)
    reference_subset = external_content_pass_subset(scenario, args.output_root)
    reference_subset.to_csv(args.output_root / "statistics/external_reference_content_pass_subset.csv", index=False)
    cross = cross_family_table(scenario, args.output_root)
    cross.to_csv(args.output_root / "tables/table_cross_family_main.csv", index=False)
    (args.output_root / "tables/table_cross_family_main.md").write_text(cross.to_markdown(index=False) + "\n", encoding="utf-8")
    make_alpha_curves(args.output_root)
    status = {
        "generated_at": utc_now(),
        "scenario_rows": len(scenario),
        "summary_rows": len(summary),
        "models": sorted(summary.model_id.unique().tolist()),
        "encoders": sorted(summary.encoder_id.unique().tolist()),
        "protocols": sorted(summary.anchor_protocol.unique().tolist()),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_iterations": 100000,
        "holm_family_definition": "all model comparisons of one fixed encoder × metric × protocol-vs-self contrast",
    }
    atomic_write_json(args.output_root / "manifests/geometry_analysis.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
