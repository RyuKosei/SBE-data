from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from analysis_common import cluster_correlation_ci, holm_adjust, stratified_cluster_mean_ci
from common import atomic_write_json, read_jsonl, utc_now


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260916


def cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    a = left.astype(bool).to_numpy()
    b = right.astype(bool).to_numpy()
    if not len(a):
        return np.nan
    observed = float(np.mean(a == b))
    p_a = float(a.mean())
    p_b = float(b.mean())
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return np.nan if math.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def load_joined(output_root: Path) -> pd.DataFrame:
    index = pd.DataFrame(read_jsonl(output_root / "data/evaluation_index.jsonl"))
    qwen = pd.DataFrame(read_jsonl(output_root / "metrics/qwen_judge_evaluations.jsonl"))
    gpt = pd.DataFrame(read_jsonl(output_root / "metrics/gpt_judge_evaluations.jsonl"))
    fields = ["evaluation_item_id", "style_B_intensity", "naturalness", "content_preservation_pass", "content_coverage_rate", "unjudgeable"]
    qwen = qwen[[column for column in fields if column in qwen]].drop_duplicates("evaluation_item_id", keep="last")
    gpt = gpt[[column for column in fields if column in gpt]].drop_duplicates("evaluation_item_id", keep="last")
    qwen = qwen.rename(columns={column: f"qwen_{column}" for column in fields if column != "evaluation_item_id"})
    gpt = gpt.rename(columns={column: f"gpt_{column}" for column in fields if column != "evaluation_item_id"})
    joined = index.merge(qwen, on="evaluation_item_id", how="left", validate="one_to_one")
    joined = joined.merge(gpt, on="evaluation_item_id", how="left", validate="one_to_one")
    joined["character_length"] = joined.generated_text.astype(str).str.len()
    return joined


def ci_columns(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_mean": result["mean"],
        f"{prefix}_ci_low": result["ci_low"],
        f"{prefix}_ci_high": result["ci_high"],
        f"{prefix}_valid_scenarios": result["valid_scenarios"],
    }


def judge_summary(main: pd.DataFrame) -> pd.DataFrame:
    records = []
    internal = main[(main.target_ratio > 0) & (main.target_ratio < 1)].copy()
    for judge in ("qwen", "gpt"):
        internal[f"{judge}_style_mae"] = (internal[f"{judge}_style_B_intensity"] / 100.0 - internal.target_ratio).abs()
        for model_id, model in internal.groupby("model_id", sort=True):
            for axis_label, group in [("ALL", model), *list(model.groupby("style_axis", sort=True))]:
                scenario = group.groupby(["scenario_id", "style_axis"], as_index=False).agg(
                    style_mae=(f"{judge}_style_mae", "mean"),
                    naturalness=(f"{judge}_naturalness", "mean"),
                    content_pass=(f"{judge}_content_preservation_pass", "mean"),
                )
                record: dict[str, Any] = {"judge": judge, "model_id": model_id, "style_axis": axis_label, "valid_output_rows": int(group[f"{judge}_style_B_intensity"].notna().sum())}
                for metric in ("style_mae", "naturalness", "content_pass"):
                    record.update(ci_columns(stratified_cluster_mean_ci(scenario, metric, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED), metric))
                records.append(record)
    return pd.DataFrame(records)


def agreement_tables(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched = joined.dropna(subset=["qwen_style_B_intensity", "gpt_style_B_intensity"]).copy()
    matched["intensity_difference_qwen_minus_gpt"] = matched.qwen_style_B_intensity - matched.gpt_style_B_intensity
    rows = []
    for label, group in [("overall", matched), *[(f"alpha={alpha:g}", group) for alpha, group in matched.groupby("target_ratio", sort=True)], *[(f"axis={axis}", group) for axis, group in matched.groupby("style_axis", sort=True)]]:
        pearson = pearsonr(group.qwen_style_B_intensity, group.gpt_style_B_intensity).statistic if len(group) > 2 and group.qwen_style_B_intensity.nunique() > 1 and group.gpt_style_B_intensity.nunique() > 1 else np.nan
        spearman = spearmanr(group.qwen_style_B_intensity, group.gpt_style_B_intensity).statistic if len(group) > 2 and group.qwen_style_B_intensity.nunique() > 1 and group.gpt_style_B_intensity.nunique() > 1 else np.nan
        rows.append({
            "stratum": label,
            "n": len(group),
            "qwen_minus_gpt_mean": group.intensity_difference_qwen_minus_gpt.mean(),
            "qwen_minus_gpt_mae": group.intensity_difference_qwen_minus_gpt.abs().mean(),
            "pearson": pearson,
            "spearman": spearman,
        })
    matched["qwen_demeaned_alpha_axis"] = matched.qwen_style_B_intensity - matched.groupby(["target_ratio", "style_axis"]).qwen_style_B_intensity.transform("mean")
    matched["gpt_demeaned_alpha_axis"] = matched.gpt_style_B_intensity - matched.groupby(["target_ratio", "style_axis"]).gpt_style_B_intensity.transform("mean")
    residual = matched.dropna(subset=["qwen_demeaned_alpha_axis", "gpt_demeaned_alpha_axis"])
    rows.append({
        "stratum": "demeaned_within_alpha_axis",
        "n": len(residual),
        "qwen_minus_gpt_mean": np.nan,
        "qwen_minus_gpt_mae": np.nan,
        "pearson": residual.qwen_demeaned_alpha_axis.corr(residual.gpt_demeaned_alpha_axis),
        "spearman": spearmanr(residual.qwen_demeaned_alpha_axis, residual.gpt_demeaned_alpha_axis).statistic,
    })
    agreement = pd.DataFrame(rows)

    content = joined.dropna(subset=["qwen_content_preservation_pass", "gpt_content_preservation_pass"]).copy()
    qpass = content.qwen_content_preservation_pass.astype(bool)
    gpass = content.gpt_content_preservation_pass.astype(bool)
    confusion = pd.DataFrame([
        {"qwen_content_pass": q, "gpt_content_pass": g, "count": int(((qpass == q) & (gpass == g)).sum())}
        for q in (False, True) for g in (False, True)
    ])
    content_stats = pd.DataFrame([{
        "n": len(content),
        "raw_agreement": float((qpass == gpass).mean()) if len(content) else np.nan,
        "cohen_kappa": cohen_kappa(qpass, gpass),
        "qwen_pass_rate": qpass.mean(),
        "gpt_pass_rate": gpass.mean(),
    }])
    return agreement, confusion, content_stats


def ranking_table(main: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = main[(main.target_ratio > 0) & (main.target_ratio < 1)].copy()
    records = []
    for judge in ("qwen", "gpt"):
        internal[f"{judge}_mae"] = (internal[f"{judge}_style_B_intensity"] / 100.0 - internal.target_ratio).abs()
        # Preserve scenario as the analysis unit before forming the descriptive
        # six-model ordering; otherwise a failed output would slightly change
        # that scenario's weight rather than only its effective denominator.
        scenario = internal.groupby(["scenario_id", "style_axis", "model_id"], as_index=False)[f"{judge}_mae"].mean()
        means = scenario.groupby("model_id")[f"{judge}_mae"].mean().sort_values()
        for rank, (model_id, value) in enumerate(means.items(), 1):
            records.append({"judge": judge, "model_id": model_id, "style_mae": value, "rank": rank})
    rankings = pd.DataFrame(records)
    pivot = rankings.pivot(index="model_id", columns="judge", values="style_mae").dropna()
    comparison = pd.DataFrame([{
        "models": len(pivot),
        "spearman_rank_correlation_descriptive": spearmanr(pivot.qwen, pivot.gpt).statistic if len(pivot) >= 3 else np.nan,
        "kendall_not_reported_as_inference": True,
    }])
    return rankings, comparison


def paired_model_style_mae(main: pd.DataFrame) -> pd.DataFrame:
    internal = main[(main.target_ratio > 0) & (main.target_ratio < 1)].copy()
    records = []
    for judge in ("qwen", "gpt"):
        metric = f"{judge}_style_mae"
        internal[metric] = (internal[f"{judge}_style_B_intensity"] / 100.0 - internal.target_ratio).abs()
        scenario = internal.groupby(["scenario_id", "style_axis", "model_id"], as_index=False)[metric].mean()
        pivot = scenario.pivot(index=["scenario_id", "style_axis"], columns="model_id", values=metric)
        models = sorted(pivot.columns)
        for index, first in enumerate(models):
            for second in models[index + 1:]:
                differences = (pivot[first] - pivot[second]).dropna()
                frame = differences.rename("difference").reset_index()
                result = stratified_cluster_mean_ci(frame, "difference", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
                sd = float(differences.std(ddof=1))
                try:
                    pvalue = float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
                except ValueError:
                    pvalue = np.nan
                records.append({
                    "correction_family": f"main_style_mae_model_pairs::{judge}",
                    "judge": judge,
                    "model_first": first,
                    "model_second": second,
                    "mean_difference_first_minus_second": result["mean"],
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "paired_scenarios": result["valid_scenarios"],
                    "paired_effect_size_dz": result["mean"] / sd if sd > 0 else np.nan,
                    "paired_wilcoxon_p_raw": pvalue,
                })
    output = pd.DataFrame(records)
    output["paired_wilcoxon_p_holm"] = np.nan
    for _family, indices in output.groupby("correction_family").groups.items():
        output.loc[indices, "paired_wilcoxon_p_holm"] = holm_adjust(output.loc[indices, "paired_wilcoxon_p_raw"])
    return output


def geometry_judge_association(output_root: Path, joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry = pd.read_parquet(output_root / "metrics/geometry/bge_m3/main_sample.parquet")
    geometry = geometry[(geometry.anchor_protocol == "external_fixed_reference") & (geometry.target_ratio > 0) & (geometry.target_ratio < 1)]
    columns = ["run_id", "qwen_style_B_intensity", "gpt_style_B_intensity"]
    data = geometry.merge(joined[columns], on="run_id", how="left", validate="one_to_one")
    trajectory_records = []
    for key, group in data.groupby(["scenario_id", "style_axis", "model_id", "replicate"], sort=True):
        group = group.sort_values("target_ratio")
        record: dict[str, Any] = {"scenario_id": key[0], "style_axis": key[1], "model_id": key[2], "replicate": key[3]}
        for judge in ("qwen", "gpt"):
            valid = group.dropna(subset=["projection_raw", f"{judge}_style_B_intensity"])
            constant = len(valid) < 2 or valid.projection_raw.nunique() <= 1 or valid[f"{judge}_style_B_intensity"].nunique() <= 1
            record[f"{judge}_valid_internal_points"] = len(valid)
            record[f"{judge}_constant_sequence"] = constant
            record[f"{judge}_pearson"] = np.nan if constant else pearsonr(valid.projection_raw, valid[f"{judge}_style_B_intensity"]).statistic
            record[f"{judge}_spearman"] = np.nan if constant else spearmanr(valid.projection_raw, valid[f"{judge}_style_B_intensity"]).statistic
        trajectory_records.append(record)
    trajectories = pd.DataFrame(trajectory_records)
    summaries = []
    for judge in ("qwen", "gpt"):
        for model_id, group in trajectories.groupby("model_id", sort=True):
            scenario = group.groupby(["scenario_id", "style_axis"], as_index=False).agg(
                pearson=(f"{judge}_pearson", "mean"), spearman=(f"{judge}_spearman", "mean")
            )
            record = {
                "judge": judge,
                "model_id": model_id,
                "constant_trajectory_count": int(group[f"{judge}_constant_sequence"].sum()),
                "incomplete_five_point_trajectory_count": int((group[f"{judge}_valid_internal_points"] < 5).sum()),
                "valid_internal_point_total": int(group[f"{judge}_valid_internal_points"].sum()),
                "trajectory_count": len(group),
            }
            for metric in ("pearson", "spearman"):
                record.update(ci_columns(stratified_cluster_mean_ci(scenario, metric, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED), metric))
            summaries.append(record)
    return trajectories, pd.DataFrame(summaries)


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(design).all(axis=1)
    output = np.full(len(values), np.nan)
    coefficients, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    output[valid] = values[valid] - design[valid] @ coefficients
    return output


def length_sensitivity(output_root: Path, joined: pd.DataFrame) -> pd.DataFrame:
    geometry = pd.read_parquet(output_root / "metrics/geometry/bge_m3/main_sample.parquet")
    geometry = geometry[(geometry.anchor_protocol == "external_fixed_reference") & (geometry.target_ratio > 0) & (geometry.target_ratio < 1)]
    data = geometry.merge(joined[["run_id", "qwen_style_B_intensity", "gpt_style_B_intensity"]], on="run_id", how="left", validate="one_to_one")
    averaged = data.groupby(["scenario_id", "style_axis", "model_id", "target_ratio"], as_index=False).agg(
        projection_raw=("projection_raw", "mean"),
        character_length=("character_length", "mean"),
        qwen_intensity=("qwen_style_B_intensity", "mean"),
        gpt_intensity=("gpt_style_B_intensity", "mean"),
    )
    categorical = averaged[["target_ratio", "model_id", "style_axis", "scenario_id"]].astype(str)
    design = pd.get_dummies(categorical, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(design)), design])
    geometry_residual = residualize(averaged.projection_raw.to_numpy(dtype=float), design)
    length_residual = residualize(averaged.character_length.to_numpy(dtype=float), design)
    records = []
    for judge in ("qwen", "gpt"):
        judge_residual = residualize(averaged[f"{judge}_intensity"].to_numpy(dtype=float), design)
        valid_length = np.isfinite(length_residual)
        length_design = np.column_stack([np.ones(valid_length.sum()), length_residual[valid_length]])
        geometry_length_controlled = np.full(len(averaged), np.nan)
        judge_length_controlled = np.full(len(averaged), np.nan)
        geometry_length_controlled[valid_length] = residualize(geometry_residual[valid_length], length_design)
        judge_length_controlled[valid_length] = residualize(judge_residual[valid_length], length_design)
        work = averaged[["scenario_id"]].copy()
        work["geometry_residual"] = geometry_residual
        work["judge_residual"] = judge_residual
        work["geometry_length_controlled"] = geometry_length_controlled
        work["judge_length_controlled"] = judge_length_controlled
        base = cluster_correlation_ci(work, "geometry_residual", "judge_residual", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
        controlled = cluster_correlation_ci(work, "geometry_length_controlled", "judge_length_controlled", iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
        records.append({
            "judge": judge,
            "seed_averaged_before_analysis": True,
            "controls": "target_ratio + model_id + style_axis + scenario_id fixed effects",
            "residual_correlation": base["correlation"],
            "residual_correlation_ci_low": base["ci_low"],
            "residual_correlation_ci_high": base["ci_high"],
            "length_controlled_correlation": controlled["correlation"],
            "length_controlled_ci_low": controlled["ci_low"],
            "length_controlled_ci_high": controlled["ci_high"],
            "valid_rows": base["valid_rows"],
            "valid_scenarios": base["valid_scenarios"],
        })
    return pd.DataFrame(records)


def make_scatter(joined: pd.DataFrame, output_root: Path) -> None:
    matched = joined.dropna(subset=["qwen_style_B_intensity", "gpt_style_B_intensity"])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    histogram = ax.hexbin(matched.qwen_style_B_intensity, matched.gpt_style_B_intensity, gridsize=40, mincnt=1, cmap="viridis")
    ax.plot([0, 100], [0, 100], color="black", linewidth=1, linestyle="--")
    ax.set(xlabel="Qwen judge intensity", ylabel="GPT judge intensity", xlim=(0, 100), ylim=(0, 100))
    fig.colorbar(histogram, ax=ax, label="Output count")
    fig.tight_layout()
    fig.savefig(output_root / "figures/figure_dual_judge_scatter.png", dpi=300)
    fig.savefig(output_root / "figures/figure_dual_judge_scatter.pdf")
    plt.close(fig)


def make_intensity_curves(main: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    for ax, judge in zip(axes, ("qwen", "gpt")):
        value = f"{judge}_style_B_intensity"
        for model_id, group in main.groupby("model_id", sort=True):
            scenario = group.groupby(["scenario_id", "style_axis", "target_ratio"], as_index=False)[value].mean()
            points = []
            for alpha, alpha_group in scenario.groupby("target_ratio", sort=True):
                result = stratified_cluster_mean_ci(alpha_group, value, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED)
                points.append({"target_ratio": alpha, **result})
            curve = pd.DataFrame(points)
            line = ax.plot(curve.target_ratio, curve["mean"] / 100.0, marker="o", label=model_id)[0]
            ax.fill_between(curve.target_ratio, curve.ci_low / 100.0, curve.ci_high / 100.0, color=line.get_color(), alpha=0.10)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{judge.upper()} judge")
        ax.set_xlabel("Requested B-style ratio (alpha)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Judged B-style intensity")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "figures/figure_judge_intensity_by_alpha.png", dpi=300)
    fig.savefig(output_root / "figures/figure_judge_intensity_by_alpha.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    joined = load_joined(args.output_root)
    tables = args.output_root / "tables"
    stats = args.output_root / "statistics"
    metrics = args.output_root / "metrics"
    figures = args.output_root / "figures"
    for path in (tables, stats, metrics, figures):
        path.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(metrics / "dual_judge_sample.parquet", index=False)
    main_data = joined[joined.source_type.isin(["main_qwen", "main_external"])].copy()
    summary = judge_summary(main_data)
    summary.to_csv(tables / "table_independent_judge.csv", index=False)
    (tables / "table_independent_judge.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    agreement, confusion, content_stats = agreement_tables(joined)
    agreement.to_csv(stats / "dual_judge_intensity_agreement.csv", index=False)
    confusion.to_csv(stats / "dual_judge_content_confusion.csv", index=False)
    content_stats.to_csv(stats / "dual_judge_content_agreement.csv", index=False)
    rankings, ranking_comparison = ranking_table(main_data)
    rankings.to_csv(stats / "dual_judge_model_rankings.csv", index=False)
    ranking_comparison.to_csv(stats / "dual_judge_ranking_comparison.csv", index=False)
    paired_models = paired_model_style_mae(main_data)
    paired_models.to_csv(stats / "dual_judge_model_pairwise_style_mae.csv", index=False)
    trajectories, association = geometry_judge_association(args.output_root, joined)
    trajectories.to_parquet(metrics / "geometry_judge_trajectory_associations.parquet", index=False)
    association.to_csv(stats / "geometry_judge_association_summary.csv", index=False)
    length = length_sensitivity(args.output_root, joined)
    length.to_csv(stats / "geometry_judge_length_sensitivity.csv", index=False)
    condition_bias = joined.dropna(subset=["qwen_style_B_intensity", "gpt_style_B_intensity"]).copy()
    condition_bias["qwen_minus_gpt_intensity"] = condition_bias.qwen_style_B_intensity - condition_bias.gpt_style_B_intensity
    condition_bias.groupby(["model_id", "source_type", "robustness_condition_id"], dropna=False).agg(
        n=("qwen_minus_gpt_intensity", "size"),
        mean_difference=("qwen_minus_gpt_intensity", "mean"),
        mean_absolute_difference=("qwen_minus_gpt_intensity", lambda values: values.abs().mean()),
    ).reset_index().to_csv(stats / "dual_judge_model_condition_differences.csv", index=False)
    make_scatter(joined, args.output_root)
    make_intensity_curves(main_data, args.output_root)
    status = {
        "generated_at": utc_now(),
        "evaluation_index_rows": len(joined),
        "qwen_valid": int(joined.qwen_style_B_intensity.notna().sum()),
        "gpt_valid": int(joined.gpt_style_B_intensity.notna().sum()),
        "matched_valid": int(joined[["qwen_style_B_intensity", "gpt_style_B_intensity"]].notna().all(axis=1).sum()),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_test": "two-sided Wilcoxon signed-rank on scenario-level matched model differences",
        "holm_family_definition": "all pairwise model comparisons within one judge for main-experiment style MAE",
        "human_validation": False,
    }
    atomic_write_json(args.output_root / "manifests/dual_judge_analysis.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
