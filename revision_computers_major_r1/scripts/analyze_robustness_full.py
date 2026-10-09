from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import holm_adjust, paired_sign_permutation_pvalue, stratified_cluster_mean_ci
from common import atomic_write_json, read_jsonl, utc_now


ITERATIONS = 10_000
SEED = 20260916
CONDITIONS = ["p1_high", "p2_high", "p1_low", "p2_low"]
METRICS = [
    "external_reference_ice",
    "external_reference_drift",
    "external_reference_out_of_range",
    "qwen_style_mae",
    "qwen_content_pass",
    "qwen_naturalness",
    "gpt_style_mae",
    "gpt_content_pass",
    "gpt_naturalness",
    "character_length",
]


def load_scores(root: Path) -> pd.DataFrame:
    index = pd.DataFrame(read_jsonl(root / "data/evaluation_index.jsonl"))
    qwen = pd.DataFrame(read_jsonl(root / "metrics/qwen_judge_evaluations.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    gpt = pd.DataFrame(read_jsonl(root / "metrics/gpt_judge_evaluations.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    fields = ["evaluation_item_id", "style_B_intensity", "content_preservation_pass", "naturalness"]
    qwen = qwen[[column for column in fields if column in qwen]].rename(columns=lambda column: column if column == "evaluation_item_id" else f"qwen_{column}")
    gpt = gpt[[column for column in fields if column in gpt]].rename(columns=lambda column: column if column == "evaluation_item_id" else f"gpt_{column}")
    scores = index.merge(qwen, on="evaluation_item_id", how="left", validate="one_to_one")
    scores = scores.merge(gpt, on="evaluation_item_id", how="left", validate="one_to_one")
    return scores


def prepare_sample(root: Path) -> pd.DataFrame:
    geometry = pd.read_parquet(root / "metrics/geometry/bge_m3/robustness_sample.parquet")
    geometry = geometry[geometry.anchor_protocol == "external_fixed_reference"].copy()
    outputs = pd.DataFrame(read_jsonl(root / "data/robustness_outputs.jsonl"))
    outputs["judge_source_run_id"] = outputs.source_run_id.fillna(outputs.run_id)
    metadata = outputs[["run_id", "judge_source_run_id"]]
    scores = load_scores(root)
    score_columns = [
        "original_source_run_id", "qwen_style_B_intensity", "qwen_content_preservation_pass", "qwen_naturalness",
        "gpt_style_B_intensity", "gpt_content_preservation_pass", "gpt_naturalness",
    ]
    scores = scores[score_columns].drop_duplicates("original_source_run_id", keep="last")
    data = geometry.merge(metadata, on="run_id", how="left", validate="many_to_one")
    data = data.merge(scores, left_on="judge_source_run_id", right_on="original_source_run_id", how="left", validate="many_to_one")
    data["external_reference_ice"] = data.projection_error_raw
    data["external_reference_drift"] = data.off_axis_drift_normalized
    data["external_reference_out_of_range"] = data.out_of_range.astype(float)
    data["qwen_style_mae"] = (data.qwen_style_B_intensity / 100.0 - data.target_ratio).abs()
    data["gpt_style_mae"] = (data.gpt_style_B_intensity / 100.0 - data.target_ratio).abs()
    data["qwen_content_pass"] = data.qwen_content_preservation_pass.astype(float)
    data["gpt_content_pass"] = data.gpt_content_preservation_pass.astype(float)
    return data


def scenario_cells(sample: pd.DataFrame) -> pd.DataFrame:
    # This is the required order of operations: average the three seeds and
    # five internal ratios within a scenario/condition before model effects.
    aggregations = {metric: (metric, "mean") for metric in METRICS}
    return sample.groupby(["scenario_id", "style_axis", "model_id", "robustness_condition_id"], as_index=False).agg(
        valid_output_count=("run_id", "nunique"),
        **aggregations,
    )


def condition_summaries(cells: pd.DataFrame) -> pd.DataFrame:
    records = []
    for metric in METRICS:
        for model_id, model in cells.groupby("model_id", sort=True):
            for condition, group in model.groupby("robustness_condition_id", sort=True):
                result = stratified_cluster_mean_ci(group, metric, iterations=ITERATIONS, seed=SEED)
                records.append({"record_type": "condition_mean", "metric": metric, "model_id": model_id, "style_axis": "ALL", "condition_or_effect": condition, **result})
        # Equal-model overall: first average the five model cells per scenario.
        overall = cells.groupby(["scenario_id", "style_axis", "robustness_condition_id"], as_index=False)[metric].mean()
        for condition, group in overall.groupby("robustness_condition_id", sort=True):
            result = stratified_cluster_mean_ci(group, metric, iterations=ITERATIONS, seed=SEED)
            records.append({"record_type": "condition_mean", "metric": metric, "model_id": "equal_weight_five_models", "style_axis": "ALL", "condition_or_effect": condition, **result})
    return pd.DataFrame(records)


def effect_frame(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = cells.pivot(index=["scenario_id", "style_axis", "model_id"], columns="robustness_condition_id", values=metric)
    pivot = pivot.reindex(columns=CONDITIONS).dropna().reset_index()
    count_pivot = cells.pivot(index=["scenario_id", "style_axis", "model_id"], columns="robustness_condition_id", values="valid_output_count").reindex(columns=CONDITIONS)
    count_pivot["all_four_cells_have_15_outputs"] = count_pivot.eq(15).all(axis=1)
    pivot = pivot.merge(count_pivot[["all_four_cells_have_15_outputs"]].reset_index(), on=["scenario_id", "style_axis", "model_id"], how="left", validate="one_to_one")
    pivot["prompt_effect"] = ((pivot.p2_high + pivot.p2_low) - (pivot.p1_high + pivot.p1_low)) / 2.0
    pivot["temperature_effect"] = ((pivot.p1_low + pivot.p2_low) - (pivot.p1_high + pivot.p2_high)) / 2.0
    pivot["interaction"] = (pivot.p2_low - pivot.p2_high) - (pivot.p1_low - pivot.p1_high)
    return pivot


def effect_summaries(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    scenario_effect_rows = []
    for metric in METRICS:
        effects = effect_frame(cells, metric)
        effects["metric"] = metric
        scenario_effect_rows.append(effects)
        for model_id, model in effects.groupby("model_id", sort=True):
            for effect in ("prompt_effect", "temperature_effect", "interaction"):
                result = stratified_cluster_mean_ci(model, effect, iterations=ITERATIONS, seed=SEED)
                records.append({"record_type": "factorial_effect", "metric": metric, "model_id": model_id, "style_axis": "ALL", "condition_or_effect": effect, "complete_15_per_condition_scenarios": int(model.all_four_cells_have_15_outputs.sum()), **result})
                for axis, axis_group in model.groupby("style_axis", sort=True):
                    axis_result = stratified_cluster_mean_ci(axis_group, effect, iterations=ITERATIONS, seed=SEED)
                    records.append({"record_type": "factorial_effect_axis", "metric": metric, "model_id": model_id, "style_axis": axis, "condition_or_effect": effect, **axis_result})
        # Strict equal-model match: only scenarios with all five model effects.
        for effect in ("prompt_effect", "temperature_effect", "interaction"):
            equal = effects.pivot(index=["scenario_id", "style_axis"], columns="model_id", values=effect).dropna()
            equal = equal.mean(axis=1).rename(effect).reset_index()
            result = stratified_cluster_mean_ci(equal, effect, iterations=ITERATIONS, seed=SEED)
            confirmatory = metric in {"external_reference_ice", "qwen_style_mae", "gpt_style_mae"}
            records.append({
                "record_type": "factorial_effect",
                "metric": metric,
                "model_id": "equal_weight_five_models",
                "style_axis": "ALL",
                "condition_or_effect": effect,
                "correction_family": f"robustness_primary::{metric}" if confirmatory else "sensitivity_no_confirmatory_test",
                "paired_permutation_p_raw": paired_sign_permutation_pvalue(equal[effect].to_numpy(), iterations=100_000, seed=SEED) if confirmatory else np.nan,
                **result,
            })
            for axis, axis_group in equal.groupby("style_axis", sort=True):
                axis_result = stratified_cluster_mean_ci(axis_group, effect, iterations=ITERATIONS, seed=SEED)
                records.append({"record_type": "factorial_effect_axis", "metric": metric, "model_id": "equal_weight_five_models", "style_axis": axis, "condition_or_effect": effect, **axis_result})
    output = pd.DataFrame(records)
    output["paired_permutation_p_holm"] = np.nan
    for family, indices in output.groupby("correction_family", dropna=False).groups.items():
        if not isinstance(family, str) or family == "sensitivity_no_confirmatory_test":
            continue
        output.loc[indices, "paired_permutation_p_holm"] = holm_adjust(output.loc[indices, "paired_permutation_p_raw"])
    return output, pd.concat(scenario_effect_rows, ignore_index=True)


def plot_effects(effects: pd.DataFrame, root: Path) -> None:
    plot = effects[(effects.record_type == "factorial_effect") & (effects.metric == "external_reference_ice")].copy()
    models = sorted([model for model in plot.model_id.unique() if model != "equal_weight_five_models"]) + ["equal_weight_five_models"]
    labels, means, lower, upper = [], [], [], []
    for effect in ("prompt_effect", "temperature_effect", "interaction"):
        for model in models:
            row = plot[(plot.model_id == model) & (plot.condition_or_effect == effect)].iloc[0]
            labels.append(f"{effect}: {model}")
            means.append(row["mean"])
            lower.append(row["mean"] - row.ci_low)
            upper.append(row.ci_high - row["mean"])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 8.5))
    ax.errorbar(means, y, xerr=np.asarray([lower, upper]), fmt="o", color="#4C78A8", capsize=2)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("Matched factorial effect on external-reference ICE")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "figures/figure_robustness_effects_ci.png", dpi=300)
    fig.savefig(root / "figures/figure_robustness_effects_ci.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    sample = prepare_sample(args.output_root)
    cells = scenario_cells(sample)
    conditions = condition_summaries(cells)
    effects, scenario_effects = effect_summaries(cells)
    table = pd.concat([conditions, effects], ignore_index=True)
    (args.output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (args.output_root / "statistics").mkdir(parents=True, exist_ok=True)
    (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
    (args.output_root / "figures").mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.output_root / "metrics/robustness_analysis_sample.parquet", index=False)
    cells.to_parquet(args.output_root / "metrics/robustness_scenario_condition.parquet", index=False)
    scenario_effects.to_parquet(args.output_root / "statistics/robustness_scenario_effects.parquet", index=False)
    table.to_csv(args.output_root / "tables/table_robustness_full.csv", index=False)
    (args.output_root / "tables/table_robustness_full.md").write_text(table.to_markdown(index=False) + "\n", encoding="utf-8")
    effects[effects.record_type == "factorial_effect_axis"].to_csv(args.output_root / "tables/table_robustness_axis_effects.csv", index=False)
    plot_effects(effects, args.output_root)
    status = {
        "generated_at": utc_now(),
        "valid_output_rows": int(sample.run_id.nunique()),
        "expected_condition_records": 96000,
        "scenario_condition_cells": len(cells),
        "expected_scenario_condition_cells": 6400,
        "bootstrap_iterations": ITERATIONS,
        "bootstrap_seed": SEED,
        "effect_definitions": {
            "prompt_effect": "mean_temperature(P2-P1)",
            "temperature_effect": "mean_prompt(low-high)",
            "interaction": "(P2_low-P2_high)-(P1_low-P1_high)",
        },
        "overall_model_weighting": "equal weight over the five original Qwen models after scenario-level matched effects",
        "confirmatory_test_families": "For each primary metric (BGE-M3 external-reference ICE, Qwen style MAE, GPT style MAE), the three equal-model factorial effects form one Holm family; model- and axis-specific effects are sensitivity summaries with CIs.",
        "paired_permutation_iterations": 100000,
    }
    atomic_write_json(args.output_root / "manifests/robustness_analysis.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
