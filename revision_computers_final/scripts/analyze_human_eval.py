"""Analyze real human ratings; stop cleanly when the annotation package is blank."""

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


SCORE_COLUMNS = [
    "style_B_intensity_0_100",
    "content_coverage_1_5",
    "naturalness_1_5",
    "adds_key_information_yes_no",
    "contradicts_source_yes_no",
    "unable_to_judge_yes_no",
]


def binary(series: pd.Series) -> pd.Series:
    mapping = {"yes": 1.0, "no": 0.0, "是": 1.0, "否": 0.0, True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}
    return series.map(lambda value: mapping.get(str(value).strip().lower(), mapping.get(value, np.nan)))


def icc_absolute(matrix: np.ndarray) -> tuple[float, float]:
    n, k = matrix.shape
    grand = matrix.mean()
    item_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    ms_items = k * np.square(item_means - grand).sum() / (n - 1)
    ms_raters = n * np.square(rater_means - grand).sum() / (k - 1)
    residual = matrix - item_means[:, None] - rater_means[None, :] + grand
    ms_error = np.square(residual).sum() / ((n - 1) * (k - 1))
    denominator = ms_items + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    single = (ms_items - ms_error) / denominator if denominator else np.nan
    average = (ms_items - ms_error) / (ms_items + (ms_raters - ms_error) / n) if denominator else np.nan
    return float(single), float(average)


def krippendorff_interval(matrix: np.ndarray) -> float:
    within = []
    for row in matrix:
        values = row[np.isfinite(row)]
        for first in range(len(values)):
            for second in range(first + 1, len(values)):
                within.append((values[first] - values[second]) ** 2)
    pooled = matrix[np.isfinite(matrix)]
    if not within or len(pooled) < 2:
        return np.nan
    expected = np.square(pooled[:, None] - pooled[None, :])[np.triu_indices(len(pooled), 1)].mean()
    return float(1 - np.mean(within) / expected) if expected else np.nan


def fleiss_kappa(matrix: np.ndarray) -> float:
    counts = np.column_stack([(matrix == 0).sum(axis=1), (matrix == 1).sum(axis=1)])
    raters = counts.sum(axis=1)
    valid = raters > 1
    counts = counts[valid]
    raters = raters[valid]
    if not len(counts):
        return np.nan
    agreement = ((np.square(counts).sum(axis=1) - raters) / (raters * (raters - 1))).mean()
    proportions = counts.sum(axis=0) / counts.sum()
    expected = np.square(proportions).sum()
    return float((agreement - expected) / (1 - expected)) if expected < 1 else np.nan


def cohen_kappa(first: np.ndarray, second: np.ndarray) -> float:
    observed = np.mean(first == second)
    p_first = np.mean(first == 1)
    p_second = np.mean(second == 1)
    expected = p_first * p_second + (1 - p_first) * (1 - p_second)
    return float((observed - expected) / (1 - expected)) if expected < 1 else np.nan


def bootstrap_geometry(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    records = []
    for encoder, group in frame.groupby("encoder_id", sort=True):
        observed = {
            "spearman": float(spearmanr(group.human_style_ratio, group.projection_raw).statistic),
            "pearson": float(pearsonr(group.human_style_ratio, group.projection_raw).statistic),
            "mae_human_ratio": float(np.abs(group.projection_raw - group.human_style_ratio).mean()),
        }
        scenarios = [(scenario, values) for scenario, values in group.groupby("scenario_id", sort=True)]
        distributions = {metric: [] for metric in observed}
        for _ in range(iterations):
            sample = pd.concat([scenarios[index][1] for index in random.integers(0, len(scenarios), len(scenarios))])
            distributions["spearman"].append(spearmanr(sample.human_style_ratio, sample.projection_raw).statistic)
            distributions["pearson"].append(pearsonr(sample.human_style_ratio, sample.projection_raw).statistic)
            distributions["mae_human_ratio"].append(np.abs(sample.projection_raw - sample.human_style_ratio).mean())
        for metric, estimate in observed.items():
            records.append(
                {
                    "encoder_id": encoder,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": float(np.nanpercentile(distributions[metric], 2.5)),
                    "ci_upper": float(np.nanpercentile(distributions[metric], 97.5)),
                    "bootstrap_unit": "scenario_id",
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=REVISION_ROOT / "human_eval/human_results.csv")
    parser.add_argument("--map", type=Path, default=REVISION_ROOT / "human_eval/randomization_map.csv")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    output = REVISION_ROOT / "statistics/human_evaluation_results.txt"
    ratings = pd.read_csv(args.results)
    if not ratings[SCORE_COLUMNS].notna().any().any():
        output.write_text(
            "status=external_pending\nNo real human ratings are present. No scores were simulated and no human-machine agreement statistic was computed.\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "external_pending", "formal_rows_expected": 960}))
        return

    ratings["unable"] = binary(ratings["unable_to_judge_yes_no"])
    attention = ratings[ratings.attention_check == "yes"].copy()
    attention["passed"] = binary(attention["unable_to_judge_yes_no"]) == 1
    attention_failures = attention.groupby("rater_slot")["passed"].apply(lambda values: int((~values).sum()))
    excluded_slots = set(attention_failures[attention_failures >= 2].index)
    formal = ratings[(ratings.attention_check == "no") & ~ratings.rater_slot.isin(excluded_slots)].copy()
    formal = formal[formal.unable != 1]
    required = ["anonymous_rater_id", "style_B_intensity_0_100", "content_coverage_1_5", "naturalness_1_5", "adds_key_information_yes_no", "contradicts_source_yes_no"]
    formal = formal.dropna(subset=required)
    formal["adds"] = binary(formal["adds_key_information_yes_no"])
    formal["contradicts"] = binary(formal["contradicts_source_yes_no"])
    formal["speed_anomaly"] = pd.to_numeric(formal.duration_seconds, errors="coerce") < 5
    duplicates = formal.duplicated(["anonymous_rater_id", "item_id"]).sum()
    if duplicates:
        raise ValueError(f"duplicate rater/item responses: {duplicates}")

    reliability = []
    for metric in ("style_B_intensity_0_100", "content_coverage_1_5", "naturalness_1_5"):
        matrix = formal.pivot(index="item_id", columns="rater_slot", values=metric).dropna().to_numpy(float)
        single, average = icc_absolute(matrix)
        reliability.append({"metric": metric, "statistic": "ICC(A,1)", "estimate": single, "item_count": len(matrix)})
        reliability.append({"metric": metric, "statistic": "ICC(A,k)", "estimate": average, "item_count": len(matrix)})
        reliability.append({"metric": metric, "statistic": "Krippendorff_alpha_interval", "estimate": krippendorff_interval(matrix), "item_count": len(matrix)})
    for metric in ("adds", "contradicts"):
        matrix = formal.pivot(index="item_id", columns="rater_slot", values=metric).to_numpy(float)
        reliability.append({"metric": metric, "statistic": "Fleiss_kappa", "estimate": fleiss_kappa(matrix), "item_count": len(matrix)})
    pd.DataFrame(reliability).to_csv(REVISION_ROOT / "statistics/human_rater_reliability.csv", index=False)

    item = formal.groupby("item_id", as_index=False).agg(
        human_style_intensity=("style_B_intensity_0_100", "mean"),
        human_content_coverage=("content_coverage_1_5", "mean"),
        human_naturalness=("naturalness_1_5", "mean"),
        human_adds=("adds", lambda values: float(values.mean() >= 0.5)),
        human_contradicts=("contradicts", lambda values: float(values.mean() >= 0.5)),
        valid_raters=("anonymous_rater_id", "nunique"),
    )
    item["human_style_ratio"] = item.human_style_intensity / 100
    item["human_content_pass"] = (item.human_content_coverage >= 4.5) & (item.human_adds == 0) & (item.human_contradicts == 0)
    mapping = pd.read_csv(args.map)
    item = item.merge(mapping, on="item_id", validate="one_to_one")
    sample = pd.read_parquet(REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    human_geometry = item.merge(sample[["run_id", "encoder_id", "projection_raw"]], on="run_id", validate="one_to_many")
    geometry = bootstrap_geometry(human_geometry, args.iterations, args.seed)
    geometry.to_csv(REVISION_ROOT / "statistics/human_geometry_agreement.csv", index=False)
    axis = human_geometry.groupby(["style_axis", "encoder_id"]).apply(
        lambda group: pd.Series(
            {
                "spearman": spearmanr(group.human_style_ratio, group.projection_raw).statistic,
                "pearson": pearsonr(group.human_style_ratio, group.projection_raw).statistic,
                "mae_human_ratio": np.abs(group.projection_raw - group.human_style_ratio).mean(),
                "item_count": len(group),
            }
        ),
        include_groups=False,
    ).reset_index()
    axis.to_csv(REVISION_ROOT / "statistics/human_geometry_by_axis.csv", index=False)

    evaluation_path = REVISION_ROOT / "metrics/llm_evaluations.jsonl"
    if evaluation_path.exists():
        automatic = pd.DataFrame(read_jsonl_tolerant(evaluation_path).rows)
        compare = item.merge(automatic[["run_id", "content_preservation_pass"]], on="run_id", validate="one_to_one")
        truth = compare.human_content_pass.astype(int).to_numpy()
        prediction = compare.content_preservation_pass.astype(int).to_numpy()
        tp = int(((truth == 1) & (prediction == 1)).sum())
        tn = int(((truth == 0) & (prediction == 0)).sum())
        fp = int(((truth == 0) & (prediction == 1)).sum())
        fn = int(((truth == 1) & (prediction == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        pd.DataFrame(
            [{
                "item_count": len(compare), "true_positive": tp, "true_negative": tn,
                "false_positive": fp, "false_negative": fn,
                "accuracy": (tp + tn) / len(compare), "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
                "cohen_kappa": cohen_kappa(truth, prediction),
            }]
        ).to_csv(REVISION_ROOT / "statistics/automatic_checker_human_validation.csv", index=False)
    lines = [
        "status=complete",
        f"formal_rows_included={len(formal)}",
        f"unique_items={item.item_id.nunique()}",
        f"excluded_rater_slots={','.join(sorted(excluded_slots)) or 'none'}",
        f"speed_anomalies={int(formal.speed_anomaly.sum())}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "formal_rows": len(formal), "items": len(item)}))


if __name__ == "__main__":
    main()
