"""Apply the preregistered failure taxonomy after thresholds were frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


FAILURES = [
    "early_overshoot",
    "late_under_response",
    "non_monotonic_reversal",
    "near_constant_response",
    "endpoint_collapse",
    "out_of_range_extrapolation",
    "high_off_axis_drift",
    "curved_trajectory",
    "length_dominated_control",
    "content_style_entanglement",
]


def correlations(group: pd.DataFrame) -> float:
    if group["target_ratio"].std() == 0 or group["character_length"].std() == 0:
        return np.nan
    return float(group["target_ratio"].corr(group["character_length"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    parser.add_argument("--generation", type=Path, default=REVISION_ROOT / "data/qwen_family_outputs.jsonl")
    parser.add_argument("--llm-evaluation", type=Path, default=REVISION_ROOT / "metrics/llm_evaluations.jsonl")
    parser.add_argument("--thresholds", type=Path, default=REVISION_ROOT / "config/failure_thresholds.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    sample = pd.read_parquet(args.sample)
    trajectory_paths = sorted((REVISION_ROOT / "metrics/fragments").glob("*/*.trajectory.parquet"))
    trajectories = pd.concat([pd.read_parquet(path) for path in trajectory_paths], ignore_index=True)

    keys = ["scenario_id", "style_axis", "model_id", "encoder_id"]
    cell_keys = keys + ["target_ratio"]
    cells = sample.groupby(cell_keys, as_index=False).agg(
        projection_raw=("projection_raw", "mean"),
        projection_error_raw=("projection_error_raw", "mean"),
        off_axis_drift=("off_axis_drift_normalized", "mean"),
        character_length=("character_length", "mean"),
        out_of_range=("out_of_range", "max"),
        endpoint_separation=("endpoint_separation", "mean"),
    )
    trajectory = trajectories.groupby(keys, as_index=False).agg(
        path_length_ratio=("trajectory_ordered_path_length_ratio", "mean"),
        endpoint_pc1_abs_cosine=("trajectory_endpoint_pc1_abs_cosine", "mean"),
    )
    records: list[dict[str, object]] = []
    for values, group in cells.groupby(keys, sort=True):
        group = group.sort_values("target_ratio")
        interior = group[(group.target_ratio > 0) & (group.target_ratio < 1)]
        lookup = dict(zip(group.target_ratio.round(10), group.projection_raw))
        early = any(
            lookup.get(round(alpha, 10), np.nan) - alpha >= float(config["early_overshoot"]["threshold"])
            for alpha in (1 / 6, 2 / 6)
        )
        late = any(
            lookup.get(round(alpha, 10), np.nan) - alpha <= -float(config["late_under_response"]["threshold"])
            for alpha in (4 / 6, 5 / 6)
        )
        projection = group.projection_raw.to_numpy(float)
        records.append(
            {
                **dict(zip(keys, values)),
                "interior_calibration_error": float(interior.projection_error_raw.mean()),
                "mean_off_axis_drift": float(interior.off_axis_drift.mean()),
                "endpoint_separation": float(group.endpoint_separation.mean()),
                "length_target_correlation": correlations(group),
                "early_overshoot": bool(early),
                "late_under_response": bool(late),
                "non_monotonic_reversal": bool((np.diff(projection) < float(config["non_monotonic_reversal"]["threshold"])).any()),
                "near_constant_response": bool(np.ptp(interior.projection_raw) <= float(config["near_constant_response"]["threshold"])),
                "out_of_range_extrapolation": bool(interior.out_of_range.any()),
            }
        )
    classified = pd.DataFrame(records).merge(trajectory, on=keys, validate="one_to_one")

    classified["endpoint_collapse"] = False
    classified["high_off_axis_drift"] = False
    classified["curved_trajectory"] = False
    for encoder, indexes in classified.groupby("encoder_id").groups.items():
        separation_cut = np.nanpercentile(
            classified.loc[indexes, "endpoint_separation"],
            float(config["endpoint_collapse"]["percentile"]),
        )
        drift_cut = np.nanpercentile(
            classified.loc[indexes, "mean_off_axis_drift"],
            float(config["high_off_axis_drift"]["percentile"]),
        )
        path_cut = np.nanpercentile(
            classified.loc[indexes, "path_length_ratio"],
            float(config["curved_trajectory"]["path_length_percentile"]),
        )
        classified.loc[indexes, "endpoint_collapse"] = classified.loc[indexes, "endpoint_separation"] <= separation_cut
        classified.loc[indexes, "high_off_axis_drift"] = classified.loc[indexes, "mean_off_axis_drift"] >= drift_cut
        classified.loc[indexes, "curved_trajectory"] = (
            (classified.loc[indexes, "path_length_ratio"] >= path_cut)
            | (classified.loc[indexes, "endpoint_pc1_abs_cosine"] < float(config["curved_trajectory"]["endpoint_pc1_abs_cosine"]))
        )
    medians = classified.groupby(["model_id", "encoder_id"])["interior_calibration_error"].transform("median")
    classified["length_dominated_control"] = (
        classified["length_target_correlation"].abs() >= float(config["length_dominated_control"]["absolute_correlation"])
    ) & (classified["interior_calibration_error"] > medians)

    classified["content_style_entanglement"] = False
    llm_status = "external_pending"
    if args.llm_evaluation.exists():
        evaluations = read_jsonl_tolerant(args.llm_evaluation)
        if not evaluations.bad_lines:
            evaluation_frame = pd.DataFrame(evaluations.rows)
            expected_ids = set(sample.run_id)
            if expected_ids.issubset(set(evaluation_frame.run_id)):
                joined = sample[sample.is_interior].merge(
                    evaluation_frame[["run_id", "content_preservation_pass"]],
                    on="run_id",
                    validate="many_to_one",
                )
                joined["entangled"] = (
                    ~joined["content_preservation_pass"].astype(bool)
                ) & (joined["projection_error_raw"] < float(config["content_style_entanglement"]["projection_error_upper"]))
                flags = joined.groupby(keys, as_index=False)["entangled"].max()
                classified = classified.drop(columns="content_style_entanglement").merge(flags, on=keys, how="left", validate="one_to_one")
                classified = classified.rename(columns={"entangled": "content_style_entanglement"})
                classified["content_style_entanglement"] = classified["content_style_entanglement"].fillna(False)
                llm_status = "complete"

    output_path = REVISION_ROOT / "metrics/failure_classifications.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classified.to_parquet(output_path, index=False)
    count_records: list[dict[str, object]] = []
    for failure in FAILURES:
        for level, columns in [("overall", []), ("model", ["model_id"]), ("style_axis", ["style_axis"]), ("encoder", ["encoder_id"])]:
            iterator = [((), classified)] if not columns else classified.groupby(columns, sort=True)
            for values, group in iterator:
                if columns and not isinstance(values, tuple):
                    values = (values,)
                record: dict[str, object] = {
                    "failure_type": failure,
                    "aggregation": level,
                    "count": int(group[failure].sum()),
                    "total": len(group),
                    "proportion": float(group[failure].mean()),
                }
                for column, value in zip(columns, values):
                    record[column] = value
                count_records.append(record)
    counts = pd.DataFrame(count_records)
    counts.to_csv(REVISION_ROOT / "metrics/failure_taxonomy_counts.csv", index=False)

    generations = read_jsonl_tolerant(args.generation)
    generation_frame = pd.DataFrame(generations.rows)[["scenario_id", "model_id", "target_ratio", "generated_text"]]
    representative = generation_frame[np.isclose(generation_frame.target_ratio, 0.5)].drop_duplicates(["scenario_id", "model_id"])
    cases: list[pd.DataFrame] = []
    for failure in FAILURES:
        positive = classified[classified[failure]].sort_values("interior_calibration_error", ascending=False).head(2).assign(case_label="positive")
        negative = classified[~classified[failure]].sort_values("interior_calibration_error").head(2).assign(case_label="negative")
        selected = pd.concat([positive, negative], ignore_index=True)
        selected["failure_type"] = failure
        cases.append(selected.merge(representative, on=["scenario_id", "model_id"], how="left"))
    case_frame = pd.concat(cases, ignore_index=True)
    case_path = REVISION_ROOT / "cases/failure_cases.csv"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_frame.to_csv(case_path, index=False)

    overall = counts[counts.aggregation == "overall"].sort_values("proportion", ascending=False)
    unexpected = overall.iloc[0]
    lines = [
        "# Failure taxonomy results",
        "",
        f"Threshold configuration: `{args.thresholds}` (frozen before new-model ranking).",
        f"Content-style entanglement status: **{llm_status}**.",
        "",
        "| Failure | Count | Proportion |",
        "|---|---:|---:|",
    ]
    for row in overall.itertuples():
        lines.append(f"| {row.failure_type} | {row.count} / {row.total} | {row.proportion:.1%} |")
    lines.extend(
        [
            "",
            "## Unexpected negative finding",
            "",
            f"The most prevalent preregistered flag was `{unexpected.failure_type}` ({unexpected.proportion:.1%} of scenario-model-encoder cells). This prevents interpreting a favorable average ICE as evidence that trajectories are generally clean or one-dimensional.",
            "",
            "Representative positive and negative cases, including the midpoint text, are stored in `cases/failure_cases.csv`.",
        ]
    )
    (REVISION_ROOT / "cases/failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(classified), "content_status": llm_status, "counts_rows": len(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
