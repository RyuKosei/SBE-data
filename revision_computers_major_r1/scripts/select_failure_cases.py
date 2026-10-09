from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import atomic_write_json, read_jsonl, utc_now


CATEGORY_LIMIT = 2
TOTAL_LIMIT = 12
THRESHOLDS = {
    "style_mae": 0.40,
    "judge_intensity_difference": 30.0,
    "anchor_coordinate_difference": 0.40,
    "nonmonotonic_spearman": 0.30,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario-file", type=Path, required=True)
    args = parser.parse_args()
    geometry = pd.read_parquet(args.output_root / "metrics/geometry/bge_m3/main_sample.parquet")
    values = geometry.pivot(index=["run_id", "scenario_id", "style_axis", "model_id", "replicate", "target_ratio", "character_length"], columns="anchor_protocol", values=["projection_raw", "projection_error_raw", "out_of_range"]).reset_index()
    values.columns = ["_".join([str(part) for part in column if part]) if isinstance(column, tuple) else column for column in values.columns]
    values = values.rename(columns={
        "projection_raw_self_anchored": "self_coordinate",
        "projection_raw_external_fixed_reference": "external_coordinate",
        "projection_error_raw_self_anchored": "self_ice",
        "projection_error_raw_external_fixed_reference": "external_ice",
        "out_of_range_external_fixed_reference": "external_out_of_range",
    })
    judge = pd.read_parquet(args.output_root / "metrics/dual_judge_sample.parquet")
    judge = judge[judge.source_type.isin(["main_qwen", "main_external"])].drop_duplicates("run_id")
    judge_columns = [
        "run_id", "generated_text", "qwen_style_B_intensity", "gpt_style_B_intensity",
        "qwen_content_preservation_pass", "gpt_content_preservation_pass",
        "qwen_content_coverage_rate", "gpt_content_coverage_rate",
        "qwen_naturalness", "gpt_naturalness",
    ]
    data = values.merge(judge[[column for column in judge_columns if column in judge]], on="run_id", how="left", validate="one_to_one")
    scenarios = pd.DataFrame(read_jsonl(args.scenario_file))[["scenario_id", "length_range_zh_chars"]]
    data = data.merge(scenarios, on="scenario_id", how="left", validate="many_to_one")
    data["length_min"] = data.length_range_zh_chars.map(lambda value: value[0])
    data["length_max"] = data.length_range_zh_chars.map(lambda value: value[1])
    data["length_violation"] = np.maximum(data.length_min - data.character_length, 0) + np.maximum(data.character_length - data.length_max, 0)
    data["qwen_style_mae"] = (data.qwen_style_B_intensity / 100.0 - data.target_ratio).abs()
    data["gpt_style_mae"] = (data.gpt_style_B_intensity / 100.0 - data.target_ratio).abs()
    data["judge_intensity_difference"] = (data.qwen_style_B_intensity - data.gpt_style_B_intensity).abs()
    data["anchor_coordinate_difference"] = (data.self_coordinate - data.external_coordinate).abs()

    monotonic = []
    internal = data[(data.target_ratio > 0) & (data.target_ratio < 1)]
    for key, group in internal.groupby(["scenario_id", "model_id", "replicate"], sort=True):
        valid = group.dropna(subset=["external_coordinate"])
        rho = spearmanr(valid.target_ratio, valid.external_coordinate).statistic if len(valid) == 5 and valid.external_coordinate.nunique() > 1 else np.nan
        chosen = valid.loc[valid.external_ice.idxmax()] if len(valid) else None
        if chosen is not None:
            monotonic.append({"run_id": chosen.run_id, "trajectory_external_spearman": rho})
    data = data.merge(pd.DataFrame(monotonic), on="run_id", how="left", validate="one_to_one")

    definitions = [
        ("content_failure", data[(data.qwen_content_preservation_pass == False) | (data.gpt_content_preservation_pass == False)], lambda frame: (2 - frame.qwen_content_coverage_rate.fillna(1) - frame.gpt_content_coverage_rate.fillna(1))),
        ("style_under_or_overshoot", data[np.maximum(data.qwen_style_mae, data.gpt_style_mae) >= THRESHOLDS["style_mae"]], lambda frame: np.maximum(frame.qwen_style_mae, frame.gpt_style_mae)),
        ("external_out_of_range", data[data.external_out_of_range == True], lambda frame: np.maximum(-frame.external_coordinate, frame.external_coordinate - 1)),
        ("non_monotonic", data[data.trajectory_external_spearman < THRESHOLDS["nonmonotonic_spearman"]], lambda frame: THRESHOLDS["nonmonotonic_spearman"] - frame.trajectory_external_spearman),
        ("length_violation", data[data.length_violation > 0], lambda frame: frame.length_violation),
        ("judge_disagreement", data[data.judge_intensity_difference >= THRESHOLDS["judge_intensity_difference"]], lambda frame: frame.judge_intensity_difference),
        ("anchor_disagreement", data[data.anchor_coordinate_difference >= THRESHOLDS["anchor_coordinate_difference"]], lambda frame: frame.anchor_coordinate_difference),
    ]
    ranked_by_category: list[tuple[str, list[dict]]] = []
    for category, candidates, severity_function in definitions:
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["selection_severity"] = severity_function(candidates)
        candidates = candidates.sort_values(["selection_severity", "run_id"], ascending=[False, True])
        ranked_by_category.append((category, candidates.to_dict("records")))

    # First take one case from every available category so a 12-case cap
    # cannot systematically exclude the final (anchor-disagreement) category.
    # A second fixed-order pass supplies at most one additional case/category.
    selected: list[dict] = []
    used: set[str] = set()
    category_counts: dict[str, int] = {category: 0 for category, _ in ranked_by_category}
    for _pass in range(CATEGORY_LIMIT):
        for category, candidates in ranked_by_category:
            if len(selected) >= TOTAL_LIMIT or category_counts[category] >= CATEGORY_LIMIT:
                continue
            choice = next((row for row in candidates if row["run_id"] not in used), None)
            if choice is None:
                continue
            selected.append({"failure_category": category, **choice})
            used.add(choice["run_id"])
            category_counts[category] += 1
    frame = pd.DataFrame(selected)
    columns = [
        "failure_category", "run_id", "scenario_id", "style_axis", "model_id", "replicate", "target_ratio",
        "generated_text", "qwen_style_B_intensity", "gpt_style_B_intensity", "qwen_content_preservation_pass",
        "gpt_content_preservation_pass", "qwen_naturalness", "gpt_naturalness", "self_coordinate", "external_coordinate",
        "self_ice", "external_ice", "external_out_of_range", "trajectory_external_spearman", "character_length",
        "length_min", "length_max", "judge_intensity_difference", "anchor_coordinate_difference", "selection_severity",
    ]
    frame = frame[[column for column in columns if column in frame]]
    frame.to_csv(args.output_root / "tables/failure_cases.csv", index=False)
    lines = [
        "# Representative failure cases",
        "",
        "Cases are selected deterministically from the BGE-M3 main experiment using the predeclared categories and thresholds in `scripts/select_failure_cases.py`. The first pass takes one unique output from every available category; the second fixed-order pass adds at most one more per category up to 12 total. Within a category, candidates are sorted by severity then run ID. These are diagnostic examples, not a hand-picked success/failure narrative. Text is kept in the original Chinese; no translated text is used for metrics.",
        "",
    ]
    for row in frame.to_dict("records"):
        lines.extend([
            f"## {row['failure_category']}: `{row['run_id']}`",
            "",
            f"- scenario/model/replicate/alpha: `{row['scenario_id']}` / `{row['model_id']}` / {row['replicate']} / {row['target_ratio']:.6g}",
            f"- Qwen/GPT intensity: {row.get('qwen_style_B_intensity')} / {row.get('gpt_style_B_intensity')}; content pass: {row.get('qwen_content_preservation_pass')} / {row.get('gpt_content_preservation_pass')}; naturalness: {row.get('qwen_naturalness')} / {row.get('gpt_naturalness')}",
            f"- self/external coordinate: {row.get('self_coordinate'):.4f} / {row.get('external_coordinate'):.4f}; self/external ICE: {row.get('self_ice'):.4f} / {row.get('external_ice'):.4f}",
            f"- character length and allowed range: {row.get('character_length')} / [{row.get('length_min')}, {row.get('length_max')}]",
            f"- original Chinese output: {row.get('generated_text')}",
            "",
        ])
    (args.output_root / "reports/failure_cases.md").write_text("\n".join(lines), encoding="utf-8")
    status = {"generated_at": utc_now(), "selected_cases": len(frame), "category_counts": frame.failure_category.value_counts().to_dict(), "thresholds": THRESHOLDS, "human_selected": False}
    atomic_write_json(args.output_root / "manifests/failure_case_selection.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
