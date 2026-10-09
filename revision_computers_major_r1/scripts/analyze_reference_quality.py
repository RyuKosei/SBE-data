from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_write_json, read_jsonl, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    references = pd.DataFrame(read_jsonl(args.output_root / "data/gpt_fixed_references.jsonl"))
    evaluations = pd.DataFrame(read_jsonl(args.output_root / "metrics/qwen_reference_evaluations.jsonl"))
    if references.empty:
        raise ValueError("no fixed references")
    references["run_id"] = references["reference_id"]
    references["character_length"] = references.reference_text.astype(str).str.len()
    references["chinese_character_count"] = references.reference_text.astype(str).map(
        lambda text: sum("\u4e00" <= char <= "\u9fff" for char in text)
    )
    references["length_min"] = references.length_range_zh_chars.map(lambda value: int(value[0]))
    references["length_max"] = references.length_range_zh_chars.map(lambda value: int(value[1]))
    references["length_pass"] = references.character_length.between(references.length_min, references.length_max)
    judge_columns = [
        "run_id",
        "content_coverage_rate",
        "content_preservation_pass",
        "style_B_intensity",
        "naturalness",
        "unjudgeable",
        "factual_contradiction",
        "task_result_changed_by_addition",
    ]
    available = [column for column in judge_columns if column in evaluations.columns]
    sample = references.merge(evaluations[available], on="run_id", how="left", validate="one_to_one")
    sample["auto_screened"] = sample.style_B_intensity.notna() if "style_B_intensity" in sample else False
    sample["human_verified"] = False
    sample["approval_status"] = None
    metrics_dir = args.output_root / "metrics"
    tables_dir = args.output_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(metrics_dir / "reference_quality_sample.parquet", index=False)

    base = sample[["scenario_id", "style_axis", "endpoint", "reference_id", "character_length", "length_pass"]].copy()
    for column in ("content_preservation_pass", "style_B_intensity", "naturalness"):
        base[column] = sample[column] if column in sample else pd.NA
    a = base[base.endpoint == "A"].set_index("scenario_id")
    b = base[base.endpoint == "B"].set_index("scenario_id")
    scenario = pd.DataFrame({
        "scenario_id": a.index,
        "style_axis": a.style_axis,
        "reference_id_A": a.reference_id,
        "reference_id_B": b.reference_id,
        "character_length_A": a.character_length,
        "character_length_B": b.character_length,
        "length_pass_A": a.length_pass,
        "length_pass_B": b.length_pass,
        "content_pass_A": a.content_preservation_pass,
        "content_pass_B": b.content_preservation_pass,
        "style_intensity_A": a.style_B_intensity,
        "style_intensity_B": b.style_B_intensity,
        "style_span_B_minus_A": b.style_B_intensity - a.style_B_intensity,
        "both_content_pass": a.content_preservation_pass.fillna(False).astype(bool) & b.content_preservation_pass.fillna(False).astype(bool),
        "both_length_pass": a.length_pass.astype(bool) & b.length_pass.astype(bool),
        "reference_status": "auto_generated",
        "auto_screened": a.style_B_intensity.notna() & b.style_B_intensity.notna(),
        "human_verified": False,
    }).reset_index(drop=True)
    scenario.to_parquet(metrics_dir / "reference_quality_scenario.parquet", index=False)
    reference_geometry: dict[str, float | int] = {}
    for encoder_id in ("bge_m3", "text2vec_base_chinese", "multilingual_e5_large_instruct"):
        archive_path = args.output_root / "embeddings" / encoder_id / "gpt_references.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            vector_by_id = {str(identifier): vector for identifier, vector in zip(archive["ids"].tolist(), archive["vectors"])}
        separations = np.asarray([
            np.linalg.norm(vector_by_id[row.reference_id_B] - vector_by_id[row.reference_id_A])
            for row in scenario.itertuples()
        ], dtype=np.float64)
        computable = separations > 1e-12
        reference_geometry.update({
            f"{encoder_id}_reference_span_mean": float(separations[computable].mean()) if computable.any() else np.nan,
            f"{encoder_id}_reference_span_median": float(np.median(separations[computable])) if computable.any() else np.nan,
            f"{encoder_id}_computable_pair_count": int(computable.sum()),
            f"{encoder_id}_collapsed_pair_count": int((~computable).sum()),
        })

    summary = pd.DataFrame([
        {
            "reference_source_model": references.reference_source_model.iloc[0],
            "reference_texts": len(sample),
            "scenario_pairs": len(scenario),
            "qwen_judge_valid": int(sample.get("style_B_intensity", pd.Series(dtype=float)).notna().sum()),
            "qwen_judge_failure_count": int(sample.get("style_B_intensity", pd.Series(dtype=float)).isna().sum()),
            "length_pass_rate": sample.length_pass.mean(),
            "content_pass_rate": sample.get("content_preservation_pass", pd.Series(dtype=float)).mean(),
            "pair_both_length_pass_count": int(scenario.both_length_pass.sum()),
            "pair_both_content_pass_count": int(scenario.both_content_pass.sum()),
            "pair_both_auto_screened_count": int(scenario.auto_screened.sum()),
            "mean_style_span_B_minus_A": scenario.style_span_B_minus_A.mean(),
            "median_style_span_B_minus_A": scenario.style_span_B_minus_A.median(),
            "nonpositive_style_span_count": int((scenario.style_span_B_minus_A <= 0).sum()),
            **reference_geometry,
            "human_verified": False,
        }
    ])
    summary.to_csv(tables_dir / "table_reference_quality.csv", index=False)
    (tables_dir / "table_reference_quality.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    status = {
        "generated_at": utc_now(),
        "reference_rows": len(sample),
        "scenario_pairs": len(scenario),
        "qwen_judge_valid": int(sample.get("style_B_intensity", pd.Series(dtype=float)).notna().sum()),
        "auto_generated": True,
        "auto_screened": bool(scenario.auto_screened.all()),
        "human_verified": False,
        "approval_status": None,
    }
    atomic_write_json(args.output_root / "manifests/reference_quality.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
