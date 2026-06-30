from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "human_eval" / "human_validation_sample_v1.jsonl"
ROOT_DOC_PATH = ROOT / "docs" / "human_validation_protocol.md"
FINAL_DOC_PATH = ROOT / "results" / "main8axis_v1_final_eval" / "docs" / "human_validation_protocol.md"

GENERATION_FILES = {
    "qwen35_4b": ROOT / "data/generations/qwen35_4b/main8axis_v1_4b.jsonl",
    "qwen35_9b": ROOT / "data/generations/qwen35_9b/main8axis_v1_9b.jsonl",
    "qwen35_35b_a3b": ROOT / "data/generations/qwen35_35b_a3b/main8axis_v1_35b.jsonl",
}

PROJECTION_FILES = {
    "bge_m3": {
        "qwen35_4b": ROOT / "data/metrics/projection_bge_m3_main8axis_v1_4b.jsonl",
        "qwen35_9b": ROOT / "data/metrics/projection_bge_m3_main8axis_v1_9b.jsonl",
        "qwen35_35b_a3b": ROOT / "data/metrics/projection_bge_m3_main8axis_v1_35b.jsonl",
    },
    "text2vec": {
        "qwen35_4b": ROOT / "data/metrics/projection_text2vec_main8axis_v1_4b.jsonl",
        "qwen35_9b": ROOT / "data/metrics/projection_text2vec_main8axis_v1_9b.jsonl",
        "qwen35_35b_a3b": ROOT / "data/metrics/projection_text2vec_main8axis_v1_35b.jsonl",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_generations() -> pd.DataFrame:
    rows = []
    for path in GENERATION_FILES.values():
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def load_projection_errors() -> pd.DataFrame:
    frames = []
    for embedding, by_model in PROJECTION_FILES.items():
        for path in by_model.values():
            df = pd.DataFrame(read_jsonl(path))
            df["embedding"] = embedding
            df[f"{embedding}_projection_error"] = (df["projection_ratio_clipped"].astype(float) - df["alpha"].astype(float)).abs()
            keep = ["run_id", f"{embedding}_projection_error", "projection_ratio_clipped", "off_axis_drift", "nearest_neighbor_rank"]
            renamed = df[keep].rename(
                columns={
                    "projection_ratio_clipped": f"{embedding}_r_proj_clipped",
                    "off_axis_drift": f"{embedding}_off_axis_drift",
                    "nearest_neighbor_rank": f"{embedding}_nearest_neighbor_rank",
                }
            )
            frames.append(renamed)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="run_id", how="outer")
    out["mean_projection_error"] = out[["bge_m3_projection_error", "text2vec_projection_error"]].mean(axis=1)
    return out


def build_sample() -> list[dict[str, Any]]:
    gens = load_generations()
    metrics = load_projection_errors()
    df = gens.merge(metrics, on="run_id", how="left")
    selected = []
    group_cols = ["model", "axis_id", "alpha"]
    for _, group in df.groupby(group_cols, sort=True):
        sorted_group = group.sort_values(["mean_projection_error", "scenario_id", "seed"])
        low = sorted_group.head(1).copy()
        high = sorted_group.tail(1).copy()
        low["selection_reason"] = "low_mean_projection_error_within_model_axis_alpha"
        high["selection_reason"] = "high_mean_projection_error_within_model_axis_alpha"
        selected.extend([low.iloc[0].to_dict(), high.iloc[0].to_dict()])

    rows = []
    for i, row in enumerate(selected, start=1):
        rows.append(
            {
                "sample_id": f"human_val_v1_{i:04d}",
                "run_id": row["run_id"],
                "model": row["model"],
                "axis_id": row["axis_id"],
                "scenario_id": row["scenario_id"],
                "alpha": float(row["alpha"]),
                "seed": int(row["seed"]),
                "content": row["content"],
                "style_a": row["style_a"],
                "style_b": row["style_b"],
                "attribute_name": row.get("attribute_name"),
                "content_checks": row.get("content_checks", []),
                "length_range_zh_chars": row.get("length_range_zh_chars", []),
                "output": row["output"],
                "bge_m3_projection_error": float(row["bge_m3_projection_error"]),
                "text2vec_projection_error": float(row["text2vec_projection_error"]),
                "mean_projection_error": float(row["mean_projection_error"]),
                "bge_m3_r_proj_clipped": float(row["bge_m3_r_proj_clipped"]),
                "text2vec_r_proj_clipped": float(row["text2vec_r_proj_clipped"]),
                "bge_m3_off_axis_drift": float(row["bge_m3_off_axis_drift"]),
                "text2vec_off_axis_drift": float(row["text2vec_off_axis_drift"]),
                "selection_reason": row["selection_reason"],
                "annotation_fields": {
                    "style_b_ratio_0_to_100": None,
                    "content_preservation": "yes/no/partial",
                    "naturalness_1_to_5": None,
                    "optional_pairwise_preference": None,
                    "notes": None,
                },
            }
        )
    return rows


def protocol_text(n: int) -> str:
    return f"""# Human Validation Protocol v1

## Purpose

Validate whether embedding projection scores align with human perception of explicit Style B ratio, while separately checking content preservation and naturalness.

## Sample

- File: `data/human_eval/human_validation_sample_v1.jsonl`
- Size: {n} outputs.
- Coverage: 3 models, 8 axes, 7 alpha ratios.
- Selection: for each `{{model, axis, alpha}}`, one low-error and one high-error sample by mean bge-m3/text2vec projection error.

## Annotation Fields

Annotate each row independently:

- `style_b_ratio_0_to_100`: perceived percentage of Style B in the output. Use 0 for pure Style A and 100 for pure Style B.
- `content_preservation`: `yes`, `partial`, or `no`.
- `naturalness_1_to_5`: 1 means unnatural or broken; 5 means fluent and natural.
- `notes`: short optional explanation for ambiguous cases.

Optional pairwise follow-up:

- If two outputs are shown for the same scenario, choose which one is closer to the requested Style B ratio.

## Annotator Instructions

Read the core content, Style A, Style B, target alpha, and model output. Judge style ratio based on the output text only. Do not reward length by itself unless length is part of the style definition. Penalize content drift in `content_preservation`, not in the style ratio field.

## Analysis Plan

1. Correlate human `style_b_ratio_0_to_100 / 100` with bge-m3 and text2vec `r_proj_clipped`.
2. Compare high-error and low-error subsets to estimate whether projection errors are perceptually meaningful.
3. Recompute model ranking on human-ratio absolute error for this sample.
4. Report content preservation and naturalness separately from style calibration.
"""


def main() -> None:
    rows = build_sample()
    write_jsonl(OUT_PATH, rows)
    text = protocol_text(len(rows))
    ROOT_DOC_PATH.write_text(text, encoding="utf-8")
    FINAL_DOC_PATH.write_text(text, encoding="utf-8")
    print(f"human_validation_rows={len(rows)} out={OUT_PATH}")


if __name__ == "__main__":
    main()
