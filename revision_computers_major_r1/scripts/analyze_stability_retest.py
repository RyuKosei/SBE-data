from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import atomic_write_json, read_jsonl, utc_now


def summarize(original: pd.DataFrame, retest: pd.DataFrame, judge: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DataFrame(read_jsonl(ARGS.output_root / "data/stability_retest_index.jsonl"))
    fields = ["evaluation_item_id", "style_B_intensity", "naturalness", "content_preservation_pass"]
    original = original[fields].rename(columns={"evaluation_item_id": "original_evaluation_item_id", **{column: f"original_{column}" for column in fields[1:]}})
    retest = retest[fields].rename(columns={column: f"retest_{column}" for column in fields if column != "evaluation_item_id"})
    merged = index.merge(original, on="original_evaluation_item_id", how="left", validate="one_to_one")
    merged = merged.merge(retest, on="evaluation_item_id", how="left", validate="one_to_one")
    valid = merged.dropna(subset=["original_style_B_intensity", "retest_style_B_intensity"])
    summary = pd.DataFrame([{
        "judge": judge,
        "selected": len(merged),
        "matched_valid": len(valid),
        "intensity_mean_absolute_difference": (valid.original_style_B_intensity - valid.retest_style_B_intensity).abs().mean(),
        "intensity_pearson": pearsonr(valid.original_style_B_intensity, valid.retest_style_B_intensity).statistic if len(valid) > 2 else float("nan"),
        "intensity_spearman": spearmanr(valid.original_style_B_intensity, valid.retest_style_B_intensity).statistic if len(valid) > 2 else float("nan"),
        "naturalness_mean_absolute_difference": (valid.original_naturalness - valid.retest_naturalness).abs().mean(),
        "content_pass_agreement": (valid.original_content_preservation_pass == valid.retest_content_preservation_pass).mean(),
    }])
    merged["judge"] = judge
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    global ARGS
    ARGS = parser.parse_args()
    qwen_original = pd.DataFrame(read_jsonl(ARGS.output_root / "metrics/qwen_judge_evaluations.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    gpt_original = pd.DataFrame(read_jsonl(ARGS.output_root / "metrics/gpt_judge_evaluations.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    qwen_retest = pd.DataFrame(read_jsonl(ARGS.output_root / "metrics/qwen_stability_retest.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    gpt_retest = pd.DataFrame(read_jsonl(ARGS.output_root / "metrics/gpt_stability_retest.jsonl")).drop_duplicates("evaluation_item_id", keep="last")
    qwen_rows, qwen_summary = summarize(qwen_original, qwen_retest, "qwen")
    gpt_rows, gpt_summary = summarize(gpt_original, gpt_retest, "gpt")
    sample = pd.concat([qwen_rows, gpt_rows], ignore_index=True)
    summary = pd.concat([qwen_summary, gpt_summary], ignore_index=True)
    sample.to_parquet(ARGS.output_root / "metrics/judge_stability_retest_sample.parquet", index=False)
    summary.to_csv(ARGS.output_root / "tables/table_judge_stability_retest.csv", index=False)
    (ARGS.output_root / "tables/table_judge_stability_retest.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    status = {"generated_at": utc_now(), "selected_per_judge": len(qwen_rows), "qwen_valid": int(qwen_summary.matched_valid.iloc[0]), "gpt_valid": int(gpt_summary.matched_valid.iloc[0])}
    atomic_write_json(ARGS.output_root / "manifests/judge_stability_retest.json", status)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
