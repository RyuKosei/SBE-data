from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "main8axis_v1_final_eval"
TABLE_DIR = OUT_ROOT / "tables"
DOC_DIR = OUT_ROOT / "docs"


PAIRWISE_FILES = [
    ("qwen35_4b", "qwen35_35b_a3b", False, ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_4b.jsonl"),
    ("qwen35_4b", "qwen35_35b_a3b", True, ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_4b_swapped.jsonl"),
    ("qwen35_9b", "qwen35_35b_a3b", False, ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_9b.jsonl"),
    ("qwen35_9b", "qwen35_35b_a3b", True, ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_9b_swapped.jsonl"),
    ("qwen35_35b_a3b", "qwen35_9b", False, ROOT / "data/judgments/pairwise_qwen35_9b_main8axis_v1_sampled_35b.jsonl"),
    ("qwen35_35b_a3b", "qwen35_9b", True, ROOT / "data/judgments/pairwise_qwen35_9b_main8axis_v1_sampled_35b_swapped.jsonl"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def semantic_outcome(row: pd.Series) -> str:
    if not bool(row.get("parse_ok")):
        return "invalid"
    winner = row.get("winner")
    if winner == "tie":
        return "tie"
    winner_run_id = row.get(f"{winner}_run_id")
    if winner_run_id == row.get("right_run_id"):
        return "higher"
    if winner_run_id == row.get("left_run_id"):
        return "lower"
    return "invalid"


def load_pairwise() -> pd.DataFrame:
    frames = []
    for model, expected_judge, expected_swap, path in PAIRWISE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.DataFrame(read_jsonl(path))
        df["model"] = model
        df["expected_judge_model"] = expected_judge
        df["swap_order"] = expected_swap
        df["source_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["parse_ok"] = df["parse_ok"].astype(bool)
    df["left_alpha"] = df["left_alpha"].astype(float)
    df["right_alpha"] = df["right_alpha"].astype(float)
    df["pair_gap"] = df["right_alpha"] - df["left_alpha"]
    df["semantic_outcome"] = df.apply(semantic_outcome, axis=1)
    df["higher_alpha_win"] = df["semantic_outcome"].eq("higher")
    df["lower_alpha_win"] = df["semantic_outcome"].eq("lower")
    df["tie"] = df["semantic_outcome"].eq("tie")
    df["text_1_win"] = df["parse_ok"] & df["winner"].eq("text_1")
    df["text_2_win"] = df["parse_ok"] & df["winner"].eq("text_2")
    df["valid_non_tie"] = df["parse_ok"] & df["winner"].isin(["text_1", "text_2"])
    df["monotonic_violation"] = df["valid_non_tie"] & df["lower_alpha_win"]
    return df


def agg_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        valid = g[g["parse_ok"]]
        non_tie = g[g["valid_non_tie"]]
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n": int(len(g)),
                "parse_ok_rate": float(g["parse_ok"].mean()) if len(g) else np.nan,
                "tie_rate": float(valid["tie"].mean()) if len(valid) else np.nan,
                "higher_alpha_win_rate": float(valid["higher_alpha_win"].mean()) if len(valid) else np.nan,
                "monotonic_violation_rate": float(non_tie["monotonic_violation"].mean()) if len(non_tie) else np.nan,
                "text_1_win_rate": float(non_tie["text_1_win"].mean()) if len(non_tie) else np.nan,
                "text_2_win_rate": float(non_tie["text_2_win"].mean()) if len(non_tie) else np.nan,
                "position_bias_rate": float(abs(non_tie["text_1_win"].mean() - non_tie["text_2_win"].mean())) if len(non_tie) else np.nan,
                "mean_confidence": float(pd.to_numeric(valid["confidence"], errors="coerce").mean()) if len(valid) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_consistency(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "expected_judge_model", "axis_id", "scenario_id", "seed", "left_run_id", "right_run_id", "left_alpha", "right_alpha"]
    normal = df[~df["swap_order"]].copy()
    swapped = df[df["swap_order"]].copy()
    keep = keys + ["parse_ok", "winner", "semantic_outcome"]
    merged = normal[keep].merge(swapped[keep], on=keys, suffixes=("_normal", "_swapped"))
    merged["both_parse_ok"] = merged["parse_ok_normal"] & merged["parse_ok_swapped"]
    merged["bidirectional_consistent"] = merged["both_parse_ok"] & merged["semantic_outcome_normal"].eq(merged["semantic_outcome_swapped"])
    merged["contradiction"] = (
        merged["both_parse_ok"]
        & merged["semantic_outcome_normal"].isin(["higher", "lower"])
        & merged["semantic_outcome_swapped"].isin(["higher", "lower"])
        & ~merged["semantic_outcome_normal"].eq(merged["semantic_outcome_swapped"])
    )
    merged["same_position_winner"] = (
        merged["both_parse_ok"]
        & merged["winner_normal"].isin(["text_1", "text_2"])
        & merged["winner_normal"].eq(merged["winner_swapped"])
    )
    return merged


def consistency_agg(matched: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = matched.groupby(group_cols, dropna=False) if group_cols else [((), matched)]
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        both = g[g["both_parse_ok"]]
        row = dict(zip(group_cols, key))
        row.update(
            {
                "matched_pairs": int(len(g)),
                "both_parse_ok_rate": float(g["both_parse_ok"].mean()) if len(g) else np.nan,
                "bidirectional_consistency": float(both["bidirectional_consistent"].mean()) if len(both) else np.nan,
                "contradiction_rate": float(both["contradiction"].mean()) if len(both) else np.nan,
                "same_position_winner_rate": float(both["same_position_winner"].mean()) if len(both) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report_section(overall: pd.DataFrame, by_axis: pd.DataFrame, incons: pd.DataFrame) -> None:
    report_path = DOC_DIR / "main8axis_v1_final_analysis_report.md"
    report = report_path.read_text(encoding="utf-8")
    start = report.index("## Pairwise Audit")
    end = report.index("## Figure Paths")
    overall_md = overall.to_markdown(index=False)
    by_axis_brief = by_axis.sort_values(["model", "monotonic_violation_rate"], ascending=[True, False]).groupby("model").head(3)
    by_axis_md = by_axis_brief[
        ["model", "axis_id", "n", "higher_alpha_win_rate", "monotonic_violation_rate", "tie_rate", "bidirectional_consistency", "contradiction_rate"]
    ].to_markdown(index=False)
    incons_md = incons.to_markdown(index=False)
    section = f"""## Pairwise Audit

Sampled bidirectional pairwise audit is complete. The sampling uses 8 axes * 20 scenarios per axis * 3 seeds * 9 local alpha pairs * 2 orders for each model, for 25,920 total judgments. This differs from the instruction's approximate total of 8,640; the written factorial design implies 25,920.

Judge assignment is asymmetric: 4B and 9B outputs are judged by 35B-A3B, while 35B-A3B outputs are judged by 9B. Pairwise is therefore an auxiliary monotonicity and preference audit; projection geometry remains the primary metric.

{overall_md}

Worst axis-level monotonicity rows by model:

{by_axis_md}

Bidirectional inconsistency summary:

{incons_md}

"""
    report_path.write_text(report[:start] + section + report[end:], encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_pairwise()
    matched = matched_consistency(df)

    base = agg_metrics(df, ["model", "expected_judge_model"])
    cons = consistency_agg(matched, ["model", "expected_judge_model"])
    overall = base.merge(cons, on=["model", "expected_judge_model"], how="left")
    overall.to_csv(TABLE_DIR / "main_pairwise_sampled_metrics.csv", index=False)

    pos = agg_metrics(df, ["model", "expected_judge_model", "swap_order"])
    pos_cons = consistency_agg(matched, ["model", "expected_judge_model"])
    pos.to_csv(TABLE_DIR / "main_pairwise_position_bias.csv", index=False)

    incons = consistency_agg(matched, ["model", "expected_judge_model"])
    incons.to_csv(TABLE_DIR / "main_pairwise_inconsistency.csv", index=False)

    by_axis_base = agg_metrics(df, ["model", "expected_judge_model", "axis_id"])
    by_axis_cons = consistency_agg(matched, ["model", "expected_judge_model", "axis_id"])
    by_axis = by_axis_base.merge(by_axis_cons, on=["model", "expected_judge_model", "axis_id"], how="left")
    by_axis.to_csv(TABLE_DIR / "main_pairwise_by_axis_model.csv", index=False)

    write_report_section(overall, by_axis, incons)
    print(f"pairwise_rows={len(df)} matched_pairs={len(matched)} out={TABLE_DIR}")


if __name__ == "__main__":
    main()
