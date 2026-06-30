from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import ROOT, read_jsonl


MODEL_ORDER = ["qwen35_4b", "qwen35_9b", "qwen35_35b_a3b"]
MODEL_LABELS = {
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
}
MODEL_COLORS = {
    "qwen35_4b": "#6B7280",
    "qwen35_9b": "#2563EB",
    "qwen35_35b_a3b": "#F97316",
}
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
PAIRWISE_PATHS = [
    ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_4b.jsonl",
    ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_4b_swapped.jsonl",
    ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_9b.jsonl",
    ROOT / "data/judgments/pairwise_qwen35_35b_a3b_main8axis_v1_sampled_9b_swapped.jsonl",
    ROOT / "data/judgments/pairwise_qwen35_9b_main8axis_v1_sampled_35b.jsonl",
    ROOT / "data/judgments/pairwise_qwen35_9b_main8axis_v1_sampled_35b_swapped.jsonl",
]


def read_many(paths: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def ensure_dirs(out_dir: Path) -> None:
    for name in ["tables", "figures", "docs"]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def embedding_short(name: str) -> str:
    if "bge-m3" in name:
        return "bge_m3"
    if "text2vec" in name:
        return "text2vec"
    return Path(str(name).split()[0]).name.replace("-", "_")


def zh_char_count(text: object) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def length_bounds(value: object) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list) and len(loaded) >= 2:
                return float(loaded[0]), float(loaded[1])
        except json.JSONDecodeError:
            pass
    return float("nan"), float("nan")


def safe_spearman(df: pd.DataFrame, measured: str) -> float:
    if len(df) < 2 or df["alpha"].nunique() < 2:
        return float("nan")
    return float(df["alpha"].corr(df[measured], method="spearman"))


def smoothness(df: pd.DataFrame, measured: str) -> float:
    ordered = df.sort_values("alpha")
    values = ordered.groupby("alpha")[measured].mean().reindex(ALPHAS).dropna().to_numpy()
    if len(values) < 3:
        return float("nan")
    return float(np.mean(np.abs(values[2:] - 2 * values[1:-1] + values[:-2])))


def endpoint_attraction(df: pd.DataFrame, measured: str) -> float:
    middle = df[df["alpha"].isin([0.25, 0.5, 0.75])]
    if middle.empty:
        return float("nan")
    return float(((middle[measured] < 0.15) | (middle[measured] > 0.85)).mean())


def bootstrap_ci_by_unit(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    unit_col: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for keys, group in df.dropna(subset=[value_col]).groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        by_unit = [u[value_col].to_numpy(dtype=float) for _, u in group.groupby(unit_col)]
        if not by_unit:
            continue
        observed = float(np.mean(np.concatenate(by_unit)))
        draws = []
        for _ in range(iterations):
            sampled = [by_unit[idx] for idx in rng.integers(0, len(by_unit), size=len(by_unit))]
            draws.append(float(np.mean(np.concatenate(sampled))))
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "metric": value_col,
                "mean": observed,
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "bootstrap_unit": unit_col,
                "iterations": iterations,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_generation(gens: pd.DataFrame) -> pd.DataFrame:
    gens = gens.copy()
    gens["alpha"] = pd.to_numeric(gens["alpha"], errors="coerce")
    gens["seed"] = pd.to_numeric(gens["seed"], errors="coerce")
    gens["zh_chars"] = gens["output"].map(zh_char_count)
    bounds = gens["length_range_zh_chars"].map(length_bounds)
    gens["min_zh_chars"] = bounds.map(lambda x: x[0])
    gens["max_zh_chars"] = bounds.map(lambda x: x[1])
    gens["too_short"] = gens["zh_chars"] < gens["min_zh_chars"]
    gens["too_long"] = gens["zh_chars"] > gens["max_zh_chars"]
    gens["length_violation"] = gens["too_short"] | gens["too_long"]
    return gens


def prepare_projection(proj: pd.DataFrame) -> pd.DataFrame:
    proj = proj.copy()
    numeric_cols = [
        "alpha",
        "projection_ratio_raw",
        "projection_ratio_clipped",
        "interpolation_distance",
        "off_axis_drift",
        "endpoint_distance",
        "nearest_neighbor_rank",
    ]
    for col in numeric_cols:
        proj[col] = pd.to_numeric(proj[col], errors="coerce")
    proj["embedding"] = proj["embedding_model"].map(embedding_short)
    proj["r_proj"] = proj["projection_ratio_raw"]
    proj["r_proj_clipped"] = proj["projection_ratio_clipped"]
    proj["projection_error"] = (proj["r_proj_clipped"] - proj["alpha"]).abs()
    proj["projection_error_unclipped"] = (proj["r_proj"] - proj["alpha"]).abs()
    proj["normalized_off_axis_drift"] = proj["off_axis_drift"] / proj["endpoint_distance"].replace(0, np.nan)
    proj["out_of_range"] = (proj["r_proj"] < 0.0) | (proj["r_proj"] > 1.0)
    proj["nearest_neighbor_top1"] = proj["nearest_neighbor_rank"] == 1
    return proj


def prepare_scalar(scalar: pd.DataFrame) -> pd.DataFrame:
    scalar = scalar.copy()
    for col in ["alpha", "style_b_ratio", "content_preservation", "style_mixture_naturalness"]:
        scalar[col] = pd.to_numeric(scalar[col], errors="coerce")
    scalar["judge_error"] = (scalar["style_b_ratio"] - scalar["alpha"]).abs()
    return scalar


def prepare_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    pairwise = pairwise.copy()
    if pairwise.empty:
        return pairwise
    for col in ["left_alpha", "right_alpha", "confidence"]:
        pairwise[col] = pd.to_numeric(pairwise[col], errors="coerce")
    pairwise["parse_ok"] = pairwise["parse_ok"].fillna(False).astype(bool)
    pairwise["swap_order"] = pairwise["swap_order"].fillna(False).astype(bool)

    def canonical_winner(row: pd.Series) -> str:
        if row["winner"] == "tie":
            return "tie"
        if row["winner"] == "text_1":
            run_id = row["text_1_run_id"]
        elif row["winner"] == "text_2":
            run_id = row["text_2_run_id"]
        else:
            return "invalid"
        if run_id == row["left_run_id"]:
            return "left"
        if run_id == row["right_run_id"]:
            return "right"
        return "invalid"

    pairwise["canonical_winner"] = pairwise.apply(canonical_winner, axis=1)
    pairwise["high_alpha_winner"] = pairwise["canonical_winner"] == "right"
    pairwise["low_alpha_winner"] = pairwise["canonical_winner"] == "left"
    pairwise["tie_winner"] = pairwise["canonical_winner"] == "tie"
    pairwise["preference_score"] = np.where(
        pairwise["high_alpha_winner"],
        1.0,
        np.where(pairwise["tie_winner"], 0.5, np.where(pairwise["low_alpha_winner"], 0.0, np.nan)),
    )
    pairwise["text_1_win"] = pairwise["winner"] == "text_1"
    pairwise["text_2_win"] = pairwise["winner"] == "text_2"
    return pairwise


def summarize_projection(group: pd.DataFrame) -> dict:
    return {
        "n": len(group),
        "projection_mae": group["projection_error"].mean(),
        "projection_mae_unclipped": group["projection_error_unclipped"].mean(),
        "projection_bias": (group["r_proj_clipped"] - group["alpha"]).mean(),
        "projection_spearman": safe_spearman(group, "r_proj_clipped"),
        "endpoint_attraction_rate": endpoint_attraction(group, "r_proj_clipped"),
        "interpolation_distance_mean": group["interpolation_distance"].mean(),
        "off_axis_drift_mean": group["off_axis_drift"].mean(),
        "normalized_off_axis_drift_mean": group["normalized_off_axis_drift"].mean(),
        "out_of_range_rate": group["out_of_range"].mean(),
        "nearest_neighbor_top1_rate": group["nearest_neighbor_top1"].mean(),
        "nearest_neighbor_mean_rank": group["nearest_neighbor_rank"].mean(),
        "smoothness": smoothness(group, "r_proj_clipped"),
    }


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(summarize_projection(group))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairwise(group: pd.DataFrame) -> dict:
    ok = group[group["parse_ok"]].copy()
    return {
        "n": len(group),
        "parse_ok_n": len(ok),
        "parse_ok_rate": float(group["parse_ok"].mean()) if len(group) else float("nan"),
        "high_alpha_win_rate": ok["high_alpha_winner"].mean() if len(ok) else float("nan"),
        "low_alpha_win_rate": ok["low_alpha_winner"].mean() if len(ok) else float("nan"),
        "tie_rate": ok["tie_winner"].mean() if len(ok) else float("nan"),
        "monotonic_preference_score": ok["preference_score"].mean() if len(ok) else float("nan"),
        "mean_confidence": ok["confidence"].mean() if len(ok) else float("nan"),
    }


def pairwise_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(summarize_pairwise(group))
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_consistency(pairwise: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["model", "judge_model", "axis_id", "scenario_id", "seed", "left_run_id", "right_run_id", "left_alpha", "right_alpha"]
    normal = pairwise[(~pairwise["swap_order"]) & pairwise["parse_ok"]][key_cols + ["canonical_winner", "confidence"]].rename(
        columns={"canonical_winner": "normal_winner", "confidence": "normal_confidence"}
    )
    swapped = pairwise[pairwise["swap_order"] & pairwise["parse_ok"]][key_cols + ["canonical_winner", "confidence"]].rename(
        columns={"canonical_winner": "swapped_winner", "confidence": "swapped_confidence"}
    )
    merged = normal.merge(swapped, on=key_cols, how="inner")
    if merged.empty:
        merged["inconsistent"] = []
        return merged
    merged["inconsistent"] = merged["normal_winner"] != merged["swapped_winner"]
    merged["both_high_alpha"] = (merged["normal_winner"] == "right") & (merged["swapped_winner"] == "right")
    merged["both_low_alpha"] = (merged["normal_winner"] == "left") & (merged["swapped_winner"] == "left")
    merged["both_tie"] = (merged["normal_winner"] == "tie") & (merged["swapped_winner"] == "tie")
    return merged


def write_pairwise_tables(pairwise: pd.DataFrame, out_dir: Path) -> bool:
    if pairwise.empty:
        return False
    tables = out_dir / "tables"
    pairwise_group_summary(pairwise, ["model", "judge_model"]).to_csv(tables / "main_pairwise_sampled_metrics.csv", index=False)

    position = pairwise.groupby(["model", "judge_model", "swap_order"], as_index=False).agg(
        n=("pair_id", "count"),
        parse_ok_rate=("parse_ok", "mean"),
        text_1_win_rate=("text_1_win", "mean"),
        text_2_win_rate=("text_2_win", "mean"),
        tie_rate=("tie_winner", "mean"),
        high_alpha_win_rate=("high_alpha_winner", "mean"),
        monotonic_preference_score=("preference_score", "mean"),
        mean_confidence=("confidence", "mean"),
    )
    normal = position[~position["swap_order"]].set_index(["model", "judge_model"])
    swapped = position[position["swap_order"]].set_index(["model", "judge_model"])
    gaps = normal[["text_1_win_rate", "high_alpha_win_rate", "monotonic_preference_score"]].join(
        swapped[["text_1_win_rate", "high_alpha_win_rate", "monotonic_preference_score"]],
        lsuffix="_normal",
        rsuffix="_swapped",
    )
    gaps["text_1_win_rate_gap_swapped_minus_normal"] = gaps["text_1_win_rate_swapped"] - gaps["text_1_win_rate_normal"]
    gaps["high_alpha_win_rate_gap_swapped_minus_normal"] = gaps["high_alpha_win_rate_swapped"] - gaps["high_alpha_win_rate_normal"]
    gaps["monotonic_score_gap_swapped_minus_normal"] = gaps["monotonic_preference_score_swapped"] - gaps["monotonic_preference_score_normal"]
    position = position.merge(gaps.reset_index(), on=["model", "judge_model"], how="left")
    position.to_csv(tables / "main_pairwise_position_bias.csv", index=False)

    consistency = pairwise_consistency(pairwise)
    if not consistency.empty:
        inconsistency = consistency.groupby(["model", "judge_model"], as_index=False).agg(
            comparable_pairs=("inconsistent", "count"),
            inconsistency_rate=("inconsistent", "mean"),
            both_high_alpha_rate=("both_high_alpha", "mean"),
            both_low_alpha_rate=("both_low_alpha", "mean"),
            both_tie_rate=("both_tie", "mean"),
            normal_mean_confidence=("normal_confidence", "mean"),
            swapped_mean_confidence=("swapped_confidence", "mean"),
        )
    else:
        inconsistency = pd.DataFrame(columns=["model", "judge_model", "comparable_pairs", "inconsistency_rate"])
    inconsistency.to_csv(tables / "main_pairwise_inconsistency.csv", index=False)

    by_axis = pairwise_group_summary(pairwise, ["model", "judge_model", "axis_id"])
    if not consistency.empty:
        by_axis_inconsistency = consistency.groupby(["model", "judge_model", "axis_id"], as_index=False).agg(
            comparable_pairs=("inconsistent", "count"),
            inconsistency_rate=("inconsistent", "mean"),
        )
        by_axis = by_axis.merge(by_axis_inconsistency, on=["model", "judge_model", "axis_id"], how="left")
    by_axis.to_csv(tables / "main_pairwise_by_axis_model.csv", index=False)
    return True


def write_tables(gens: pd.DataFrame, proj: pd.DataFrame, scalar: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path, bootstrap_iterations: int) -> bool:
    tables = out_dir / "tables"

    by_embedding = group_summary(proj, ["embedding", "embedding_model", "model"])
    by_embedding.to_csv(tables / "main_projection_metrics_by_embedding.csv", index=False)
    by_embedding.to_csv(tables / "main_geometry_metrics_by_model_embedding.csv", index=False)

    group_summary(proj, ["embedding", "embedding_model", "model", "axis_id"]).to_csv(
        tables / "main_geometry_metrics_by_axis_model_embedding.csv", index=False
    )
    group_summary(proj, ["embedding", "embedding_model", "model", "alpha"]).to_csv(
        tables / "main_geometry_metrics_by_alpha_model_embedding.csv", index=False
    )
    group_summary(proj, ["embedding", "embedding_model", "model", "axis_id", "alpha"]).to_csv(
        tables / "main_axis_metrics.csv", index=False
    )

    nn = (
        proj.groupby(["embedding", "embedding_model", "model"], as_index=False)
        .agg(
            nearest_neighbor_top1_rate=("nearest_neighbor_top1", "mean"),
            nearest_neighbor_mean_rank=("nearest_neighbor_rank", "mean"),
            nearest_neighbor_median_rank=("nearest_neighbor_rank", "median"),
            n=("run_id", "count"),
        )
        .sort_values(["embedding", "nearest_neighbor_mean_rank"])
    )
    nn.to_csv(tables / "main_nearest_neighbor_metrics.csv", index=False)

    oor = (
        proj.groupby(["embedding", "embedding_model", "model", "axis_id"], as_index=False)
        .agg(
            out_of_range_rate=("out_of_range", "mean"),
            r_proj_min=("r_proj", "min"),
            r_proj_max=("r_proj", "max"),
            n=("run_id", "count"),
        )
        .sort_values(["embedding", "model", "axis_id"])
    )
    oor.to_csv(tables / "main_out_of_range_projection_metrics.csv", index=False)

    boot = bootstrap_ci_by_unit(
        proj,
        value_col="projection_error",
        group_cols=["embedding", "embedding_model", "model"],
        unit_col="scenario_id",
        iterations=bootstrap_iterations,
        seed=20260624,
    )
    boot.to_csv(tables / "main_bootstrap_ci_by_embedding.csv", index=False)

    length_by = (
        gens.groupby(["model", "axis_id", "alpha"], as_index=False)
        .agg(
            n=("run_id", "count"),
            mean_zh_chars=("zh_chars", "mean"),
            length_violation_rate=("length_violation", "mean"),
            too_short_rate=("too_short", "mean"),
            too_long_rate=("too_long", "mean"),
        )
        .sort_values(["model", "axis_id", "alpha"])
    )
    length_by.to_csv(tables / "main_length_by_model_axis_alpha.csv", index=False)

    length_model = (
        gens.groupby(["model"], as_index=False)
        .agg(
            n=("run_id", "count"),
            mean_zh_chars=("zh_chars", "mean"),
            length_violation_rate=("length_violation", "mean"),
            too_short_rate=("too_short", "mean"),
            too_long_rate=("too_long", "mean"),
        )
    )
    length_model.to_csv(tables / "main_length_metrics.csv", index=False)

    proj_with_len = proj.merge(gens[["run_id", "length_violation", "too_short", "too_long", "zh_chars"]], on="run_id", how="left")
    compliant = group_summary(proj_with_len[~proj_with_len["length_violation"].fillna(False)], ["embedding", "embedding_model", "model"])
    compliant.to_csv(tables / "main_length_compliant_subset_projection.csv", index=False)

    corr_rows = []
    proj_with_len["length_error_amount"] = np.where(
        proj_with_len["zh_chars"] < gens.set_index("run_id").reindex(proj_with_len["run_id"])["min_zh_chars"].to_numpy(),
        gens.set_index("run_id").reindex(proj_with_len["run_id"])["min_zh_chars"].to_numpy() - proj_with_len["zh_chars"],
        np.where(
            proj_with_len["zh_chars"] > gens.set_index("run_id").reindex(proj_with_len["run_id"])["max_zh_chars"].to_numpy(),
            proj_with_len["zh_chars"] - gens.set_index("run_id").reindex(proj_with_len["run_id"])["max_zh_chars"].to_numpy(),
            0.0,
        ),
    )
    for keys, group in proj_with_len.groupby(["embedding", "model"]):
        corr_rows.append(
            {
                "embedding": keys[0],
                "model": keys[1],
                "pearson_length_error_projection_error": group["length_error_amount"].corr(group["projection_error"], method="pearson"),
                "spearman_length_error_projection_error": group["length_error_amount"].corr(group["projection_error"], method="spearman"),
                "n": len(group),
            }
        )
    pd.DataFrame(corr_rows).to_csv(tables / "main_length_projection_correlation.csv", index=False)

    if not scalar.empty:
        scalar_rows = []
        for keys, group in scalar.groupby(["model", "judge_model"]):
            scalar_rows.append(
                {
                    "model": keys[0],
                    "judge_model": keys[1],
                    "n": len(group),
                    "parse_ok_rate": group["parse_ok"].mean() if "parse_ok" in group else np.nan,
                    "judge_mae": group["judge_error"].mean(),
                    "judge_bias": (group["style_b_ratio"] - group["alpha"]).mean(),
                    "judge_spearman": safe_spearman(group, "style_b_ratio"),
                    "content_preservation_mean": group["content_preservation"].mean(),
                    "naturalness_mean": group["style_mixture_naturalness"].mean(),
                    "endpoint_attraction_rate": endpoint_attraction(group, "style_b_ratio"),
                    "smoothness": smoothness(group, "style_b_ratio"),
                }
            )
        pd.DataFrame(scalar_rows).to_csv(tables / "main_scalar_metrics.csv", index=False)
    return write_pairwise_tables(pairwise, out_dir)


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / "figures" / f"{stem}.pdf")
    fig.savefig(out_dir / "figures" / f"{stem}.png")


def plot_calibration(proj: pd.DataFrame, out_dir: Path, embedding: str, stem: str) -> None:
    plt = setup_matplotlib()
    data = proj[proj["embedding"] == embedding]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1, label="Ideal")
    for model in MODEL_ORDER:
        group = data[data["model"] == model]
        curve = group.groupby("alpha", as_index=False)["r_proj_clipped"].mean().sort_values("alpha")
        ax.plot(
            curve["alpha"],
            curve["r_proj_clipped"],
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            linewidth=2,
        )
    ax.set_title(f"Calibration Curves ({embedding})")
    ax.set_xlabel("Target style-B ratio")
    ax.set_ylabel("Measured projection ratio")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, stem)
    plt.close(fig)


def heatmap(df: pd.DataFrame, value: str, title: str, stem: str, out_dir: Path) -> None:
    plt = setup_matplotlib()
    pivot = df.pivot_table(index="axis_id", columns="model", values=value, aggfunc="mean").reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)), [MODEL_LABELS[c] for c in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value.replace("_", " "))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > np.nanmean(pivot.to_numpy()) else "black", fontsize=8)
    save_figure(fig, out_dir, stem)
    plt.close(fig)


def write_figures(gens: pd.DataFrame, proj: pd.DataFrame, scalar: pd.DataFrame, out_dir: Path) -> list[tuple[str, str]]:
    plt = setup_matplotlib()
    notes: list[tuple[str, str]] = []
    plot_calibration(proj, out_dir, "bge_m3", "calibration_curves_by_model_bge_m3")
    notes.append(("calibration_curves_by_model_bge_m3", "Mean clipped projection ratio by target ratio for bge-m3. The dashed line is ideal calibration."))
    plot_calibration(proj, out_dir, "text2vec", "calibration_curves_by_model_text2vec")
    notes.append(("calibration_curves_by_model_text2vec", "Mean clipped projection ratio by target ratio for text2vec."))

    bge = proj[proj["embedding"] == "bge_m3"]
    heatmap(bge, "projection_error", "Projection MAE by Axis (bge-m3)", "axis_heatmap_projection_mae", out_dir)
    notes.append(("axis_heatmap_projection_mae", "Axis-level projection absolute error under bge-m3. Lower is better."))
    heatmap(bge, "off_axis_drift", "Off-Axis Drift by Axis (bge-m3)", "axis_heatmap_off_axis_drift", out_dir)
    notes.append(("axis_heatmap_off_axis_drift", "Axis-level off-axis drift under bge-m3. Lower indicates generations stay closer to the endpoint interpolation line."))

    boot_path = out_dir / "tables" / "main_bootstrap_ci_by_embedding.csv"
    boot = pd.read_csv(boot_path)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    for offset, embedding in [(-width / 2, "bge_m3"), (width / 2, "text2vec")]:
        sub = boot[boot["embedding"] == embedding].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + offset, sub["mean"], width=width, label=embedding, color="#93C5FD" if embedding == "bge_m3" else "#FDBA74")
        ax.errorbar(
            x + offset,
            sub["mean"],
            yerr=[sub["mean"] - sub["ci95_low"], sub["ci95_high"] - sub["mean"]],
            fmt="none",
            ecolor="#111827",
            capsize=3,
            linewidth=1,
        )
    ax.set_title("Projection MAE Bootstrap CI")
    ax.set_ylabel("Projection MAE")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, out_dir, "model_comparison_bootstrap_ci")
    plt.close(fig)
    notes.append(("model_comparison_bootstrap_ci", "Scenario-bootstrap 95% confidence intervals for projection MAE by embedding model."))

    if not scalar.empty:
        merged = proj.merge(scalar[["run_id", "style_b_ratio", "judge_model"]], on="run_id", how="inner")
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        for model in MODEL_ORDER:
            sub = merged[(merged["model"] == model) & (merged["embedding"] == "bge_m3")]
            ax.scatter(sub["r_proj_clipped"], sub["style_b_ratio"], s=6, alpha=0.18, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1)
        ax.set_title("Projection vs Scalar Judge")
        ax.set_xlabel("Projection ratio (bge-m3)")
        ax.set_ylabel("Scalar judge style-B ratio")
        ax.legend(markerscale=2)
        ax.grid(True, alpha=0.25)
        save_figure(fig, out_dir, "projection_vs_scalar_judge_scatter")
        plt.close(fig)
        notes.append(("projection_vs_scalar_judge_scatter", "Scatter of bge-m3 projection ratio against scalar judge ratio. Scalar is auxiliary and judge assignment is asymmetric for 35B-A3B."))

    heatmap(gens, "length_violation", "Length Violation Rate by Axis", "length_violation_heatmap", out_dir)
    notes.append(("length_violation_heatmap", "Length violation rate by model and axis."))

    nn = proj.groupby(["embedding", "model"], as_index=False)["nearest_neighbor_rank"].mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for embedding, marker in [("bge_m3", "o"), ("text2vec", "s")]:
        sub = nn[nn["embedding"] == embedding].set_index("model").reindex(MODEL_ORDER)
        ax.plot([MODEL_LABELS[m] for m in MODEL_ORDER], sub["nearest_neighbor_rank"], marker=marker, linewidth=2, label=embedding)
    ax.set_title("Nearest-Neighbor Mean Rank by Model")
    ax.set_ylabel("Mean rank")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, out_dir, "nearest_neighbor_rank_by_model")
    plt.close(fig)
    notes.append(("nearest_neighbor_rank_by_model", "Mean nearest-neighbor rank of each output against its target interpolation point. Lower is better."))

    oor = proj.groupby(["model", "axis_id"], as_index=False)["out_of_range"].mean()
    heatmap(oor, "out_of_range", "Out-of-Range Projection Rate by Axis", "out_of_range_rate_by_model_axis", out_dir)
    notes.append(("out_of_range_rate_by_model_axis", "Rate of raw projection ratios outside [0, 1], averaged across embeddings."))

    return notes


def write_markdown_docs(gens: pd.DataFrame, proj: pd.DataFrame, scalar: pd.DataFrame, out_dir: Path, figure_notes: list[tuple[str, str]]) -> None:
    docs = out_dir / "docs"
    tables = out_dir / "tables"
    by_emb = pd.read_csv(tables / "main_projection_metrics_by_embedding.csv")
    axis = pd.read_csv(tables / "main_geometry_metrics_by_axis_model_embedding.csv")
    alpha = pd.read_csv(tables / "main_geometry_metrics_by_alpha_model_embedding.csv")
    scalar_metrics = pd.read_csv(tables / "main_scalar_metrics.csv") if (tables / "main_scalar_metrics.csv").exists() else pd.DataFrame()
    length = pd.read_csv(tables / "main_length_metrics.csv")
    corr = pd.read_csv(tables / "main_length_projection_correlation.csv")

    def best_table_text(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
        return df[cols].head(n).to_markdown(index=False)

    bge = by_emb[by_emb["embedding"] == "bge_m3"].sort_values("projection_mae")
    text2vec = by_emb[by_emb["embedding"] == "text2vec"].sort_values("projection_mae")
    axis_worst = axis[axis["embedding"] == "bge_m3"].sort_values("projection_mae", ascending=False)
    drift_worst = axis[axis["embedding"] == "bge_m3"].sort_values("normalized_off_axis_drift_mean", ascending=False)

    summary = [
        "# Main 8-Axis Geometry Summary",
        "",
        "## Headline",
        "",
        "Both bge-m3 and text2vec preserve the same projection ranking: qwen35_9b is best, qwen35_35b_a3b is second, and qwen35_4b is weakest.",
        "",
        "## Projection Metrics By Embedding",
        "",
        "bge-m3:",
        "",
        best_table_text(bge, ["model", "projection_mae", "projection_spearman", "interpolation_distance_mean", "off_axis_drift_mean", "normalized_off_axis_drift_mean"]),
        "",
        "text2vec:",
        "",
        best_table_text(text2vec, ["model", "projection_mae", "projection_spearman", "interpolation_distance_mean", "off_axis_drift_mean", "normalized_off_axis_drift_mean"]),
        "",
        "## Required Questions",
        "",
        "### Is 9B's advantage lower projection error or lower off-axis drift?",
        "",
        "9B's advantage is primarily lower projection error and higher rank correlation. It does not have the lowest off-axis drift; 35B-A3B often has slightly lower drift, especially under text2vec. This means 9B is better calibrated along the style-ratio axis, while 35B-A3B can be geometrically close to the interpolation line but less accurately placed along it.",
        "",
        "### Why does 35B-A3B not exceed 9B?",
        "",
        "35B-A3B does not exceed 9B because its projection-ratio placement is less calibrated. Its off-axis drift is competitive, and out-of-range rates are low, so the gap is better explained by projection proportion bias and axis-specific calibration error rather than broad geometric instability.",
        "",
        "### Where is 4B weakest?",
        "",
        "Under bge-m3, the weakest 4B axes by projection MAE are:",
        "",
        best_table_text(axis_worst[axis_worst["model"] == "qwen35_4b"], ["axis_id", "projection_mae", "projection_spearman", "normalized_off_axis_drift_mean"], 8),
        "",
        "4B also has the highest length violation rate, mainly too-short outputs, which should be treated as a practical limitation even though prior diagnostics indicate length does not dominate the projection ranking.",
        "",
        "### Which axes are most linear and which are most off-axis?",
        "",
        "The most linear axes are those with low smoothness penalty and low projection MAE. The easiest off-axis failure cases are captured by normalized off-axis drift. Highest bge-m3 normalized drift cases:",
        "",
        best_table_text(drift_worst, ["model", "axis_id", "normalized_off_axis_drift_mean", "off_axis_drift_mean", "projection_mae"], 12),
        "",
        "### Do low and high ratio regions show overshoot or endpoint attraction?",
        "",
        "Endpoint attraction is present but not dominant. It should be interpreted from `main_geometry_metrics_by_alpha_model_embedding.csv` and `main_out_of_range_projection_metrics.csv`. Out-of-range projection rates are low overall, so the more important issue is mild under/over-placement within the [0, 1] interval rather than systematic extreme overshoot.",
        "",
        "## Scalar Judge Caveat",
        "",
    ]
    if not scalar_metrics.empty:
        summary.extend(
            [
                scalar_metrics.to_markdown(index=False),
                "",
                "Scalar judge is auxiliary. 4B/9B are judged by 35B-A3B, while 35B-A3B is judged by 9B, so scalar metrics are not fully same-scale for 9B vs 35B-A3B.",
                "",
            ]
        )
    summary.extend(
        [
            "## Length Diagnostics",
            "",
            length.to_markdown(index=False),
            "",
            "Length-projection correlations:",
            "",
            corr.to_markdown(index=False),
            "",
        ]
    )
    (docs / "main8axis_geometry_summary.md").write_text("\n".join(summary), encoding="utf-8")

    fig_lines = ["# Main 8-Axis Figure Notes", ""]
    for stem, note in figure_notes:
        fig_lines.append(f"## {stem}")
        fig_lines.append("")
        fig_lines.append(f"- PDF: `../figures/{stem}.pdf`")
        fig_lines.append(f"- PNG: `../figures/{stem}.png`")
        fig_lines.append(f"- Note: {note}")
        fig_lines.append("")
    (docs / "main8axis_figure_notes.md").write_text("\n".join(fig_lines), encoding="utf-8")


def write_plan_docs(out_dir: Path) -> None:
    docs = out_dir / "docs"
    models_root = Path("/home/data_cpfs/zicheng/models")
    model_rows = []
    for cfg in sorted(models_root.glob("*/config.json")):
        model_dir = cfg.parent
        if "qwen" in model_dir.name.lower():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        model_rows.append(
            {
                "path": str(model_dir),
                "model_type": data.get("model_type", "unknown"),
                "hidden_size": data.get("hidden_size"),
                "num_hidden_layers": data.get("num_hidden_layers"),
                "notes": "embedding model; not suitable as generative cross-family baseline" if "bge" in model_dir.name.lower() or "text2vec" in model_dir.name.lower() else "needs manual review",
            }
        )
    available = pd.DataFrame(model_rows)
    if available.empty:
        available_md = "No non-Qwen local model config was found under `/home/data_cpfs/zicheng/models`."
    else:
        available_md = available.to_markdown(index=False)

    cross = f"""# Cross-Family Experiment Plan

## Status

Do not run this experiment until the main 8-axis final report is complete.

## Available Non-Qwen Model Paths Found

{available_md}

## Current Assessment

The scanned non-Qwen paths under `/home/data_cpfs/zicheng/models` appear to be embedding models rather than instruction-tuned generative LLMs. They are useful for measurement, not as tested generator baselines.

Before running a cross-family generation experiment, identify at least one instruction-tuned non-Qwen model, such as a Llama-family, Mistral-family, Gemma-family, Yi-family, or DeepSeek-chat/instruct model, with a local path and license suitable for evaluation.

## Recommended Subset

- Axes: `formality_request`, `emotion_apology`, `empathy_clinical`, `expertise_explain`.
- Scenarios: 20 per axis.
- Ratios: 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0.
- Seeds: 2.
- Scale: `4 axes * 20 scenarios * 7 ratios * 2 seeds * N models = 1120 * N generations`.

## Goals

- Test whether explicit style-ratio calibration exists outside the Qwen family.
- Compare Qwen3.5-9B against similarly sized 7B/8B/9B instruction models.
- Keep bge-m3/text2vec projection as the main metric for comparability.

## Execution Recommendation

Do not launch until:

1. Main 8-axis final report is complete.
2. A non-Qwen instruction-tuned model path is confirmed.
3. A short smoke test verifies no reasoning/meta leakage and acceptable Chinese output quality.
"""
    (docs / "cross_family_experiment_plan.md").write_text(cross, encoding="utf-8")

    temp = """# Temperature Ablation Plan

## Status

Do not run this ablation until the main 8-axis final report is complete.

## Proposed Settings

- Temperatures: 0.2, 0.7, 1.0.
- Models: `qwen35_9b`, `qwen35_35b_a3b`.
- Axes: choose 3 axes from `formality_request`, `emotion_apology`, `empathy_clinical`, `expertise_explain`.
- Scenarios: 20 per axis.
- Ratios: 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0.
- Seeds: 2.

Scale:

`3 axes * 20 scenarios * 7 ratios * 2 seeds * 2 models * 3 temps = 5040 generations`.

## Questions

1. Does low temperature improve projection calibration?
2. Does high temperature increase naturalness but increase off-axis drift?
3. Does temperature change the 9B vs 35B-A3B ranking?
4. Does length compliance improve or degrade at lower temperature?

## Recommended Metrics

- bge-m3 and text2vec projection MAE/Spearman.
- Interpolation distance.
- Off-axis drift and normalized off-axis drift.
- Length violation rate.
- Scalar judge on a subset only, if needed.

## Execution Recommendation

Run only after the final report identifies whether the current main result needs robustness support. The ablation should be framed as a targeted robustness check, not as part of the primary benchmark.
"""
    (docs / "temperature_ablation_plan.md").write_text(temp, encoding="utf-8")


def write_final_report(out_dir: Path, pairwise_available: bool) -> None:
    tables = out_dir / "tables"
    docs = out_dir / "docs"
    by_emb = pd.read_csv(tables / "main_projection_metrics_by_embedding.csv")
    scalar = pd.read_csv(tables / "main_scalar_metrics.csv")
    length = pd.read_csv(tables / "main_length_metrics.csv")
    boot = pd.read_csv(tables / "main_bootstrap_ci_by_embedding.csv")
    pairwise = pd.read_csv(tables / "main_pairwise_sampled_metrics.csv") if pairwise_available else pd.DataFrame()
    position = pd.read_csv(tables / "main_pairwise_position_bias.csv") if pairwise_available else pd.DataFrame()
    inconsistency = pd.read_csv(tables / "main_pairwise_inconsistency.csv") if pairwise_available else pd.DataFrame()

    report = [
        "# Main 8-Axis v1 Final Analysis Report",
        "",
        "## Status",
        "",
        "Main generation, projection evaluation, scalar judge, geometry diagnostics, length diagnostics, and paper-grade figures are complete. Sampled pairwise audit is included only if pairwise files are present.",
        "",
        "## Main Projection Results",
        "",
        by_emb.sort_values(["embedding", "projection_mae"])[["embedding", "model", "projection_mae", "projection_spearman", "interpolation_distance_mean", "off_axis_drift_mean", "normalized_off_axis_drift_mean", "out_of_range_rate"]].to_markdown(index=False),
        "",
        "## Bootstrap CI",
        "",
        boot.to_markdown(index=False),
        "",
        "## Scalar Judge Auxiliary Results",
        "",
        scalar.to_markdown(index=False),
        "",
        "Scalar is auxiliary because judge assignment is asymmetric: 4B/9B are judged by 35B-A3B, while 35B-A3B is judged by 9B.",
        "",
        "## Length Diagnostics",
        "",
        length.to_markdown(index=False),
        "",
        "## Pairwise Audit",
        "",
        "Sampled bidirectional pairwise audit has been completed and included in `main_pairwise_*.csv`." if pairwise_available else "Sampled bidirectional pairwise audit is not yet included in this report. Projection remains the primary metric.",
        "",
    ]
    if pairwise_available:
        report.extend(
            [
                "Overall sampled pairwise metrics:",
                "",
                pairwise.to_markdown(index=False),
                "",
                "Position-bias summary:",
                "",
                position.to_markdown(index=False),
                "",
                "Bidirectional inconsistency summary:",
                "",
                inconsistency.to_markdown(index=False),
                "",
                "Pairwise is an auxiliary monotonicity audit: for each sampled pair, the higher-alpha text should be closer to style B. The audit should not replace projection geometry as the primary metric.",
                "",
            ]
        )
    report.extend(
        [
        "## Figure Paths",
        "",
        "- `figures/calibration_curves_by_model_bge_m3.pdf/png`",
        "- `figures/calibration_curves_by_model_text2vec.pdf/png`",
        "- `figures/axis_heatmap_projection_mae.pdf/png`",
        "- `figures/axis_heatmap_off_axis_drift.pdf/png`",
        "- `figures/model_comparison_bootstrap_ci.pdf/png`",
        "- `figures/projection_vs_scalar_judge_scatter.pdf/png`",
        "- `figures/length_violation_heatmap.pdf/png`",
        "- `figures/nearest_neighbor_rank_by_model.pdf/png`",
        "- `figures/out_of_range_rate_by_model_axis.pdf/png`",
        "",
        "## Cross-Family Recommendation",
        "",
        "Do not launch yet. First confirm local non-Qwen instruction-tuned generative model paths. The currently scanned non-Qwen paths under `/home/data_cpfs/zicheng/models` are embedding models, not generator baselines.",
        "",
        "## Temperature Ablation Recommendation",
        "",
        "Do not launch yet. Use it as a targeted robustness check after finalizing the main report and pairwise decision.",
        "",
        "## Paper-Ready Findings",
        "",
        "1. Explicit style-ratio prompting shows measurable monotonic calibration across 8 Chinese style axes.",
        "2. Qwen3.5-9B is the best calibrated generator among the tested Qwen-family models.",
        "3. Qwen3.5-35B-A3B is close to 9B but does not surpass it on projection calibration.",
        "4. Qwen3.5-4B is consistently weaker, especially in projection error and length compliance.",
        "5. bge-m3 and text2vec agree on the model ranking, supporting measurement robustness.",
        "6. Length violations remain substantial but prior diagnostics and current subset tables indicate they do not explain away the projection ranking.",
        "7. Scalar judge results support the broad ranking but remain auxiliary due to asymmetric judge assignment.",
        "8. Pairwise, when used, should be treated as an auxiliary preference and monotonicity audit rather than the main metric.",
        "",
        ]
    )
    (docs / "main8axis_v1_final_analysis_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=ROOT / "results/main8axis_v1_final_eval")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    gens = prepare_generation(
        read_many(
            [
                ROOT / "data/generations/qwen35_4b/main8axis_v1_4b.jsonl",
                ROOT / "data/generations/qwen35_9b/main8axis_v1_9b.jsonl",
                ROOT / "data/generations/qwen35_35b_a3b/main8axis_v1_35b.jsonl",
            ]
        )
    )
    proj = prepare_projection(
        read_many(
            [
                ROOT / "data/metrics/projection_bge_m3_main8axis_v1_4b.jsonl",
                ROOT / "data/metrics/projection_bge_m3_main8axis_v1_9b.jsonl",
                ROOT / "data/metrics/projection_bge_m3_main8axis_v1_35b.jsonl",
                ROOT / "data/metrics/projection_text2vec_main8axis_v1_4b.jsonl",
                ROOT / "data/metrics/projection_text2vec_main8axis_v1_9b.jsonl",
                ROOT / "data/metrics/projection_text2vec_main8axis_v1_35b.jsonl",
            ]
        )
    )
    scalar = prepare_scalar(
        read_many(
            [
                ROOT / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_4b.jsonl",
                ROOT / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_9b.jsonl",
                ROOT / "data/judgments/scalar_qwen35_9b_main8axis_v1_35b.jsonl",
            ]
        )
    )
    pairwise_paths = [path for path in PAIRWISE_PATHS if path.exists()]
    pairwise = prepare_pairwise(read_many(pairwise_paths)) if pairwise_paths else pd.DataFrame()

    pairwise_available = write_tables(gens, proj, scalar, pairwise, out_dir, args.bootstrap_iterations)
    figure_notes = write_figures(gens, proj, scalar, out_dir)
    write_markdown_docs(gens, proj, scalar, out_dir, figure_notes)
    write_plan_docs(out_dir)
    write_final_report(out_dir, pairwise_available=pairwise_available)
    print(f"Wrote final analysis outputs under {out_dir}")


if __name__ == "__main__":
    main()
