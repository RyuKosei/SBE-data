from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # Keep the asset build runnable on minimal cluster images.
    sns = None


# Inputs: existing final-analysis outputs only. This script does not run generation,
# judging, or embedding.
ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "main8axis_v1_final_eval"
FINAL_TABLES = FINAL_DIR / "tables"
FINAL_DOCS = FINAL_DIR / "docs"
OUT_DIR = ROOT / "results" / "main8axis_v1_paper_assets"
OUT_FIGURES = OUT_DIR / "figures"
OUT_TABLES = OUT_DIR / "tables"
OUT_CAPTIONS = OUT_DIR / "captions"
OUT_SCRIPTS = OUT_DIR / "scripts"
OUT_DOCS = OUT_DIR / "docs"

INPUT_TABLES = {
    "projection": FINAL_TABLES / "main_projection_metrics_by_embedding.csv",
    "axis_projection": FINAL_TABLES / "main_geometry_metrics_by_axis_model_embedding.csv",
    "bootstrap": FINAL_TABLES / "main_bootstrap_ci_by_embedding.csv",
    "length": FINAL_TABLES / "main_length_metrics.csv",
    "length_axis_alpha": FINAL_TABLES / "main_length_by_model_axis_alpha.csv",
    "length_compliant": FINAL_TABLES / "main_length_compliant_subset_projection.csv",
    "length_corr": FINAL_TABLES / "main_length_projection_correlation.csv",
    "scalar": FINAL_TABLES / "main_scalar_metrics.csv",
    "pairwise": FINAL_TABLES / "main_pairwise_sampled_metrics.csv",
    "nearest_neighbor": FINAL_TABLES / "main_nearest_neighbor_metrics.csv",
    "out_of_range": FINAL_TABLES / "main_out_of_range_projection_metrics.csv",
}

PROJECTION_JSONL = [
    ROOT / "data/metrics/projection_bge_m3_main8axis_v1_4b.jsonl",
    ROOT / "data/metrics/projection_bge_m3_main8axis_v1_9b.jsonl",
    ROOT / "data/metrics/projection_bge_m3_main8axis_v1_35b.jsonl",
    ROOT / "data/metrics/projection_text2vec_main8axis_v1_4b.jsonl",
    ROOT / "data/metrics/projection_text2vec_main8axis_v1_9b.jsonl",
    ROOT / "data/metrics/projection_text2vec_main8axis_v1_35b.jsonl",
]
SCALAR_JSONL = [
    ROOT / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_4b.jsonl",
    ROOT / "data/judgments/scalar_qwen35_35b_a3b_main8axis_v1_9b.jsonl",
    ROOT / "data/judgments/scalar_qwen35_9b_main8axis_v1_35b.jsonl",
]

MODEL_ORDER = ["qwen35_4b", "qwen35_9b", "qwen35_35b_a3b"]
MODEL_LABELS = {
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
}
MODEL_COLORS = {
    "qwen35_4b": "#7f8c8d",
    "qwen35_9b": "#1f77b4",
    "qwen35_35b_a3b": "#ff7f0e",
}
AXIS_LABELS = {
    "concision_detail": "Detail",
    "emotion_apology": "Emotion",
    "empathy_clinical": "Empathy",
    "expertise_explain": "Expertise",
    "formality_request": "Formality",
    "humor_neutral": "Humor",
    "objectivity_cat": "Objectivity",
    "politeness_refusal": "Politeness",
}
RNG_SEED = 20260624


def ensure_dirs() -> None:
    for path in [OUT_FIGURES, OUT_TABLES, OUT_CAPTIONS, OUT_SCRIPTS, OUT_DOCS]:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_many_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def embedding_short(value: str) -> str:
    if "bge-m3" in value:
        return "bge_m3"
    if "text2vec" in value:
        return "text2vec"
    return value


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def axis_label(axis: str) -> str:
    return AXIS_LABELS.get(axis, axis)


def ordered_model_index(df: pd.DataFrame, col: str = "model") -> pd.DataFrame:
    out = df.copy()
    out["_model_order"] = out[col].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values("_model_order").drop(columns=["_model_order"])


def write_table(df: pd.DataFrame, stem: str, md_note: str = "") -> tuple[Path, Path]:
    csv_path = OUT_TABLES / f"{stem}.csv"
    md_path = OUT_TABLES / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md = df.to_markdown(index=False)
    if md_note:
        md += "\n\n" + md_note.strip() + "\n"
    md_path.write_text(md + "\n", encoding="utf-8")
    return csv_path, md_path


def savefig(fig, stem: str) -> list[Path]:
    pdf = OUT_FIGURES / f"{stem}.pdf"
    png = OUT_FIGURES / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def setup_plots() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    tables = {name: pd.read_csv(path) for name, path in INPUT_TABLES.items()}
    projection = read_many_jsonl(PROJECTION_JSONL)
    for col in ["alpha", "projection_ratio_raw", "projection_ratio_clipped", "interpolation_distance", "off_axis_drift", "endpoint_distance", "nearest_neighbor_rank"]:
        projection[col] = pd.to_numeric(projection[col], errors="coerce")
    projection["embedding"] = projection["embedding_model"].map(embedding_short)
    projection["projection_error"] = (projection["projection_ratio_clipped"] - projection["alpha"]).abs()
    projection["normalized_off_axis_drift"] = projection["off_axis_drift"] / projection["endpoint_distance"].replace(0, np.nan)
    projection["out_of_range"] = (projection["projection_ratio_raw"] < 0) | (projection["projection_ratio_raw"] > 1)
    projection["low_alpha_overshoot"] = projection["alpha"].isin([0.1, 0.25]) & (projection["projection_ratio_raw"] > projection["alpha"] + 0.15)
    projection["high_alpha_undershoot"] = projection["alpha"].isin([0.75, 0.9]) & (projection["projection_ratio_raw"] < projection["alpha"] - 0.15)

    scalar = read_many_jsonl(SCALAR_JSONL)
    for col in ["alpha", "style_b_ratio", "content_preservation", "style_mixture_naturalness"]:
        scalar[col] = pd.to_numeric(scalar[col], errors="coerce")
    return tables, projection, scalar


def prepare_tables(tables: dict[str, pd.DataFrame], projection: pd.DataFrame) -> tuple[list[Path], dict[str, pd.DataFrame]]:
    written: list[Path] = []
    outputs: dict[str, pd.DataFrame] = {}
    projection_metrics = tables["projection"]
    length = tables["length"]
    length_axis = tables["length_axis_alpha"]
    length_compliant = tables["length_compliant"]
    length_corr = tables["length_corr"]
    scalar = tables["scalar"]
    pairwise = tables["pairwise"]

    table1 = (
        projection_metrics.groupby("model", as_index=False)
        .agg(
            **{
                "Projection MAE mean across embeddings": ("projection_mae", "mean"),
                "Projection Spearman mean across embeddings": ("projection_spearman", "mean"),
                "Interpolation distance": ("interpolation_distance_mean", "mean"),
                "Off-axis drift": ("off_axis_drift_mean", "mean"),
                "Normalized drift": ("normalized_off_axis_drift_mean", "mean"),
                "NN mean rank": ("nearest_neighbor_mean_rank", "mean"),
                "Out-of-range rate": ("out_of_range_rate", "mean"),
            }
        )
        .merge(length[["model", "length_violation_rate"]], on="model", how="left")
        .rename(columns={"model": "Model", "length_violation_rate": "Length violation rate"})
        .sort_values("Projection MAE mean across embeddings")
    )
    table1["Model"] = table1["Model"].map(model_label)
    outputs["table_1_main_model_comparison"] = table1
    written.extend(write_table(table1, "table_1_main_model_comparison"))

    table2 = projection_metrics[
        ["embedding", "model", "projection_mae", "projection_spearman", "off_axis_drift_mean", "normalized_off_axis_drift_mean"]
    ].copy()
    table2 = table2.rename(
        columns={
            "embedding": "Embedding",
            "model": "Model",
            "projection_mae": "Projection MAE",
            "projection_spearman": "Projection Spearman",
            "off_axis_drift_mean": "Off-axis drift",
            "normalized_off_axis_drift_mean": "Normalized drift",
        }
    ).sort_values(["Embedding", "Projection MAE"])
    table2["Model"] = table2["Model"].map(model_label)
    outputs["table_2_embedding_robustness"] = table2
    written.extend(write_table(table2, "table_2_embedding_robustness"))

    axis_proj = tables["axis_projection"]
    length_axis_model = (
        length_axis.groupby(["model", "axis_id"], as_index=False)
        .agg(length_violation_rate=("length_violation_rate", "mean"))
    )
    table3 = (
        axis_proj.groupby(["axis_id", "model"], as_index=False)
        .agg(
            **{
                "Projection MAE": ("projection_mae", "mean"),
                "Projection Spearman": ("projection_spearman", "mean"),
                "Off-axis drift": ("off_axis_drift_mean", "mean"),
                "NN mean rank": ("nearest_neighbor_mean_rank", "mean"),
            }
        )
        .merge(length_axis_model, on=["model", "axis_id"], how="left")
        .rename(columns={"axis_id": "Axis", "model": "Model", "length_violation_rate": "Length violation rate"})
    )
    table3["Axis"] = table3["Axis"].map(axis_label)
    table3["Model"] = table3["Model"].map(model_label)
    table3 = table3.sort_values(["Axis", "Projection MAE"])
    outputs["table_3_axis_level_projection"] = table3
    written.extend(write_table(table3, "table_3_axis_level_projection"))

    pivot = table3.pivot_table(index="Axis", columns="Model", values="Projection MAE", aggfunc="mean")
    pivot = pivot[[MODEL_LABELS[m] for m in MODEL_ORDER]]
    pivot = pivot.reset_index()
    outputs["table_3_axis_projection_mae_pivot"] = pivot
    written.extend(write_table(pivot, "table_3_axis_projection_mae_pivot"))

    table4 = (
        projection_metrics.groupby("model", as_index=False)
        .agg(
            **{
                "Projection MAE": ("projection_mae", "mean"),
                "Interpolation distance": ("interpolation_distance_mean", "mean"),
                "Off-axis drift": ("off_axis_drift_mean", "mean"),
                "Normalized drift": ("normalized_off_axis_drift_mean", "mean"),
                "Out-of-range rate": ("out_of_range_rate", "mean"),
            }
        )
        .sort_values("Projection MAE")
        .rename(columns={"model": "Model"})
    )
    table4["Model"] = table4["Model"].map(model_label)
    table4_note = (
        "Interpretation: Qwen3.5-9B has the lowest projection MAE, so its advantage is primarily better ratio placement. "
        "Qwen3.5-35B-A3B has the lowest off-axis and normalized drift, showing that lower drift alone does not imply better calibration. "
        "Qwen3.5-4B has the worst projection MAE and normalized drift. "
        "Out-of-range projection is rare for all models, so the dominant errors are in-range under/over-placement rather than extreme overshoot."
    )
    outputs["table_4_geometry_decomposition"] = table4
    written.extend(write_table(table4, "table_4_geometry_decomposition", table4_note))

    compliant = (
        length_compliant.groupby("model", as_index=False)
        .agg(**{"Length-compliant Projection MAE": ("projection_mae", "mean")})
    )
    corr = (
        length_corr.groupby("model", as_index=False)
        .agg(
            **{
                "Length-error vs Projection-error Pearson": ("pearson_length_error_projection_error", "mean"),
                "Length-error vs Projection-error Spearman": ("spearman_length_error_projection_error", "mean"),
            }
        )
    )
    table5 = (
        length.rename(
            columns={
                "model": "Model",
                "mean_zh_chars": "Mean Chinese chars",
                "length_violation_rate": "Length violation rate",
                "too_short_rate": "Too short rate",
                "too_long_rate": "Too long rate",
            }
        )
        .merge(compliant.rename(columns={"model": "Model"}), on="Model", how="left")
        .merge(corr.rename(columns={"model": "Model"}), on="Model", how="left")
    )
    table5["Model"] = table5["Model"].map(model_label)
    table5 = table5.sort_values("Length violation rate")
    outputs["table_5_length_diagnostics"] = table5
    written.extend(write_table(table5, "table_5_length_diagnostics"))

    table6 = (
        scalar[["model", "judge_model", "judge_mae", "judge_spearman"]]
        .merge(
            pairwise[
                [
                    "model",
                    "expected_judge_model",
                    "higher_alpha_win_rate",
                    "monotonic_violation_rate",
                    "bidirectional_consistency",
                    "tie_rate",
                ]
            ],
            on="model",
            how="left",
        )
        .rename(
            columns={
                "model": "Model",
                "judge_model": "Scalar judge model",
                "expected_judge_model": "Pairwise judge model",
                "judge_mae": "Scalar judge MAE",
                "judge_spearman": "Scalar judge Spearman",
                "higher_alpha_win_rate": "Higher-alpha win rate",
                "monotonic_violation_rate": "Pairwise monotonic violation",
                "bidirectional_consistency": "Bidirectional consistency",
                "tie_rate": "Tie rate",
            }
        )
    )
    table6["Judge model"] = table6["Scalar judge model"] + " / " + table6["Pairwise judge model"]
    table6 = table6[
        [
            "Model",
            "Scalar judge MAE",
            "Scalar judge Spearman",
            "Higher-alpha win rate",
            "Pairwise monotonic violation",
            "Bidirectional consistency",
            "Tie rate",
            "Judge model",
        ]
    ]
    table6["Model"] = table6["Model"].map(model_label)
    table6 = table6.sort_values("Scalar judge MAE")
    table6_note = "Note: scalar and pairwise are auxiliary due to judge asymmetry and order effects."
    outputs["table_6_scalar_and_pairwise_auxiliary"] = table6
    written.extend(write_table(table6, "table_6_scalar_and_pairwise_auxiliary", table6_note))

    table7 = (
        projection.groupby(["model", "embedding"], as_index=False)
        .agg(
            **{
                "NN mean rank": ("nearest_neighbor_rank", "mean"),
                "NN top-1 accuracy": ("nearest_neighbor_rank", lambda s: float((s == 1).mean())),
                "Out-of-range rate": ("out_of_range", "mean"),
                "Low-alpha over-shoot rate": ("low_alpha_overshoot", "mean"),
                "High-alpha under-shoot rate": ("high_alpha_undershoot", "mean"),
            }
        )
        .rename(columns={"model": "Model", "embedding": "Embedding"})
        .sort_values(["Embedding", "NN mean rank"])
    )
    table7["Model"] = table7["Model"].map(model_label)
    outputs["table_7_nearest_neighbor_and_out_of_range"] = table7
    written.extend(write_table(table7, "table_7_nearest_neighbor_and_out_of_range"))
    return written, outputs


def add_error_band(ax, grouped: pd.DataFrame, color: str) -> None:
    ax.fill_between(
        grouped["alpha"].to_numpy(dtype=float),
        (grouped["mean"] - 1.96 * grouped["sem"].fillna(0)).to_numpy(dtype=float),
        (grouped["mean"] + 1.96 * grouped["sem"].fillna(0)).to_numpy(dtype=float),
        color=color,
        alpha=0.12,
        linewidth=0,
    )


def make_figures(tables: dict[str, pd.DataFrame], projection: pd.DataFrame, scalar: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    setup_plots()

    for embedding, stem, title in [
        ("bge_m3", "fig_1_model_calibration_curves_bge_m3", "Calibration Curves (bge-m3)"),
        ("text2vec", "fig_2_model_calibration_curves_text2vec", "Calibration Curves (text2vec)"),
    ]:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1, label="Ideal")
        data = projection[projection["embedding"] == embedding]
        for model in MODEL_ORDER:
            sub = data[data["model"] == model]
            grouped = (
                sub.groupby("alpha", as_index=False)["projection_ratio_clipped"]
                .agg(mean="mean", sem="sem")
                .sort_values("alpha")
            )
            ax.plot(grouped["alpha"], grouped["mean"], marker="o", color=MODEL_COLORS[model], label=model_label(model), linewidth=2)
            add_error_band(ax, grouped, MODEL_COLORS[model])
        ax.set_title(title)
        ax.set_xlabel("Target alpha")
        ax.set_ylabel("Measured projection ratio")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        paths.extend(savefig(fig, stem))

    boot = tables["bootstrap"]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for offset, embedding in [(-width / 2, "bge_m3"), (width / 2, "text2vec")]:
        sub = boot[boot["embedding"] == embedding].set_index("model").reindex(MODEL_ORDER)
        means = sub["mean"].to_numpy()
        low = means - sub["ci95_low"].to_numpy()
        high = sub["ci95_high"].to_numpy() - means
        color = "#9ecae1" if embedding == "bge_m3" else "#fdae6b"
        ax.bar(x + offset, means, width=width, label=embedding, color=color, edgecolor="#333333", linewidth=0.5)
        ax.errorbar(x + offset, means, yerr=[low, high], fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    ax.set_title("Projection MAE with Scenario-bootstrap 95% CI")
    ax.set_ylabel("Projection MAE")
    ax.set_xticks(x, [model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    paths.extend(savefig(fig, "fig_3_model_comparison_projection_mae_ci"))

    axis_proj = tables["axis_projection"]
    for value, stem, title, cbar_label in [
        ("projection_mae", "fig_4_axis_heatmap_projection_mae", "Projection MAE by Axis", "Projection MAE"),
        ("normalized_off_axis_drift_mean", "fig_5_axis_heatmap_off_axis_drift", "Normalized Off-axis Drift by Axis", "Normalized drift"),
    ]:
        data = axis_proj.groupby(["axis_id", "model"], as_index=False)[value].mean()
        pivot = data.pivot_table(index="axis_id", columns="model", values=value).reindex(columns=MODEL_ORDER)
        pivot.index = [axis_label(i) for i in pivot.index]
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xticks(range(len(MODEL_ORDER)), [model_label(m) for m in MODEL_ORDER], rotation=18, ha="right")
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        threshold = np.nanmean(pivot.to_numpy())
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black" if val > threshold else "white")
        paths.extend(savefig(fig, stem))

    geom = (
        tables["projection"].groupby("model", as_index=False)
        .agg(
            projection_mae=("projection_mae", "mean"),
            normalized_off_axis_drift_mean=("normalized_off_axis_drift_mean", "mean"),
            interpolation_distance_mean=("interpolation_distance_mean", "mean"),
        )
    )
    metrics = [
        ("projection_mae", "Projection MAE"),
        ("normalized_off_axis_drift_mean", "Normalized drift"),
        ("interpolation_distance_mean", "Interpolation distance"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for idx, (col, label) in enumerate(metrics):
        sub = geom.set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + (idx - 1) * width, sub[col], width=width, label=label, color=["#bdd7e7", "#fdae6b", "#bcbddc"][idx])
    ax.set_title("Geometry Decomposition by Model")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x, [model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    paths.extend(savefig(fig, "fig_6_geometry_decomposition_bar"))

    length_axis = tables["length_axis_alpha"].groupby(["axis_id", "model"], as_index=False)["length_violation_rate"].mean()
    pivot = length_axis.pivot_table(index="axis_id", columns="model", values="length_violation_rate").reindex(columns=MODEL_ORDER)
    pivot.index = [axis_label(i) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_title("Length Violation Rate by Axis")
    ax.set_xticks(range(len(MODEL_ORDER)), [model_label(m) for m in MODEL_ORDER], rotation=18, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Violation rate")
    threshold = np.nanmean(pivot.to_numpy())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black" if val > threshold else "white")
    paths.extend(savefig(fig, "fig_7_length_violation_heatmap"))

    scatter = projection[projection["embedding"] == "bge_m3"].merge(
        scalar[["run_id", "style_b_ratio"]], on="run_id", how="inner"
    )
    if len(scatter) > 3000:
        scatter = scatter.sample(n=3000, random_state=RNG_SEED)
    pearson = scatter["projection_ratio_clipped"].corr(scatter["style_b_ratio"], method="pearson")
    spearman = scatter["projection_ratio_clipped"].corr(scatter["style_b_ratio"], method="spearman")
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    for model in MODEL_ORDER:
        sub = scatter[scatter["model"] == model]
        ax.scatter(sub["projection_ratio_clipped"], sub["style_b_ratio"], s=8, alpha=0.28, color=MODEL_COLORS[model], label=model_label(model), linewidths=0)
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    ax.text(0.03, 0.95, f"Pearson={pearson:.2f}\nSpearman={spearman:.2f}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_title("Projection Ratio vs Scalar Judge Ratio")
    ax.set_xlabel("Projection ratio (bge-m3)")
    ax.set_ylabel("Scalar judge ratio")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.25)
    paths.extend(savefig(fig, "fig_8_projection_vs_scalar_judge_scatter"))

    nn = tables["nearest_neighbor"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for offset, embedding in [(-width / 2, "bge_m3"), (width / 2, "text2vec")]:
        sub = nn[nn["embedding"] == embedding].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + offset, sub["nearest_neighbor_mean_rank"], width=width, label=embedding, color="#9ecae1" if embedding == "bge_m3" else "#fdae6b", edgecolor="#333333", linewidth=0.5)
    ax.set_title("Nearest-neighbor Mean Rank")
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_xticks(x, [model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    paths.extend(savefig(fig, "fig_9_nearest_neighbor_rank_by_model"))

    pairwise = tables["pairwise"].set_index("model").reindex(MODEL_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    cols = [
        ("higher_alpha_win_rate", "Higher-alpha win"),
        ("monotonic_violation_rate", "Monotonic violation"),
        ("bidirectional_consistency", "Bidirectional consistency"),
    ]
    for idx, (col, label) in enumerate(cols):
        ax.bar(x + (idx - 1) * width, pairwise[col], width=width, label=label, color=["#74c476", "#fb6a4a", "#6baed6"][idx], edgecolor="#333333", linewidth=0.5)
    ax.set_title("Pairwise Auxiliary Audit")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [model_label(m) for m in MODEL_ORDER], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    paths.extend(savefig(fig, "fig_10_pairwise_auxiliary_audit"))
    return paths


def write_captions() -> tuple[Path, Path]:
    figure_text = """# Figure Captions

## Figure 1. Calibration Curves under bge-m3
Caption: Mean measured projection ratio by target alpha for each model, with 95% normal-approximation standard-error bands over generated samples. The dashed diagonal is ideal calibration.
Key takeaway: All models are broadly monotonic, but Qwen3.5-9B stays closest to the diagonal.
Use in paper section: Results.

## Figure 2. Calibration Curves under text2vec
Caption: Same calibration view as Figure 1 using text2vec embeddings.
Key takeaway: text2vec reproduces the same model ordering as bge-m3, supporting embedding robustness.
Use in paper section: Results / Robustness.

## Figure 3. Projection MAE with Bootstrap Confidence Intervals
Caption: Projection MAE by model and embedding with scenario-bootstrap 95% confidence intervals.
Key takeaway: Qwen3.5-9B has the lowest projection MAE under both embeddings.
Use in paper section: Results.

## Figure 4. Axis Heatmap of Projection MAE
Caption: Axis-level projection MAE averaged across embeddings, with darker cells indicating larger calibration error.
Key takeaway: Axis difficulty varies substantially; detail/concision is a recurring hard axis.
Use in paper section: Results / Axis analysis.

## Figure 5. Axis Heatmap of Normalized Off-axis Drift
Caption: Axis-level normalized off-axis drift averaged across embeddings.
Key takeaway: Drift patterns differ from projection MAE, separating ratio-placement errors from off-axis geometry.
Use in paper section: Diagnostics.

## Figure 6. Geometry Decomposition
Caption: Model-level grouped bars for projection MAE, normalized off-axis drift, and interpolation distance.
Key takeaway: 35B-A3B has lower drift but worse projection calibration than 9B, so 9B's advantage is not simply lower off-axis movement.
Use in paper section: Diagnostics.

## Figure 7. Length Violation Heatmap
Caption: Length violation rate by axis and model.
Key takeaway: 9B has the best length compliance, while 4B is often too short and 35B-A3B is often too long.
Use in paper section: Diagnostics.

## Figure 8. Projection vs Scalar Judge Scatter
Caption: Sampled bge-m3 projection ratios against scalar judge style-B ratios, colored by model.
Key takeaway: Projection and scalar judgments are correlated, but scalar remains auxiliary due to judge asymmetry.
Use in paper section: Auxiliary validation.

## Figure 9. Nearest-neighbor Mean Rank
Caption: Mean nearest-neighbor rank of generated outputs relative to target interpolation points, faceted by embedding through grouped bars.
Key takeaway: Nearest-neighbor behavior agrees with the main projection ranking.
Use in paper section: Diagnostics.

## Figure 10. Pairwise Auxiliary Audit
Caption: Pairwise higher-alpha win rate, monotonic violation rate, and bidirectional consistency by model.
Key takeaway: Pairwise judgments broadly support monotonicity but show noise and order/judge effects.
Use in paper section: Auxiliary validation.
"""
    table_text = """# Table Captions

## Table 1. Main Model Comparison
Caption: Model-level projection and geometry metrics averaged across bge-m3 and text2vec, with length violation rates.
Key takeaway: Qwen3.5-9B has the best style-ratio calibration.
Use in paper section: Results.

## Table 2. Embedding Robustness
Caption: Projection MAE, Spearman correlation, off-axis drift, and normalized drift by embedding and model.
Key takeaway: bge-m3 and text2vec agree on the model ranking.
Use in paper section: Robustness.

## Table 3. Axis-level Projection
Caption: Axis-level projection, drift, nearest-neighbor, and length metrics by model.
Key takeaway: Style axes differ in difficulty, with detail/concision among the harder axes.
Use in paper section: Axis analysis.

## Table 4. Geometry Decomposition
Caption: Model-level decomposition of projection MAE, interpolation distance, off-axis drift, normalized drift, and out-of-range rate.
Key takeaway: 9B's advantage is ratio-placement calibration, not lower off-axis drift.
Use in paper section: Diagnostics.

## Table 5. Length Diagnostics
Caption: Length-compliance metrics and length/projection-error correlations.
Key takeaway: Length violations are material but do not explain away the projection ranking.
Use in paper section: Diagnostics.

## Table 6. Scalar and Pairwise Auxiliary Metrics
Caption: Scalar judge and sampled pairwise metrics with judge model assignment.
Key takeaway: Auxiliary metrics support broad monotonicity but should not be treated as primary due to judge asymmetry and order effects.
Use in paper section: Auxiliary validation.

## Table 7. Nearest-neighbor and Out-of-range Diagnostics
Caption: Nearest-neighbor rank, top-1 accuracy, out-of-range projection rate, and low/high-alpha overshoot diagnostics by embedding.
Key takeaway: Extreme out-of-range failures are rare; most errors are in-range calibration errors.
Use in paper section: Diagnostics.
"""
    fig_path = OUT_CAPTIONS / "figure_captions.md"
    table_path = OUT_CAPTIONS / "table_captions.md"
    fig_path.write_text(figure_text, encoding="utf-8")
    table_path.write_text(table_text, encoding="utf-8")
    return fig_path, table_path


def write_digest(outputs: dict[str, pd.DataFrame]) -> Path:
    table1 = outputs["table_1_main_model_comparison"]
    t1_by_model = table1.set_index("Model")
    digest = f"""# Paper Results Digest

## One-paragraph Result Summary

StyleBlend-Bench main8axis_v1 shows that explicit style-ratio prompting yields measurable monotonic calibration across eight Chinese style axes, but model scale does not translate monotonically into better ratio control. Across bge-m3 and text2vec projection metrics, Qwen3.5-9B is the best calibrated model, Qwen3.5-35B-A3B is close but less accurately placed along the style-ratio axis, and Qwen3.5-4B is weakest. Auxiliary scalar and pairwise audits broadly support the projection-based findings but should not be used as primary metrics because judge assignment is asymmetric and pairwise order effects are visible.

## Paper-ready Findings

### Finding 1: Qwen3.5-9B achieves the best style-ratio calibration.
Evidence: `tables/table_1_main_model_comparison.md`, `figures/fig_3_model_comparison_projection_mae_ci.pdf`.
Numbers: Qwen3.5-9B has mean projection MAE {t1_by_model.loc['Qwen3.5-9B', 'Projection MAE mean across embeddings']:.3f} and mean Spearman {t1_by_model.loc['Qwen3.5-9B', 'Projection Spearman mean across embeddings']:.3f}.
Interpretation: The mid-sized model follows requested style ratios better than both smaller and larger Qwen-family variants.

### Finding 2: The embedding choice does not change the model ranking.
Evidence: `tables/table_2_embedding_robustness.md`, `figures/fig_1_model_calibration_curves_bge_m3.pdf`, `figures/fig_2_model_calibration_curves_text2vec.pdf`.
Numbers: 9B has the lowest projection MAE under both bge-m3 and text2vec.
Interpretation: The core ranking is robust to the two independent embedding spaces used here.

### Finding 3: 9B's advantage is ratio placement rather than lower off-axis drift.
Evidence: `tables/table_4_geometry_decomposition.md`, `figures/fig_6_geometry_decomposition_bar.pdf`.
Numbers: 35B-A3B has lower normalized drift than 9B, but 9B has lower projection MAE.
Interpretation: Staying near the interpolation line is not sufficient; the model must land at the correct location along that line.

### Finding 4: Axis difficulty is heterogeneous.
Evidence: `tables/table_3_axis_level_projection.md`, `tables/table_3_axis_projection_mae_pivot.md`, `figures/fig_4_axis_heatmap_projection_mae.pdf`.
Numbers: Projection MAE varies visibly by axis and model.
Interpretation: Style-ratio controllability should be reported per axis, not only as a global average.

### Finding 5: Length compliance is a real diagnostic but not the main explanation.
Evidence: `tables/table_5_length_diagnostics.md`, `figures/fig_7_length_violation_heatmap.pdf`.
Numbers: 9B has the lowest length violation rate; 4B has the highest.
Interpretation: Length behavior affects benchmark reliability and should be reported, but current length-error correlations do not explain away the projection ranking.

### Finding 6: Scalar judge agrees broadly but remains auxiliary.
Evidence: `tables/table_6_scalar_and_pairwise_auxiliary.md`, `figures/fig_8_projection_vs_scalar_judge_scatter.pdf`.
Numbers: Scalar MAE ranks 9B and 35B-A3B close together and 4B weakest.
Interpretation: Scalar judgments are useful validation but should not replace geometry because the judge model differs across evaluated models.

### Finding 7: Pairwise judgments support monotonicity but reveal noise and bias.
Evidence: `tables/table_6_scalar_and_pairwise_auxiliary.md`, `figures/fig_10_pairwise_auxiliary_audit.pdf`.
Numbers: Higher-alpha win rates are above chance for all models, but bidirectional consistency is imperfect.
Interpretation: Pairwise audit is best framed as an auxiliary sanity check, not a primary benchmark score.

## Claims That Are Safe to Write

- Qwen3.5-9B is best calibrated among the three tested Qwen-family models under the current main8axis_v1 setup.
- bge-m3 and text2vec agree on the main ranking.
- 35B-A3B's lower off-axis drift does not translate into better projection calibration than 9B.
- 4B is weakest and has the largest length-compliance issue.
- Scalar and pairwise audits are auxiliary checks.

## Claims That Should Not Be Written Yet

- Do not claim universal superiority of 9B across model families; no cross-family generation has been run.
- Do not claim human-validated calibration; the human validation sample is prepared but not annotated.
- Do not treat scalar judge or pairwise judge scores as primary metrics.
- Do not claim temperature robustness; the temperature ablation plan exists but has not been executed.

## Missing Evidence / Next Work

- Human annotations for `data/human_eval/human_validation_sample_v1.jsonl`.
- Cross-family subset generation after confirming a non-Qwen instruction-tuned local model path.
- Temperature ablation if reviewers ask whether sampling temperature drives calibration.
- Optional external judge to reduce asymmetric judge caveats in scalar/pairwise validation.
"""
    path = OUT_DOCS / "paper_results_digest.md"
    path.write_text(digest, encoding="utf-8")
    return path


def write_manifest(table_paths: list[Path], figure_paths: list[Path], caption_paths: list[Path], digest_path: Path, missing_inputs: list[str]) -> Path:
    copied_script = OUT_SCRIPTS / Path(__file__).name
    shutil.copy2(Path(__file__), copied_script)
    lines = [
        "# main8axis_v1 Paper Assets Manifest",
        "",
        f"Generated from: `{FINAL_DIR}`",
        f"Output directory: `{OUT_DIR}`",
        "",
        f"missing_inputs: {missing_inputs}",
        f"created_tables: {len(table_paths)}",
        f"created_figures: {len(figure_paths)}",
        "",
        "## Tables",
        "",
    ]
    lines.extend(f"- `{path}`" for path in sorted(table_paths))
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{path}`" for path in sorted(figure_paths))
    lines.extend(["", "## Captions", ""])
    lines.extend(f"- `{path}`" for path in sorted(caption_paths))
    lines.extend(["", "## Docs", "", f"- `{digest_path}`", f"- `{copied_script}`"])
    manifest = OUT_DOCS / "asset_manifest.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ensure_dirs()
    missing_inputs = [str(path) for path in list(INPUT_TABLES.values()) + PROJECTION_JSONL + SCALAR_JSONL if not path.exists()]
    if missing_inputs:
        print(f"created_figures: 0")
        print(f"created_tables: 0")
        print(f"missing_inputs: {missing_inputs}")
        print(f"missing_count = {len(missing_inputs)}")
        raise SystemExit(1)

    tables, projection, scalar = load_inputs()
    table_paths, table_outputs = prepare_tables(tables, projection)
    figure_paths = make_figures(tables, projection, scalar)
    caption_paths = list(write_captions())
    digest_path = write_digest(table_outputs)
    manifest = write_manifest(table_paths, figure_paths, caption_paths, digest_path, missing_inputs)

    print(f"created_figures: {len(figure_paths)}")
    print(f"created_tables: {len(table_paths)}")
    print(f"missing_inputs: {missing_inputs}")
    print(f"missing_count = {len(missing_inputs)}")
    print(f"asset_manifest: {manifest}")


if __name__ == "__main__":
    main()
