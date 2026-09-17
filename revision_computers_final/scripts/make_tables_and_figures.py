"""Create publication tables and accessible PNG/PDF figures from frozen results."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
MODEL_LABELS = {
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "qwen35_35b_a3b": "Qwen3.5-35B-A3B",
    "qwen36_27b": "Qwen3.6-27B",
    "qwen36_35b_a3b": "Qwen3.6-35B-A3B",
}
ENCODER_LABELS = {
    "bge_m3": "BGE-M3",
    "text2vec_base_chinese": "text2vec",
    "multilingual_e5_large_instruct": "mE5-large",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    root = REVISION_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_mean(values_by_axis: list[np.ndarray], iterations: int, random: np.random.Generator) -> tuple[float, float]:
    total = sum(len(values) for values in values_by_axis)
    distribution = np.zeros(iterations)
    for values in values_by_axis:
        positions = random.integers(0, len(values), size=(iterations, len(values)))
        distribution += values[positions].sum(axis=1)
    distribution /= total
    return float(np.nanpercentile(distribution, 2.5)), float(np.nanpercentile(distribution, 97.5))


def pending_figure(name: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.39, "EXTERNAL PENDING", ha="center", va="center", fontsize=13, color=COLORS[4], weight="bold")
    ax.text(0.5, 0.20, reason, ha="center", va="center", fontsize=9, wrap=True)
    save(fig, name)


def framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 1.45, 1.8, 1.0, "320 scenarios\n8 style axes"),
        (2.4, 1.45, 1.8, 1.0, "Qwen panel\n5 checkpoints"),
        (4.6, 1.45, 1.8, 1.0, "7 ratios x 3 seeds\n33,600 texts"),
        (6.8, 2.35, 1.8, 1.0, "Self/shared\nanchors"),
        (6.8, 0.55, 1.8, 1.0, "3 encoders\n+ fixed judge"),
        (9.0, 1.45, 1.8, 1.0, "ICE, drift, path,\nquality, efficiency"),
    ]
    for index, (x, y, w, h, label) in enumerate(boxes):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=COLORS[index % len(COLORS)] + "22", ec=COLORS[index % len(COLORS)], lw=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")
    arrows = [((2.0, 1.95), (2.4, 1.95)), ((4.2, 1.95), (4.6, 1.95)), ((6.4, 1.95), (6.8, 2.75)), ((6.4, 1.95), (6.8, 1.05)), ((8.6, 2.75), (9.0, 2.05)), ((8.6, 1.05), (9.0, 1.85))]
    for first, second in arrows:
        ax.add_patch(FancyArrowPatch(first, second, arrowstyle="-|>", mutation_scale=12, color="#555555", lw=1.2))
    ax.set_title("Anchor-aware evaluation pipeline", fontsize=14, weight="bold")
    save(fig, "fig01_method_framework")


def projection_schematic() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    point = np.array([0.63, 0.48])
    projection = np.array([point[0], 0.0])
    ideal = np.array([0.5, 0.0])
    ax.plot([0, 1], [0, 0], color=COLORS[0], lw=2.5, label="Endpoint line")
    ax.scatter(*a, color=COLORS[0], s=55); ax.text(-0.03, -0.10, r"$z_A$")
    ax.scatter(*b, color=COLORS[0], s=55); ax.text(0.98, -0.10, r"$z_B$")
    ax.scatter(*point, color=COLORS[4], s=65, zorder=4, label=r"Generated $z_\alpha$")
    ax.scatter(*ideal, color=COLORS[2], marker="x", s=80, zorder=4, label="Ideal location")
    ax.plot([point[0], projection[0]], [point[1], projection[1]], "--", color=COLORS[4], lw=1.5)
    ax.annotate("off-axis drift", (0.65, 0.25), color=COLORS[4])
    ax.annotate("raw projection error", xy=projection, xytext=ideal, arrowprops={"arrowstyle": "<->", "color": COLORS[2]}, ha="center", va="bottom", color=COLORS[2])
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.15, 0.75)
    ax.set_xlabel("Endpoint-axis coordinate"); ax.set_ylabel("Orthogonal component")
    ax.set_title("Raw projection and normalized off-axis drift")
    ax.legend(loc="upper left")
    save(fig, "fig02_projection_drift_schematic")


def pca_trajectory() -> None:
    trajectory_paths = sorted((REVISION_ROOT / "metrics/fragments/bge_m3").glob("*.trajectory.parquet"))
    trajectories = pd.concat([pd.read_parquet(path) for path in trajectory_paths], ignore_index=True)
    subset = trajectories[(trajectories.model_id == "qwen35_9b") & (trajectories.seed == 1)].copy()
    target = subset.iloc[(subset.interior_calibration_error - subset.interior_calibration_error.median()).abs().argmin()]
    generation = [row for row in read_jsonl_tolerant(REVISION_ROOT / "data/fragments/main/qwen35_9b.jsonl").rows if row["scenario_id"] == target.scenario_id and int(row["seed"]) == 1]
    generation.sort(key=lambda row: float(row["target_ratio"]))
    archive = np.load(REVISION_ROOT / "data/embeddings/bge_m3/qwen35_9b.npz")
    by_id = dict(zip(map(str, archive["run_ids"]), np.asarray(archive["vectors"], dtype=float)))
    matrix = np.stack([by_id[row["run_id"]] for row in generation])
    centered = matrix - matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vh[:2].T
    ratios = [float(row["target_ratio"]) for row in generation]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for index in range(len(points) - 1):
        ax.annotate("", points[index + 1], points[index], arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2})
    scatter = ax.scatter(points[:, 0], points[:, 1], c=ratios, cmap="viridis", s=75, edgecolor="white", linewidth=0.7, zorder=3)
    for point, ratio in zip(points, ratios):
        # Place the rightmost internal-ratio label to the left so that the
        # complete value remains visible beside the color bar.
        if ratio >= 2 / 3 - 1e-9 and point[0] == points[:, 0].max():
            ax.annotate(f"α={ratio:.2f}", point, xytext=(-5, 4), textcoords="offset points", fontsize=8, ha="right")
        else:
            ax.annotate(f"α={ratio:.2f}", point, xytext=(5, 4), textcoords="offset points", fontsize=8)
    fig.colorbar(scatter, ax=ax, label="Target ratio α")
    ax.set_xlabel("Trajectory PC1"); ax.set_ylabel("Trajectory PC2")
    ax.set_title(f"Representative seven-point trajectory\n{target.scenario_id}, Qwen3.5-9B, BGE-M3")
    save(fig, "fig03_typical_pca_trajectory")


def endpoint_ablation_figure() -> None:
    frame = pd.read_csv(REVISION_ROOT / "tables/ablation_5point_7point_raw_clipped.csv")
    frame = frame[frame.encoder_id == "bge_m3"]
    models = list(MODEL_LABELS)
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for offset, (metric, label, color) in enumerate([("mae_7_point_raw", "7 points (endpoints included)", COLORS[1]), ("mae_5_point_raw", "5 interior points", COLORS[0])]):
        data = frame[frame.metric == metric].set_index("model_id").reindex(models)
        position = x + (offset - 0.5) * 0.25
        ax.errorbar(position, data.estimate, yerr=[data.estimate - data.ci_lower, data.ci_upper - data.estimate], fmt="o", capsize=3, color=color, label=label)
    ax.set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=20, ha="right")
    ax.set_ylabel("Mean absolute projection error")
    ax.set_title("Endpoint inclusion mechanically lowers reported error (BGE-M3)")
    ax.legend()
    save(fig, "fig04_5point_vs_7point_error")


def encoder_heatmap() -> None:
    summary = pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv")
    encoders = list(ENCODER_LABELS)
    pivot = summary.pivot(index="model_id", columns="encoder_id", values="interior_calibration_error_estimate")
    matrix = np.empty((3, 3))
    for i, first in enumerate(encoders):
        for j, second in enumerate(encoders):
            matrix[i, j] = spearman_rank(pivot[first].to_numpy(), pivot[second].to_numpy())
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if abs(matrix[i, j]) > 0.6 else "black", weight="bold")
    ax.set_xticks(range(3), [ENCODER_LABELS[e] for e in encoders], rotation=25, ha="right")
    ax.set_yticks(range(3), [ENCODER_LABELS[e] for e in encoders])
    ax.set_title("Model-ranking agreement for ICE")
    fig.colorbar(image, ax=ax, label="Spearman rank correlation")
    save(fig, "fig06_three_encoder_rank_heatmap")


def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def scale_figure() -> None:
    summary = pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv")
    registry = pd.read_csv(REVISION_ROOT / "config/model_registry.csv").set_index("model_id")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for encoder_index, encoder in enumerate(ENCODER_LABELS):
        data = summary[summary.encoder_id == encoder].set_index("model_id").reindex(MODEL_LABELS)
        total = registry.reindex(MODEL_LABELS).parameter_count_total / 1e9
        active = registry.reindex(MODEL_LABELS).parameter_count_active / 1e9
        axes[0].errorbar(total, data.interior_calibration_error_estimate, yerr=[data.interior_calibration_error_estimate - data.interior_calibration_error_ci_lower, data.interior_calibration_error_ci_upper - data.interior_calibration_error_estimate], fmt="o", capsize=2, color=COLORS[encoder_index], label=ENCODER_LABELS[encoder])
        axes[1].errorbar(active, data.normalized_off_axis_drift_estimate, yerr=[data.normalized_off_axis_drift_estimate - data.normalized_off_axis_drift_ci_lower, data.normalized_off_axis_drift_ci_upper - data.normalized_off_axis_drift_estimate], fmt="o", capsize=2, color=COLORS[encoder_index], label=ENCODER_LABELS[encoder])
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("Total parameters (billions)"); axes[0].set_ylabel("ICE")
    axes[1].set_xlabel("Active parameters (billions)"); axes[1].set_ylabel("Normalized off-axis drift")
    axes[0].set_title("Scale vs calibration error"); axes[1].set_title("Active scale vs drift")
    axes[0].legend()
    fig.suptitle("Scale trends are descriptive; version and architecture are confounded")
    save(fig, "fig07_scale_ice_drift")


def robustness_figure() -> None:
    path = REVISION_ROOT / "statistics/robustness_factorial_effects.csv"
    if not path.exists():
        pending_figure("fig08_prompt_temperature_robustness", "Prompt/temperature robustness", "Fixed-judge analysis has not completed.")
        return
    frame = pd.read_csv(path)
    frame = frame[(frame.scope == "all_models") & (frame.style_axis == "all") & frame.metric.isin(["style_scalar_mae", "naturalness"])]
    effects = ["prompt_P2_minus_P1", "temperature_low_minus_high", "prompt_temperature_interaction"]
    labels = ["Prompt", "Temperature", "Interaction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, metric, title in zip(axes, ["style_scalar_mae", "naturalness"], ["Style-ratio absolute error", "Naturalness"]):
        data = frame[frame.metric == metric].set_index("effect").reindex(effects)
        ax.errorbar(np.arange(3), data.estimate, yerr=[data.estimate - data.ci_lower, data.ci_upper - data.estimate], fmt="o", capsize=4, color=COLORS[0])
        ax.axhline(0, color="#555555", lw=1)
        ax.set_xticks(range(3), labels, rotation=20)
        ax.set_ylabel("Factorial effect")
        ax.set_title(title)
    fig.suptitle("Prompt and temperature robustness (fixed scalar/content judge)")
    save(fig, "fig08_prompt_temperature_robustness")


def failure_figure() -> None:
    path = REVISION_ROOT / "metrics/failure_taxonomy_counts.csv"
    if not path.exists():
        pending_figure("fig09_failure_taxonomy", "Failure taxonomy", "Classification awaits complete content judgments.")
        return
    frame = pd.read_csv(path)
    frame = frame[frame.aggregation == "overall"].sort_values("proportion")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(frame.failure_type.str.replace("_", " "), frame.proportion, color=COLORS[0])
    ax.set_xlabel("Proportion of scenario-model-encoder cells")
    ax.set_xlim(0, min(1, frame.proportion.max() * 1.15))
    ax.set_title("Preregistered control failure taxonomy")
    for index, value in enumerate(frame.proportion):
        ax.text(value, index, f" {value:.1%}", va="center", fontsize=8)
    save(fig, "fig09_failure_taxonomy")


def pareto_figure() -> None:
    path = REVISION_ROOT / "statistics/efficiency_pareto.csv"
    if not path.exists():
        pending_figure("fig10_quality_efficiency_pareto", "Quality-efficiency Pareto frontier", "Efficiency aggregation has not completed.")
        return
    frame = pd.read_csv(path)
    summary = pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv")
    ci = summary.groupby("model_id").agg(lower=("interior_calibration_error_ci_lower", "min"), upper=("interior_calibration_error_ci_upper", "max")).reset_index()
    frame = frame.merge(ci, on="model_id", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for index, row in frame.iterrows():
        color = COLORS[2] if row.global_descriptive_pareto_ICE_latency else COLORS[0]
        ax.errorbar(row.request_latency_p95_seconds, row.mean_three_encoder_ICE, yerr=[[row.mean_three_encoder_ICE - row.lower], [row.upper - row.mean_three_encoder_ICE]], fmt="o", color=color, capsize=3)
        ax.annotate(MODEL_LABELS[row.model_id], (row.request_latency_p95_seconds, row.mean_three_encoder_ICE), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("P95 request latency (seconds)"); ax.set_ylabel("Mean three-encoder ICE")
    ax.set_title("Descriptive quality–latency view\nDirect comparisons only within identical concurrency/TP groups")
    save(fig, "fig10_quality_efficiency_pareto")


def tables() -> None:
    generation = pd.DataFrame(read_jsonl_tolerant(REVISION_ROOT / "data/qwen_family_outputs.jsonl").rows)
    registry = pd.read_csv(REVISION_ROOT / "config/model_registry.csv")
    config_rows = []
    for model, group in generation.groupby("model_id", sort=True):
        config_rows.append(
            {
                "model_id": model,
                "outputs": len(group),
                "scenarios": group.scenario_id.nunique(),
                "style_axes": group.style_axis.nunique(),
                "target_ratios": group.target_ratio.nunique(),
                "seeds": group.seed.nunique(),
                "temperature": group.temperature.unique()[0],
                "top_p": group.top_p.unique()[0],
                "max_new_tokens": group.max_new_tokens.unique()[0],
                "empty_outputs": int((group.generated_text.str.strip() == "").sum()),
                "length_finished": int((group.finish_reason == "length").sum()),
            }
        )
    pd.DataFrame(config_rows).merge(registry, on="model_id", validate="one_to_one").to_csv(REVISION_ROOT / "tables/table01_dataset_models_generation.csv", index=False)
    pd.read_csv(REVISION_ROOT / "metrics/model_level_summary.csv").to_csv(REVISION_ROOT / "tables/table02_all_qwen_main_metrics.csv", index=False)

    trajectory_paths = sorted((REVISION_ROOT / "metrics/fragments").glob("*/*.trajectory.parquet"))
    trajectory = pd.concat([pd.read_parquet(path) for path in trajectory_paths], ignore_index=True)
    trajectory = trajectory.groupby(["scenario_id", "style_axis", "model_id", "encoder_id"], as_index=False).mean(numeric_only=True)
    metrics = ["trajectory_pca_pc1_explained_variance", "trajectory_endpoint_pc1_abs_cosine", "trajectory_ordered_path_length_ratio", "trajectory_discrete_curvature_mean_degrees", "trajectory_negative_local_slope_count"]
    records = []
    random = np.random.default_rng(20260916)
    for (model, encoder), group in trajectory.groupby(["model_id", "encoder_id"], sort=True):
        for metric in metrics:
            arrays = [axis[metric].to_numpy(float) for _, axis in group.groupby("style_axis", sort=True)]
            lower, upper = bootstrap_mean(arrays, 10000, random)
            records.append({"model_id": model, "encoder_id": encoder, "metric": metric, "estimate": group[metric].mean(), "ci_lower": lower, "ci_upper": upper})
    pd.DataFrame(records).to_csv(REVISION_ROOT / "tables/table05_linearity_diagnostics.csv", index=False)

    pd.read_csv(REVISION_ROOT / "tables/three_encoder_ranking_consistency.csv").to_csv(
        REVISION_ROOT / "tables/table04_three_encoder_robustness.csv", index=False
    )
    shared_status = pd.read_csv(REVISION_ROOT / "statistics/shared_anchor_status.csv")
    shared_status.to_csv(REVISION_ROOT / "tables/table03_self_shared_anchor_status.csv", index=False)
    human_reliability = REVISION_ROOT / "statistics/human_rater_reliability.csv"
    if human_reliability.exists():
        human_table = pd.read_csv(human_reliability)
    else:
        human_table = pd.DataFrame(
            [{
                "status": "external_pending",
                "required_unique_texts": 320,
                "required_formal_ratings": 960,
                "note": "No real ratings were simulated; ICC, alpha, kappa, and human-machine agreement await annotation.",
            }]
        )
    human_table.to_csv(REVISION_ROOT / "tables/table07_human_machine_rater_agreement.csv", index=False)
    pd.read_csv(REVISION_ROOT / "statistics/efficiency_pareto.csv").to_csv(
        REVISION_ROOT / "tables/table08_efficiency_pareto.csv", index=False
    )
    robustness_path = REVISION_ROOT / "statistics/robustness_factorial_effects.csv"
    if robustness_path.exists():
        pd.read_csv(robustness_path).to_csv(
            REVISION_ROOT / "tables/table09_prompt_temperature_robustness.csv", index=False
        )
    failure_path = REVISION_ROOT / "metrics/failure_taxonomy_counts.csv"
    if failure_path.exists():
        pd.read_csv(failure_path).to_csv(
            REVISION_ROOT / "tables/table10_failure_taxonomy.csv", index=False
        )


def main() -> None:
    style()
    tables()
    framework_figure()
    projection_schematic()
    pca_trajectory()
    endpoint_ablation_figure()
    pending_figure("fig05_self_vs_shared_ranking", "Self-anchor vs shared-anchor model ranking", "The 80 shared endpoint pairs await independent rewriting and two-author approval; no reference text was fabricated.")
    encoder_heatmap()
    scale_figure()
    robustness_figure()
    failure_figure()
    pareto_figure()
    print(json.dumps({"png_figures": len(list((REVISION_ROOT / 'figures').glob('*.png'))), "pdf_figures": len(list((REVISION_ROOT / 'figures').glob('*.pdf')))}))


if __name__ == "__main__":
    main()
