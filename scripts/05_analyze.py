from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import read_jsonl


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def safe_spearman(df: pd.DataFrame, measured: str) -> float:
    if len(df) < 2 or df["alpha"].nunique() < 2:
        return float("nan")
    return float(df["alpha"].corr(df[measured], method="spearman"))


def smoothness(group: pd.DataFrame, measured: str) -> float:
    ordered = group.sort_values("alpha")
    values = ordered.groupby("alpha")[measured].mean().dropna().to_numpy()
    if len(values) < 3:
        return float("nan")
    return float(np.mean(np.abs(values[2:] - 2 * values[1:-1] + values[:-2])))


def endpoint_attraction(df: pd.DataFrame, measured: str) -> float:
    middle = df[df["alpha"].isin([0.25, 0.5, 0.75])].copy()
    if middle.empty:
        return float("nan")
    attracted = (middle[measured] < 0.15) | (middle[measured] > 0.85)
    return float(attracted.mean())


def zh_char_count(text: object) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def length_bounds(value: object) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return float("nan"), float("nan")


def bootstrap_ci_by_scenario(df: pd.DataFrame, value_col: str, iterations: int, seed: int) -> tuple[float, float, float]:
    valid = df[["scenario_id", value_col]].dropna()
    if valid.empty:
        return float("nan"), float("nan"), float("nan")
    by_scenario = [group[value_col].to_numpy(dtype=float) for _scenario, group in valid.groupby("scenario_id")]
    if not by_scenario:
        return float("nan"), float("nan"), float("nan")
    observed = float(np.mean(np.concatenate(by_scenario)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        sampled = [by_scenario[idx] for idx in rng.integers(0, len(by_scenario), size=len(by_scenario))]
        draws.append(float(np.mean(np.concatenate(sampled))))
    return observed, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def read_many_jsonl(paths: list[str] | None) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def write_generation_summary(gens: pd.DataFrame, out_dir: Path) -> None:
    gens = gens.copy()
    if "output" in gens and "length_range_zh_chars" in gens:
        gens["zh_chars"] = gens["output"].map(zh_char_count)
        bounds = gens["length_range_zh_chars"].map(length_bounds)
        gens["min_zh_chars"] = bounds.map(lambda item: item[0])
        gens["max_zh_chars"] = bounds.map(lambda item: item[1])
        gens["length_violation"] = (gens["zh_chars"] < gens["min_zh_chars"]) | (gens["zh_chars"] > gens["max_zh_chars"])
    else:
        gens["length_violation"] = np.nan
    rows = []
    for (model, axis_id), group in gens.groupby(["model", "axis_id"]):
        rows.append(
            {
                "model": model,
                "axis_id": axis_id,
                "n_generations": len(group),
                "n_scenarios": group["scenario_id"].nunique(),
                "n_seeds": group["seed"].nunique(),
                "n_ratios": group["alpha"].nunique(),
                "mean_output_chars": group["output"].fillna("").map(len).mean(),
                "mean_zh_chars": group["zh_chars"].mean() if "zh_chars" in group else np.nan,
                "length_violation_rate": group["length_violation"].mean() if "length_violation" in group else np.nan,
                "think_block_rate": group["output"].fillna("").str.contains("<think", case=False, regex=False).mean(),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "tables/generation_summary.csv", index=False)


def write_scalar_metrics(gens: pd.DataFrame, scalar: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if scalar.empty:
        return pd.DataFrame()
    scalar["style_b_ratio"] = pd.to_numeric(scalar["style_b_ratio"], errors="coerce")
    scalar["alpha"] = pd.to_numeric(scalar["alpha"], errors="coerce")
    scalar["abs_error"] = (scalar["style_b_ratio"] - scalar["alpha"]).abs()

    main_rows = []
    for model, group in scalar.groupby("model"):
        main_rows.append(
            {
                "model": model,
                "judge_mae": group["abs_error"].mean(),
                "judge_spearman": safe_spearman(group, "style_b_ratio"),
                "content_preservation_mean": pd.to_numeric(group["content_preservation"], errors="coerce").mean(),
                "naturalness_mean": pd.to_numeric(group["style_mixture_naturalness"], errors="coerce").mean(),
                "endpoint_attraction": endpoint_attraction(group, "style_b_ratio"),
                "parse_ok_rate": group["parse_ok"].mean() if "parse_ok" in group else np.nan,
                "n": len(group),
            }
        )
    main = pd.DataFrame(main_rows)
    main.to_csv(out_dir / "tables/main_metrics.csv", index=False)

    axis_rows = []
    for (model, axis_id), group in scalar.groupby(["model", "axis_id"]):
        axis_rows.append(
            {
                "model": model,
                "axis_id": axis_id,
                "judge_mae": group["abs_error"].mean(),
                "judge_spearman": safe_spearman(group, "style_b_ratio"),
                "smoothness": smoothness(group, "style_b_ratio"),
                "endpoint_attraction": endpoint_attraction(group, "style_b_ratio"),
                "content_preservation_mean": pd.to_numeric(group["content_preservation"], errors="coerce").mean(),
                "n": len(group),
            }
        )
    pd.DataFrame(axis_rows).to_csv(out_dir / "tables/axis_metrics.csv", index=False)
    return scalar


def write_projection_metrics(projection: pd.DataFrame, out_dir: Path) -> None:
    if projection.empty:
        return
    projection["projection_ratio_clipped"] = pd.to_numeric(projection["projection_ratio_clipped"], errors="coerce")
    projection["alpha"] = pd.to_numeric(projection["alpha"], errors="coerce")
    projection["abs_error"] = (projection["projection_ratio_clipped"] - projection["alpha"]).abs()
    projection["interpolation_distance"] = pd.to_numeric(projection.get("interpolation_distance"), errors="coerce")
    projection["off_axis_drift"] = pd.to_numeric(projection.get("off_axis_drift"), errors="coerce")
    projection["nearest_neighbor_rank"] = pd.to_numeric(projection.get("nearest_neighbor_rank"), errors="coerce")
    rows = []
    for model, group in projection.groupby("model"):
        rows.append(
            {
                "model": model,
                "projection_mae": group["abs_error"].mean(),
                "projection_spearman": safe_spearman(group, "projection_ratio_clipped"),
                "projection_endpoint_attraction": endpoint_attraction(group, "projection_ratio_clipped"),
                "interpolation_distance_mean": group["interpolation_distance"].mean(),
                "off_axis_drift_mean": group["off_axis_drift"].mean(),
                "nearest_neighbor_top1_rate": (group["nearest_neighbor_rank"] == 1).mean(),
                "nearest_neighbor_mean_rank": group["nearest_neighbor_rank"].mean(),
                "n": len(group),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "tables/projection_metrics.csv", index=False)

    axis_rows = []
    for (model, axis_id), group in projection.groupby(["model", "axis_id"]):
        axis_rows.append(
            {
                "model": model,
                "axis_id": axis_id,
                "projection_mae": group["abs_error"].mean(),
                "projection_spearman": safe_spearman(group, "projection_ratio_clipped"),
                "smoothness": smoothness(group, "projection_ratio_clipped"),
                "endpoint_attraction": endpoint_attraction(group, "projection_ratio_clipped"),
                "interpolation_distance_mean": group["interpolation_distance"].mean(),
                "off_axis_drift_mean": group["off_axis_drift"].mean(),
                "nearest_neighbor_top1_rate": (group["nearest_neighbor_rank"] == 1).mean(),
                "n": len(group),
            }
        )
    pd.DataFrame(axis_rows).to_csv(out_dir / "tables/projection_axis_metrics.csv", index=False)


def write_pairwise_metrics(pairwise: pd.DataFrame, out_dir: Path) -> None:
    if pairwise.empty:
        return
    pairwise["left_alpha"] = pd.to_numeric(pairwise["left_alpha"], errors="coerce")
    pairwise["right_alpha"] = pd.to_numeric(pairwise["right_alpha"], errors="coerce")
    comparable = pairwise[pairwise["left_alpha"] != pairwise["right_alpha"]].copy()
    if comparable.empty:
        return
    valid_winner = comparable["winner"].isin(["text_1", "text_2", "tie"])
    comparable["valid_pairwise"] = comparable.get("parse_ok", True).astype(bool) & valid_winner
    if "swap_order" in comparable:
        swapped = comparable["swap_order"].fillna(False).astype(bool)
    else:
        swapped = pd.Series(False, index=comparable.index)
    expected_normal = np.where(comparable["right_alpha"] > comparable["left_alpha"], "text_2", "text_1")
    expected_swapped = np.where(comparable["right_alpha"] > comparable["left_alpha"], "text_1", "text_2")
    expected = np.where(swapped, expected_swapped, expected_normal)
    comparable["monotonic_violation"] = comparable["valid_pairwise"] & (comparable["winner"] != "tie") & (comparable["winner"] != expected)
    comparable["tie"] = comparable["winner"] == "tie"
    rows = []
    for model, group in comparable.groupby("model"):
        valid = group[group["valid_pairwise"]]
        rows.append(
            {
                "model": model,
                "pairwise_monotonic_violation": valid["monotonic_violation"].mean() if not valid.empty else np.nan,
                "pairwise_tie_rate": valid["tie"].mean() if not valid.empty else np.nan,
                "parse_ok_rate": group["parse_ok"].mean() if "parse_ok" in group else np.nan,
                "n": len(group),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "tables/pairwise_metrics.csv", index=False)


def write_bootstrap_ci(scalar: pd.DataFrame, projection: pd.DataFrame, out_dir: Path, iterations: int, seed: int) -> None:
    rows = []
    if not projection.empty:
        projection = projection.copy()
        projection["alpha"] = pd.to_numeric(projection["alpha"], errors="coerce")
        projection["projection_ratio_clipped"] = pd.to_numeric(projection["projection_ratio_clipped"], errors="coerce")
        projection["projection_abs_error"] = (projection["projection_ratio_clipped"] - projection["alpha"]).abs()
        for model, group in projection.groupby("model"):
            mean, low, high = bootstrap_ci_by_scenario(group, "projection_abs_error", iterations, seed)
            rows.append({"model": model, "metric": "projection_mae", "mean": mean, "ci95_low": low, "ci95_high": high, "bootstrap_unit": "scenario", "iterations": iterations})
    if not scalar.empty:
        scalar = scalar.copy()
        scalar["alpha"] = pd.to_numeric(scalar["alpha"], errors="coerce")
        scalar["style_b_ratio"] = pd.to_numeric(scalar["style_b_ratio"], errors="coerce")
        scalar["judge_abs_error"] = (scalar["style_b_ratio"] - scalar["alpha"]).abs()
        for model, group in scalar.groupby("model"):
            mean, low, high = bootstrap_ci_by_scenario(group, "judge_abs_error", iterations, seed)
            rows.append({"model": model, "metric": "judge_mae", "mean": mean, "ci95_low": low, "ci95_high": high, "bootstrap_unit": "scenario", "iterations": iterations})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "tables/bootstrap_ci.csv", index=False)


def write_length_error_metrics(gens: pd.DataFrame, scalar: pd.DataFrame, projection: pd.DataFrame, out_dir: Path) -> None:
    if gens.empty:
        return
    gens = gens.copy()
    if "output" not in gens or "length_range_zh_chars" not in gens:
        return
    gens["zh_chars"] = gens["output"].map(zh_char_count)
    bounds = gens["length_range_zh_chars"].map(length_bounds)
    gens["min_zh_chars"] = bounds.map(lambda item: item[0])
    gens["max_zh_chars"] = bounds.map(lambda item: item[1])
    gens["length_violation"] = (gens["zh_chars"] < gens["min_zh_chars"]) | (gens["zh_chars"] > gens["max_zh_chars"])
    base = gens[["run_id", "model", "axis_id", "scenario_id", "alpha", "zh_chars", "min_zh_chars", "max_zh_chars", "length_violation"]].copy()
    if not projection.empty:
        proj = projection[["run_id", "projection_ratio_clipped"]].copy()
        proj["projection_ratio_clipped"] = pd.to_numeric(proj["projection_ratio_clipped"], errors="coerce")
        base = base.merge(proj, on="run_id", how="left")
        base["projection_abs_error"] = (base["projection_ratio_clipped"] - pd.to_numeric(base["alpha"], errors="coerce")).abs()
    if not scalar.empty:
        scal = scalar[["run_id", "style_b_ratio"]].copy()
        scal["style_b_ratio"] = pd.to_numeric(scal["style_b_ratio"], errors="coerce")
        base = base.merge(scal, on="run_id", how="left")
        base["judge_abs_error"] = (base["style_b_ratio"] - pd.to_numeric(base["alpha"], errors="coerce")).abs()
    rows = []
    metric_cols = [col for col in ["projection_abs_error", "judge_abs_error"] if col in base]
    for (model, axis_id, violation), group in base.groupby(["model", "axis_id", "length_violation"]):
        row = {
            "model": model,
            "axis_id": axis_id,
            "length_violation": bool(violation),
            "n": len(group),
            "mean_zh_chars": group["zh_chars"].mean(),
        }
        for col in metric_cols:
            row[col] = group[col].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "tables/length_error_metrics.csv", index=False)


def write_combined_main_table(scalar: pd.DataFrame, projection: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path) -> None:
    frames = []
    projection_path = out_dir / "tables/projection_metrics.csv"
    scalar_path = out_dir / "tables/main_metrics.csv"
    pairwise_path = out_dir / "tables/pairwise_metrics.csv"
    if projection_path.exists():
        frames.append(pd.read_csv(projection_path).set_index("model"))
    if scalar_path.exists() and not scalar.empty:
        frames.append(pd.read_csv(scalar_path).set_index("model"))
    if pairwise_path.exists() and not pairwise.empty:
        frames.append(pd.read_csv(pairwise_path).set_index("model"))
    if not frames:
        return
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined.reset_index().to_csv(out_dir / "tables/main_metrics_combined.csv", index=False)


def write_figures(scalar: pd.DataFrame, projection: pd.DataFrame, out_dir: Path) -> None:
    curve_source = scalar.copy()
    measured = "style_b_ratio"
    ylabel = "Judge style B ratio"
    if curve_source.empty and not projection.empty:
        curve_source = projection.copy()
        measured = "projection_ratio_clipped"
        ylabel = "Projection style B ratio"
    if curve_source.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return

    curve_source[measured] = pd.to_numeric(curve_source[measured], errors="coerce")
    curve_source["alpha"] = pd.to_numeric(curve_source["alpha"], errors="coerce")
    curve = curve_source.groupby(["model", "alpha"], as_index=False)[measured].mean()
    plt.figure(figsize=(7, 5))
    for model, group in curve.groupby("model"):
        group = group.sort_values("alpha")
        plt.plot(group["alpha"], group[measured], marker="o", label=model)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Target style B ratio")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "figures/calibration_curves.pdf")
    plt.close()

    axis = curve_source.copy()
    axis["abs_error"] = (axis[measured] - axis["alpha"]).abs()
    heat = axis.groupby(["model", "axis_id"])["abs_error"].mean().reset_index()
    pivot = heat.pivot(index="model", columns="axis_id", values="abs_error").fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(3, len(pivot.index) * 0.8)))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i, model in enumerate(pivot.index):
        for j, axis_id in enumerate(pivot.columns):
            ax.text(j, i, f"{pivot.loc[model, axis_id]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "figures/axis_heatmap.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", nargs="+", required=True)
    parser.add_argument("--scalar-judge-file", nargs="*", default=None)
    parser.add_argument("--projection-file", nargs="*", default=None)
    parser.add_argument("--pairwise-judge-file", nargs="*", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260622)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    gens = read_many_jsonl(args.generation_file)
    if gens.empty:
        raise SystemExit(f"No generation rows found: {args.generation_file}")
    write_generation_summary(gens, out_dir)

    scalar = read_many_jsonl(args.scalar_judge_file)
    scalar = write_scalar_metrics(gens, scalar, out_dir)

    projection = read_many_jsonl(args.projection_file)
    write_projection_metrics(projection, out_dir)

    pairwise = read_many_jsonl(args.pairwise_judge_file)
    write_pairwise_metrics(pairwise, out_dir)
    write_bootstrap_ci(scalar, projection, out_dir, args.bootstrap_iterations, args.bootstrap_seed)
    write_length_error_metrics(gens, scalar, projection, out_dir)
    write_combined_main_table(scalar, projection, pairwise, out_dir)
    write_figures(scalar, projection, out_dir)

    ablation_path = out_dir / "tables/ablation_metrics.csv"
    if not ablation_path.exists():
        pd.DataFrame(columns=["ablation", "model", "metric", "value", "notes"]).to_csv(ablation_path, index=False)
    print(f"Wrote analysis tables under {out_dir / 'tables'}")


if __name__ == "__main__":
    main()
