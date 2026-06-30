from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ROOT, read_jsonl


KEYS = ["model", "axis_id", "scenario_id", "seed", "alpha"]


def zh_char_count(text: object) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in str(text))


def length_bounds(value: object) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return float("nan"), float("nan")


def read_many(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return pd.DataFrame(rows)


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    if x not in df.columns or y not in df.columns:
        return float("nan")
    valid = df[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def summarize_geometry(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "n": len(group),
            "projection_mae": group["projection_error"].mean(),
            "projection_mae_unclipped": group["projection_error_unclipped"].mean(),
            "projection_spearman": safe_corr(group, "alpha", "projection_ratio_clipped", "spearman"),
            "projection_spearman_unclipped": safe_corr(group, "alpha", "projection_ratio_raw", "spearman"),
            "interpolation_distance_mean": group["interpolation_distance"].mean(),
            "off_axis_drift_mean": group["off_axis_drift"].mean(),
            "normalized_off_axis_drift_mean": group["normalized_off_axis_drift"].mean(),
            "endpoint_distance_mean": group["endpoint_distance"].mean(),
            "out_of_range_rate": group["out_of_range"].mean(),
            "nn_top1_accuracy": (group["nn_rank"] == 1).mean(),
            "nn_mean_rank": group["nn_rank"].mean(),
        }
    )


def prepare_projection(paths: list[str]) -> pd.DataFrame:
    projection = read_many(paths).copy()
    numeric_cols = [
        "alpha",
        "endpoint_distance",
        "projection_ratio_raw",
        "projection_ratio_clipped",
        "interpolation_distance",
        "off_axis_drift",
        "nearest_neighbor_rank",
    ]
    for col in numeric_cols:
        projection[col] = pd.to_numeric(projection[col], errors="coerce")
    projection["projection_error"] = (projection["projection_ratio_clipped"] - projection["alpha"]).abs()
    projection["projection_error_unclipped"] = (projection["projection_ratio_raw"] - projection["alpha"]).abs()
    projection["normalized_off_axis_drift"] = np.where(
        projection["endpoint_distance"] > 0,
        projection["off_axis_drift"] / projection["endpoint_distance"],
        np.nan,
    )
    projection["out_of_range"] = (projection["projection_ratio_raw"] < 0) | (projection["projection_ratio_raw"] > 1)
    projection["nn_rank"] = projection["nearest_neighbor_rank"]
    projection["nn_top1"] = projection["nn_rank"] == 1
    return projection


def prepare_generations(paths: list[str]) -> pd.DataFrame:
    gens = read_many(paths).copy()
    gens["alpha"] = pd.to_numeric(gens["alpha"], errors="coerce")
    gens["zh_chars"] = gens["output"].map(zh_char_count)
    bounds = gens["length_range_zh_chars"].map(length_bounds)
    gens["min_zh_chars"] = bounds.map(lambda item: item[0])
    gens["max_zh_chars"] = bounds.map(lambda item: item[1])
    gens["target_mid_zh_chars"] = (gens["min_zh_chars"] + gens["max_zh_chars"]) / 2.0
    gens["length_violation"] = (gens["zh_chars"] < gens["min_zh_chars"]) | (gens["zh_chars"] > gens["max_zh_chars"])
    gens["too_short"] = gens["zh_chars"] < gens["min_zh_chars"]
    gens["too_long"] = gens["zh_chars"] > gens["max_zh_chars"]
    gens["length_deviation"] = np.select(
        [gens["too_short"], gens["too_long"]],
        [gens["min_zh_chars"] - gens["zh_chars"], gens["zh_chars"] - gens["max_zh_chars"]],
        default=0.0,
    )
    gens["length_deviation_abs"] = (gens["zh_chars"] - gens["target_mid_zh_chars"]).abs()
    return gens


def write_geometry_tables(projection: pd.DataFrame, out_dir: Path) -> None:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    projection.groupby(["embedding_model", "model"]).apply(summarize_geometry, include_groups=False).reset_index().to_csv(
        tables / "geometry_metrics_by_model_embedding.csv", index=False
    )
    projection.groupby(["embedding_model", "axis_id", "model"]).apply(summarize_geometry, include_groups=False).reset_index().to_csv(
        tables / "geometry_metrics_by_axis_model_embedding.csv", index=False
    )
    projection.groupby(["embedding_model", "model", "alpha"]).apply(summarize_geometry, include_groups=False).reset_index().to_csv(
        tables / "geometry_metrics_by_alpha_model_embedding.csv", index=False
    )

    projection.groupby(["embedding_model", "model"]).agg(
        n=("nn_rank", "size"),
        nn_top1_accuracy=("nn_top1", "mean"),
        nn_mean_rank=("nn_rank", "mean"),
        nn_median_rank=("nn_rank", "median"),
        nn_rank_std=("nn_rank", "std"),
    ).reset_index().to_csv(tables / "nearest_neighbor_metrics.csv", index=False)

    projection.groupby(["embedding_model", "model"]).agg(
        n=("out_of_range", "size"),
        out_of_range_rate=("out_of_range", "mean"),
        raw_projection_min=("projection_ratio_raw", "min"),
        raw_projection_max=("projection_ratio_raw", "max"),
        raw_projection_mean=("projection_ratio_raw", "mean"),
        unclipped_error_mean=("projection_error_unclipped", "mean"),
        clipped_error_mean=("projection_error", "mean"),
    ).reset_index().to_csv(tables / "out_of_range_projection_metrics.csv", index=False)


def write_length_tables(gens: pd.DataFrame, projection: pd.DataFrame, out_dir: Path) -> None:
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    merged = projection.merge(
        gens[
            [
                "run_id",
                "zh_chars",
                "min_zh_chars",
                "max_zh_chars",
                "target_mid_zh_chars",
                "length_violation",
                "too_short",
                "too_long",
                "length_deviation",
                "length_deviation_abs",
            ]
        ],
        on="run_id",
        how="left",
    )

    gens.groupby(["model", "axis_id", "alpha"]).agg(
        n=("run_id", "size"),
        mean_zh_chars=("zh_chars", "mean"),
        min_zh_chars=("min_zh_chars", "mean"),
        max_zh_chars=("max_zh_chars", "mean"),
        length_violation_rate=("length_violation", "mean"),
        too_short_rate=("too_short", "mean"),
        too_long_rate=("too_long", "mean"),
        mean_length_deviation=("length_deviation", "mean"),
    ).reset_index().to_csv(tables / "length_diagnostics_by_model_axis_alpha.csv", index=False)

    corr_rows = []
    for (embedding, model), group in merged.groupby(["embedding_model", "model"]):
        corr_rows.append(
            {
                "embedding_model": embedding,
                "model": model,
                "n": len(group),
                "pearson_zh_chars_projection_error": safe_corr(group, "zh_chars", "projection_error", "pearson"),
                "spearman_zh_chars_projection_error": safe_corr(group, "zh_chars", "projection_error", "spearman"),
                "pearson_length_deviation_projection_error": safe_corr(group, "length_deviation", "projection_error", "pearson"),
                "spearman_length_deviation_projection_error": safe_corr(group, "length_deviation", "projection_error", "spearman"),
                "pearson_abs_mid_deviation_projection_error": safe_corr(group, "length_deviation_abs", "projection_error", "pearson"),
                "spearman_abs_mid_deviation_projection_error": safe_corr(group, "length_deviation_abs", "projection_error", "spearman"),
            }
        )
    pd.DataFrame(corr_rows).to_csv(tables / "length_error_correlation_with_projection.csv", index=False)

    compliant = merged[~merged["length_violation"].fillna(True)].copy()
    if not compliant.empty:
        compliant.groupby(["embedding_model", "model"]).apply(summarize_geometry, include_groups=False).reset_index().to_csv(
            tables / "length_compliant_subset_metrics.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(tables / "length_compliant_subset_metrics.csv", index=False)

    gens.groupby(["model", "axis_id"]).agg(
        n=("run_id", "size"),
        length_violation_rate=("length_violation", "mean"),
        too_short_rate=("too_short", "mean"),
        too_long_rate=("too_long", "mean"),
        too_short_count=("too_short", "sum"),
        too_long_count=("too_long", "sum"),
        mean_zh_chars=("zh_chars", "mean"),
    ).reset_index().to_csv(tables / "too_short_too_long_breakdown.csv", index=False)


def write_geometry_report(projection: pd.DataFrame, out_dir: Path) -> None:
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    model_table = projection.groupby(["embedding_model", "model"]).apply(summarize_geometry, include_groups=False).reset_index()
    best_proj = model_table.sort_values(["embedding_model", "projection_mae"])
    lines = [
        "# Curated 3-Axis v2 Geometry Diagnostics",
        "",
        "Created: 2026-06-23 UTC",
        "",
        "## Scope",
        "",
        "This diagnostic report uses existing curated 3-axis v2 projection outputs. No generation was rerun.",
        "",
        "## Main Geometry Summary",
        "",
        "| embedding | model | projection MAE | Spearman | interp dist | off-axis drift | norm off-axis | out-of-range | NN top1 | NN mean rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_proj.itertuples(index=False):
        embedding = str(row.embedding_model).replace("/home/data_cpfs/zicheng/models/", "")
        lines.append(
            f"| {embedding} | {row.model} | {row.projection_mae:.3f} | {row.projection_spearman:.3f} | "
            f"{row.interpolation_distance_mean:.3f} | {row.off_axis_drift_mean:.3f} | "
            f"{row.normalized_off_axis_drift_mean:.3f} | {row.out_of_range_rate:.3f} | "
            f"{row.nn_top1_accuracy:.3f} | {row.nn_mean_rank:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- Both embeddings keep the same projection ranking: qwen35_9b best, qwen35_35b_a3b second, qwen35_4b weakest.",
            "- 9B's advantage is mainly projection calibration. It has the lowest projection MAE and highest projection Spearman under both embeddings. Its normalized off-axis drift is not consistently lower than 35B-A3B.",
            "- 35B-A3B does not show a higher out-of-range rate; its out-of-range rate is lower than or comparable to 9B. Its off-axis drift is often slightly smaller, but its projection calibration is worse.",
            "- 4B's weakness is primarily ratio calibration and monotonicity/projection ordering rather than uniquely larger off-axis drift. Its nearest-neighbor mean rank is also worst under both embeddings.",
            "",
            "## Output Tables",
            "",
            "- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_model_embedding.csv`",
            "- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_axis_model_embedding.csv`",
            "- `results/curated3axis_v2_full_eval/tables/geometry_metrics_by_alpha_model_embedding.csv`",
            "- `results/curated3axis_v2_full_eval/tables/nearest_neighbor_metrics.csv`",
            "- `results/curated3axis_v2_full_eval/tables/out_of_range_projection_metrics.csv`",
        ]
    )
    (docs_dir / "curated3axis_v2_geometry_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-file", nargs="+", required=True)
    parser.add_argument("--projection-file", nargs="+", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "results/curated3axis_v2_full_eval"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    projection = prepare_projection(args.projection_file)
    gens = prepare_generations(args.generation_file)

    write_geometry_tables(projection, out_dir)
    write_length_tables(gens, projection, out_dir)
    write_geometry_report(projection, out_dir)
    print(f"Wrote blocking diagnostics under {out_dir / 'tables'}")
    print(f"Wrote {ROOT / 'docs/curated3axis_v2_geometry_diagnostics.md'}")


if __name__ == "__main__":
    main()
