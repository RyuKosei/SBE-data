"""Analyze the preregistered 2x2 prompt/temperature robustness experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT, read_jsonl_tolerant  # type: ignore
else:
    from .common import REVISION_ROOT, read_jsonl_tolerant


METRICS = ["style_scalar_mae", "naturalness", "content_pass", "character_length"]


def bootstrap_effect(values: pd.DataFrame, iterations: int, seed: int) -> tuple[float, float, float, float]:
    random = np.random.default_rng(seed)
    arrays = [axis.effect.to_numpy(float) for _, axis in values.groupby("style_axis", sort=True)]
    distribution = np.zeros(iterations)
    total = sum(len(array) for array in arrays)
    for array in arrays:
        positions = random.integers(0, len(array), size=(iterations, len(array)))
        distribution += array[positions].sum(axis=1)
    distribution /= total
    observed = float(values.effect.mean())
    deviation = float(values.effect.std(ddof=1))
    return observed, float(np.percentile(distribution, 2.5)), float(np.percentile(distribution, 97.5)), observed / deviation if deviation else np.nan


def calculate_effects(cells: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    records = []
    for metric in METRICS:
        pivot = cells.pivot(
            index=["scenario_id", "style_axis", "model_id", "target_ratio"],
            columns=["prompt_level", "temperature_level"],
            values=metric,
        )
        effects = {
            "prompt_P2_minus_P1": ((pivot[("P2", "low")] + pivot[("P2", "high")]) - (pivot[("P1", "low")] + pivot[("P1", "high")])) / 2,
            "temperature_low_minus_high": ((pivot[("P1", "low")] + pivot[("P2", "low")]) - (pivot[("P1", "high")] + pivot[("P2", "high")])) / 2,
            "prompt_temperature_interaction": (pivot[("P2", "low")] - pivot[("P2", "high")]) - (pivot[("P1", "low")] - pivot[("P1", "high")]),
        }
        for effect_name, values in effects.items():
            effect_frame = values.rename("effect").reset_index().groupby(["scenario_id", "style_axis", "model_id"], as_index=False).effect.mean()
            overall = effect_frame.groupby(["scenario_id", "style_axis"], as_index=False).effect.mean()
            for scope, scoped in [("all_models", overall), *[(model, group) for model, group in effect_frame.groupby("model_id", sort=True)]]:
                estimate, lower, upper, effect_size = bootstrap_effect(scoped, iterations, seed)
                records.append(
                    {
                        "scope": scope,
                        "style_axis": "all",
                        "metric": metric,
                        "effect": effect_name,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "paired_effect_size_dz": effect_size,
                        "scenario_count": scoped.scenario_id.nunique(),
                    }
                )
            for (axis_name, model), scoped in effect_frame.groupby(["style_axis", "model_id"], sort=True):
                estimate, lower, upper, effect_size = bootstrap_effect(scoped, iterations, seed)
                records.append(
                    {
                        "scope": model,
                        "style_axis": axis_name,
                        "metric": metric,
                        "effect": effect_name,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "paired_effect_size_dz": effect_size,
                        "scenario_count": scoped.scenario_id.nunique(),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, default=REVISION_ROOT / "data/robustness_outputs.jsonl")
    parser.add_argument("--evaluation", type=Path, default=REVISION_ROOT / "metrics/robustness_llm_evaluations.jsonl")
    parser.add_argument("--quality", type=Path, default=REVISION_ROOT / "metrics/robustness_text_quality.parquet")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    outputs = pd.DataFrame(read_jsonl_tolerant(args.generation).rows)
    evaluations = pd.DataFrame(read_jsonl_tolerant(args.evaluation).rows)
    if outputs.run_id.nunique() != 9600 or evaluations.run_id.nunique() != 9600:
        raise SystemExit(f"robustness data incomplete: outputs={outputs.run_id.nunique()}, evaluations={evaluations.run_id.nunique()}")
    quality = pd.read_parquet(args.quality)
    frame = outputs.merge(
        evaluations[["run_id", "style_B_intensity", "naturalness", "content_preservation_pass"]],
        on="run_id",
        validate="one_to_one",
    ).merge(quality[["run_id", "character_length"]], on="run_id", validate="one_to_one")
    frame["style_scalar_mae"] = (frame.style_B_intensity / 100 - frame.target_ratio).abs()
    frame["content_pass"] = frame.content_preservation_pass.astype(float)
    frame["prompt_level"] = frame.prompt_template_id.map({"computers_explicit_ratio_v1": "P1", "computers_explicit_ratio_v2": "P2"})
    frame["temperature_level"] = np.where(np.isclose(frame.temperature, 0.2), "low", "high")
    cells = frame.groupby(
        ["scenario_id", "style_axis", "model_id", "target_ratio", "prompt_level", "temperature_level"],
        as_index=False,
    )[METRICS].mean()
    if len(cells) != 3200:
        raise ValueError(f"expected 3200 seed-aggregated factorial cells, found {len(cells)}")
    effects = calculate_effects(cells, args.iterations, args.seed)
    effects.to_csv(REVISION_ROOT / "statistics/robustness_factorial_effects.csv", index=False)

    condition = cells.groupby(["model_id", "prompt_level", "temperature_level"], as_index=False)[METRICS].mean()
    condition.to_csv(REVISION_ROOT / "metrics/robustness_condition_summary.csv", index=False)
    rank_records = []
    conditions = sorted({(row.prompt_level, row.temperature_level) for row in condition.itertuples()})
    for metric in METRICS:
        pivot = condition.pivot(index="model_id", columns=["prompt_level", "temperature_level"], values=metric)
        for first, second in itertools.combinations(conditions, 2):
            rank_records.append(
                {
                    "metric": metric,
                    "condition_first": "/".join(first),
                    "condition_second": "/".join(second),
                    "spearman_model_ranking": float(spearmanr(pivot[first], pivot[second]).statistic),
                    "model_count": len(pivot),
                }
            )
    pd.DataFrame(rank_records).to_csv(REVISION_ROOT / "statistics/robustness_model_ranking.csv", index=False)
    shared = pd.read_csv(REVISION_ROOT / "data/shared_anchors_final.csv")
    geometry_status = "complete" if (shared.approval_status == "approved").all() else "external_pending"
    pd.DataFrame(
        [{"status": geometry_status, "note": "The required shared-anchor geometric factorial analysis awaits the two-author reference endpoint review. Fixed-judge scalar, content, naturalness, and length effects are complete."}]
    ).to_csv(REVISION_ROOT / "statistics/robustness_shared_anchor_status.csv", index=False)
    print(json.dumps({"factorial_cells": len(cells), "effect_rows": len(effects), "geometry_status": geometry_status}))


if __name__ == "__main__":
    main()
