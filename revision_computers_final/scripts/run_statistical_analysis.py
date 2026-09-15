"""Scenario-level paired inference, fixed hypotheses, and model-scale summaries."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import REVISION_ROOT  # type: ignore
else:
    from .common import REVISION_ROOT


PAIR_METRICS = ["interior_calibration_error", "normalized_off_axis_drift"]
SCALE_METRICS = ["interior_calibration_error", "normalized_off_axis_drift", "spearman_monotonicity"]


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted


def stratified_bootstrap_difference(
    paired: pd.DataFrame, iterations: int, seed: int
) -> tuple[float, float, float, float]:
    random = np.random.default_rng(seed)
    pieces = []
    for _, group in paired.groupby("style_axis", sort=True):
        values = group["difference"].to_numpy(float)
        positions = random.integers(0, len(values), size=(iterations, len(values)))
        pieces.append(values[positions])
    distribution = np.nanmean(np.concatenate(pieces, axis=1), axis=1)
    return (
        float(np.mean(paired["difference"])),
        float(np.median(paired["difference"])),
        float(np.percentile(distribution, 2.5)),
        float(np.percentile(distribution, 97.5)),
    )


def paired_tests(scenario: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    models = sorted(scenario["model_id"].unique())
    for encoder, encoder_frame in scenario.groupby("encoder_id", sort=True):
        for metric in PAIR_METRICS:
            for first, second in itertools.combinations(models, 2):
                keys = ["scenario_id", "style_axis"]
                left = encoder_frame[encoder_frame.model_id == first][keys + [metric]]
                right = encoder_frame[encoder_frame.model_id == second][keys + [metric]]
                pair = left.merge(right, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
                pair["difference"] = pair[f"{metric}_first"] - pair[f"{metric}_second"]
                mean, median, lower, upper = stratified_bootstrap_difference(pair, iterations, seed)
                difference = pair["difference"].to_numpy(float)
                standard_deviation = float(np.std(difference, ddof=1))
                effect = mean / standard_deviation if standard_deviation > 0 else np.nan
                try:
                    pvalue = float(wilcoxon(difference, zero_method="zsplit").pvalue)
                except ValueError:
                    pvalue = 1.0
                records.append(
                    {
                        "encoder_id": encoder,
                        "metric": metric,
                        "model_first": first,
                        "model_second": second,
                        "difference_definition": "first_minus_second",
                        "scenario_count": len(pair),
                        "mean_difference": mean,
                        "median_difference": median,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "paired_effect_size_dz": effect,
                        "wilcoxon_p_raw": pvalue,
                    }
                )
    frame = pd.DataFrame(records)
    frame["wilcoxon_p_holm"] = np.nan
    for _, indexes in frame.groupby(["encoder_id", "metric"]).groups.items():
        frame.loc[indexes, "wilcoxon_p_holm"] = holm_adjust(frame.loc[indexes, "wilcoxon_p_raw"].to_numpy())
    return frame


def bootstrap_correlation(
    frame: pd.DataFrame,
    first: str,
    second: str,
    function: Callable[..., object],
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    random = np.random.default_rng(seed)
    observed = float(function(frame[first], frame[second]).statistic)
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    models = sorted(frame["model_id"].unique())
    for _, group in frame.groupby("style_axis", sort=True):
        first_pivot = group.pivot(index="scenario_id", columns="model_id", values=first).reindex(columns=models)
        second_pivot = group.pivot(index="scenario_id", columns="model_id", values=second).reindex(columns=models)
        arrays.append((first_pivot.to_numpy(float), second_pivot.to_numpy(float)))
    distribution = np.empty(iterations)
    for index in range(iterations):
        sampled_first: list[np.ndarray] = []
        sampled_second: list[np.ndarray] = []
        for first_values, second_values in arrays:
            positions = random.integers(0, len(first_values), size=len(first_values))
            sampled_first.append(first_values[positions].ravel())
            sampled_second.append(second_values[positions].ravel())
        distribution[index] = function(
            np.concatenate(sampled_first), np.concatenate(sampled_second)
        ).statistic
    return observed, float(np.nanpercentile(distribution, 2.5)), float(np.nanpercentile(distribution, 97.5))


def hypothesis_tests(scenario: pd.DataFrame, iterations: int, seed: int) -> tuple[pd.DataFrame, str]:
    records: list[dict[str, object]] = []
    for encoder, frame in scenario.groupby("encoder_id", sort=True):
        # Endpoint errors are exactly zero by coordinate construction, so 7-point
        # MAE is 5/7 of the endpoint-free value for every complete trajectory.
        for model_id, model_frame in frame.groupby("model_id", sort=True):
            h1_difference = model_frame["interior_calibration_error"] - (5.0 / 7.0) * model_frame["interior_calibration_error"]
            h1_frame = model_frame[["style_axis"]].copy()
            h1_frame["difference"] = h1_difference.to_numpy()
            h1_mean, _, h1_lower, h1_upper = stratified_bootstrap_difference(h1_frame, iterations, seed)
            records.append(
                {
                    "hypothesis": "H1",
                    "encoder_id": encoder,
                    "model_id": model_id,
                    "test": "paired 5-point minus 7-point ICE",
                    "estimate": h1_mean,
                    "ci_lower": h1_lower,
                    "ci_upper": h1_upper,
                    "supported": bool((h1_difference > 0).all()),
                    "note": "Seven-point value includes two mechanically zero endpoint errors.",
                }
            )
        for hypothesis, first, second, test in [
            ("H2", "endpoint_separation", "interior_calibration_error", "Spearman endpoint separation vs ICE"),
            ("H2", "endpoint_separation", "normalized_off_axis_drift", "Spearman endpoint separation vs drift"),
            ("H5", "interior_calibration_error", "normalized_off_axis_drift", "Spearman ICE vs drift"),
        ]:
            observed, lower, upper = bootstrap_correlation(frame, first, second, spearmanr, iterations, seed)
            if hypothesis == "H2":
                supported = upper < 0
            else:
                supported = lower > 0 and observed < 0.8
            records.append(
                {
                    "hypothesis": hypothesis,
                    "encoder_id": encoder,
                    "test": test,
                    "estimate": observed,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "supported": supported,
                    "note": "Scenario-level, style-axis-stratified bootstrap.",
                }
            )
    text = """# Preregistered hypothesis status

- H1 is evaluated from the paired endpoint-included and endpoint-excluded definitions. Because endpoint projection errors are mechanically zero, the complete seven-point ICE is exactly five sevenths of endpoint-free ICE.
- H2 and H5 are evaluated below with scenario-level, style-axis-stratified bootstrap confidence intervals.
- H3 (self versus shared-anchor ranking changes) is **external pending** until the two-author shared endpoint review is complete.
- H4 is treated as a non-monotonicity/descriptive claim rather than a universal scaling-law test; only five models are available, and architecture is confounded with scale.
- H6 is **external pending** for human correlations. Encoder dependence is available from the three-encoder tables, but it cannot substitute for real human ratings.

Machine-readable results are in `statistics/hypothesis_tests.csv`.
"""
    return pd.DataFrame(records), text


def scale_analysis(scenario: pd.DataFrame, registry_path: Path, iterations: int, seed: int) -> pd.DataFrame:
    registry = pd.read_csv(registry_path)
    registry = registry[registry["included"].astype(str).str.lower() == "true"].copy()
    registry["log_total_parameters"] = np.log(registry["parameter_count_total"].astype(float))
    registry["log_active_parameters"] = np.log(registry["parameter_count_active"].astype(float))
    model_means = scenario.groupby(["model_id", "encoder_id"], as_index=False)[SCALE_METRICS].mean()
    merged = model_means.merge(registry, on="model_id", validate="many_to_one")
    records: list[dict[str, object]] = []
    scopes = {"all_models_descriptive": merged, "Qwen3.5": merged[merged.family_version == "Qwen3.5"]}
    random = np.random.default_rng(seed)
    for scope, frame in scopes.items():
        for (encoder,), encoder_frame in frame.groupby(["encoder_id"], sort=True):
            for metric in SCALE_METRICS:
                for predictor in ("log_total_parameters", "log_active_parameters"):
                    x = encoder_frame[predictor].to_numpy(float)
                    y = encoder_frame[metric].to_numpy(float)
                    if len(x) < 3 or np.std(x) == 0:
                        continue
                    slope, intercept = np.polyfit(x, y, 1)
                    fitted = intercept + slope * x
                    denominator = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - np.sum((y - fitted) ** 2) / denominator if denominator else np.nan
                    rho = float(spearmanr(x, y).statistic)
                    # Scenario bootstrap preserves the same scenario draw across models.
                    source = scenario[scenario.encoder_id == encoder]
                    source = source[source.model_id.isin(encoder_frame.model_id)]
                    axis_arrays = []
                    ordered_models = encoder_frame.model_id.tolist()
                    for _, axis_frame in source.groupby("style_axis", sort=True):
                        pivot = axis_frame.pivot(index="scenario_id", columns="model_id", values=metric)
                        axis_arrays.append(pivot.reindex(columns=ordered_models).to_numpy(float))
                    boot_means = np.zeros((iterations, len(x)), dtype=float)
                    total_rows = 0
                    for values in axis_arrays:
                        positions = random.integers(0, len(values), size=(iterations, len(values)))
                        boot_means += values[positions].sum(axis=1)
                        total_rows += len(values)
                    boot_means /= total_rows
                    centered_x = x - x.mean()
                    slopes = ((boot_means - boot_means.mean(axis=1, keepdims=True)) * centered_x).sum(axis=1) / np.square(centered_x).sum()
                    records.append(
                        {
                            "scope": scope,
                            "encoder_id": encoder,
                            "metric": metric,
                            "predictor": predictor,
                            "model_count": len(x),
                            "spearman_rho": rho,
                            "regression_slope": float(slope),
                            "slope_ci_lower": float(np.percentile(slopes, 2.5)),
                            "slope_ci_upper": float(np.percentile(slopes, 97.5)),
                            "r_squared": float(r2),
                            "causal_interpretation_allowed": False,
                        }
                    )
    return pd.DataFrame(records)


def mixed_effects(sample_path: Path, registry_path: Path, output_path: Path) -> None:
    sample = pd.read_parquet(sample_path)
    sample = sample[sample["is_interior"]].copy()
    registry = pd.read_csv(registry_path)[["model_id", "parameter_count_active"]]
    sample = sample.merge(registry, on="model_id", validate="many_to_one")
    # Honor the independent unit: first average generation seeds in each cell.
    keys = ["scenario_id", "style_axis", "model_id", "family_version", "architecture", "encoder_id", "target_ratio"]
    cells = sample.groupby(keys, as_index=False).agg(
        projection_error_raw=("projection_error_raw", "mean"),
        off_axis_drift_normalized=("off_axis_drift_normalized", "mean"),
        endpoint_separation=("endpoint_separation", "mean"),
        character_length=("character_length", "mean"),
        parameter_count_active=("parameter_count_active", "first"),
    )
    cells["log_active_parameters"] = np.log(cells["parameter_count_active"].astype(float))
    formula_tail = (
        "C(family_version) + log_active_parameters + C(architecture) + target_ratio "
        "+ C(style_axis) + C(encoder_id) + endpoint_separation + character_length "
        "+ C(family_version):target_ratio"
    )
    lines = [
        "Mixed-effects models use seed-averaged interior cells and a random intercept for scenario_id.",
        "Architecture, version, and parameter count are strongly confounded in this five-model panel; coefficients are associative, not causal.",
        "",
    ]
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        output_path.write_text("statsmodels unavailable: " + repr(exc) + "\n", encoding="utf-8")
        return
    for outcome in ("projection_error_raw", "off_axis_drift_normalized"):
        formula = f"{outcome} ~ {formula_tail}"
        lines.extend(["=" * 80, formula])
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(formula, cells, groups=cells["scenario_id"])
                result = model.fit(reml=False, method="lbfgs", maxiter=300)
            lines.append(result.summary().as_text())
            lines.append(f"converged={result.converged}")
            lines.append("warnings=" + " | ".join(str(item.message) for item in caught))
            exog = result.model.exog
            residual = np.asarray(result.resid)
            try:
                from scipy.stats import chi2

                squared = np.square(residual)
                coefficients = np.linalg.lstsq(exog, squared, rcond=None)[0]
                fitted = exog @ coefficients
                denominator = np.square(squared - squared.mean()).sum()
                r_squared_aux = 1 - np.square(squared - fitted).sum() / denominator
                lm = len(squared) * r_squared_aux
                degrees = max(exog.shape[1] - 1, 1)
                lines.append(f"Breusch-Pagan LM={lm:.6g}, df={degrees}, p={chi2.sf(lm, degrees):.6g}")
            except Exception as exc:
                lines.append("Breusch-Pagan failed: " + repr(exc))
            condition = np.linalg.cond(exog)
            lines.append(f"design_matrix_condition_number={condition:.6g}")
        except Exception as exc:
            lines.append("MixedLM failed: " + repr(exc))
            lines.append("Fallback: OLS with scenario-clustered standard errors.")
            try:
                result = smf.ols(formula, cells).fit(cov_type="cluster", cov_kwds={"groups": cells["scenario_id"]})
                lines.append(result.summary().as_text())
            except Exception as fallback_exc:
                lines.append("Fallback also failed: " + repr(fallback_exc))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=REVISION_ROOT / "metrics/scenario_level_metrics.parquet")
    parser.add_argument("--sample", type=Path, default=REVISION_ROOT / "metrics/sample_level_metrics.parquet")
    parser.add_argument("--registry", type=Path, default=REVISION_ROOT / "config/model_registry.csv")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    scenario = pd.read_parquet(args.scenario)
    output = REVISION_ROOT / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    paired = paired_tests(scenario, args.iterations, args.seed)
    paired.to_csv(output / "paired_tests.csv", index=False)
    hypotheses, hypothesis_text = hypothesis_tests(scenario, args.iterations, args.seed)
    hypotheses.to_csv(output / "hypothesis_tests.csv", index=False)
    (output / "hypothesis_results.md").write_text(hypothesis_text, encoding="utf-8")
    scale = scale_analysis(scenario, args.registry, args.iterations, args.seed)
    scale.to_csv(output / "scale_version_architecture.csv", index=False)
    mixed_effects(args.sample, args.registry, output / "mixed_effects_results.txt")
    print(json.dumps({"paired_rows": len(paired), "hypothesis_rows": len(hypotheses), "scale_rows": len(scale)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
