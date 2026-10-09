from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def stratified_cluster_mean_ci(
    frame: pd.DataFrame,
    value: str,
    *,
    cluster: str = "scenario_id",
    strata: str = "style_axis",
    iterations: int = 10_000,
    seed: int = 20260916,
) -> dict[str, float | int]:
    data = frame[[cluster, strata, value]].dropna().copy()
    cluster_values = data.groupby([strata, cluster], as_index=False)[value].mean()
    if cluster_values.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "valid_scenarios": 0}
    estimate = float(cluster_values[value].mean())
    rng = np.random.default_rng(seed)
    draws = np.zeros(iterations, dtype=np.float64)
    total = 0
    for _stratum, group in cluster_values.groupby(strata, sort=True):
        values = group[value].to_numpy(dtype=np.float64)
        n = len(values)
        sampled = values[rng.integers(0, n, size=(iterations, n))].sum(axis=1)
        draws += sampled
        total += n
    draws /= total
    return {
        "mean": estimate,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "valid_scenarios": int(cluster_values[cluster].nunique()),
    }


def cluster_correlation_ci(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    cluster: str = "scenario_id",
    iterations: int = 10_000,
    seed: int = 20260916,
) -> dict[str, float | int]:
    data = frame[[cluster, x, y]].dropna().copy()
    if data.empty or data[x].nunique() <= 1 or data[y].nunique() <= 1:
        return {"correlation": np.nan, "ci_low": np.nan, "ci_high": np.nan, "valid_scenarios": data[cluster].nunique(), "valid_rows": len(data)}
    correlation = float(data[x].corr(data[y]))
    aggregates = data.groupby(cluster).agg(
        n=(x, "size"), sx=(x, "sum"), sy=(y, "sum"),
        sxx=(x, lambda values: float(np.square(values).sum())),
        syy=(y, lambda values: float(np.square(values).sum())),
        sxy=(x, lambda values: 0.0),
    )
    # pandas named aggregation cannot see the other column, so compute sxy separately.
    aggregates["sxy"] = data.assign(_xy=data[x] * data[y]).groupby(cluster)._xy.sum()
    values = aggregates[["n", "sx", "sy", "sxx", "syy", "sxy"]].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_clusters = len(values)
    draws = np.empty(iterations, dtype=np.float64)
    chunk = 1000
    for start in range(0, iterations, chunk):
        end = min(iterations, start + chunk)
        sampled = values[rng.integers(0, n_clusters, size=(end - start, n_clusters))].sum(axis=1)
        n, sx, sy, sxx, syy, sxy = sampled.T
        covariance = sxy - sx * sy / n
        variance_x = sxx - sx * sx / n
        variance_y = syy - sy * sy / n
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        draws[start:end] = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0)
    finite = draws[np.isfinite(draws)]
    return {
        "correlation": correlation,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
        "valid_scenarios": int(n_clusters),
        "valid_rows": int(len(data)),
    }


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=np.float64)
    output = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return output
    order = finite_indices[np.argsort(values[finite_indices])]
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(order) - rank) * values[index] for rank, index in enumerate(order)], dtype=np.float64)
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    output[order] = adjusted_sorted
    return output


def paired_sign_permutation_pvalue(values: np.ndarray, *, iterations: int = 100_000, seed: int = 20260916) -> float:
    differences = np.asarray(values, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return np.nan
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 1000
    for start in range(0, iterations, chunk):
        count = min(chunk, iterations - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-15))
    return (exceed + 1.0) / (iterations + 1.0)
