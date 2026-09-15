"""Scenario-grouped proportion-estimation baselines without sklearn."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def constant_half(count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    return np.full(count, 0.5, dtype=np.float64)


def endpoint_distance_ratio(
    vectors: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray:
    """Estimate B intensity as d(z, A) / (d(z, A) + d(z, B))."""

    matrix = np.asarray(vectors, dtype=np.float64)
    distance_a = np.linalg.norm(matrix - np.asarray(endpoint_a), axis=1)
    distance_b = np.linalg.norm(matrix - np.asarray(endpoint_b), axis=1)
    denominator = distance_a + distance_b
    output = np.full(len(matrix), 0.5, dtype=np.float64)
    valid = denominator > epsilon
    output[valid] = distance_a[valid] / denominator[valid]
    return output


def scenario_group_folds(
    scenario_ids: Iterable[str], axes: Iterable[str], fold_count: int = 5, seed: int = 20260916
) -> np.ndarray:
    """Assign whole scenarios to deterministic axis-stratified folds."""

    scenario_array = np.asarray(list(scenario_ids), dtype=object)
    axis_array = np.asarray(list(axes), dtype=object)
    if len(scenario_array) != len(axis_array):
        raise ValueError("scenario_ids and axes must have equal length")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    scenario_axis: dict[str, str] = {}
    for scenario, axis in zip(scenario_array, axis_array):
        prior = scenario_axis.setdefault(str(scenario), str(axis))
        if prior != str(axis):
            raise ValueError(f"scenario {scenario} appears under multiple axes")
    assignment: dict[str, int] = {}
    for axis in sorted(set(scenario_axis.values())):
        scenarios = [key for key, value in scenario_axis.items() if value == axis]
        scenarios.sort(
            key=lambda value: hashlib.sha256(f"{seed}:{axis}:{value}".encode()).digest()
        )
        for position, scenario in enumerate(scenarios):
            assignment[scenario] = position % fold_count
    return np.asarray([assignment[str(value)] for value in scenario_array], dtype=np.int64)


def _linear_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    coefficients = np.linalg.lstsq(design, train_y, rcond=None)[0]
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def grouped_length_only_cv(
    length_coordinate: Iterable[float],
    target_ratio: Iterable[float],
    scenario_ids: Iterable[str],
    axes: Iterable[str],
    *,
    fold_count: int = 5,
    seed: int = 20260916,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated OLS using only the endpoint-normalized length coordinate."""

    feature = np.asarray(list(length_coordinate), dtype=np.float64)
    target = np.asarray(list(target_ratio), dtype=np.float64)
    scenarios = list(scenario_ids)
    axis_values = list(axes)
    if not (len(feature) == len(target) == len(scenarios) == len(axis_values)):
        raise ValueError("all inputs must have equal length")
    folds = scenario_group_folds(scenarios, axis_values, fold_count, seed)
    predictions = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(fold_count):
        test = folds == fold
        train = ~test
        predictions[test] = _linear_fit_predict(feature[train], target[train], feature[test])
    return predictions, folds

@dataclass(frozen=True)
class IsotonicModel:
    x: np.ndarray
    y: np.ndarray

    def predict(self, values: Iterable[float]) -> np.ndarray:
        query = np.asarray(list(values), dtype=np.float64)
        if len(self.x) == 1:
            return np.full(len(query), self.y[0], dtype=np.float64)
        return np.interp(query, self.x, self.y, left=self.y[0], right=self.y[-1])


def fit_isotonic(x: Iterable[float], y: Iterable[float]) -> IsotonicModel:
    """Fit an increasing isotonic calibration curve using weighted PAVA."""

    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    if len(x_values) == 0 or len(x_values) != len(y_values):
        raise ValueError("x and y must be nonempty and have equal length")
    order = np.argsort(x_values, kind="mergesort")
    sorted_x = x_values[order]
    sorted_y = y_values[order]
    unique_x, inverse, counts = np.unique(sorted_x, return_inverse=True, return_counts=True)
    means = np.zeros(len(unique_x), dtype=np.float64)
    np.add.at(means, inverse, sorted_y)
    means /= counts

    blocks: list[list[float]] = []
    for index, (mean, weight) in enumerate(zip(means, counts)):
        blocks.append([float(index), float(index), float(weight), float(mean)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[2] + right[2]
            pooled = (left[2] * left[3] + right[2] * right[3]) / total_weight
            blocks.append([left[0], right[1], total_weight, pooled])
    fitted = np.empty(len(unique_x), dtype=np.float64)
    for start, stop, _, value in blocks:
        fitted[int(start) : int(stop) + 1] = value
    return IsotonicModel(x=unique_x, y=fitted)


def grouped_isotonic_cv(
    raw_projection: Iterable[float],
    target_ratio: Iterable[float],
    scenario_ids: Iterable[str],
    axes: Iterable[str],
    *,
    fold_count: int = 5,
    seed: int = 20260916,
) -> tuple[np.ndarray, np.ndarray]:
    projection = np.asarray(list(raw_projection), dtype=np.float64)
    target = np.asarray(list(target_ratio), dtype=np.float64)
    scenarios = list(scenario_ids)
    axis_values = list(axes)
    if not (len(projection) == len(target) == len(scenarios) == len(axis_values)):
        raise ValueError("all inputs must have equal length")
    folds = scenario_group_folds(scenarios, axis_values, fold_count, seed)
    predictions = np.full(len(target), np.nan, dtype=np.float64)
    for fold in range(fold_count):
        test = folds == fold
        model = fit_isotonic(projection[~test], target[~test])
        predictions[test] = model.predict(projection[test])
    return predictions, folds
