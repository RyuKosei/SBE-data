"""Endpoint-aware geometry metrics for the Computers revision.

The functions in this module are deliberately independent of model inference
and embedding code. Primary summaries must filter to interior ratios before
aggregation; endpoints are retained only to define the local coordinate axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr


DEFAULT_EPSILON = 1.0e-12


@dataclass(frozen=True)
class GeometryResult:
    target_ratio: np.ndarray
    projection_raw: np.ndarray
    projection_clipped: np.ndarray
    projection_error_raw: np.ndarray
    projection_error_clipped: np.ndarray
    off_axis_drift_normalized: np.ndarray
    interpolation_distance_normalized: np.ndarray
    out_of_range: np.ndarray
    nearest_neighbor_rank: np.ndarray
    nearest_neighbor_rank_normalized: np.ndarray
    decomposition_residual: np.ndarray
    endpoint_separation: float
    endpoint_unstable: bool


def _as_float_matrix(vectors: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("vectors must be a two-dimensional array with at least two rows")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain NaN or infinite values")
    return matrix


def _as_ratios(ratios: np.ndarray | Iterable[float], row_count: int) -> np.ndarray:
    values = np.asarray(list(ratios), dtype=np.float64)
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError("ratios must be one-dimensional and match vector rows")
    if not np.isfinite(values).all():
        raise ValueError("ratios contain NaN or infinite values")
    if len(np.unique(values)) != len(values):
        raise ValueError("one trajectory may contain only one point per target ratio")
    return values


def _endpoint_index(ratios: np.ndarray, target: float) -> int:
    matches = np.flatnonzero(np.isclose(ratios, target, atol=1.0e-12, rtol=0.0))
    if len(matches) != 1:
        raise ValueError(f"trajectory requires exactly one endpoint at ratio {target}")
    return int(matches[0])


def midpoint_rank(distances: np.ndarray, candidate_index: int) -> float:
    """Return a deterministic 1-based midrank, including the candidate itself.

    The candidate set is all outputs in the same seven-point trajectory. Ties
    use their average rank, matching scipy's ``rankdata(method='average')``.
    """

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or not 0 <= candidate_index < len(values):
        raise ValueError("invalid distances or candidate index")
    return float(rankdata(values, method="average")[candidate_index])


def compute_geometry(
    vectors: np.ndarray | Iterable[Iterable[float]],
    ratios: np.ndarray | Iterable[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> GeometryResult:
    """Compute sample-level projection and normalized distance diagnostics.

    Out-of-range status and raw error are computed before clipping. If the two
    endpoint embeddings collapse (separation <= epsilon), all coordinate-based
    values are NaN and ``endpoint_unstable`` is true; callers must not silently
    replace those coordinates by 0.5.
    """

    matrix = _as_float_matrix(vectors)
    alpha = _as_ratios(ratios, len(matrix))
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    index_a = _endpoint_index(alpha, 0.0)
    index_b = _endpoint_index(alpha, 1.0)
    endpoint_a = matrix[index_a]
    delta = matrix[index_b] - endpoint_a
    separation = float(np.linalg.norm(delta))

    if separation <= epsilon:
        missing = np.full(len(matrix), np.nan, dtype=np.float64)
        return GeometryResult(
            target_ratio=alpha,
            projection_raw=missing.copy(),
            projection_clipped=missing.copy(),
            projection_error_raw=missing.copy(),
            projection_error_clipped=missing.copy(),
            off_axis_drift_normalized=missing.copy(),
            interpolation_distance_normalized=missing.copy(),
            out_of_range=np.zeros(len(matrix), dtype=bool),
            nearest_neighbor_rank=missing.copy(),
            nearest_neighbor_rank_normalized=missing.copy(),
            decomposition_residual=missing.copy(),
            endpoint_separation=separation,
            endpoint_unstable=True,
        )

    displacement = matrix - endpoint_a
    coordinate = displacement @ delta / float(delta @ delta)
    axial = coordinate[:, None] * delta
    orthogonal = displacement - axial
    reconstructed = axial + orthogonal
    residual = np.linalg.norm(displacement - reconstructed, axis=1)
    residual /= np.maximum(np.linalg.norm(displacement, axis=1), 1.0)

    ideal = endpoint_a + alpha[:, None] * delta
    interpolation = np.linalg.norm(matrix - ideal, axis=1) / (separation + epsilon)
    off_axis = np.linalg.norm(orthogonal, axis=1) / (separation + epsilon)
    clipped = np.clip(coordinate, 0.0, 1.0)

    ranks = np.empty(len(matrix), dtype=np.float64)
    for row_index, target in enumerate(ideal):
        distances = np.linalg.norm(matrix - target, axis=1)
        ranks[row_index] = midpoint_rank(distances, row_index)
    normalized_ranks = (ranks - 1.0) / max(len(matrix) - 1, 1)

    return GeometryResult(
        target_ratio=alpha,
        projection_raw=coordinate,
        projection_clipped=clipped,
        projection_error_raw=np.abs(coordinate - alpha),
        projection_error_clipped=np.abs(clipped - alpha),
        off_axis_drift_normalized=off_axis,
        interpolation_distance_normalized=interpolation,
        out_of_range=(coordinate < 0.0) | (coordinate > 1.0),
        nearest_neighbor_rank=ranks,
        nearest_neighbor_rank_normalized=normalized_ranks,
        decomposition_residual=residual,
        endpoint_separation=separation,
        endpoint_unstable=False,
    )


def interior_mask(ratios: np.ndarray | Iterable[float]) -> np.ndarray:
    values = np.asarray(list(ratios), dtype=np.float64)
    return (values > 0.0) & (values < 1.0)


def summarize_interior(result: GeometryResult) -> dict[str, float | int | bool]:
    """Summarize exactly the non-endpoint rows of a trajectory."""

    mask = interior_mask(result.target_ratio)
    if int(mask.sum()) == 0:
        raise ValueError("trajectory has no interior ratios")
    if result.endpoint_unstable:
        return {
            "interior_point_count": int(mask.sum()),
            "endpoint_separation": result.endpoint_separation,
            "endpoint_unstable": True,
            "interior_calibration_error": np.nan,
            "interior_calibration_error_clipped": np.nan,
            "normalized_off_axis_drift_mean": np.nan,
            "interpolation_distance_mean": np.nan,
            "interpolation_distance_rms": np.nan,
            "out_of_range_rate": np.nan,
            "nearest_neighbor_mean_rank": np.nan,
            "nearest_neighbor_mean_rank_normalized": np.nan,
            "spearman_monotonicity": np.nan,
            "projection_decomposition_max_residual": np.nan,
        }

    monotonicity = spearmanr(
        result.target_ratio[mask], result.projection_raw[mask]
    ).statistic
    distances = result.interpolation_distance_normalized[mask]
    return {
        "interior_point_count": int(mask.sum()),
        "endpoint_separation": result.endpoint_separation,
        "endpoint_unstable": False,
        "interior_calibration_error": float(result.projection_error_raw[mask].mean()),
        "interior_calibration_error_clipped": float(
            result.projection_error_clipped[mask].mean()
        ),
        "normalized_off_axis_drift_mean": float(
            result.off_axis_drift_normalized[mask].mean()
        ),
        "interpolation_distance_mean": float(distances.mean()),
        "interpolation_distance_rms": float(np.sqrt(np.mean(np.square(distances)))),
        "out_of_range_rate": float(result.out_of_range[mask].mean()),
        "nearest_neighbor_mean_rank": float(result.nearest_neighbor_rank[mask].mean()),
        "nearest_neighbor_mean_rank_normalized": float(
            result.nearest_neighbor_rank_normalized[mask].mean()
        ),
        "spearman_monotonicity": float(monotonicity),
        "projection_decomposition_max_residual": float(
            result.decomposition_residual.max()
        ),
    }


def trajectory_diagnostics(
    vectors: np.ndarray | Iterable[Iterable[float]],
    ratios: np.ndarray | Iterable[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float | int | bool | list[float]]:
    """Diagnose how well an ordered trajectory follows its endpoint line."""

    matrix = _as_float_matrix(vectors)
    alpha = _as_ratios(ratios, len(matrix))
    order = np.argsort(alpha)
    matrix = matrix[order]
    alpha = alpha[order]
    geometry = compute_geometry(matrix, alpha, epsilon=epsilon)
    if geometry.endpoint_unstable:
        return {
            "endpoint_unstable": True,
            "endpoint_separation": geometry.endpoint_separation,
            "pca_pc1_explained_variance": np.nan,
            "endpoint_pc1_abs_cosine": np.nan,
            "endpoint_pc1_angle_degrees": np.nan,
            "ordered_path_length_ratio": np.nan,
            "discrete_curvature_mean_degrees": np.nan,
            "discrete_curvature_max_degrees": np.nan,
            "projection_spearman_all_points": np.nan,
            "negative_local_slope_count": 0,
            "turning_point_count": 0,
            "local_projection_slopes": [],
        }

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values)
    pc1_variance = float(variance[0] / variance.sum()) if variance.sum() else np.nan
    endpoint_direction = matrix[-1] - matrix[0]
    pc1 = vh[0]
    cosine = float(
        abs(endpoint_direction @ pc1)
        / (np.linalg.norm(endpoint_direction) * np.linalg.norm(pc1))
    )
    cosine = min(1.0, max(0.0, cosine))

    segments = np.diff(matrix, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    path_ratio = float(segment_lengths.sum() / geometry.endpoint_separation)
    angles: list[float] = []
    for first, second in zip(segments[:-1], segments[1:]):
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= epsilon:
            continue
        turn_cosine = float(np.clip((first @ second) / denominator, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(turn_cosine))))

    local_slopes = np.diff(geometry.projection_raw) / np.diff(alpha)
    nonzero_slope_signs = np.sign(local_slopes[np.abs(local_slopes) > epsilon])
    turning_points = int(np.sum(nonzero_slope_signs[1:] != nonzero_slope_signs[:-1]))
    rho = spearmanr(alpha, geometry.projection_raw).statistic
    return {
        "endpoint_unstable": False,
        "endpoint_separation": geometry.endpoint_separation,
        "pca_pc1_explained_variance": pc1_variance,
        "endpoint_pc1_abs_cosine": cosine,
        "endpoint_pc1_angle_degrees": float(np.degrees(np.arccos(cosine))),
        "ordered_path_length_ratio": path_ratio,
        "discrete_curvature_mean_degrees": float(np.mean(angles)) if angles else np.nan,
        "discrete_curvature_max_degrees": float(np.max(angles)) if angles else np.nan,
        "projection_spearman_all_points": float(rho),
        "negative_local_slope_count": int(np.sum(local_slopes < 0.0)),
        "turning_point_count": turning_points,
        "local_projection_slopes": [float(value) for value in local_slopes],
    }


def _closest_polyline_location(
    point: np.ndarray,
    centers: np.ndarray,
    ratios: np.ndarray,
    epsilon: float,
) -> tuple[float, float]:
    best_distance = np.inf
    best_ratio = np.nan
    for index in range(len(centers) - 1):
        start = centers[index]
        segment = centers[index + 1] - start
        denominator = float(segment @ segment)
        fraction = 0.0 if denominator <= epsilon else float((point - start) @ segment / denominator)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        closest = start + fraction * segment
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_ratio = float(ratios[index] + fraction * (ratios[index + 1] - ratios[index]))
    return best_ratio, best_distance


def leave_one_seed_out_diagnostics(
    seed_vectors: dict[int, np.ndarray],
    ratios: np.ndarray | Iterable[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> list[dict[str, float | int | bool]]:
    """Evaluate each seed against empirical references formed from other seeds.

    Each mapping value has shape ``(ratio_count, embedding_dimension)`` in the
    same ratio order. The intended design has three seeds, so each held-out seed
    is evaluated against two-seed centers.
    """

    if len(seed_vectors) < 3:
        raise ValueError("leave-one-seed-out diagnostics require at least three seeds")
    alpha = np.asarray(list(ratios), dtype=np.float64)
    order = np.argsort(alpha)
    alpha = alpha[order]
    _endpoint_index(alpha, 0.0)
    _endpoint_index(alpha, 1.0)
    matrices: dict[int, np.ndarray] = {}
    for seed, vectors in seed_vectors.items():
        matrix = _as_float_matrix(vectors)
        if len(matrix) != len(alpha):
            raise ValueError("each seed matrix must match the ratio count")
        matrices[int(seed)] = matrix[order]
    mask = interior_mask(alpha)
    records: list[dict[str, float | int | bool]] = []
    for test_seed, test in sorted(matrices.items()):
        train = np.mean(
            np.stack([matrix for seed, matrix in matrices.items() if seed != test_seed]),
            axis=0,
        )
        delta = train[-1] - train[0]
        separation = float(np.linalg.norm(delta))
        if separation <= epsilon:
            records.append(
                {
                    "test_seed": test_seed,
                    "training_seed_count": len(matrices) - 1,
                    "endpoint_unstable": True,
                    "training_endpoint_separation": separation,
                }
            )
            continue
        line_targets = train[0] + alpha[:, None] * delta
        line_error = np.linalg.norm(test - line_targets, axis=1) / (separation + epsilon)
        centroid_error = np.linalg.norm(test - train, axis=1) / (separation + epsilon)
        line_nearest: list[bool] = []
        centroid_nearest: list[bool] = []
        piecewise_nearest: list[bool] = []
        piecewise_along_error: list[float] = []
        piecewise_drift: list[float] = []
        line_drift: list[float] = []
        for index in np.flatnonzero(mask):
            line_choice = int(np.argmin(np.linalg.norm(line_targets - test[index], axis=1)))
            centroid_choice = int(np.argmin(np.linalg.norm(train - test[index], axis=1)))
            estimated_ratio, distance = _closest_polyline_location(
                test[index], train, alpha, epsilon
            )
            piecewise_choice = int(np.argmin(np.abs(alpha - estimated_ratio)))
            line_coordinate = float((test[index] - train[0]) @ delta / (delta @ delta))
            line_orthogonal = (test[index] - train[0]) - line_coordinate * delta
            line_nearest.append(line_choice == index)
            centroid_nearest.append(centroid_choice == index)
            piecewise_nearest.append(piecewise_choice == index)
            piecewise_along_error.append(abs(estimated_ratio - alpha[index]))
            piecewise_drift.append(distance / (separation + epsilon))
            line_drift.append(float(np.linalg.norm(line_orthogonal) / (separation + epsilon)))
        records.append(
            {
                "test_seed": test_seed,
                "training_seed_count": len(matrices) - 1,
                "endpoint_unstable": False,
                "training_endpoint_separation": separation,
                "endpoint_line_error_mean": float(line_error[mask].mean()),
                "endpoint_line_nearest_ratio_accuracy": float(np.mean(line_nearest)),
                "endpoint_line_off_trajectory_mean": float(np.mean(line_drift)),
                "empirical_centroid_error_mean": float(centroid_error[mask].mean()),
                "empirical_centroid_nearest_ratio_accuracy": float(np.mean(centroid_nearest)),
                "piecewise_along_error_mean": float(np.mean(piecewise_along_error)),
                "piecewise_nearest_ratio_accuracy": float(np.mean(piecewise_nearest)),
                "piecewise_off_trajectory_mean": float(np.mean(piecewise_drift)),
            }
        )
    return records
