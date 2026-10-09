from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compute_anchor_metrics import fixed_geometry, trajectory_ranks  # noqa: E402


def test_fixed_geometry_known_projection_and_drift() -> None:
    result = fixed_geometry(
        np.asarray([0.25, 0.5]),
        0.25,
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
    )
    assert result["projection_raw"] == 0.25
    assert result["projection_error_raw"] == 0.0
    assert np.isclose(result["off_axis_drift_normalized_exact"], 0.5)
    assert np.isclose(result["interpolation_distance_normalized_exact"], 0.5)
    assert result["out_of_range"] is False


def test_nearest_target_rank_has_old_seven_candidate_denominator() -> None:
    ratios = [index / 6 for index in range(7)]
    vectors = [np.asarray([value, 0.0]) for value in ratios]
    ranks = trajectory_ranks(vectors, ratios, np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]))
    assert ranks == [(1.0, 0.0)] * 7


def test_nearest_target_rank_is_missing_for_partial_trajectory() -> None:
    ratios = [index / 6 for index in range(7) if index != 5]
    vectors = [np.asarray([value, 0.0]) for value in ratios]
    ranks = trajectory_ranks(vectors, ratios, np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]))
    assert len(ranks) == 6
    assert all(np.isnan(rank) and np.isnan(normalized) for rank, normalized in ranks)


def test_collapsed_fixed_anchor_is_nan_not_half() -> None:
    result = fixed_geometry(np.asarray([1.0, 2.0]), 0.5, np.asarray([0.0, 0.0]), np.asarray([0.0, 0.0]))
    assert result["endpoint_unstable"] is True
    assert np.isnan(result["projection_raw"])
    assert result["out_of_range"] is None
