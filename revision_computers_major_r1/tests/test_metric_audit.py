from __future__ import annotations

import numpy as np


def test_self_anchor_seven_point_mae_is_five_sevenths_internal_mae() -> None:
    rng = np.random.default_rng(20260916)
    for _ in range(100):
        internal_errors = rng.uniform(0.0, 2.0, size=5)
        seven = np.concatenate(([0.0], internal_errors, [0.0]))
        assert np.isclose(seven.mean(), (5.0 / 7.0) * internal_errors.mean(), atol=1e-15)


def test_relative_underestimate_is_undefined_at_zero_internal_error() -> None:
    internal_mae = 0.0
    relative = np.nan if internal_mae == 0.0 else 1.0 - ((5.0 / 7.0) * internal_mae) / internal_mae
    assert np.isnan(relative)


def test_rms_does_not_obey_five_sevenths_factor() -> None:
    internal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    rms_internal = np.sqrt(np.mean(np.square(internal)))
    rms_seven = np.sqrt(np.mean(np.square(np.concatenate(([0.0], internal, [0.0])))))
    assert not np.isclose(rms_seven, (5.0 / 7.0) * rms_internal)
    assert np.isclose(rms_seven, np.sqrt(5.0 / 7.0) * rms_internal)
