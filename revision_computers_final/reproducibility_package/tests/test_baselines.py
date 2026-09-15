from __future__ import annotations

import unittest

import numpy as np

from revision_computers_final.scripts.baselines import (
    constant_half,
    endpoint_distance_ratio,
    fit_isotonic,
    grouped_isotonic_cv,
    scenario_group_folds,
)


class BaselinesTest(unittest.TestCase):
    def test_constant_theoretical_mae_distinguishes_grids(self) -> None:
        legacy = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
        sixths = np.asarray([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
        self.assertAlmostEqual(np.mean(np.abs(constant_half(5) - legacy)), 0.26)
        self.assertAlmostEqual(np.mean(np.abs(constant_half(5) - sixths)), 0.20)

    def test_endpoint_distance_ratio(self) -> None:
        vectors = np.asarray([[0.0, 0.0], [0.25, 0.0], [1.0, 0.0]])
        predicted = endpoint_distance_ratio(vectors, vectors[0], vectors[-1])
        np.testing.assert_allclose(predicted, [0.0, 0.25, 1.0])

    def test_group_folds_never_split_scenario(self) -> None:
        scenarios = [f"s{index // 3}" for index in range(30)]
        axes = [f"a{(index // 3) % 2}" for index in range(30)]
        folds = scenario_group_folds(scenarios, axes)
        for scenario in set(scenarios):
            self.assertEqual(len(set(folds[np.asarray(scenarios) == scenario])), 1)

    def test_isotonic_fit_is_monotone(self) -> None:
        model = fit_isotonic([0, 1, 2, 3], [0, 0.8, 0.4, 1])
        self.assertTrue(np.all(np.diff(model.y) >= 0))

    def test_grouped_isotonic_predictions_are_complete(self) -> None:
        scenarios = [f"s{index // 5}" for index in range(50)]
        axes = [f"a{(index // 25)}" for index in range(50)]
        x = np.tile(np.linspace(0.1, 0.9, 5), 10)
        y = x.copy()
        prediction, _ = grouped_isotonic_cv(x, y, scenarios, axes)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
