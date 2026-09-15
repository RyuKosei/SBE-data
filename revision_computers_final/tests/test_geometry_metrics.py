from __future__ import annotations

import unittest

import numpy as np

from revision_computers_final.scripts.geometry_metrics import (
    compute_geometry,
    leave_one_seed_out_diagnostics,
    summarize_interior,
    trajectory_diagnostics,
)


class GeometryMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ratios = np.asarray([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])

    def test_ideal_line_has_zero_primary_errors(self) -> None:
        vectors = np.column_stack([self.ratios, np.zeros(7)])
        result = compute_geometry(vectors, self.ratios)
        summary = summarize_interior(result)
        self.assertLess(summary["interior_calibration_error"], 1e-12)
        self.assertLess(summary["normalized_off_axis_drift_mean"], 1e-12)
        self.assertLess(summary["projection_decomposition_max_residual"], 1e-12)
        self.assertEqual(summary["interior_point_count"], 5)

    def test_raw_error_and_out_of_range_precede_clipping(self) -> None:
        x = self.ratios.copy()
        x[1] = -0.2
        vectors = np.column_stack([x, np.zeros(7)])
        result = compute_geometry(vectors, self.ratios)
        self.assertTrue(result.out_of_range[1])
        self.assertGreater(result.projection_error_raw[1], result.projection_error_clipped[1])

    def test_orthogonal_drift_and_interpolation_identity(self) -> None:
        vectors = np.column_stack([self.ratios, np.zeros(7)])
        vectors[3, 1] = 0.25
        result = compute_geometry(vectors, self.ratios)
        self.assertAlmostEqual(result.off_axis_drift_normalized[3], 0.25)
        self.assertAlmostEqual(result.interpolation_distance_normalized[3], 0.25)
        self.assertLess(result.decomposition_residual.max(), 1e-12)

    def test_collapsed_endpoints_are_explicitly_unstable(self) -> None:
        vectors = np.zeros((7, 3))
        result = compute_geometry(vectors, self.ratios)
        self.assertTrue(result.endpoint_unstable)
        self.assertTrue(np.isnan(result.projection_raw).all())

    def test_trajectory_diagnostics_on_line(self) -> None:
        vectors = np.column_stack([self.ratios, np.zeros(7)])
        metrics = trajectory_diagnostics(vectors, self.ratios)
        self.assertAlmostEqual(metrics["pca_pc1_explained_variance"], 1.0)
        self.assertAlmostEqual(metrics["endpoint_pc1_abs_cosine"], 1.0)
        self.assertAlmostEqual(metrics["ordered_path_length_ratio"], 1.0)

    def test_leave_one_seed_out_on_shared_line(self) -> None:
        base = np.column_stack([self.ratios, np.zeros(7)])
        seeds = {1: base.copy(), 2: base.copy(), 3: base.copy()}
        records = leave_one_seed_out_diagnostics(seeds, self.ratios)
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertAlmostEqual(record["endpoint_line_error_mean"], 0.0)
            self.assertAlmostEqual(record["empirical_centroid_error_mean"], 0.0)
            self.assertAlmostEqual(record["piecewise_along_error_mean"], 0.0)
            self.assertAlmostEqual(record["piecewise_nearest_ratio_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
