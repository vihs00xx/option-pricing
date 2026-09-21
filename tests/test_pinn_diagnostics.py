import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

from src.neural.pinn_diagnostics import (
    acceptance_checks,
    full_diagnostics,
    prediction_errors,
    region_masks,
    reference_values,
    shape_violations,
    violation_summary,
)


class ShapeViolationTests(unittest.TestCase):
    def setUp(self):
        self.spots = np.array([50.0, 80.0, 100.0, 120.0, 200.0])
        self.tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.k, self.r, self.q = 100.0, 0.05, 0.0

    def test_exact_black_scholes_satisfies_all_bounds(self):
        reference = reference_values(
            self.spots, self.tau, self.k, self.r, 0.2, self.q
        )
        smooth = reference["smooth"]
        price = reference["price"]
        delta = np.where(smooth, reference["greeks"]["delta"], 0.0)
        gamma = np.where(smooth, reference["greeks"]["gamma"], 0.0)
        # Exploit the analytical identity: at tau=0 price equals the
        # intrinsic, delta is the indicator, gamma is a delta mass; use
        # only the smooth points for Greek-sensitive checks.
        violations = shape_violations(
            price[smooth], delta[smooth], gamma[smooth],
            self.spots[smooth], self.tau[smooth], self.k, self.r, self.q,
        )
        for name, magnitudes in violations.items():
            assert_allclose(magnitudes, 0.0, atol=1e-12,
                            err_msg=f"{name} violated by exact values")

    def test_violation_summary_counts_proportions_and_sizes(self):
        magnitudes = np.array([0.0, 0.0, 2.0, 4.0, 0.0])
        summary = violation_summary(magnitudes)
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["proportion"], 0.4)
        self.assertEqual(summary["maximum"], 4.0)
        self.assertAlmostEqual(summary["mean_size"], 1.2)

    def test_violation_summary_respects_masks(self):
        magnitudes = np.array([1.0, 0.0, 3.0])
        mask = np.array([True, False, False])
        summary = violation_summary(magnitudes, mask)
        self.assertEqual(summary["points"], 1)
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["maximum"], 1.0)

    def test_synthetic_bounds_violations_are_detected(self):
        intrinsic = np.maximum(
            self.spots - self.k * np.exp(-self.r * self.tau), 0.0
        )
        price = intrinsic + 1.0
        price[3] = 20.0   # below intrinsic (intrinsic ~ 23.68)
        price[4] = 250.0  # above the discounted spot (200 with q = 0)
        delta = np.array([-0.5, 0.5, 1.5, 0.5, 0.5])
        gamma = np.array([0.01, -0.01, 0.01, 0.01, -0.02])
        violations = shape_violations(
            price, delta, gamma, self.spots, self.tau, self.k, self.r, self.q
        )
        self.assertEqual(np.sum(violations["price_below_intrinsic"] > 0), 1)
        self.assertAlmostEqual(
            violations["price_below_intrinsic"][3], intrinsic[3] - 20.0)
        self.assertEqual(
            np.sum(violations["price_above_discounted_spot"] > 0), 1)
        self.assertAlmostEqual(
            violations["price_above_discounted_spot"][4], 50.0)
        self.assertEqual(np.sum(violations["delta_below_zero"] > 0), 1)
        self.assertEqual(
            np.sum(violations["delta_above_discount_factor"] > 0), 1)
        self.assertEqual(np.sum(violations["gamma_negative"] > 0), 2)


class RegionMaskTests(unittest.TestCase):
    def test_masks_partition_expected_points(self):
        spots = np.array([50.0, 90.0, 100.0, 110.0, 200.0])
        tau = np.array([0.01, 0.5, 1.0, 0.02, 1.0])
        masks = region_masks(
            spots, tau, 100.0, (0.5, 1.5), 0.05, 0.3, 0.05
        )
        self.assertEqual(masks["full"].sum(), 5)
        self.assertEqual(masks["near_expiry"].sum(), 2)
        self.assertEqual(masks["near_strike"].sum(), 3)
        self.assertEqual(masks["low_tail"].sum(), 1)
        self.assertEqual(masks["high_tail"].sum(), 1)
        self.assertEqual(
            np.sum(masks["low_tail"] & masks["near_strike"]), 0)


class MetricTests(unittest.TestCase):
    def test_prediction_errors_matches_manual_values(self):
        estimate = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 2.0, 5.0])
        metrics = prediction_errors(estimate, reference)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["maximum_absolute_error"], 2.0)
        self.assertAlmostEqual(metrics["bias"], -2.0 / 3.0)

    def test_prediction_errors_with_empty_mask(self):
        metrics = prediction_errors(np.array([1.0]), np.array([1.0]),
                                    np.array([False], dtype=bool))
        self.assertIsNone(metrics["rmse"])

    def test_acceptance_checks_use_core_region_thresholds(self):
        errors = {
            "core": {
                "price": {"rmse": 0.01},
                "delta": {"rmse": 0.005},
                "gamma": {"rmse": 0.0005},
                "theta": {"rmse": 0.05},
            }
        }
        acceptance = {
            "core_price_rmse": 0.02, "core_delta_rmse": 0.01,
            "core_gamma_rmse": 0.001, "core_theta_rmse": 0.1,
        }
        self.assertTrue(all(acceptance_checks(errors, acceptance).values()))
        errors["core"]["gamma"]["rmse"] = 0.002
        self.assertFalse(acceptance_checks(errors, acceptance)["core_gamma_rmse"])


if __name__ == "__main__":
    unittest.main()
