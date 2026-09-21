import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

from src.neural.fixed_pinn import FixedBlackScholesPINN, FixedBlackScholesProblem
from src.neural.pinn_training import (
    VariantTrainer,
    causal_bin_weights,
    category_gradient_norm,
    compute_constraint_violations,
    effective_bin_count,
    update_category_weights,
    violation_point_weights,
    violation_total_magnitude,
)


class ConstraintViolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        self.model = FixedBlackScholesPINN(self.problem, hidden_layers=2, hidden_width=8)

    def test_violations_are_positive_and_finite(self):
        y = torch.log(torch.tensor([[0.5], [1.0], [2.0]]))
        tau = torch.tensor([[0.1], [0.5], [1.0]])
        y.requires_grad_(True)
        tau.requires_grad_(True)
        violations = compute_constraint_violations(self.model, y, tau)
        for name, magnitudes in violations.items():
            self.assertTrue(torch.all(magnitudes >= 0), name)
            self.assertTrue(torch.all(torch.isfinite(magnitudes)), name)

    def test_violations_require_grad_for_training(self):
        y = torch.log(torch.tensor([[1.2]]))
        y.requires_grad_(True)
        tau = torch.tensor([[0.5]])
        tau.requires_grad_(True)
        violations = compute_constraint_violations(self.model, y, tau)
        total = sum(magnitude.sum() for magnitude in violations.values())
        total.backward()
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(tau.grad)

    def test_total_magnitude_is_sum_of_parts(self):
        y = torch.log(torch.tensor([[0.4], [1.0], [3.0]]))
        tau = torch.tensor([[0.05], [0.5], [1.0]])
        y.requires_grad_(True)
        tau.requires_grad_(True)
        violations = compute_constraint_violations(self.model, y, tau)
        total = violation_total_magnitude(violations)
        expected = sum(m for m in violations.values())
        assert_allclose(total.detach().numpy(), expected.detach().numpy())


class CausalWeightTests(unittest.TestCase):
    def test_earlier_bins_get_larger_weights(self):
        bin_losses = torch.tensor([0.1, 10.0, 10.0, 10.0])
        weights, cumulative = causal_bin_weights(bin_losses, epsilon=1.0)
        self.assertTrue(torch.all(weights[1:] < weights[0]))
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=12)
        self.assertEqual(float(cumulative[0]), 0.0)
        self.assertAlmostEqual(float(cumulative[1]), 0.1, places=12)

    def test_equal_losses_give_geometric_weights(self):
        bin_losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        weights, _ = causal_bin_weights(bin_losses, epsilon=5.0)
        # Cumulative sums are 0, 1, 2, 3 and the total is 4, so the
        # normalised cumulative is 0, 0.25, 0.5, 0.75 and the weights
        # decay geometrically as exp(-5 * 0.25) per step.
        ratios = weights[1:] / weights[:-1]
        assert_allclose(
            ratios.detach().numpy(),
            np.full(3, float(torch.exp(torch.tensor(-1.25)))),
            rtol=1e-10,
        )

    def test_effective_bin_count_matches_participation_ratio(self):
        uniform = torch.ones(8)
        self.assertAlmostEqual(effective_bin_count(uniform), 8.0, places=10)
        concentrated = torch.zeros(8)
        concentrated[0] = 1.0
        self.assertAlmostEqual(effective_bin_count(concentrated), 1.0, places=10)

    def test_causal_concentration_is_scale_free(self):
        # Rescaling every bin loss by a common factor must not change the
        # weights, because epsilon acts on cumulative / total.
        base, _ = causal_bin_weights(torch.tensor([0.1, 0.2, 0.3, 0.4]), epsilon=2.0)
        scaled, _ = causal_bin_weights(torch.tensor([1.0, 2.0, 3.0, 4.0]), epsilon=2.0)
        assert_allclose(base.detach().numpy(), scaled.detach().numpy(), rtol=1e-10)

    def test_larger_epsilon_sharpens_focus(self):
        bin_losses = torch.tensor([0.01, 5.0, 5.0])
        gentle, _ = causal_bin_weights(bin_losses, epsilon=0.1)
        sharp, _ = causal_bin_weights(bin_losses, epsilon=10.0)
        self.assertGreater(float(sharp[0] / sharp[1]),
                           float(gentle[0] / gentle[1]))


class CategoryWeightTests(unittest.TestCase):
    def test_price_bounds_do_not_dominate(self):
        weights, magnitudes = update_category_weights(
            {"price_bounds": 1.0, "delta_bounds": 1.0, "gamma": 1.0},
            {"price_bounds": torch.tensor(0.001),
             "delta_bounds": torch.tensor(0.8),
             "gamma": torch.tensor(1.2)},
            ema=0.0, minimum=1.0, maximum=10.0,
        )
        contributions = {name: weights[name] * magnitudes[name] for name in weights}
        # reference = mean(0.001, 0.8, 1.2) = 0.667. The price-bound
        # weight saturates at the ceiling (10, not the old 100) and its
        # contribution stays far below Delta's.
        self.assertAlmostEqual(weights["price_bounds"], 10.0)
        self.assertAlmostEqual(weights["delta_bounds"], 1.0)
        self.assertLess(contributions["price_bounds"], contributions["delta_bounds"])

    def test_floor_prevents_weight_collapse(self):
        weights, _ = update_category_weights(
            {"gamma": 2.0}, {"gamma": torch.tensor(100.0)},
            ema=0.5, minimum=1.0, maximum=10.0,
        )
        # reference = 100.0, target = 1.0 clipped to the floor.
        # EMA: 0.5 * 2.0 + 0.5 * 1.0 = 1.5 (never ~0.01)
        self.assertAlmostEqual(weights["gamma"], 1.5)

    def test_weights_respect_bounds(self):
        weights, _ = update_category_weights(
            {"gamma": 1.0}, {"gamma": torch.tensor(1000.0)},
            ema=0.0, minimum=1.0, maximum=10.0,
        )
        # target = 1000 / 1000 clipped to floor 1.0
        self.assertAlmostEqual(weights["gamma"], 1.0)

    def test_category_gradient_norm_is_mean_absolute_entry(self):
        torch.set_default_dtype(torch.float64)
        a = torch.tensor([1.0, -2.0], requires_grad=True)
        loss = (a**2).sum()
        norm = category_gradient_norm(loss, [a])
        self.assertAlmostEqual(float(norm), 3.0, places=12)


class PointWeightTests(unittest.TestCase):
    def test_point_weights_are_mean_one_and_increasing(self):
        violation_total = torch.tensor([0.0, 1.0, 3.0, 10.0])
        weights = violation_point_weights(violation_total, eta=1.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=12)
        self.assertTrue(torch.all(weights[1:] > weights[:-1]))
        self.assertTrue(torch.all(weights > 0))


class TrainerSmokeTests(unittest.TestCase):
    def _budget(self, epochs):
        return {
            "adam_epochs": epochs, "adam_learning_rate": 0.01,
            "adam_batch_size": 64, "terminal_batch_size": 64,
            "terminal_loss_weight": 25.0, "near_strike_fraction": 0.25,
            "near_strike_half_width": 0.3, "minimum_tau": 0.0001,
            "log_every": 1, "wall_clock_cap_seconds": 60,
        }

    def test_all_variants_complete_and_log_rows(self):
        torch.set_default_dtype(torch.float64)
        problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        constrained_settings = {
            "constraint_points": 32, "violation_candidates": 64,
            "violation_top_k": 8, "resample_every": 2, "balance_every": 2,
            "balance_ema": 0.5, "category_weight_min": 1.0,
            "category_weight_max": 10.0, "point_weight_eta": 1.0,
            "constraint_weight": 1.0,
        }
        causal_settings = {
            "causal_bins": 4, "epsilon_start": 1.0,
            "epsilon_final": 10.0, "epsilon_anneal_fraction": 0.75,
        }
        for variant, extra in (
            ("baseline_matched", {}),
            ("constrained", constrained_settings),
            ("causal", causal_settings),
        ):
            torch.manual_seed(7)
            model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
            trainer = VariantTrainer(
                model, self._budget(5), variant, extra, seed=11,
                causal_settings=causal_settings,
            )
            history, metadata = trainer.train(log_every=1, log=lambda *_: None)
            self.assertEqual(len(history), 5, variant)
            self.assertEqual(metadata["adam_epochs_completed"], 5, variant)
            self.assertFalse(metadata["hit_wall_cap"], variant)
            self.assertTrue(all(np.isfinite(row["loss"]) for row in history))

    def test_wall_clock_cap_stops_training(self):
        torch.set_default_dtype(torch.float64)
        problem = FixedBlackScholesProblem(100, 1, 0.05, 0.2, 0, 0.05, 4)
        torch.manual_seed(7)
        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        trainer = VariantTrainer(
            model, self._budget(50), "baseline_matched", {}, seed=11,
        )
        history, metadata = trainer.train(
            wall_clock_cap=0.0, log_every=1, log=lambda *_: None
        )
        self.assertEqual(metadata["adam_epochs_completed"], 1)
        self.assertTrue(metadata["hit_wall_cap"])


if __name__ == "__main__":
    unittest.main()
