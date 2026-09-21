import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.neural.anchor_data import (
    build_dataset, component_counts, config_hash, load_dataset, nested_prefix_check,
    problem_from, reference_scales, sample_points, save_dataset, subset,
    verify_disjoint, verify_domain,
)
from src.neural.anchored_pinn import (
    AnchoredTrainer, MODES, bound_weight, quantities, stage_active_terms,
    term_losses, training_batches, validation_batches, weighted_total,
)
from src.neural.fixed_pinn import FixedBlackScholesPINN

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = {
    "study_version": 2,
    "problem": {"K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0,
                "x_min": 0.05, "x_max": 4.0},
    "network": {"hidden_layers": 2, "hidden_width": 8, "dtype": "float64"},
    "output": {
        "results_dir": "results/pinn/anchored_study_v2",
        "checkpoint_dir": "checkpoints/pinn/anchored_study_v2"
    },
    "acceptance": {
        "absolute_core_grid": {
            "price_rmse": 0.0098, "delta_rmse": 0.00247,
            "gamma_rmse": 0.000872, "theta_rmse": 0.0874
        },
        "improvement_vs_reference_broad_test": {
            "price_rmse": True, "delta_rmse": True,
            "gamma_rmse": True, "theta_rmse": True
        },
        "financial_categories": {
            "criterion": "per category proportion and magnitude",
            "categories": {
                "price_below_lower_bound": {"units": "price (dollars, K=100)"},
                "price_above_upper_bound": {"units": "price (dollars, K=100)"},
                "delta_below_zero": {"units": "delta (dimensionless)"},
                "delta_above_upper_bound": {"units": "delta (dimensionless)"},
                "gamma_negative": {"units": "gamma (1/dollars, K=100 scale)"}
            }
        }
    },
    "seed": 7,
    "anchor_counts": [128, 256],
    "greek_anchor_divisor": 4,
    "greek_anchor_max": 64,
    "validation_points": 32,
    "test_points": 32,
    "pde_points": 32,
    "terminal_points": 32,
    "minimum_tau": 0.01,
    "density": {"uniform": 0.5, "near_strike": 0.2, "near_expiry": 0.2,
                "high_gamma": 0.1, "boundaries": 0.0},
    "greek_density": {"uniform": 0.2, "near_strike": 0.4, "near_expiry": 0.2,
                      "high_gamma": 0.2, "boundaries": 0.0},
    "near_strike_half_width": 0.3,
    "near_expiry_tau_span": 0.05,
    "high_gamma_half_width": 0.15,
    "high_gamma_tau_span": 0.1,
    "boundary_fraction_of_domain": 0.1,
    "scale_epsilon": 1e-12,
    "weight_min": 0.05,
    "weight_max": 100.0,
    "adam_learning_rate": 0.01,
    "validation_every": 2,
    "lbfgs_history_size": 10,
    "lbfgs_tolerance_grad": 1e-12,
    "lbfgs_tolerance_change": 1e-14,
    "lbfgs_validation_every": 10,
    "selection": {
        "physics_only": {"pde": 1.0, "terminal": 1.0},
        "data_only": {"price": 1.0, "theta": 1.0},
        "hybrid": {"price": 1.0, "theta": 1.0},
        "v2": {"price": 1.0, "theta": 1.0, "delta": 1.0, "gamma": 1.0},
        "supervised_hybrid_control": {"price": 1.0, "theta": 1.0},
        "supervised_v2_control": {"price": 1.0, "theta": 1.0, "delta": 1.0,
                                  "gamma": 1.0},
    },
    "stages": [
        {"name": "warmup_price_and_payoff", "adam_epochs": 2, "lbfgs_iterations": 2,
         "weights": {"price": 1.0, "terminal": 1.0, "pde": 0.0,
                     "theta": 0.0, "delta": 0.0, "gamma": 0.0}},
        {"name": "pde_balance", "adam_epochs": 2, "lbfgs_iterations": 0,
         "weights": {"price": 1.0, "terminal": 1.0, "pde": 1.0,
                     "theta": 0.0, "delta": 0.0, "gamma": 0.0}},
    ],
}


class GreekTransformationTests(unittest.TestCase):
    """Verify the transformation algebra against numerically differentiated
    analytical Black--Scholes prices, not against another implementation of
    the same formula."""

    def test_transformations_reproduce_analytical_greeks(self):
        problem = problem_from(SETTINGS)
        spots = np.array([70.0, 100.0, 130.0])
        tau = np.array([0.3, 0.7, 1.0])
        step_y, step_tau = 1e-5, 1e-5

        def u(y_value, tau_value):
            s = problem.K * np.exp(y_value)
            return black_scholes_price(s, problem.K, tau_value, problem.r,
                                       problem.sigma, problem.q, "call") / problem.K

        y = np.log(spots / problem.K)
        u0 = u(y, tau)
        u_y = (u(y + step_y, tau) - u(y - step_y, tau)) / (2 * step_y)
        u_yy = (u(y + step_y, tau) - 2 * u0 + u(y - step_y, tau)) / step_y ** 2
        u_tau = (u(y, tau + step_tau) - u(y, tau - step_tau)) / (2 * step_tau)

        delta = u_y / np.exp(y)
        gamma = (u_yy - u_y) / (problem.K * np.exp(2 * y))
        theta = -problem.K * u_tau

        exact = black_scholes_greeks(spots, problem.K, tau, problem.r,
                                     problem.sigma, problem.q, "call")
        np.testing.assert_allclose(delta, exact["delta"], rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(gamma, exact["gamma"], rtol=2e-4, atol=2e-8)
        np.testing.assert_allclose(theta, exact["theta"], rtol=2e-4, atol=2e-6)

    def test_model_qualities_match_analytical_greeks(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        problem = problem_from(SETTINGS)
        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        y = torch.log(torch.tensor([[0.7], [1.0]], dtype=torch.float64)).requires_grad_(True)
        tau = torch.tensor([[0.5], [0.9]], dtype=torch.float64).requires_grad_(True)
        values = quantities(model, y, tau, {"delta", "gamma", "theta"})
        manual = model.price_and_greeks(y, tau)
        np.testing.assert_allclose(values["delta"].detach().numpy(),
                                   manual["delta"].detach().numpy(), rtol=1e-9)
        np.testing.assert_allclose(values["gamma"].detach().numpy(),
                                   manual["gamma"].detach().numpy(), rtol=1e-9)
        np.testing.assert_allclose(
            (values["theta"] * problem.K).detach().numpy(),
            manual["theta"].detach().numpy(), rtol=1e-9)


class WeightHandlingTests(unittest.TestCase):
    def test_exact_zero_weight_is_disabled(self):
        self.assertEqual(bound_weight(0.0, SETTINGS), 0.0)
        self.assertEqual(bound_weight(0, SETTINGS), 0.0)

    def test_positive_weights_are_clipped_into_band(self):
        self.assertEqual(bound_weight(0.001, SETTINGS), SETTINGS["weight_min"])
        self.assertEqual(bound_weight(1e6, SETTINGS), SETTINGS["weight_max"])
        self.assertEqual(bound_weight(2.0, SETTINGS), 2.0)

    def test_negative_or_nonfinite_weights_raise(self):
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bound_weight(bad, SETTINGS)

    def test_stage_active_terms_use_permitted_and_positive_weights(self):
        stage = SETTINGS["stages"][0]
        self.assertEqual(stage_active_terms("data_only", stage["weights"], SETTINGS),
                         ("price",))
        self.assertEqual(stage_active_terms("physics_only", stage["weights"], SETTINGS),
                         ("terminal",))
        self.assertEqual(stage_active_terms("hybrid", stage["weights"], SETTINGS),
                         ("price", "terminal"))
        self.assertEqual(stage_active_terms("v2", stage["weights"], SETTINGS),
                         ("price", "terminal"))
        self.assertEqual(
            stage_active_terms("v2", {"pde": 1.0, "theta": 1.0, "gamma": 1.0}, SETTINGS),
            ("theta", "gamma", "pde"))

    def test_disabled_losses_do_not_contribute_gradients(self):
        """Gradients of a terminal-only objective must ignore other terms."""
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(1)
        problem = problem_from(SETTINGS)
        dataset = build_dataset(SETTINGS)
        batches = training_batches(dataset, torch.float64)
        denominators = {name: 1.0 for name in
                        ("price", "delta", "gamma", "theta")}

        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        losses = term_losses(model, batches, denominators, ("terminal",))
        self.assertEqual(set(losses), {"terminal"})
        weighted_total(losses, {"terminal": 1.0}).backward()
        restricted = [p.grad.clone() for p in model.parameters()]

        model.zero_grad(set_to_none=True)
        ty, ttau, tpayoff = batches["terminal"]
        torch.mean((model(ty, ttau) - tpayoff) ** 2).backward()
        direct = [p.grad.clone() for p in model.parameters()]
        for left, right in zip(restricted, direct):
            np.testing.assert_allclose(left.numpy(), right.numpy(),
                                       rtol=1e-12, atol=1e-14)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.problem = problem_from(SETTINGS)

    def test_density_must_sum_to_one(self):
        bad = dict(SETTINGS["density"], uniform=0.9)
        with self.assertRaises(ValueError):
            sample_points(self.problem, 16, 1, bad, SETTINGS)

    def test_counts_are_exact_and_points_inside_domain(self):
        y, tau, component = sample_points(self.problem, 97, 3,
                                          SETTINGS["density"], SETTINGS)
        self.assertEqual(len(y), 97)
        self.assertEqual(len(component), 97)
        self.assertTrue(np.all(np.exp(y) >= self.problem.x_min - 1e-12))
        self.assertTrue(np.all(np.exp(y) <= self.problem.x_max + 1e-12))
        self.assertTrue(np.all(tau >= SETTINGS["minimum_tau"] - 1e-12))
        self.assertTrue(np.all(tau <= self.problem.T + 1e-12))

    def test_sampling_is_dtype_explicit(self):
        torch.set_default_dtype(torch.float32)  # hostile process default
        try:
            y, tau, _ = sample_points(self.problem, 64, 5,
                                      SETTINGS["density"], SETTINGS)
        finally:
            torch.set_default_dtype(torch.float64)
        self.assertEqual(y.dtype, np.float64)
        self.assertEqual(tau.dtype, np.float64)
        self.assertGreater(np.unique(y).size, 60)

    def test_boundary_strips_are_symmetric(self):
        density = {"uniform": 0.0, "near_strike": 0.0, "near_expiry": 0.0,
                   "high_gamma": 0.0, "boundaries": 1.0}
        y, _, _ = sample_points(self.problem, 200, 9, density, SETTINGS)
        y_min = np.log(self.problem.x_min)
        y_max = np.log(self.problem.x_max)
        left = y[y < 0.5 * (y_min + y_max)]
        right = y[y >= 0.5 * (y_min + y_max)]
        self.assertEqual(left.size, right.size)
        edge = SETTINGS["boundary_fraction_of_domain"] * (y_max - y_min)
        self.assertTrue(np.all(left <= y_min + edge + 1e-12))
        self.assertTrue(np.all(right >= y_max - edge - 1e-12))

    def test_dataset_builds_and_verifies(self):
        dataset = build_dataset(SETTINGS)
        self.assertEqual(len(dataset["anchor_y"]), 256)
        self.assertEqual(len(dataset["greek_y"]), 64)
        self.assertEqual(len(dataset["validation_terminal_y"]), 32)
        self.assertEqual(verify_domain(dataset, SETTINGS), [])
        self.assertEqual(verify_disjoint(dataset), {})

    def test_terminal_samples_are_independent(self):
        dataset = build_dataset(SETTINGS)
        training = set(zip(dataset["terminal_y"].tolist(),
                           dataset["terminal_tau"].tolist()))
        validation = set(zip(dataset["validation_terminal_y"].tolist(),
                             dataset["validation_terminal_tau"].tolist()))
        self.assertEqual(training & validation, set())
        self.assertFalse(np.allclose(dataset["terminal_y"],
                                     dataset["validation_terminal_y"]))

    def test_subsets_are_nested_and_cover_every_component(self):
        dataset = build_dataset(SETTINGS)
        small = subset(dataset, 128, SETTINGS)
        large = subset(dataset, 256, SETTINGS)
        self.assertEqual(len(small["anchor_y"]), 128)
        self.assertEqual(len(small["greek_y"]), 32)
        # Stratified prefixes are approximately proportional, not exactly.
        uniform = component_counts(small["anchor_component"])["uniform"]
        self.assertLessEqual(abs(uniform - 64), 3)
        np.testing.assert_allclose(small["anchor_y"], large["anchor_y"][:128])
        self.assertTrue(nested_prefix_check(dataset, [128, 256])["nested"])

    def test_disjointness_detects_reuse(self):
        dataset = build_dataset(SETTINGS)
        dataset["test_y"] = dataset["anchor_y"].copy()
        dataset["test_tau"] = dataset["anchor_tau"].copy()
        self.assertIn("anchor_train|test", verify_disjoint(dataset))

    def test_provenance_mismatch_is_detected(self, ):
        dataset = build_dataset(SETTINGS)
        path = ROOT / "results" / "pinn" / "anchored_study_v2" / "data" / "test_provenance.npz"
        save_dataset(path, dataset, SETTINGS)
        try:
            load_dataset(path, SETTINGS)  # matching config is accepted
            changed = dict(SETTINGS, seed=SETTINGS["seed"] + 1)
            with self.assertRaises(ValueError):
                load_dataset(path, changed)
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".meta.json").unlink(missing_ok=True)

    def test_smoke_and_production_dataset_paths_differ(self):
        from experiments.pinn.train_anchored_ablation import dataset_paths
        smoke = dataset_paths(SETTINGS, True)
        production = dataset_paths(SETTINGS, False)
        self.assertNotEqual(smoke, production)
        self.assertIn("smoke", smoke.name)

    def test_config_hash_is_sensitive_to_data_settings(self):
        changed = dict(SETTINGS, validation_points=SETTINGS["validation_points"] + 1)
        self.assertNotEqual(config_hash(SETTINGS), config_hash(changed))


class TrainerSelectionTests(unittest.TestCase):
    def _trainer(self, mode="hybrid", stages=None, settings=None):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(11)
        settings = settings or SETTINGS
        problem = problem_from(settings)
        model = FixedBlackScholesPINN(problem, hidden_layers=2, hidden_width=8)
        dataset = build_dataset(settings)
        return AnchoredTrainer(model, dataset, reference_scales(dataset),
                               settings, mode, seed=11), stages or settings["stages"]

    def test_stage_weights_do_not_change_selection_score(self):
        trainer, _ = self._trainer()
        before, _ = trainer.selection_score()
        mutated = json.loads(json.dumps(SETTINGS))
        mutated["stages"][0]["weights"] = {"price": 7.0, "terminal": 3.0,
                                           "pde": 0.0, "theta": 0.0,
                                           "delta": 0.0, "gamma": 0.0}
        trainer.settings = mutated
        after, _ = trainer.selection_score()
        self.assertAlmostEqual(before, after, places=12)

    def test_selection_ignores_terms_outside_the_declared_objective(self):
        trainer, _ = self._trainer(mode="hybrid")
        self.assertEqual(set(trainer.selection_terms), {"price", "theta"})
        physics, _ = self._trainer(mode="physics_only")
        self.assertEqual(set(physics.selection_terms), {"pde", "terminal"})

    def test_best_checkpoint_is_retained_across_stages(self):
        trainer, stages = self._trainer()
        history, metadata, states = trainer.train(stages, log=lambda *_: None)
        self.assertLessEqual(metadata["selected_selection_score"],
                             metadata["last_selection_score"] + 1e-12)
        self.assertTrue(np.isfinite(metadata["selected_selection_score"]))
        for stage_end in [row for row in history if row["phase"] == "stage_end"]:
            self.assertLessEqual(metadata["selected_selection_score"],
                                 stage_end["selection_score"] + 1e-12)
        # The selected and last states are distinct objects.
        self.assertIsNot(states["selected"], states["last"])

    def test_skipped_stage_is_reported(self):
        """A stage whose weights disable every permitted term is skipped."""
        trainer, _ = self._trainer(mode="data_only")
        stages = [{"name": "physics_only_stage", "adam_epochs": 2,
                   "lbfgs_iterations": 0,
                   "weights": {"price": 0.0, "terminal": 1.0, "pde": 1.0,
                               "theta": 0.0, "delta": 0.0, "gamma": 0.0}}]
        messages = []
        history, metadata, _ = trainer.train(stages, log=messages.append)
        self.assertTrue(any("skipping stage" in m for m in messages))
        self.assertIn("physics_only_stage", metadata["stage_active_terms"])

    def test_time_budget_interruption_leaves_valid_state(self):
        trainer, _ = self._trainer()
        stages = [{"name": "long", "adam_epochs": 5, "lbfgs_iterations": 50,
                   "weights": {"price": 1.0, "terminal": 1.0, "pde": 1.0,
                               "theta": 0.1, "delta": 0.1, "gamma": 0.1}}]
        history, metadata, states = trainer.train(stages, wall_clock_cap=0.0,
                                                  log=lambda *_: None)
        self.assertTrue(metadata["hit_wall_cap"])
        self.assertLessEqual(metadata["selected_selection_score"],
                             metadata["last_selection_score"] + 1e-12)
        for value in states["selected"].values():
            self.assertTrue(torch.all(torch.isfinite(value)))


class EvaluatorTests(unittest.TestCase):
    """Acceptance must use core-grid metrics for absolute thresholds and
    broad-test metrics for improvement, and aggregation must be exact."""

    def _reference_grid(self, any_proportion=0.48):
        names = ("price_below_lower_bound", "price_above_upper_bound",
                 "delta_below_zero", "delta_above_upper_bound",
                 "gamma_negative", "any_constraint")
        violations = {}
        for name in names:
            violations[name] = {
                "full": {"points": 100, "count": 48,
                         "proportion": any_proportion if name == "any_constraint" else 0.2,
                         "maximum_magnitude": 0.1, "mean_magnitude": 0.01},
                "tolerance": 1e-7, "maximum_region": "full",
            }
        return {"label": "core_grid", "points": 100, "errors": {},
                "violations": violations}

    def _row(self, model, kind="selected", **overrides):
        row = {"model": model, "checkpoint_kind": kind, "protocol_id": "p",
               "has_training_metadata": True,
               "core_grid_price_rmse": 0.5, "core_grid_delta_rmse": 0.2,
               "core_grid_gamma_rmse": 0.1, "core_grid_theta_rmse": 0.9,
               "broad_test_price_rmse": 0.01, "broad_test_delta_rmse": 0.001,
               "broad_test_gamma_rmse": 0.001, "broad_test_theta_rmse": 0.05,
               "core_grid_any_constraint_proportion": 0.4}
        for category in ("price_below_lower_bound", "price_above_upper_bound",
                         "delta_below_zero", "delta_above_upper_bound",
                         "gamma_negative"):
            row[f"core_grid_{category}_proportion"] = 0.1
            row[f"core_grid_{category}_max_magnitude"] = 0.05
        row.update(overrides)
        return row

    def _reference_row(self):
        return {"model": "reference_existing_best", "checkpoint_kind": "reference",
                "broad_test_price_rmse": 0.1, "broad_test_delta_rmse": 0.01,
                "broad_test_gamma_rmse": 0.01, "broad_test_theta_rmse": 0.5,
                "core_grid_price_rmse": 0.0098}

    def test_absolute_threshold_applies_to_core_grid_only(self):
        from experiments.pinn.evaluate_anchored_ablation import acceptance_rows
        rows = [self._row("hybrid_n8192_s1")]
        records = acceptance_rows(rows, self._reference_row(),
                                  self._reference_grid(), SETTINGS)
        record = records[0]
        # Fails the absolute core-grid thresholds...
        self.assertFalse(record["absolute_core_grid_pass"])
        # ...but beats the reference on the broad test mixture.
        self.assertTrue(record["broad_test_improvement_pass"])

    def test_broad_test_error_is_not_judged_against_core_threshold(self):
        from experiments.pinn.evaluate_anchored_ablation import acceptance_rows
        row = self._row("hybrid_n8192_s1", core_grid_price_rmse=0.001,
                        core_grid_delta_rmse=0.001, core_grid_gamma_rmse=0.0001,
                        core_grid_theta_rmse=0.01, broad_test_price_rmse=0.2,
                        broad_test_delta_rmse=0.2, broad_test_gamma_rmse=0.2,
                        broad_test_theta_rmse=0.2)
        record = acceptance_rows([row], self._reference_row(),
                                 self._reference_grid(), SETTINGS)[0]
        # A large broad-test error must not fail the absolute core-grid check.
        self.assertTrue(record["absolute_core_grid_pass"])
        self.assertFalse(record["broad_test_improvement_pass"])

    def test_constraint_failure_fails_acceptance(self):
        from experiments.pinn.evaluate_anchored_ablation import acceptance_rows
        worse = self._row("hybrid_n8192_s1",
                          core_grid_any_constraint_proportion=0.9)
        better = self._row("hybrid_n8192_s2",
                           core_grid_any_constraint_proportion=0.2)
        records = acceptance_rows([worse, better], self._reference_row(),
                                  self._reference_grid(), SETTINGS)
        self.assertFalse(records[0]["financial_constraints_pass"])
        self.assertTrue(records[1]["financial_constraints_pass"])

    def test_fewer_but_larger_violations_fail_acceptance(self):
        """A low proportion must not rescue an outsized magnitude."""
        from experiments.pinn.evaluate_anchored_ablation import acceptance_rows
        # Fewer violating points than the reference, but each violation is
        # far larger than the reference's maximum for that category.
        row = self._row("hybrid_n8192_s3",
                        core_grid_price_below_lower_bound_proportion=0.01,
                        core_grid_price_below_lower_bound_max_magnitude=5.0)
        record = acceptance_rows([row], self._reference_row(),
                                 self._reference_grid(), SETTINGS)[0]
        self.assertTrue(record["price_below_lower_bound_proportion_ok"])
        self.assertFalse(record["price_below_lower_bound_magnitude_ok"])
        self.assertFalse(record["financial_constraints_pass"])

    def test_per_category_units_and_tolerance_are_reported(self):
        from experiments.pinn.evaluate_anchored_ablation import acceptance_rows
        record = acceptance_rows([self._row("hybrid_n8192_s4")],
                                 self._reference_row(),
                                 self._reference_grid(), SETTINGS)[0]
        self.assertEqual(record["gamma_negative_units"],
                         "gamma (1/dollars, K=100 scale)")
        self.assertEqual(record["price_below_lower_bound_units"],
                         "price (dollars, K=100)")
        self.assertAlmostEqual(record["gamma_negative_tolerance"], 1e-7)

    def test_aggregate_matches_hand_calculation(self):
        from experiments.pinn.evaluate_anchored_ablation import aggregate
        rows = [self._row(f"hybrid_n8192_s{seed}",
                          broad_test_price_rmse=value)
                for seed, value in ((111, 1.0), (222, 2.0), (333, 3.0))]
        records = aggregate(rows, None)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["seeds"], 3)
        self.assertAlmostEqual(record["broad_test_price_rmse_mean"], 2.0)
        self.assertAlmostEqual(record["broad_test_price_rmse_std"], 1.0)
        self.assertEqual(record["std_convention"], "sample")

    def test_acceptance_summary_counts_seed_passes(self):
        from experiments.pinn.evaluate_anchored_ablation import (
            acceptance_rows, acceptance_summary)
        rows = [self._row("hybrid_n8192_s111", core_grid_any_constraint_proportion=0.1),
                self._row("hybrid_n8192_s222", core_grid_any_constraint_proportion=0.9)]
        acceptance = acceptance_rows(rows, self._reference_row(),
                                     self._reference_grid(), SETTINGS)
        summary = acceptance_summary(acceptance)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["financial_constraints_pass_count"], 1)
        self.assertAlmostEqual(summary[0]["financial_constraints_pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()