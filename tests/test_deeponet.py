import tempfile
import unittest
from pathlib import Path

import torch

from src.neural.deeponet import BlackScholesDeepONet, ParityDeepONet, load_deeponet
from experiments.deeponet.train import train_model
from experiments.deeponet.compare_supervision import bound_violations, supervised_loss
from experiments.deeponet.sweep_greek_weight import choose_weight
from experiments.deeponet.refine_optimizer import full_batch_closure, lbfgs_refine


class DeepONetTests(unittest.TestCase):
    def setUp(self):
        self.model = BlackScholesDeepONet(width=12, depth=2, rank=8)
        self.branch = torch.tensor([[100., 1., .05, .01, .2, 1.],
                                    [95., .8, .02, .03, .3, -1.]], dtype=torch.float64)
        self.trunk = torch.tensor([[105., .5], [90., .4]], dtype=torch.float64)

    def test_greeks_match_finite_differences(self):
        values = self.model.price_and_greeks(self.branch, self.trunk)
        for name, source, column, bump, sign in (
            ("delta", "trunk", 0, .001, 1),
            ("theta", "trunk", 1, .00001, -1),
            ("vega", "branch", 4, .00001, 1),
            ("rho", "branch", 2, .000001, 1),
        ):
            b_plus, b_minus = self.branch.clone(), self.branch.clone()
            t_plus, t_minus = self.trunk.clone(), self.trunk.clone()
            plus, minus = (b_plus, b_minus) if source == "branch" else (t_plus, t_minus)
            plus[:, column] += bump
            minus[:, column] -= bump
            estimate = sign * (self.model(b_plus, t_plus) - self.model(b_minus, t_minus)) / (2 * bump)
            torch.testing.assert_close(values[name], estimate, rtol=1e-5, atol=1e-6)
        plus, minus = self.trunk.clone(), self.trunk.clone()
        plus[:, 0] += .01
        minus[:, 0] -= .01
        gamma = (self.model(self.branch, plus) - 2 * self.model(self.branch, self.trunk)
                 + self.model(self.branch, minus)) / .01 ** 2
        torch.testing.assert_close(values["gamma"], gamma, rtol=1e-4, atol=1e-6)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"architecture": self.model.architecture,
                        "state_dict": self.model.state_dict()}, path)
            restored = load_deeponet(path)
            torch.testing.assert_close(self.model(self.branch, self.trunk),
                                       restored(self.branch, self.trunk))

    def test_chunked_full_batch_gradient_matches_unsplit_loss(self):
        targets = torch.zeros((2, 6), dtype=torch.float64)
        scales = torch.ones(5, dtype=torch.float64)
        expected = supervised_loss(self.model, self.branch, self.trunk, targets, scales, .01)*1e4
        expected.backward()
        gradients = [p.grad.clone() for p in self.model.parameters()]
        actual = full_batch_closure(self.model, (self.branch, self.trunk, targets), scales, chunk_size=1)
        torch.testing.assert_close(actual, expected.detach())
        for parameter, gradient in zip(self.model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient)

    def test_lbfgs_zero_budget_preserves_checkpoint(self):
        targets = torch.zeros((2, 6), dtype=torch.float64)
        original = {key: value.clone() for key, value in self.model.state_dict().items()}
        _, _, summary = lbfgs_refine(self.model, (self.branch, self.trunk, targets),
                                    (self.branch, self.trunk, targets[:, 0]),
                                    torch.ones(5, dtype=torch.float64), 0)
        self.assertEqual(summary['optimizer_steps'], 0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_zero_greek_weight_recovers_price_loss(self):
        targets = torch.zeros((2, 6), dtype=torch.float64)
        expected = (self.model(self.branch, self.trunk) / self.branch[:, 0]).square().mean()
        actual = supervised_loss(self.model, self.branch, self.trunk, targets,
                                 torch.ones(5, dtype=torch.float64), greek_weight=0.)
        torch.testing.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected, self.model.trunk[0].weight)[0]
        actual_grad = torch.autograd.grad(actual, self.model.trunk[0].weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_weight_selection_respects_price_gate_and_fallback(self):
        def candidate(weight, price, greek):
            metrics = {name: {'rmse': greek} for name in ('delta', 'gamma', 'vega', 'theta', 'rho')}
            metrics['price'] = {'rmse': price}
            return dict(weight=weight, metrics=metrics)
        candidates = [candidate(.01, .19, .5), candidate(.1, .21, .2), candidate(1., .3, .1)]
        selected, _ = choose_weight(candidates, .22, [1.] * 5)
        self.assertEqual(selected['weight'], .1)
        self.assertTrue(selected['price_gate_passed'])
        selected, _ = choose_weight(candidates, .1, [1.] * 5)
        self.assertEqual(selected['weight'], .01)
        self.assertFalse(selected['price_gate_passed'])

    def test_greek_loss_parameter_gradient_matches_finite_difference(self):
        targets = torch.zeros((2, 6), dtype=torch.float64)
        scales = torch.ones(5, dtype=torch.float64)
        loss = supervised_loss(self.model, self.branch, self.trunk, targets, scales)
        loss.backward()
        parameter = self.model.trunk[0].weight
        derivative = parameter.grad[0, 0].item()
        original = parameter[0, 0].item()
        bump = 1e-6
        with torch.no_grad():
            parameter[0, 0] = original + bump
        plus = supervised_loss(self.model, self.branch, self.trunk, targets, scales).item()
        with torch.no_grad():
            parameter[0, 0] = original - bump
        minus = supervised_loss(self.model, self.branch, self.trunk, targets, scales).item()
        with torch.no_grad():
            parameter[0, 0] = original
        self.assertAlmostEqual(derivative, (plus - minus) / (2 * bump),
                               delta=1e-4 * max(1., abs(derivative)))

    def test_constructor_does_not_change_default_dtype(self):
        original = torch.get_default_dtype()
        BlackScholesDeepONet(width=4, depth=1, rank=4)
        self.assertEqual(torch.get_default_dtype(), original)

    def test_training_selects_validation_best_and_respects_zero_budget(self):
        target = torch.tensor([10., 8.], dtype=torch.float64)
        split = (self.branch, self.trunk, target)
        settings = dict(seed=123, learning_rate=.001, epochs=20,
                        batch_size=2, validation_every=1, wall_seconds=30)
        history, last, metadata = train_model(self.model, split, split, settings)
        self.assertLess(metadata["selected_validation_mse"], history[0]["validation_mse"])
        with torch.no_grad():
            score = float(((self.model(*split[:2]) - target) / self.branch[:, 0]).square().mean())
        self.assertAlmostEqual(score, min(row["validation_mse"] for row in history))
        settings["wall_seconds"] = 0
        initial = {key: value.clone() for key, value in self.model.state_dict().items()}
        _, _, metadata = train_model(self.model, split, split, settings)
        self.assertEqual(metadata["optimizer_steps"], 0)
        self.assertEqual(metadata["stop_reason"], "wall_budget")
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, initial[key])

    def test_parity_holds_by_construction(self):
        model = ParityDeepONet(width=12, depth=2, rank=8)
        call = self.branch.clone()
        call[:, 5] = 1.
        put = self.branch.clone()
        put[:, 5] = -1.
        with_call = model.price_and_greeks(call, self.trunk)
        with_put = model.price_and_greeks(put, self.trunk)
        strike, tau = self.branch[:, 0], self.trunk[:, 1]
        discount_k = strike * torch.exp(-self.branch[:, 2] * tau)
        difference = self.trunk[:, 0] * torch.exp(-self.branch[:, 3] * tau) - discount_k
        torch.testing.assert_close(with_call["price"] - with_put["price"], difference, rtol=0, atol=1e-12)
        torch.testing.assert_close(with_call["delta"] - with_put["delta"],
                                   torch.exp(-self.branch[:, 3] * tau), rtol=0, atol=1e-12)
        torch.testing.assert_close(with_call["gamma"], with_put["gamma"], rtol=0, atol=1e-12)
        torch.testing.assert_close(with_call["vega"], with_put["vega"], rtol=0, atol=1e-12)
        torch.testing.assert_close(with_call["theta"] - with_put["theta"],
                                   self.branch[:, 3] * self.trunk[:, 0] * torch.exp(-self.branch[:, 3] * tau)
                                   - self.branch[:, 2] * discount_k, rtol=0, atol=1e-12)
        torch.testing.assert_close(with_call["rho"] - with_put["rho"], tau * discount_k, rtol=0, atol=1e-12)

    def test_parity_checkpoint_round_trip(self):
        model = ParityDeepONet(width=8, depth=1, rank=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"architecture": model.architecture,
                        "state_dict": model.state_dict()}, path)
            restored = load_deeponet(path)
        self.assertIsInstance(restored, ParityDeepONet)
        torch.testing.assert_close(restored(self.branch, self.trunk), model(self.branch, self.trunk))

    def test_bound_violations_measure_constraint_gaps(self):
        strike, tau = self.branch[:, 0], self.trunk[:, 1]
        discount = torch.exp(-self.branch[:, 3] * tau)
        disc_s = self.trunk[:, 0] * discount
        disc_k = strike * torch.exp(-self.branch[:, 2] * tau)
        sign = self.branch[:, 5]
        lower = torch.clamp(sign * (disc_s - disc_k), min=0.)
        upper = torch.where(sign > 0, disc_s, disc_k)
        violating = {"price": upper + strike,
                     "delta": sign * 2. * discount,
                     "gamma": -torch.ones(2, dtype=torch.float64)}
        violations = bound_violations(violating, self.branch, self.trunk, gamma_scale=2.)
        torch.testing.assert_close(violations["price"], torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-12)
        torch.testing.assert_close(violations["delta"], discount, rtol=0, atol=1e-12)
        torch.testing.assert_close(violations["gamma"], .5 * torch.ones(2, dtype=torch.float64),
                                   rtol=0, atol=1e-12)
        satisfied = {"price": .5 * (lower + upper),
                     "delta": sign * .5 * discount,
                     "gamma": torch.ones(2, dtype=torch.float64)}
        for name, value in bound_violations(satisfied, self.branch, self.trunk).items():
            torch.testing.assert_close(value, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-12)

    def test_penalty_term_changes_parameter_gradients(self):
        torch.manual_seed(5)
        model = BlackScholesDeepONet(width=8, depth=1, rank=4)
        targets = torch.zeros((2, 6), dtype=torch.float64)
        scales = torch.ones(5, dtype=torch.float64)
        plain = supervised_loss(model, self.branch, self.trunk, targets, scales, .01)
        penalized = supervised_loss(model, self.branch, self.trunk, targets, scales, .01,
                                    penalty_weight=1., penalty_types=("price",))
        self.assertGreater(float((penalized - plain).detach()), 0.)
        plain.backward()
        without = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        penalized.backward()
        self.assertTrue(any(not torch.equal(without[name], parameter.grad)
                            for name, parameter in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
