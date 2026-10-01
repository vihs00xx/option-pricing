import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.2, q=0.02)

    def test_plain_and_antithetic_uncertainty(self):
        for method in ("plain", "antithetic"):
            result = monte_carlo_price(**self.inputs, method=method, n_samples=4096, seed=7)
            rng = np.random.default_rng(np.random.SeedSequence(7).spawn(2)[0])
            n = 2048 if method == "antithetic" else 4096
            z = rng.standard_normal(n)
            def payoff(z):
                terminal = 100 * np.exp(0.01 + 0.2 * z)
                return np.exp(-0.05) * np.maximum(terminal - 100, 0)
            observations = payoff(z)
            if method == "antithetic":
                observations = (observations + payoff(-z)) / 2
            self.assertAlmostEqual(result.price, observations.mean(), places=12)
            self.assertAlmostEqual(result.standard_error, observations.std(ddof=1) / np.sqrt(n), places=12)
            self.assertEqual(result.independent_observations, n)
            self.assertEqual(result.payoff_evaluations, 4096)

    def test_control_uses_independent_pilot(self):
        production, pilot = np.random.SeedSequence(19).spawn(2)
        def xy(rng, n):
            terminal = 100 * np.exp(0.01 + 0.2 * rng.standard_normal(n))
            return np.exp(-0.05) * terminal, np.exp(-0.05) * np.maximum(terminal - 100, 0)
        x, y = xy(np.random.default_rng(pilot), 256)
        beta = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1)
        x, y = xy(np.random.default_rng(production), 4096 - 256)
        adjusted = y - beta * (x - 100 * np.exp(-0.02))
        result = monte_carlo_price(**self.inputs, method="control", seed=19,
                                  n_samples=4096, pilot_samples=256, batch_size=97)
        self.assertAlmostEqual(result.beta, beta, places=12)
        self.assertAlmostEqual(result.price, adjusted.mean(), places=12)
        self.assertAlmostEqual(result.standard_error, adjusted.std(ddof=1) / np.sqrt(adjusted.size), places=12)
        self.assertEqual(result.payoff_evaluations, 4096)
        self.assertEqual(result.independent_observations, 3840)
        self.assertEqual(result.pilot_samples, 256)

    def test_prices_agree_with_analytical_reference(self):
        for method in ("plain", "antithetic", "control", "sobol"):
            for option, S, r in (("call", 100, 0.05), ("put", 120, -0.01)):
                with self.subTest(method=method, option=option):
                    inputs = {**self.inputs, "option": option, "S": S, "r": r}
                    result = monte_carlo_price(**inputs, method=method, n_samples=131072, seed=24)
                    reference = black_scholes_price(**inputs)
                    self.assertLess(abs(result.price - reference), 6 * result.standard_error + 1e-10)
                    self.assertGreater(result.standard_error, 0)

    def test_reproducibility_and_batching(self):
        for method in ("plain", "antithetic", "control", "sobol"):
            first = monte_carlo_price(**self.inputs, method=method, batch_size=31)
            second = monte_carlo_price(**self.inputs, method=method, batch_size=65536)
            assert_allclose([first.price, first.standard_error],
                            [second.price, second.standard_error], rtol=1e-12)
            other = monte_carlo_price(**self.inputs, method=method, seed=43)
            self.assertNotEqual(first.price, other.price)

    def test_sobol_counts_and_interval(self):
        result = monte_carlo_price(**self.inputs, method="sobol", n_samples=8192, n_scrambles=8)
        self.assertEqual(result.independent_observations, 8)
        self.assertEqual(result.payoff_evaluations, 8192)
        self.assertAlmostEqual((result.ci_high - result.price) / result.standard_error,
                               2.3646242516, places=8)

    def test_plain_error_scale_over_independent_runs(self):
        reference = black_scholes_price(**self.inputs)
        estimates = [monte_carlo_price(**self.inputs, n_samples=4096, seed=seed)
                     for seed in range(64)]
        errors = np.array([result.price - reference for result in estimates])
        predicted = np.mean([result.standard_error**2 for result in estimates])
        self.assertLess(abs(errors.mean()), 4 * np.sqrt(predicted / 64))
        self.assertTrue(0.5 < np.mean(errors**2) / predicted < 1.8)

    def test_deterministic_limits(self):
        for change in ({"tau": 0}, {"sigma": 0}, {"S": 0}):
            for option in ("call", "put"):
                inputs = {**self.inputs, **change, "option": option}
                for method in ("plain", "antithetic", "control", "sobol"):
                    result = monte_carlo_price(**inputs, method=method)
                    self.assertAlmostEqual(result.price, black_scholes_price(**inputs))
                    self.assertEqual(result.standard_error, 0)
                    self.assertEqual(result.ci_low, result.ci_high)
                    self.assertEqual(result.payoff_evaluations, 0)

    def test_invalid_inputs(self):
        changes = [dict(S=-1), dict(K=0), dict(tau=-1), dict(sigma=-0.1), dict(q=np.nan),
                   dict(S=[100, 120]), dict(option="other"), dict(method="other"),
                   dict(n_samples=1), dict(n_samples=True), dict(batch_size=0),
                   dict(method="antithetic", n_samples=3),
                   dict(method="control", n_samples=1025, pilot_samples=1024),
                   dict(method="sobol", n_samples=1000), dict(method="sobol", n_scrambles=1)]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                monte_carlo_price(**{**self.inputs, **change})


if __name__ == "__main__":
    unittest.main()
