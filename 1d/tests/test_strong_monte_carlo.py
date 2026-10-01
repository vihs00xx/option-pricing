import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.monte_carlo_greeks import monte_carlo_greeks
from src.sampling import METHODS, estimate, sobol_normals


class StrongMonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(S=100.0, K=100.0, tau=0.7, r=-0.01, sigma=0.3, q=0.02)

    def test_combined_estimators_and_counts(self):
        for method, pilot, observations in (("antithetic_control", 1024, 32256), ("sobol_control", 32768, 16)):
            for option in ("call", "put"):
                result = monte_carlo_price(**self.inputs, option=option, method=method,
                                          n_samples=65536, pilot_samples=pilot, n_scrambles=16)
                reference = black_scholes_price(**self.inputs, option=option)
                self.assertLess(abs(result.price - reference), 6 * result.standard_error + 1e-10)
                self.assertEqual(result.payoff_evaluations, 65536)
                self.assertEqual(result.pilot_samples, pilot)
                self.assertEqual(result.independent_observations, observations)

    def test_sobol_control_regression_uses_independent_means(self):
        production_seed, pilot_seed = np.random.SeedSequence(23).spawn(2)
        def evaluate(z):
            x = np.exp(0.2 * z)
            y = np.maximum(x - 1.1, 0)
            return x[:, None], y[:, None]
        pilot_means = [tuple(v.mean() for v in evaluate(sobol_normals(child, 256)))
                       for child in pilot_seed.spawn(8)]
        x, y = np.array(pilot_means).T
        beta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        means = []
        for child in production_seed.spawn(8):
            x, y = evaluate(sobol_normals(child, 256))
            means.append(float((y - beta * (x - np.exp(0.02))).mean()))
        actual = estimate(evaluate, np.array([np.exp(0.02)]), 4096, "sobol_control", 23, 65536, 2048, 8)
        assert_allclose(actual.beta, beta, rtol=1e-10)
        assert_allclose(actual.mean, np.mean(means), rtol=1e-10)
        assert_allclose(actual.se, np.std(means, ddof=1) / np.sqrt(8), rtol=1e-9)

    def test_antithetic_control_pair_regression(self):
        def evaluate(z):
            x = np.exp(0.3 * z)
            return x[:, None], np.maximum(x - 1, 0)[:, None]
        def pairs(rng, n):
            z = rng.standard_normal(n)
            x, y = evaluate(z)
            opposite_x, opposite_y = evaluate(-z)
            return ((x + opposite_x) / 2).ravel(), ((y + opposite_y) / 2).ravel()
        production, pilot = np.random.SeedSequence(9).spawn(2)
        x, y = pairs(np.random.default_rng(pilot), 256)
        beta = np.cov(x, y)[0, 1] / x.var(ddof=1)
        x, y = pairs(np.random.default_rng(production), 1792)
        adjusted = y - beta * (x - np.exp(0.045))
        actual = estimate(evaluate, np.array([np.exp(0.045)]), 4096, "antithetic_control", 9, 37, 512, 8)
        assert_allclose(actual.beta, beta, rtol=1e-12)
        assert_allclose(actual.mean, adjusted.mean(), rtol=1e-12)
        assert_allclose(actual.se, adjusted.std(ddof=1) / np.sqrt(len(adjusted)), rtol=1e-12)

    def test_hybrid_greeks_against_analytical(self):
        for option in ("call", "put"):
            for method in METHODS:
                inputs = {**self.inputs, "option": option}
                result = monte_carlo_greeks(**inputs, method=method, n_samples=131072,
                                           pilot_samples=65536 if method == "sobol_control" else 4096,
                                           n_scrambles=16, seed=57)
                reference = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
                for name, value in reference.items():
                    with self.subTest(option=option, method=method, greek=name):
                        self.assertLess(abs(result.values[name] - value), 7 * result.standard_errors[name] + 1e-8)
                self.assertEqual(result.payoff_evaluations, 131072)

    def test_finite_difference_common_randoms_and_count(self):
        result = monte_carlo_greeks(**self.inputs, estimator="finite_difference", method="plain",
                                   n_samples=8192, relative_bump=0.003, seed=12)
        def price(**changes):
            return monte_carlo_price(**{**self.inputs, **changes}, n_samples=8192, seed=12).price
        h = 0.003 * self.inputs["S"]
        numerical = (price(S=100 + h) - 2 * price() + price(S=100 - h)) / h**2
        self.assertAlmostEqual(result.values["gamma"], numerical, places=9)
        self.assertEqual(result.sample_evaluations, 8192)
        self.assertEqual(result.payoff_evaluations, 9 * 8192)

    def test_likelihood_ratio_greeks(self):
        for method in METHODS:
            for option in ("call", "put"):
                inputs = {**self.inputs, "option": option}
                result = monte_carlo_greeks(**inputs, method=method, estimator="likelihood_ratio",
                                           n_samples=131072, n_scrambles=16, seed=119,
                                           pilot_samples=65536 if method == "sobol_control" else 4096)
                reference = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
                for name in reference:
                    with self.subTest(method=method, option=option, quantity=name):
                        self.assertLess(abs(result.values[name] - reference[name]),
                                        7 * result.standard_errors[name] + 1e-8)

    def test_conditional_smoothing_greeks(self):
        for method in ("plain", "sobol", "sobol_control"):
            for option in ("call", "put"):
                inputs = {**self.inputs, "option": option}
                result = monte_carlo_greeks(**inputs, method=method, estimator="conditional",
                                           n_samples=65536, n_scrambles=16, seed=219,
                                           pilot_samples=32768 if method == "sobol_control" else 4096)
                reference = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
                for name in reference:
                    with self.subTest(method=method, option=option, quantity=name):
                        self.assertLess(abs(result.values[name] - reference[name]),
                                        7 * result.standard_errors[name] + 1e-8)
                self.assertEqual(result.payoff_evaluations, 65536)

    def test_joint_price_matches_price_only(self):
        for method in METHODS:
            arguments = dict(method=method, n_samples=8192, pilot_samples=4096 if method == "sobol_control" else 512)
            price = monte_carlo_price(**self.inputs, **arguments)
            greeks = monte_carlo_greeks(**self.inputs, **arguments)
            assert_allclose([price.price, price.standard_error],
                            [greeks.values["price"], greeks.standard_errors["price"]], rtol=1e-10)

    def test_invalid_combination_and_greek_inputs(self):
        for options in (dict(method="sobol_control", n_samples=8192, pilot_samples=1024),
                        dict(method="antithetic_control", pilot_samples=3)):
            with self.assertRaises(ValueError):
                monte_carlo_price(**self.inputs, **options)
        for changes in (dict(tau=0), dict(sigma=0), dict(S=0), dict(relative_bump=1),
                        dict(rate_bump=0), dict(estimator="wrong"), dict(conditional_fraction=0),
                        dict(conditional_fraction=1), dict(conditional_fraction=np.nan)):
            with self.assertRaises(ValueError):
                monte_carlo_greeks(**{**self.inputs, **changes})


if __name__ == "__main__":
    unittest.main()
