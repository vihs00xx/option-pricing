import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.crank_nicolson import crank_nicolson_grid, crank_nicolson_price, crank_nicolson_values


class CrankNicolsonTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.2, q=0.0)

    def test_price_against_black_scholes(self):
        cases = (
            {},
            {"option": "put", "q": 0.03},
            {"S": 90.0, "tau": 0.15, "sigma": 0.35},
            {"option": "put", "S": 112.0, "r": -0.015, "q": 0.04, "tau": 1.4},
        )
        for changes in cases:
            inputs = {**self.inputs, **changes}
            actual = crank_nicolson_price(**inputs, n_space=500, n_time=500)
            expected = black_scholes_price(**inputs)
            with self.subTest(inputs=inputs):
                self.assertLess(abs(actual - expected), 0.002)

    def test_greeks_against_black_scholes(self):
        for option in ("call", "put"):
            inputs = {**self.inputs, "option": option, "q": 0.02, "tau": 0.8, "sigma": 0.27}
            actual = crank_nicolson_values(**inputs, n_space=600, n_time=600)
            expected = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
            tolerances = {"price": 0.002, "delta": 2e-4, "gamma": 3e-5,
                          "vega": 0.02, "theta": 0.01, "rho": 0.02}
            for name in expected:
                with self.subTest(option=option, quantity=name):
                    self.assertLess(abs(actual[name] - expected[name]), tolerances[name])

    def test_grid_boundaries_and_parity(self):
        call = crank_nicolson_grid(100, 0.7, -0.01, 0.3, 0.02, "call",
                                   s_max=400, n_space=300, n_time=300)
        put = crank_nicolson_grid(100, 0.7, -0.01, 0.3, 0.02, "put",
                                  s_max=400, n_space=300, n_time=300)
        expected = call.spots * np.exp(-0.02 * 0.7) - 100 * np.exp(0.01 * 0.7)
        assert_allclose(call.values - put.values, expected, atol=3e-4)

    def test_rannacher_convergence(self):
        exact = black_scholes_price(**self.inputs)
        coarse = crank_nicolson_price(**self.inputs, n_space=80, n_time=80)
        fine = crank_nicolson_price(**self.inputs, n_space=320, n_time=320)
        self.assertLess(abs(fine - exact), abs(coarse - exact))

    def test_richardson_extrapolation(self):
        inputs = {**self.inputs, "S": 103.0, "q": 0.02, "sigma": 0.27}
        exact = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
        standard = crank_nicolson_values(**inputs, n_space=120, n_time=120)
        extrapolated = crank_nicolson_values(
            **inputs, n_space=120, n_time=120, richardson=True,
        )
        for quantity in ("price", "delta", "gamma"):
            with self.subTest(quantity=quantity):
                self.assertLess(
                    abs(extrapolated[quantity] - exact[quantity]),
                    abs(standard[quantity] - exact[quantity]),
                )

    def test_strike_clustered_mesh(self):
        inputs = {**self.inputs, "S": 108.0, "tau": 1.8, "sigma": 0.55, "option": "put"}
        grid = crank_nicolson_grid(
            inputs["K"], inputs["tau"], inputs["r"], inputs["sigma"],
            inputs["q"], inputs["option"], n_space=200,
            mesh="strike_clustered",
        )
        self.assertTrue(np.any(grid.spots == inputs["K"]))
        self.assertTrue(np.all(np.diff(grid.spots) > 0))
        exact = black_scholes_price(**inputs)
        estimate = crank_nicolson_price(
            **inputs, n_space=200, n_time=200,
            mesh="strike_clustered", richardson=True,
        )
        self.assertLess(abs(estimate - exact), 5e-6)

    def test_invalid_inputs(self):
        changes = (
            {"K": 0}, {"tau": -1}, {"sigma": 0}, {"option": "wrong"},
            {"n_space": 3}, {"n_time": 0}, {"rannacher_steps": 3, "n_time": 2},
            {"S": -1}, {"s_max": 50},
            {"richardson": "yes"},
            {"mesh": "wrong"}, {"mesh": "strike_clustered", "concentration": 0},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                crank_nicolson_price(**{**self.inputs, **change})


if __name__ == "__main__":
    unittest.main()
