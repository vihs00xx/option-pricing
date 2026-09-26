import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.black_scholes import black_scholes_greeks, black_scholes_price


class BlackScholesTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(S=100.0, K=100.0, tau=1.0, r=0.05, sigma=0.2)

    def test_standard_reference_values(self):
        self.assertAlmostEqual(black_scholes_price(**self.inputs), 10.450583572185565)
        self.assertAlmostEqual(
            black_scholes_price(**self.inputs, option="put"), 5.573526022256971
        )
        expected = dict(
            delta=0.6368306511756191, gamma=0.018762017345846895,
            vega=37.52403469169379, theta=-6.414027546438197,
            rho=53.232481545376345,
        )
        actual = black_scholes_greeks(**self.inputs)
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=9)

    def test_dividend_parity_and_bounds(self):
        S = np.array([50.0, 80.0, 100.0, 120.0, 200.0])
        for r in (-0.02, 0.05):
            inputs = {**self.inputs, "S": S, "r": r, "q": 0.03}
            call = black_scholes_price(**inputs)
            put = black_scholes_price(**inputs, option="put")
            stock = S * np.exp(-0.03)
            strike = 100 * np.exp(-r)
            assert_allclose(call - put, stock - strike, atol=1e-12)
            self.assertTrue(np.all(call >= np.maximum(stock - strike, 0) - 1e-12))
            self.assertTrue(np.all(call <= stock + 1e-12))
            self.assertTrue(np.all(put >= np.maximum(strike - stock, 0) - 1e-12))
            self.assertTrue(np.all(put <= strike + 1e-12))

    def test_expiry_zero_spot_and_zero_volatility(self):
        for option, expected in (("call", [0, 0, 20]), ("put", [20, 0, 0])):
            inputs = {**self.inputs, "S": [80, 100, 120], "tau": 0}
            assert_allclose(black_scholes_price(**inputs, option=option), expected)
        inputs = {**self.inputs, "S": 0}
        self.assertEqual(black_scholes_price(**inputs), 0)
        self.assertAlmostEqual(black_scholes_price(**inputs, option="put"), 100 * np.exp(-0.05))
        inputs = {**self.inputs, "sigma": 0, "q": 0.02}
        self.assertAlmostEqual(
            black_scholes_price(**inputs), 100 * (np.exp(-0.02) - np.exp(-0.05))
        )
        self.assertEqual(black_scholes_price(**inputs, option="put"), 0)

    def test_mixed_boundary_array_and_broadcasting(self):
        inputs = {**self.inputs, "S": [0, 100, 100, 100],
                  "tau": [1, 0, 1, 1], "sigma": [0.2, 0.2, 0, 0.2]}
        assert_allclose(black_scholes_price(**inputs),
                        [0, 0, 100 * (1 - np.exp(-0.05)), 10.450583572185565])
        inputs = {**self.inputs, "S": np.array([[80], [120]]), "tau": [0.25, 0.5, 1]}
        prices = black_scholes_price(**inputs)
        self.assertEqual(prices.shape, (2, 3))
        for i, S in enumerate((80, 120)):
            for j, tau in enumerate((0.25, 0.5, 1)):
                expected = black_scholes_price(**{**self.inputs, "S": S, "tau": tau})
                self.assertAlmostEqual(prices[i, j], expected)
        self.assertIsInstance(black_scholes_price(**self.inputs), float)

    def test_greeks_against_price_derivatives(self):
        for option in ("call", "put"):
            for S, r, q, tau in ((80, 0.05, 0, 0.5), (100, -0.01, 0.03, 1),
                                  (120, 0.04, 0.02, 2)):
                inputs = {**self.inputs, "S": S, "r": r, "q": q, "tau": tau, "option": option}
                greeks = black_scholes_greeks(**inputs)
                for name, parameter, bump, sign in (
                    ("delta", "S", 0.001, 1), ("vega", "sigma", 1e-5, 1),
                    ("rho", "r", 1e-5, 1), ("theta", "tau", 1e-5, -1),
                ):
                    plus = black_scholes_price(**{**inputs, parameter: inputs[parameter] + bump})
                    minus = black_scholes_price(**{**inputs, parameter: inputs[parameter] - bump})
                    numerical = sign * (plus - minus) / (2 * bump)
                    assert_allclose(greeks[name], numerical, rtol=2e-6, atol=1e-7)
                bump = 0.01
                plus = black_scholes_price(**{**inputs, "S": S + bump})
                minus = black_scholes_price(**{**inputs, "S": S - bump})
                gamma = (plus - 2 * black_scholes_price(**inputs) + minus) / bump**2
                assert_allclose(greeks["gamma"], gamma, rtol=2e-5, atol=1e-8)

    def test_greek_parity_and_pricing_pde(self):
        inputs = {**self.inputs, "S": np.array([80, 100, 120]), "q": 0.03}
        call = black_scholes_greeks(**inputs)
        put = black_scholes_greeks(**inputs, option="put")
        assert_allclose(call["delta"] - put["delta"], np.exp(-0.03))
        assert_allclose(call["gamma"], put["gamma"])
        assert_allclose(call["vega"], put["vega"])
        for option, greeks in (("call", call), ("put", put)):
            S = inputs["S"]
            price = black_scholes_price(**inputs, option=option)
            residual = (greeks["theta"] + 0.5 * 0.2**2 * S**2 * greeks["gamma"]
                        + (0.05 - 0.03) * S * greeks["delta"] - 0.05 * price)
            assert_allclose(residual, 0, atol=1e-12)

    def test_invalid_inputs(self):
        for name, value in (("S", -1), ("K", 0), ("tau", -0.1),
                            ("sigma", -0.2), ("r", np.nan), ("q", np.inf)):
            with self.subTest(name=name):
                for function in (black_scholes_price, black_scholes_greeks):
                    with self.assertRaises(ValueError):
                        function(**{**self.inputs, name: value})
        for function in (black_scholes_price, black_scholes_greeks):
            with self.assertRaises(ValueError):
                function(**self.inputs, option="invalid")
        for name in ("S", "tau", "sigma"):
            with self.assertRaisesRegex(ValueError, "Greeks require positive"):
                black_scholes_greeks(**{**self.inputs, name: 0})


if __name__ == "__main__":
    unittest.main()
