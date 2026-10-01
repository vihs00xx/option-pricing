import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.basket import (BasketContract, arithmetic_greeks, geometric_greeks,
                        geometric_price, quadrature_price, randomized_price,
                        simulate_terminal, solve_adi, randomized_greeks)
from src.black_scholes import black_scholes_price
from src.basket.integration import common_factor


class BasketTests(unittest.TestCase):
    def setUp(self):
        self.c = BasketContract(dividends=(0.01, 0.03))

    def test_contract_validation_and_payoff(self):
        for kwargs in (dict(weights=(0.7, 0.7)), dict(correlation=1.1),
                       dict(tau=-1), dict(spots=(-1, 100)), dict(volatilities=(0.2, float("nan")))):
            with self.assertRaises(ValueError):
                BasketContract(**kwargs)
        assert_allclose(self.c.payoff([[150, 300], [50, 100]]), [125, 0])
        self.assertEqual(self.c.changed(option="put").payoff([50, 100]), 25)

    def test_simulated_drift_variance_and_correlation(self):
        terminal = simulate_terminal(self.c, 200000, seed=20260930, sampler="mc")
        logs = np.log(terminal/np.asarray(self.c.spots))
        expected_mean = (self.c.rate-np.asarray(self.c.dividends)-0.5*np.asarray(self.c.volatilities)**2)*self.c.tau
        assert_allclose(logs.mean(axis=0), expected_mean, atol=0.002)
        assert_allclose(logs.var(axis=0), np.asarray(self.c.volatilities)**2*self.c.tau, rtol=0.015)
        self.assertLess(abs(np.corrcoef(logs.T)[0, 1]-self.c.correlation), 0.01)

    def test_geometric_formula_reduces_to_vanilla(self):
        c = self.c.changed(volatilities=(0.2, 0.2), dividends=(0.01, 0.01), correlation=1)
        expected = black_scholes_price(100, 100, 1, c.rate, 0.2, 0.01)
        self.assertAlmostEqual(geometric_price(c), expected, places=11)
        self.assertAlmostEqual(quadrature_price(c).price, expected, places=9)

    def test_geometric_formula_against_conditional_quadrature(self):
        for correlation in (-1, -0.4, 0, 0.7, 1):
            for option in ("call", "put"):
                c = self.c.changed(correlation=correlation, option=option, spots=(80, 130), weights=(0.3, 0.7))
                estimate = quadrature_price(c, kind="geometric")
                self.assertLess(abs(estimate.price-geometric_price(c)), 2e-8)

    def test_geometric_greeks_against_finite_differences(self):
        for option in ("call", "put"):
            c = self.c.changed(option=option, spots=(90, 120), weights=(0.4, 0.6), correlation=-0.2)
            g = geometric_greeks(c)
            base = geometric_price(c)
            for i in range(2):
                h = c.spots[i]*1e-4
                upper, lower = [geometric_price(c.bumped("spots", shift, i)) for shift in (h, -h)]
                self.assertAlmostEqual(g[f"delta_{i+1}"], (upper-lower)/(2*h), delta=2e-7)
                self.assertAlmostEqual(g[f"gamma_{i+1}{i+1}"], (upper-2*base+lower)/h**2, delta=2e-7)
            h = 0.01
            cross = sum(a*b*geometric_price(c.changed(spots=(c.spots[0]+a*h, c.spots[1]+b*h)))
                        for a in (-1, 1) for b in (-1, 1))/(4*h*h)
            self.assertAlmostEqual(g["gamma_12"], cross, delta=2e-7)
            for name, field, index, sign in (("vega_1", "volatilities", 0, 1),
                                            ("vega_2", "volatilities", 1, 1), ("theta", "tau", None, -1),
                                            ("rho", "rate", None, 1), ("correlation", "correlation", None, 1)):
                h = 1e-5
                fd = sign*(geometric_price(c.bumped(field, h, index))-geometric_price(c.bumped(field, -h, index)))/(2*h)
                self.assertAlmostEqual(g[name], fd, delta=2e-6)

    def test_arithmetic_symmetry_parity_and_geometric_lower_bound(self):
        for correlation in (-0.7, 0, 0.95):
            c = self.c.changed(correlation=correlation, spots=(80, 130), weights=(0.3, 0.7))
            swapped = c.changed(**{name: getattr(c, name)[::-1] for name in ("spots", "volatilities", "dividends", "weights")})
            call, put = quadrature_price(c).price, quadrature_price(c.changed(option="put")).price
            self.assertAlmostEqual(call, quadrature_price(swapped).price, places=8)
            self.assertAlmostEqual(call-put, c.forward_value()-c.strike*np.exp(-c.rate*c.tau), places=9)
            self.assertGreaterEqual(call+1e-10, geometric_price(c))

    def test_common_factor_covariance_and_independent_conditioning(self):
        for correlation in (-0.95, 0, 0.95):
            c = self.c.changed(correlation=correlation)
            variance, loading = common_factor(c)
            expected = np.outer(c.volatilities, c.volatilities)*np.array([[1, correlation], [correlation, 1]])*c.tau
            assert_allclose(variance*np.ones((2, 2))+np.outer(loading, loading), expected, atol=1e-14)
            self.assertAlmostEqual(quadrature_price(c).price, quadrature_price(c, conditioning="asset").price, places=9)

    def test_price_truncation_bound(self):
        c = self.c.changed(volatilities=(0.5, 0.65), tau=2)
        full, truncated = quadrature_price(c), quadrature_price(c, limit=6)
        self.assertLess(abs(full.price-truncated.price), truncated.tail_bound+full.integration_error+truncated.integration_error)

    def test_degenerate_price_cases(self):
        for option in ("call", "put"):
            expiry = self.c.changed(tau=0, option=option)
            self.assertEqual(quadrature_price(expiry).price, expiry.payoff(expiry.spots))
            c = self.c.changed(option=option, volatilities=(0, 0))
            expected = max((1 if option == "call" else -1)*(c.forward_value()-c.strike*np.exp(-c.rate*c.tau)), 0)
            self.assertAlmostEqual(quadrature_price(c).price, expected, places=10)
            c = self.c.changed(option=option, spots=(0, 100))
            expected = black_scholes_price(50, 100, c.tau, c.rate, 0.3, 0.03, option)
            self.assertAlmostEqual(quadrature_price(c).price, expected, places=9)

    def test_arithmetic_spot_greeks_and_bump_stability(self):
        c = self.c.changed(spots=(90, 120))
        greeks = arithmetic_greeks(c)
        base = quadrature_price(c).price
        h = 0.02
        for i in range(2):
            upper, lower = [quadrature_price(c.bumped("spots", shift, i)).price for shift in (h, -h)]
            self.assertAlmostEqual(greeks[f"delta_{i+1}"], (upper-lower)/(2*h), delta=2e-7)
            self.assertAlmostEqual(greeks[f"gamma_{i+1}{i+1}"], (upper-2*base+lower)/h**2, delta=2e-7)
        cross = sum(a*b*quadrature_price(c.changed(spots=(c.spots[0]+a*h, c.spots[1]+b*h))).price
                    for a in (-1, 1) for b in (-1, 1))/(4*h*h)
        self.assertAlmostEqual(greeks["gamma_12"], cross, delta=2e-7)
        refined = arithmetic_greeks(c, bump_scale=0.5)
        for key in greeks:
            self.assertAlmostEqual(greeks[key], refined[key], delta=2e-5)
        gamma = np.array([[greeks["gamma_11"], greeks["gamma_12"]], [greeks["gamma_12"], greeks["gamma_22"]]])
        self.assertGreaterEqual(np.linalg.eigvalsh(gamma).min(), -1e-10)

    def test_rqmc_reference_control_and_budget_accounting(self):
        reference = quadrature_price(self.c).price
        for estimator in ("raw", "conditional"):
            result = randomized_price(self.c, n_points=4096, n_replicates=16, estimator=estimator, control=True)
            self.assertLess(abs(result.price-reference), max(4*result.standard_error, 0.004))
            self.assertEqual(result.production_points, 65536)
            self.assertEqual(result.pilot_points, 4096)
            self.assertAlmostEqual(result.standard_error, np.std(result.replicate_means, ddof=1)/4)
        geometric = randomized_price(self.c, n_points=4096, kind="geometric", estimator="raw")
        self.assertLess(abs(geometric.price-geometric_price(self.c)), max(4*geometric.standard_error, 0.004))
        with self.assertRaises(ValueError):
            randomized_price(self.c, n_points=1000)

    def test_adi_mesh_convergence_and_greeks(self):
        reference = quadrature_price(self.c).price
        coarse = solve_adi(self.c, n_space=40, n_steps=80)
        fine = solve_adi(self.c, n_space=80, n_steps=160)
        e0, e1 = abs(coarse.evaluate(self.c.spots)-reference), abs(fine.evaluate(self.c.spots)-reference)
        self.assertLess(e1, e0)
        self.assertLess(e1, 0.04)
        greeks = arithmetic_greeks(self.c)
        for name, value in fine.spot_greeks(self.c.spots).items():
            self.assertLess(abs(value-greeks[name]), 0.002)

    def test_conditional_rqmc_greeks(self):
        reference = arithmetic_greeks(self.c)
        result = randomized_greeks(self.c, n_points=4096, control=True)
        for name, quantity in result["quantities"].items():
            tolerance = 2e-5 if name.startswith(("delta", "gamma")) else 0.006
            self.assertLess(abs(quantity["value"]-reference[name]), max(4*quantity["standard_error"], tolerance))
        self.assertEqual(result["production_points"], 65536)
        self.assertEqual(result["pilot_points"], 4096)

    def test_arithmetic_put_greek_parity(self):
        call = arithmetic_greeks(self.c)
        put = arithmetic_greeks(self.c.changed(option="put"))
        for i in range(2):
            self.assertAlmostEqual(call[f"delta_{i+1}"]-put[f"delta_{i+1}"],
                                   self.c.weights[i]*np.exp(-self.c.dividends[i]*self.c.tau), places=10)
        for name in ("gamma_11", "gamma_22", "gamma_12", "vega_1", "vega_2", "correlation"):
            self.assertAlmostEqual(call[name], put[name], delta=2e-7)
        discounted_strike = self.c.strike*np.exp(-self.c.rate*self.c.tau)
        self.assertAlmostEqual(call["rho"]-put["rho"], self.c.tau*discounted_strike, delta=2e-7)
        expected_theta = sum(w*q*s*np.exp(-q*self.c.tau) for w, q, s in
                             zip(self.c.weights, self.c.dividends, self.c.spots))-self.c.rate*discounted_strike
        self.assertAlmostEqual(call["theta"]-put["theta"], expected_theta, delta=2e-7)

    def test_adi_zero_stock_boundary_and_put_parity(self):
        call = solve_adi(self.c, n_space=40, n_steps=80)
        put = solve_adi(self.c.changed(option="put"), n_space=40, n_steps=80)
        expected = black_scholes_price(50, 100, self.c.tau, self.c.rate, 0.3, 0.03)
        self.assertAlmostEqual(call.evaluate((0, 100)), expected, places=10)
        self.assertAlmostEqual(call.evaluate(self.c.spots)-put.evaluate(self.c.spots),
                               self.c.forward_value()-100*np.exp(-self.c.rate*self.c.tau), places=9)


if __name__ == "__main__":
    unittest.main()
