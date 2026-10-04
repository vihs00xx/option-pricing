import unittest
from dataclasses import asdict

import numpy as np
from numpy.testing import assert_allclose

from src.basket import BasketContract, arithmetic_greeks, quadrature_price
from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.pricing import (Basket, adi_values, factorization, gauss_hermite_values, geometric_greeks,
                         geometric_known_values, geometric_price, quantity_names, rqmc_estimate,
                         solve_adi)

CORRELATION_3 = np.array([[1, 0.5, -0.3], [0.5, 1, 0.2], [-0.3, 0.2, 1]])


def three_stocks(**kwargs):
    values = dict(spots=(90, 100, 115), strike=100, tau=0.8, rate=0.04,
                  volatilities=(0.2, 0.35, 0.25), dividends=(0.0, 0.01, 0.03),
                  weights=(0.2, 0.3, 0.5), correlation=CORRELATION_3)
    values.update(kwargs)
    return Basket(**values)


def two_asset_reference(v1):
    reference = {"price": quadrature_price(v1).price, **arithmetic_greeks(v1)}
    reference["correlation_12"] = reference.pop("correlation")
    return reference


class ContractTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in (dict(weights=(0.2, 0.2, 0.2)), dict(correlation=np.eye(2)),
                       dict(correlation=1.1), dict(spots=(0, 100, 115)), dict(volatilities=(0.2, 0, 0.3)),
                       dict(correlation=[[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])):
            with self.assertRaises(ValueError):
                three_stocks(**kwargs)

    def test_scalar_correlation_and_symmetric_bump(self):
        c = three_stocks(correlation=0.4)
        expected = np.full((3, 3), 0.4)
        np.fill_diagonal(expected, 1)
        assert_allclose(c.correlation_matrix(), expected)
        bumped = c.bumped("correlation", 0.01, (0, 2)).correlation_matrix()
        self.assertEqual(bumped[0, 2], bumped[2, 0])
        self.assertAlmostEqual(bumped[0, 2], 0.41)


class FactorizationTests(unittest.TestCase):
    def test_reconstructs_covariance_with_positive_loadings(self):
        for c in (Basket(**asdict(BasketContract(correlation=-0.8))), three_stocks()):
            for direction in ("equal", "volatility"):
                b, A = factorization(c, direction)
                self.assertGreater(b.min(), 0)
                assert_allclose(np.outer(b, b) + A @ A.T, c.covariance(), atol=1e-14)

    def test_residual_coordinates_change_smoothly(self):
        c = three_stocks()
        _, A = factorization(c)
        _, moved = factorization(c.bumped("correlation", 1e-7, (0, 1)))
        self.assertLess(np.abs(moved - A).max(), 1e-6)

    def test_equal_loadings_match_two_asset_shared_factor(self):
        from src.basket.integration import common_factor
        v1 = BasketContract(correlation=0.3)
        b, _ = factorization(Basket(**asdict(v1)), "equal")
        assert_allclose(b**2, common_factor(v1)[0], rtol=1e-12)

    def test_gauss_hermite_node_limit(self):
        with self.assertRaises(ValueError):
            gauss_hermite_values(three_stocks(), 400)


class ConditionalTests(unittest.TestCase):
    def test_one_stock_reduces_to_black_scholes(self):
        for option in ("call", "put"):
            c = Basket(spots=(105,), strike=100, tau=0.7, rate=0.03, volatilities=(0.25,),
                       dividends=(0.01,), weights=(1,), option=option)
            values = dict(zip(quantity_names(c), gauss_hermite_values(c, 4)))
            args = (105, 100, 0.7, 0.03, 0.25, 0.01, option)
            greeks = black_scholes_greeks(*args)
            self.assertAlmostEqual(values["price"], black_scholes_price(*args), places=12)
            for ours, theirs in (("delta_1", "delta"), ("gamma_11", "gamma")):
                self.assertAlmostEqual(values[ours], greeks[theirs], places=12)
            for ours, theirs in (("vega_1", "vega"), ("theta", "theta"), ("rho", "rho")):
                self.assertAlmostEqual(values[ours], greeks[theirs], delta=1e-6)

    def test_two_stocks_match_independent_quadrature(self):
        for kwargs in (dict(), dict(option="put", spots=(80, 130), weights=(0.3, 0.7)),
                       dict(correlation=-0.7, dividends=(0.01, 0.03)), dict(tau=0.02),
                       dict(tau=2, volatilities=(0.5, 0.65))):
            v1 = BasketContract(**kwargs)
            reference = two_asset_reference(v1)
            c = Basket(**asdict(v1))
            for direction in ("equal", "volatility"):
                values = dict(zip(quantity_names(c), gauss_hermite_values(c, 128, direction=direction)))
                for name, expected in reference.items():
                    self.assertAlmostEqual(values[name], expected, delta=1e-8 * max(1, abs(expected)),
                                           msg=f"{kwargs} {direction} {name}")

    def test_geometric_conditional_integrates_to_closed_form(self):
        c = three_stocks()
        known = geometric_known_values(c)
        for direction in ("equal", "volatility"):
            values = gauss_hermite_values(c, 40, kind="geometric", direction=direction)
            # Bumped entries divide quadrature roundoff by steps as small as 1e-5.
            assert_allclose(values, known, rtol=1e-8, atol=1e-11)

    def test_geometric_greeks_against_finite_differences(self):
        c = three_stocks(option="put")
        greeks = geometric_greeks(c)
        base = geometric_price(c)
        for i in range(3):
            h = c.spots[i] * 1e-4
            upper, lower = (geometric_price(c.bumped("spots", s, i)) for s in (h, -h))
            self.assertAlmostEqual(greeks[f"delta_{i + 1}"], (upper - lower) / (2 * h), delta=1e-8)
            self.assertAlmostEqual(greeks[f"gamma_{i + 1}{i + 1}"], (upper - 2 * base + lower) / h**2, delta=1e-6)
        h = 0.01
        cross = sum(a * b * geometric_price(c.bumped("spots", a * h, 0).bumped("spots", b * h, 2))
                    for a in (-1, 1) for b in (-1, 1)) / (4 * h * h)
        self.assertAlmostEqual(greeks["gamma_13"], cross, delta=1e-7)
        for name, field, index, sign in (("vega_2", "volatilities", 1, 1), ("theta", "tau", None, -1),
                                         ("rho", "rate", None, 1), ("correlation_23", "correlation", (1, 2), 1)):
            h = 1e-5
            fd = sign * (geometric_price(c.bumped(field, h, index))
                         - geometric_price(c.bumped(field, -h, index))) / (2 * h)
            self.assertAlmostEqual(greeks[name], fd, delta=1e-5)

    def test_parity_symmetry_convexity_and_geometric_bound(self):
        c = three_stocks()
        names = quantity_names(c)
        call = dict(zip(names, gauss_hermite_values(c, 48)))
        put = dict(zip(names, gauss_hermite_values(c.changed(option="put"), 48)))
        discounted_strike = c.strike * np.exp(-c.rate * c.tau)
        self.assertAlmostEqual(call["price"] - put["price"], c.forward_value() - discounted_strike, places=9)
        for i in range(3):
            self.assertAlmostEqual(call[f"delta_{i + 1}"] - put[f"delta_{i + 1}"],
                                   c.weights[i] * np.exp(-c.dividends[i] * c.tau), places=10)
        for name in names:
            if name.startswith(("gamma", "vega", "correlation")):
                self.assertAlmostEqual(call[name], put[name], delta=1e-7, msg=name)
        self.assertAlmostEqual(call["rho"] - put["rho"], c.tau * discounted_strike, delta=1e-6)
        gamma = np.zeros((3, 3))
        for i in range(3):
            for j in range(i, 3):
                gamma[i, j] = gamma[j, i] = call[f"gamma_{i + 1}{j + 1}"]
        self.assertGreater(np.linalg.eigvalsh(gamma).min(), -1e-12)
        self.assertGreater(call["price"], geometric_price(c))
        order = [2, 0, 1]
        permuted = c.changed(spots=np.asarray(c.spots)[order], volatilities=np.asarray(c.volatilities)[order],
                             dividends=np.asarray(c.dividends)[order], weights=np.asarray(c.weights)[order],
                             correlation=CORRELATION_3[np.ix_(order, order)])
        self.assertAlmostEqual(gauss_hermite_values(permuted, 48, greeks=False)[0], call["price"], places=9)


class RQMCTests(unittest.TestCase):
    def test_estimates_cover_deterministic_reference(self):
        c = three_stocks()
        names = quantity_names(c)
        reference = gauss_hermite_values(c, 48)
        for direction, control in (("equal", False), ("equal", True), ("volatility", True)):
            estimate = rqmc_estimate(c, 2048, 16, 11, direction=direction, control=control)
            self.assertEqual(estimate.names, tuple(names))
            for name, expected in zip(names, reference):
                self.assertLess(abs(estimate.values[name] - expected),
                                5 * estimate.standard_errors[name] + 1e-12, msg=f"{direction} {name}")
            self.assertEqual(estimate.production_points, 2048 * 16)
            self.assertEqual(estimate.pilot_points, 256 * 16 if control else 0)
            assert_allclose(estimate.standard_errors["price"],
                            estimate.replicate_means[:, 0].std(ddof=1) / 4, rtol=1e-12)

    def test_uninformative_control_is_not_fitted(self):
        # With equal loadings, a fully symmetric basket has a constant
        # conditional geometric value, so the control must be switched off.
        c = Basket(spots=(100,) * 4, strike=100, tau=0.5, rate=0.03, volatilities=(0.3,) * 4,
                   dividends=(0,) * 4, weights=(0.25,) * 4, correlation=0.6, option="put")
        estimate = rqmc_estimate(c, 256, 8, 5, greeks=False, control=True)
        self.assertEqual(estimate.beta[0], 0.0)
        reference = gauss_hermite_values(c, 24, greeks=False)[0]
        self.assertLess(abs(estimate.values["price"] - reference), 5 * estimate.standard_errors["price"])

    def test_control_reduces_price_error_and_one_stock_is_exact(self):
        c = three_stocks()
        plain = rqmc_estimate(c, 1024, 16, 3, greeks=False)
        controlled = rqmc_estimate(c, 1024, 16, 3, greeks=False, control=True)
        self.assertLess(controlled.standard_errors["price"], plain.standard_errors["price"])
        one = Basket(spots=(100,), strike=100, tau=1, rate=0.05, volatilities=(0.2,), dividends=(0,), weights=(1,))
        estimate = rqmc_estimate(one, 16, 4, greeks=False)
        self.assertEqual(estimate.standard_errors["price"], 0.0)
        self.assertAlmostEqual(estimate.values["price"], black_scholes_price(100, 100, 1, 0.05, 0.2), places=12)
        with self.assertRaises(ValueError):
            rqmc_estimate(c, 1000)


class ADITests(unittest.TestCase):
    def test_convergence_and_richardson(self):
        v1 = BasketContract(dividends=(0.01, 0.03))
        reference = two_asset_reference(v1)
        c = Basket(**asdict(v1))
        errors = [abs(adi_values(c, n)["price"] - reference["price"]) for n in (40, 80)]
        self.assertLess(errors[1], errors[0] / 3)
        extrapolated = adi_values(c, 40, richardson=True)
        for name in ("price", "delta_1", "delta_2", "gamma_11", "gamma_12", "gamma_22", "theta"):
            self.assertLess(abs(extrapolated[name] - reference[name]), 5e-4 * max(1, abs(reference[name])), msg=name)

    def test_near_expiry_and_high_volatility(self):
        for kwargs in (dict(tau=0.02), dict(tau=2, volatilities=(0.5, 0.65))):
            v1 = BasketContract(**kwargs)
            reference = quadrature_price(v1).price
            self.assertLess(abs(adi_values(Basket(**asdict(v1)), 80)["price"] - reference), 1e-2, msg=str(kwargs))

    def test_parameter_greeks(self):
        v1 = BasketContract(spots=(90, 110), weights=(0.4, 0.6), correlation=0.2)
        reference = two_asset_reference(v1)
        values = adi_values(Basket(**asdict(v1)), 40, richardson=True, parameter_greeks=True)
        for name in ("vega_1", "vega_2", "rho", "correlation_12"):
            self.assertLess(abs(values[name] - reference[name]), 2e-3 * max(1, abs(reference[name])), msg=name)

    def test_put_parity_and_zero_stock_face(self):
        c = Basket(**asdict(BasketContract(dividends=(0.01, 0.03))))
        call, put = solve_adi(c), solve_adi(c.changed(option="put"))
        i, j = call.spot_index
        self.assertEqual((call.stock_1[i], call.stock_2[j]), c.spots)
        parity = call.values[i, j] - put.values[i, j]
        self.assertAlmostEqual(parity, c.forward_value() - c.strike * np.exp(-c.rate * c.tau), places=9)
        expected = black_scholes_price(0.5 * call.stock_2, 100, c.tau, c.rate, 0.3, 0.03)
        assert_allclose(call.values[0], expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
