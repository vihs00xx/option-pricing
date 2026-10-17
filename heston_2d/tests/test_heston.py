"""The Heston basket engine, its labels, the reduced coordinates, the PDE and the models."""

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.heston_basket.engine import heston_estimate
from src.heston_basket.model import HestonBasket, parameter_bumps, quantity_names
from src.neural.coordinates import (contract_of, features_from_log_forwards, levy_call, log_forwards, permute,
                                    reduced_coordinates, row_of)
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.pde import raw_residual, reduced_residual
from src.neural.reference import labels, permute_labels, price_and_greeks, quantities, reference_labels
from src.neural.training import Trainer, labelled_data, sample_reduced, reduced_box
from src.pricing import Basket, gauss_hermite_values
from src.pricing import quantity_names as gbm_names
from tests.helpers import (COMPUTATIONAL, D, DOMAIN, SHORT_RUN, STOCKS, STRIKE, basket, contracts,
                           finite_difference_check, small_model)

ONE_STOCK = Path(__file__).resolve().parents[2]/"heston"
LABELS = dict(points=1024, steps=16, seed=3)


def z_check(test, value, reference, se, limit=5.0, slack=0.0, name=""):
    test.assertLessEqual(abs(value-reference), limit*se+slack, f"{name}: {value} vs {reference} (se {se})")


class EngineTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            basket(leverage=(1.0,)*D)
        with self.assertRaises(ValueError):    # P not a correlation matrix
            basket(correlation=-0.6, vol_correlation=0.9, leverage=(-0.9,)*D)
        self.assertEqual(len(quantity_names(D)), 1+D+D*(D+1)//2+2+5*D+D*(D-1))
        self.assertEqual(quantities(STOCKS), tuple(quantity_names(D)))

    def test_bumps_keep_contracts_valid(self):
        c = basket()
        for name, field, index, step in parameter_bumps(c):
            self.assertGreater(step, 0, name)
            c.bumped(field, step, index), c.bumped(field, -step, index)

    def test_no_vol_of_vol_is_the_black_scholes_basket(self):
        # xi -> 0 with v0 = theta: constant variances, stock correlation R.
        v = tuple(np.linspace(0.04, 0.09, D))
        R = np.full((D, D), 0.4) + 0.6*np.eye(D)
        c = basket(v0=v, theta=v, xi=(1e-4,)*D, correlation=R)
        e = heston_estimate(c, 2048, 8, 1, dt=1/16)
        g = Basket(spots=c.spots, strike=100, tau=1.0, rate=0.03, volatilities=np.sqrt(v), dividends=c.dividends,
                   weights=c.weights, correlation=R)
        reference = dict(zip(gbm_names(g), gauss_hermite_values(g, 32 if D <= 3 else 16)))
        for q in ["price", "delta_1", f"gamma_1{D}", "theta", "rho", "correlation_12"]:
            z_check(self, e.values[q], reference[q], e.standard_errors[q], slack=1e-3*abs(reference[q]), name=q)

    def test_put_call_parity(self):
        call, put = basket(), basket(option="put")
        a = heston_estimate(call, 1024, 8, 5, dt=1/16, greeks=False)
        b = heston_estimate(put, 1024, 8, 5, dt=1/16, greeks=False)
        parity = call.forward_value()-call.discounted_strike()
        z_check(self, a.values["price"]-b.values["price"], parity,
                np.hypot(a.standard_errors["price"], b.standard_errors["price"]), name="parity")

    @unittest.skipUnless((ONE_STOCK/"src/heston/reference.py").exists(), "needs the one-stock study in ../heston")
    def test_one_weighted_stock_is_one_stock_heston(self):
        spec = importlib.util.spec_from_file_location("heston_src", ONE_STOCK/"src/__init__.py",
                                                      submodule_search_locations=[str(ONE_STOCK/"src")])
        if "heston_src" not in sys.modules:
            package = importlib.util.module_from_spec(spec)
            sys.modules["heston_src"] = package
            spec.loader.exec_module(package)
        model = importlib.import_module("heston_src.heston.model")
        lewis = importlib.import_module("heston_src.heston.reference")
        weights = (1.0,)+(0.0,)*(D-1)
        c = basket(weights=weights)
        one = model.Contracts(S=c.spots[0]/100, tau=1.0, r=0.03, q=c.dividends[0], v0=c.v0[0], theta=c.theta[0],
                              kappa=c.kappa[0], xi=c.xi[0], rho=c.leverage[0], call=True)
        exact, _ = lewis.reference(one)
        e = heston_estimate(c, 2048, 8, 2, dt=1/32)
        for mine, theirs, scale in (("price", "price", 100), ("delta_1", "delta", 1), ("gamma_11", "gamma", 0.01),
                                    ("vega_1", "vega", 100), ("d_kappa_1", "d_kappa", 100),
                                    ("d_leverage_1", "d_rho", 100), ("theta", "theta_cal", 100)):
            reference = float(exact[theirs][0])*scale
            z_check(self, e.values[mine], reference, e.standard_errors[mine], limit=6,
                    slack=3e-3*abs(reference)+1e-4, name=mine)

    def test_labels_match_the_engine(self):
        c = basket()
        x = torch.as_tensor(row_of(c)[None])
        features, scale = reduced_coordinates(STOCKS, x)
        values = reference_labels(STOCKS, features, dict(points=4096, steps=32, seed=7))
        e = heston_estimate(c, 4096, 8, 9, dt=1/32)
        price = float(scale[0]*values[0, 0])
        z_check(self, price, e.values["price"], e.standard_errors["price"], limit=6, slack=2e-3*e.values["price"],
                name="price")
        # Delta_i = K e^{-r tau} c_{y_i} / S_i.
        for i in range(D):
            delta = float(scale[0]*values[0, 1+i])/c.spots[i]
            z_check(self, delta, e.values[f"delta_{i+1}"], e.standard_errors[f"delta_{i+1}"], limit=6,
                    slack=2e-3*abs(e.values[f"delta_{i+1}"]), name=f"delta_{i+1}")


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.x, self.put = contracts(32, seed=3)

    def test_domain_contracts_are_valid(self):
        for k in range(len(self.x)):
            c = contract_of(STOCKS, self.x[k], self.put[k])
            for name in ("tau", "rate"):
                low, high = DOMAIN[name]
                self.assertTrue(low <= getattr(c, name) <= high)
            self.assertGreaterEqual(min(c.weights), DOMAIN["weight_floor"]-1e-12)
            assert_allclose(row_of(c), self.x[k], rtol=0, atol=1e-12)

    def test_round_trip_and_permutation(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        torch.testing.assert_close(features_from_log_forwards(STOCKS, log_forwards(STOCKS, features),
                                                              features[:, D:]), features)
        order = torch.argsort(torch.rand(len(features), D, generator=torch.Generator().manual_seed(1)), dim=1)
        back = torch.argsort(order, dim=1)
        torch.testing.assert_close(permute(STOCKS, permute(STOCKS, features, order), back), features,
                                   rtol=0, atol=1e-15)

    def test_relabelled_labels(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x[:4]))
        order = torch.argsort(torch.rand(4, D, generator=torch.Generator().manual_seed(2)), dim=1)
        values = reference_labels(STOCKS, features, LABELS)
        moved = permute_labels(STOCKS, values, order)
        back = permute_labels(STOCKS, moved, torch.argsort(order, dim=1))
        torch.testing.assert_close(back, values, rtol=0, atol=1e-15)
        # The relabelled contract's labels agree up to the simulation error.
        relabelled = reference_labels(STOCKS, permute(STOCKS, features, order), dict(LABELS, points=4096))
        self.assertLess(((relabelled[:, 0]-moved[:, 0])/moved[:, 0]).abs().max().item(), 0.03)

    def test_levy_is_exact_at_expiry(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        expiring = features.clone()
        expiring[:, STOCKS.tau] = 1e-14
        torch.testing.assert_close(levy_call(STOCKS, expiring), torch.relu(torch.expm1(features[:, 0])),
                                   atol=1e-12, rtol=0)


class PDETests(unittest.TestCase):
    def setUp(self):
        box = reduced_box(STOCKS, DOMAIN, COMPUTATIONAL)
        self.features = sample_reduced(STOCKS, torch.Generator().manual_seed(4), 32, box, {"v_power": 1.0})
        self.features[:, STOCKS.v] += 0.01

    def test_exact_solutions(self):
        # The forward e^z - 1 and the expected variance theta_i + (v_i - theta_i) e^{-kappa_i tau} solve the PDE.
        residual, _ = reduced_residual(STOCKS, lambda f: torch.expm1(f[:, 0]), self.features)
        self.assertLess(residual.abs().max().item(), 1e-12)
        for i in range(D):
            def expected_variance(f, i=i):
                p = STOCKS.split(f)
                return p["theta"][:, i]+(p["v"][:, i]-p["theta"][:, i])*torch.exp(-p["kappa"][:, i]*p["tau"])
            residual, _ = reduced_residual(STOCKS, expected_variance, self.features)
            self.assertLess(residual.abs().max().item(), 1e-12)

    def test_reduced_and_raw_forms_agree(self):
        # For any smooth call function, the raw residual of V / K = e^{-r tau} c equals e^{-r tau} times the reduced one.
        model = small_model("pinn", output="plain")
        x, put = contracts(16, seed=6)
        x = torch.as_tensor(x)
        features, scale = reduced_coordinates(STOCKS, x)
        reduced, _ = reduced_residual(STOCKS, model.call, features)
        calls = torch.zeros(len(x), dtype=torch.bool)
        raw = raw_residual(STOCKS, lambda xs, p: model.price(xs, p)/xs[:, STOCKS.column["strike"]], x, calls)
        torch.testing.assert_close(raw, scale/x[:, STOCKS.column["strike"]]*reduced, rtol=1e-8, atol=1e-10)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.x, self.put = contracts(8, seed=9)

    def test_parity_and_homogeneity(self):
        col = STOCKS.column
        for family in ("pinn", "deeponet"):
            model = small_model(family, v_input="sqrt")
            x = torch.as_tensor(self.x)
            n = len(x)
            call = model.price(x, torch.zeros(n, dtype=torch.bool))
            put = model.price(x, torch.ones(n, dtype=torch.bool))
            tau = x[:, col["tau"]]
            forward = (STOCKS.weights(x)*STOCKS.take(x, "spot_")*torch.exp(-STOCKS.take(x, "dividend_")*tau[:, None])).sum(1)
            torch.testing.assert_close(call-put, forward-x[:, col["strike"]]*torch.exp(-x[:, col["rate"]]*tau))
            scaled = x.clone()
            for column in [f"spot_{i}" for i in range(1, D+1)]+["strike"]:
                scaled[:, col[column]] *= 1.7
            torch.testing.assert_close(model.price(scaled, torch.zeros(n, dtype=torch.bool)), 1.7*call)

    def test_levy_output_with_zero_network_is_levy(self):
        for family, output in (("pinn", "levy"), ("deeponet", "levy"), ("pinn", "levy_envelope")):
            model = small_model(family, output=output)
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
            torch.testing.assert_close(model.call(features), levy_call(STOCKS, features))

    def test_autograd_greeks_match_finite_differences(self):
        for family, layout, output, v_input in (("pinn", "raw", "plain", "plain"),
                                                ("pinn", "reduced", "levy_envelope", "sqrt"),
                                                ("deeponet", "raw", "plain", "plain"),
                                                ("deeponet", "reduced", "levy", "sqrt")):
            model = small_model(family, layout=layout, output=output, v_input=v_input)
            finite_difference_check(model, self.x[:4], self.put[:4])

    def test_save_load_and_ensemble(self):
        members = [small_model("deeponet", seed=1, similarity=0.3), small_model("deeponet", seed=2, similarity=0.3)]
        for model in (small_model("pinn", v_input="sqrt"), small_model("pinn", layout="raw", output="plain"),
                      Ensemble(members)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"model.pt"
                save_model(model, path)
                loaded = load_model(path)
            x, put = torch.as_tensor(self.x), torch.as_tensor(self.put)
            torch.testing.assert_close(loaded.price(x, put), model.price(x, put), rtol=0, atol=0)
        ensemble = price_and_greeks(STOCKS, Ensemble(members).price, self.x, self.put)
        separate = [price_and_greeks(STOCKS, m.price, self.x, self.put) for m in members]
        for name in quantities(STOCKS):
            assert_allclose(ensemble[name], (separate[0][name]+separate[1][name])/2, rtol=1e-12, atol=1e-14)

    def test_price_at_expiry_is_the_payoff(self):
        x = self.x.copy()
        x[:, STOCKS.column["tau"]] = 0.0
        xt = torch.as_tensor(x)
        basket_value = (STOCKS.weights(xt)*STOCKS.take(xt, "spot_")).sum(1).numpy()
        strike = x[:, STOCKS.column["strike"]]
        payoff = np.where(self.put, np.maximum(strike-basket_value, 0), np.maximum(basket_value-strike, 0))
        for settings in (dict(output="levy"), dict(output="levy_envelope", similarity="levy", v_input="sqrt")):
            with torch.no_grad():
                price = small_model(**settings).price(xt, torch.as_tensor(self.put)).numpy()
            assert_allclose(price, payoff, atol=1e-12)


class TrainerTests(unittest.TestCase):
    """Short runs of every kind of ladder level: losses stay finite."""

    @classmethod
    def setUpClass(cls):
        x, put = contracts(8, seed=21)
        values = price_and_greeks(STOCKS, small_model("pinn").price, x, put)   # any finite monitor will do
        cls.monitor = dict(inputs=(x, put), reference={k: v+1.0 for k, v in values.items()})
        cls.data = labelled_data(STOCKS, DOMAIN, STRIKE, 32, seed=23, settings=dict(LABELS, points=256))

    def run_level(self, family, model_settings, loss, augment=None, data=True):
        model = small_model(family, **model_settings)
        settings = dict(SHORT_RUN, loss=loss, augment=augment)
        trainer = Trainer(model, settings, DOMAIN, STRIKE, self.monitor, self.data if data else None,
                          log=lambda message: None)
        record = trainer.run(seed=0)
        self.assertTrue(np.isfinite(record["best_monitor"]))
        for entry in record["history"]:
            self.assertTrue(all(np.isfinite(v) for v in entry.values() if isinstance(v, float)))

    def test_raw_pinn(self):
        self.run_level("pinn", dict(layout="raw", output="plain"), dict(pde=1.0, terminal=1.0), data=False)

    def test_reduced_pinn_with_variance_boundary_and_anchors(self):
        self.run_level("pinn", dict(output="levy_envelope", similarity="levy", v_input="sqrt"),
                       dict(pde=1.0, boundary=1.0, zero_variance=1.0, variance_cap=1.0, price=1.0),
                       augment="permutation")

    def test_raw_deeponet(self):
        self.run_level("deeponet", dict(layout="raw", output="plain"), dict(price=1.0))

    def test_physics_informed_differential_deeponet(self):
        self.run_level("deeponet", dict(output="levy_envelope", similarity=0.3, v_input="sqrt"),
                       dict(price=1.0, derivatives=1.0, pde=1.0, boundary=1.0, zero_variance=1.0, variance_cap=1.0),
                       augment="permutation")


if __name__ == "__main__":
    unittest.main()
