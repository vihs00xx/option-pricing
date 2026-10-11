"""Neural-pricer machinery for four stocks: torch reference, coordinates, PDE, Monte Carlo, models."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.neural.coordinates import (features_from_log_forwards, levy_call, log_forwards, permute, protocol_domain,
                                    reduced_coordinates)
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.pde import raw_residual, reduced_residual
from src.neural.reference import (call_function, call_pricer, labels, permute_labels, price_and_greeks,
                                  quantities, reference_labels, reference_values)
from src.neural.training import Trainer, labelled_data, score
from src.pricing import gauss_hermite_values
from src.pricing.contract import quantity_names
from tests.neural_helpers import (DOMAIN, STOCKS, STRIKE, SHORT_RUN, c_star, contract, contracts,
                                  finite_difference_check, row, small_model)

SPOT_GREEKS = 15            # price, 4 Deltas and 10 Gammas


class ReferenceTests(unittest.TestCase):
    """The torch twin against the numpy engine (same rule, autograd instead of bumps)."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.put = contracts(6, seed=11)
        cls.reference = reference_values(STOCKS, cls.x, cls.put, nodes=32)

    def test_quantities_follow_the_engine_order(self):
        self.assertEqual(quantities(STOCKS), tuple(quantity_names(c_star())))
        self.assertEqual(len(labels(STOCKS)), 26)

    def test_matches_engine_at_32_nodes(self):
        cases = [(row(c), c.option == "put") for c in (c_star(), c_star(option="put"), c_star(tau=0.01))]
        cases += [(self.x[i], self.put[i]) for i in range(3)]
        for r, put in cases:
            engine = gauss_hermite_values(contract(r, put), 32)
            twin = reference_values(STOCKS, r[None], np.array([put]), nodes=32)
            twin = np.array([twin[q][0] for q in quantities(STOCKS)])
            # Same rule: price and spot Greeks to rounding (the twin drops corner nodes of
            # mass below 1e-24).
            assert_allclose(twin[:SPOT_GREEKS], engine[:SPOT_GREEKS], rtol=1e-12, atol=1e-12)
            # Parameter Greeks: exact derivatives against the engine's central bumps.
            assert_allclose(twin[SPOT_GREEKS:], engine[SPOT_GREEKS:], rtol=1e-6, atol=5e-6)

    def test_reduced_pde_holds_for_reference(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        residual, _ = reduced_residual(STOCKS, call_function(STOCKS, 40), features)
        self.assertLess(residual.abs().max().item(), 1e-8)

    def test_raw_pde_holds_for_reference(self):
        price = call_pricer(STOCKS, call_function(STOCKS, 40))
        residual = raw_residual(STOCKS, lambda x, put: price(x, put)/x[:, STOCKS.column["strike"]],
                                torch.as_tensor(self.x[:3]), torch.as_tensor(self.put[:3]))
        self.assertLess(residual.abs().max().item(), 1e-8)

    def test_labels_satisfy_reduced_pde(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        values = reference_labels(STOCKS, features, nodes=40)
        named = dict(zip(labels(STOCKS), values.T))
        _, _, _, sigma, rho = STOCKS.split(features)
        generator = sum(0.5*sigma[:, i]**2*(named[f"c_{i+1}{i+1}"]-named[f"c_{i+1}"]) for i in range(4))
        generator = generator+sum(rho[:, k]*sigma[:, i]*sigma[:, j]*named[f"c_{i+1}{j+1}"]
                                  for k, (i, j) in enumerate(STOCKS.pairs))
        assert_allclose(named["c_tau"], generator, atol=1e-8)

    def test_relabelling_the_stocks_leaves_the_call_unchanged(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        order = torch.argsort(torch.rand(len(features), 4, generator=torch.Generator().manual_seed(3)), dim=1)
        moved = permute(STOCKS, features, order)
        original, relabelled = reference_labels(STOCKS, features, 40), reference_labels(STOCKS, moved, 40)
        torch.testing.assert_close(relabelled, permute_labels(STOCKS, original, order), rtol=0, atol=1e-8)
        back = torch.argsort(order, dim=1)
        torch.testing.assert_close(permute(STOCKS, moved, back), features, rtol=0, atol=1e-15)

    def test_coordinate_round_trip(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        y = log_forwards(STOCKS, features)
        torch.testing.assert_close(features_from_log_forwards(STOCKS, y, features[:, 4:]), features)

    def test_levy_is_exact_at_expiry_and_close_before(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        expiring = features.clone()
        expiring[:, STOCKS.tau] = 1e-14
        torch.testing.assert_close(levy_call(STOCKS, expiring), torch.relu(torch.expm1(features[:, 0])),
                                   atol=1e-12, rtol=0)
        call = call_function(STOCKS, 32)(features)
        self.assertLess((levy_call(STOCKS, features)-call).abs().max().item(), 0.02)

    def test_monte_carlo_is_unbiased(self):
        for i in range(2):
            estimates = monte_carlo_estimates(STOCKS, self.x[i], self.put[i], 400_000, seed=7+i)
            z = [(estimates[q][0]-self.reference[q][i])/estimates[q][1] for q in quantities(STOCKS)]
            self.assertLess(max(abs(v) for v in z), 4.5)

    def test_labelled_data_prices_match_reference(self):
        data = labelled_data(STOCKS, DOMAIN, STRIKE, 4, seed=5)
        values = reference_values(STOCKS, data["x"].numpy(), data["put"].numpy(), greeks=False)
        assert_allclose(data["normalized_price"].numpy()*STRIKE, values["price"], rtol=1e-12, atol=1e-12)

    def test_score_is_zero_for_reference(self):
        self.assertEqual(score(self.reference, self.reference)[0], 0.0)


class DomainTests(unittest.TestCase):
    def test_protocol_domain_is_the_tests_domain(self):
        domain, strike = protocol_domain()
        self.assertEqual(strike, STRIKE)
        self.assertEqual({k: list(v) if isinstance(v, (list, tuple)) else v for k, v in domain.items()}, DOMAIN)

    def test_contracts_lie_in_the_training_domain(self):
        x, put = contracts(256, seed=3)
        p = {name: x[:, k] for k, name in enumerate(STOCKS.fields)}
        weights = np.column_stack([p[f"weight_{i}"] for i in range(1, 4)])
        weights = np.column_stack((weights, 1-weights.sum(1)))
        self.assertGreaterEqual(weights.min(), 0.1-1e-12)
        correlations = np.tile(np.eye(4), (len(x), 1, 1))
        for i, j in STOCKS.pairs:
            correlations[:, i, j] = correlations[:, j, i] = p[f"correlation_{i+1}{j+1}"]
        self.assertGreaterEqual(np.linalg.eigvalsh(correlations)[:, 0].min(), 0.05)
        self.assertTrue(np.all((p["tau"] >= 0.05) & (p["tau"] <= 2.0)))
        self.assertEqual(int(put.sum()), 128)


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        self.x, self.put = contracts(16, seed=9)

    def test_reduced_models_keep_parity_and_homogeneity(self):
        col = STOCKS.column
        for family in ("pinn", "deeponet"):
            model = small_model(family)
            x = torch.as_tensor(self.x)
            n = len(x)
            call = model.price(x, torch.zeros(n, dtype=torch.bool))
            put = model.price(x, torch.ones(n, dtype=torch.bool))
            tau = x[:, col["tau"]]
            dividends = STOCKS.take(x, "dividend_")
            forward = (STOCKS.weights(x)*STOCKS.take(x, "spot_")*torch.exp(-dividends*tau[:, None])).sum(1)
            torch.testing.assert_close(call-put, forward-x[:, col["strike"]]*torch.exp(-x[:, col["rate"]]*tau))
            scaled = x.clone()
            for column in ("spot_1", "spot_2", "spot_3", "spot_4", "strike"):
                scaled[:, col[column]] *= 1.7
            torch.testing.assert_close(model.price(scaled, torch.zeros(n, dtype=torch.bool)), 1.7*call)

    def test_reduced_models_with_permutation_symmetric_data_are_not_forced_symmetric(self):
        # Relabelling is learned from augmented data, not built in: a random network is
        # generally not symmetric, which keeps the augmentation level meaningful.
        model = small_model("deeponet")
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        order = torch.tensor([[1, 0, 3, 2]]).expand(len(features), -1)
        self.assertGreater((model.call(permute(STOCKS, features, order))-model.call(features)).abs().max().item(),
                           1e-6)

    def test_levy_output_with_zero_network_is_levy(self):
        for family, output in (("pinn", "levy"), ("deeponet", "levy"), ("pinn", "levy_envelope")):
            model = small_model(family, output=output)
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
            torch.testing.assert_close(model.call(features), levy_call(STOCKS, features))

    def test_autograd_greeks_match_finite_differences(self):
        for family, layout, output in (("pinn", "raw", "plain"), ("pinn", "reduced", "levy_envelope"),
                                       ("deeponet", "raw", "plain"), ("deeponet", "reduced", "levy")):
            model = small_model(family, layout=layout, output=output,
                                similarity=(0.35 if family == "deeponet" and layout == "reduced" else None))
            finite_difference_check(self, model, self.x[:4], self.put[:4])

    def test_save_and_load_round_trip(self):
        for model in (small_model("pinn"), small_model("deeponet", similarity=0.35),
                      small_model("pinn", layout="raw", output="plain"),
                      Ensemble([small_model("deeponet", seed=1), small_model("deeponet", seed=2)])):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"model.pt"
                save_model(model, path)
                loaded = load_model(path)
            x, put = torch.as_tensor(self.x), torch.as_tensor(self.put)
            torch.testing.assert_close(loaded.price(x, put), model.price(x, put), rtol=0, atol=0)

    def test_ensemble_greeks_are_member_averages(self):
        members = [small_model(seed=s) for s in (1, 2)]
        ensemble = price_and_greeks(STOCKS, Ensemble(members).price, self.x, self.put)
        separate = [price_and_greeks(STOCKS, m.price, self.x, self.put) for m in members]
        for name in quantities(STOCKS):
            assert_allclose(ensemble[name], (separate[0][name]+separate[1][name])/2, rtol=1e-12, atol=1e-14)


class TrainerTests(unittest.TestCase):
    """Short runs of every kind of ladder level: losses stay finite and the monitor improves or holds."""

    @classmethod
    def setUpClass(cls):
        x, put = contracts(16, seed=21)
        cls.monitor = dict(inputs=(x, put), reference=reference_values(STOCKS, x, put, nodes=24))
        cls.data = labelled_data(STOCKS, DOMAIN, STRIKE, 64, seed=23, nodes=24)

    def run_level(self, family, model_settings, loss, augment=None, data=True):
        model = small_model(family, **model_settings)
        settings = dict(SHORT_RUN, loss=loss, augment=augment)
        trainer = Trainer(model, settings, DOMAIN, STRIKE, self.monitor, self.data if data else None,
                          log=lambda message: None)
        record = trainer.run(seed=0)
        self.assertTrue(np.isfinite(record["best_monitor"]))
        for entry in record["history"]:
            self.assertTrue(all(np.isfinite(v) for k, v in entry.items() if isinstance(v, float)))
        return record

    def test_raw_pinn(self):
        self.run_level("pinn", dict(layout="raw", output="plain"), dict(pde=1.0, terminal=1.0), data=False)

    def test_reduced_pinn_with_rad_and_anchors(self):
        self.run_level("pinn", dict(output="levy_envelope", similarity="levy"),
                       dict(pde=1.0, boundary=1.0, price=1.0), augment="permutation")

    def test_raw_deeponet(self):
        self.run_level("deeponet", dict(layout="raw", output="plain"), dict(price=1.0))

    def test_physics_informed_differential_deeponet(self):
        self.run_level("deeponet", dict(output="levy_envelope", similarity=0.35),
                       dict(price=1.0, derivatives=1.0, pde=1.0, boundary=1.0), augment="permutation")


class ExpiryTests(unittest.TestCase):
    """At tau = 0 the reduced-layout outputs return the payoff instead of 0 / 0."""

    MODELS = (dict(family="pinn", output="levy"), dict(family="pinn", output="levy_envelope", similarity="levy"),
              dict(family="deeponet", output="levy", similarity=0.3))

    @classmethod
    def setUpClass(cls):
        base = row(c_star(tau=0.0))
        col = STOCKS.column
        basket = sum(base[col[f"weight_{i}"]]*base[col[f"spot_{i}"]] for i in range(1, 4))
        basket += (1-sum(base[col[f"weight_{i}"]] for i in range(1, 4)))*base[col["spot_4"]]
        rows = []
        for strike in (0.9*basket, basket, 1.1*basket):      # in, at and out of the money for a call
            r = base.copy()
            r[col["strike"]] = strike
            rows.append(r)
        cls.x = np.array(rows*2)
        cls.put = np.array([False]*3+[True]*3)
        strikes = cls.x[:, col["strike"]]
        cls.payoff = np.where(cls.put, np.maximum(strikes-basket, 0), np.maximum(basket-strikes, 0))

    def test_price_is_the_payoff(self):
        for settings in self.MODELS:
            model = small_model(**settings)
            with torch.no_grad():
                price = model.price(torch.as_tensor(self.x), torch.as_tensor(self.put)).numpy()
            assert_allclose(price, self.payoff, atol=1e-12, err_msg=str(settings))

    def test_greeks_are_finite(self):
        for settings in self.MODELS:
            values = price_and_greeks(STOCKS, small_model(**settings).price, self.x, self.put)
            for name, value in values.items():
                self.assertTrue(np.all(np.isfinite(value)), f"{settings} {name}")

    def test_positive_maturities_are_unchanged(self):
        x, _ = contracts(8, seed=5)
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(x))
        for settings in self.MODELS:
            model = small_model(**settings)
            with torch.no_grad():
                self.assertTrue(torch.equal(model.call(features), model._call(features)), str(settings))


if __name__ == "__main__":
    unittest.main()
