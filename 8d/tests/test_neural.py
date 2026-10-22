"""Neural-pricer machinery for eight stocks: torch reference, coordinates, PDE, Monte Carlo, models."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.neural.coordinates import (factor_loadings, features_from_log_forwards, levy_call, log_forwards, permute,
                                    protocol_domain, reduced_coordinates)
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.pde import raw_residual, reduced_residual
from src.neural.reference import (call_function, call_pricer, labels, normalized_call, permute_labels,
                                  price_and_greeks, quantities, reference_labels, reference_values, sobol_rule)
from src.neural.training import Trainer, _correlations, labelled_data, score
from src.pricing import gauss_hermite_values
from src.pricing.conditional import sample_values
from src.pricing.contract import quantity_names
from src.pricing.rqmc import normal_points
from tests.neural_helpers import (DOMAIN, DOMAIN_4, MODEL, RULE, STOCKS, STOCKS_4, STRIKE, SHORT_RUN, c_star,
                                  contract, contracts, finite_difference_check, row, small_model)

SPOT_GREEKS = 45            # price, 8 Deltas and 36 Gammas
SPOT_GREEKS_4 = 15


class SobolReferenceTests(unittest.TestCase):
    """The eight-stock torch reference: a Sobol rule, the engine's estimator on the same points."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.put = contracts(6, seed=11)

    def test_quantities_follow_the_engine_order(self):
        self.assertEqual(quantities(STOCKS), tuple(quantity_names(c_star())))
        self.assertEqual(len(quantities(STOCKS)), 83)
        self.assertEqual(len(labels(STOCKS)), 82)

    def test_plain_rule_is_the_engine_on_the_same_points(self):
        rule = dict(kind="sobol", points=512, seed=9, control=False)
        U = normal_points(512, 7, 9)
        # Calls only: the twin prices puts by exact parity, which a finite rule satisfies only
        # up to its own error, while the engine integrates the put payoff on the points.
        cases = [(row(c), False) for c in (c_star(), c_star(tau=0.01))]
        cases += [(self.x[i], False) for i in range(3)]
        for r, put in cases:
            c = contract(r, put)
            engine = sample_values(c, U, "arithmetic").mean(axis=0)
            twin = reference_values(STOCKS, r[None], np.array([put]), nodes=rule)
            twin = np.array([twin[q][0] for q in quantities(STOCKS)])
            assert_allclose(twin[:SPOT_GREEKS], engine[:SPOT_GREEKS], rtol=1e-11, atol=1e-12)
            # Parameter Greeks: exact derivatives of the rule against the engine's central bumps.
            assert_allclose(twin[SPOT_GREEKS:], engine[SPOT_GREEKS:], rtol=1e-5, atol=2e-5)

    def test_geometric_control_has_the_exact_mean(self):
        # The controlled and plain rules estimate the same value; with many points they agree
        # far below either rule's error at 1,024 points.
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        big = sobol_rule(1 << 17, 7, 3)
        plain = normalized_call(STOCKS, features, big)
        controlled = normalized_call(STOCKS, features, big+(True,))
        small = normalized_call(STOCKS, features, sobol_rule(1024, 7, 4)+(True,))
        self.assertLess(((plain-controlled)/plain).abs().max().item(), 2e-4)
        self.assertLess(((small-controlled)/controlled).abs().max().item(), 2e-2)

    def test_rule_settings(self):
        function = call_function(STOCKS, RULE)
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        torch.testing.assert_close(function(features), normalized_call(STOCKS, features, sobol_rule(1024, 7, 5)+(True,)))

    def test_relabelling_moves_the_rule_value_only_by_rule_error(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        order = torch.argsort(torch.rand(len(features), 8, generator=torch.Generator().manual_seed(3)), dim=1)
        rule = dict(kind="sobol", points=16384, seed=2)
        original = reference_labels(STOCKS, features, rule)[:, 0]
        moved = reference_labels(STOCKS, permute(STOCKS, features, order), rule)[:, 0]
        self.assertLess(((moved-original)/original).abs().max().item(), 2e-3)
        back = torch.argsort(order, dim=1)
        torch.testing.assert_close(permute(STOCKS, permute(STOCKS, features, order), back), features,
                                   rtol=0, atol=1e-15)

    def test_coordinate_round_trip(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        y = log_forwards(STOCKS, features)
        torch.testing.assert_close(features_from_log_forwards(STOCKS, y, features[:, 8:]), features)

    def test_levy_is_exact_at_expiry_and_close_before(self):
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        expiring = features.clone()
        expiring[:, STOCKS.tau] = 1e-14
        torch.testing.assert_close(levy_call(STOCKS, expiring), torch.relu(torch.expm1(features[:, 0])),
                                   atol=1e-12, rtol=0)
        call = call_function(STOCKS, RULE)(features)
        self.assertLess((levy_call(STOCKS, features)-call).abs().max().item(), 0.02)

    def test_monte_carlo_is_unbiased(self):
        reference = reference_values(STOCKS, self.x[:2], self.put[:2], nodes=dict(kind="sobol", points=32768, seed=1))
        for i in range(2):
            estimates = monte_carlo_estimates(STOCKS, self.x[i], self.put[i], 200_000, seed=7+i)
            z = [(estimates[q][0]-reference[q][i])/estimates[q][1] for q in quantities(STOCKS)]
            self.assertLess(max(abs(v) for v in z), 4.5)

    def test_labelled_data_prices_match_reference(self):
        data = labelled_data(STOCKS, DOMAIN, STRIKE, 4, seed=5, nodes=RULE)
        values = reference_values(STOCKS, data["x"].numpy(), data["put"].numpy(), greeks=False, nodes=RULE)
        assert_allclose(data["normalized_price"].numpy()*STRIKE, values["price"], rtol=1e-12, atol=1e-12)

    def test_score_is_zero_for_reference(self):
        reference = reference_values(STOCKS, self.x[:2], self.put[:2], nodes=RULE)
        self.assertEqual(score(reference, reference)[0], 0.0)


class GaussHermiteReferenceTests(unittest.TestCase):
    """The general code's exact checks with four stocks, where tensor Gauss-Hermite is feasible."""

    @classmethod
    def setUpClass(cls):
        cls.x, cls.put = contracts(6, seed=11, stocks=STOCKS_4)

    def test_matches_engine_at_32_nodes(self):
        for i in range(3):
            engine = gauss_hermite_values(contract(self.x[i], self.put[i], STOCKS_4), 32)
            twin = reference_values(STOCKS_4, self.x[i][None], np.array([self.put[i]]), nodes=32)
            twin = np.array([twin[q][0] for q in quantities(STOCKS_4)])
            assert_allclose(twin[:SPOT_GREEKS_4], engine[:SPOT_GREEKS_4], rtol=1e-12, atol=1e-12)
            assert_allclose(twin[SPOT_GREEKS_4:], engine[SPOT_GREEKS_4:], rtol=1e-6, atol=5e-6)

    def test_reduced_pde_holds_for_reference(self):
        features, _ = reduced_coordinates(STOCKS_4, torch.as_tensor(self.x))
        residual, _ = reduced_residual(STOCKS_4, call_function(STOCKS_4, 40), features)
        self.assertLess(residual.abs().max().item(), 1e-8)

    def test_raw_pde_holds_for_reference(self):
        price = call_pricer(STOCKS_4, call_function(STOCKS_4, 40))
        residual = raw_residual(STOCKS_4, lambda x, put: price(x, put)/x[:, STOCKS_4.column["strike"]],
                                torch.as_tensor(self.x[:3]), torch.as_tensor(self.put[:3]))
        self.assertLess(residual.abs().max().item(), 1e-8)

    def test_labels_satisfy_reduced_pde(self):
        features, _ = reduced_coordinates(STOCKS_4, torch.as_tensor(self.x))
        values = reference_labels(STOCKS_4, features, nodes=40)
        named = dict(zip(labels(STOCKS_4), values.T))
        _, _, _, sigma, rho = STOCKS_4.split(features)
        generator = sum(0.5*sigma[:, i]**2*(named[f"c_{i+1}{i+1}"]-named[f"c_{i+1}"]) for i in range(4))
        generator = generator+sum(rho[:, k]*sigma[:, i]*sigma[:, j]*named[f"c_{i+1}{j+1}"]
                                  for k, (i, j) in enumerate(STOCKS_4.pairs))
        assert_allclose(named["c_tau"], generator, atol=1e-8)

    def test_relabelling_the_stocks_leaves_the_call_unchanged(self):
        features, _ = reduced_coordinates(STOCKS_4, torch.as_tensor(self.x))
        order = torch.argsort(torch.rand(len(features), 4, generator=torch.Generator().manual_seed(3)), dim=1)
        moved = permute(STOCKS_4, features, order)
        original, relabelled = reference_labels(STOCKS_4, features, 40), reference_labels(STOCKS_4, moved, 40)
        torch.testing.assert_close(relabelled, permute_labels(STOCKS_4, original, order), rtol=0, atol=1e-8)

    def test_sobol_rule_converges_to_gauss_hermite(self):
        features, _ = reduced_coordinates(STOCKS_4, torch.as_tensor(self.x))
        exact = call_function(STOCKS_4, 48)(features)
        sobol = call_function(STOCKS_4, dict(kind="sobol", points=1 << 15, seed=1))(features)
        self.assertLess(((sobol-exact)/exact).abs().max().item(), 1e-3)


class DomainTests(unittest.TestCase):
    def test_protocol_domain_is_the_tests_domain(self):
        domain, strike = protocol_domain()
        self.assertEqual(strike, STRIKE)
        self.assertEqual({k: list(v) if isinstance(v, (list, tuple)) else v for k, v in domain.items()}, DOMAIN)

    def test_contracts_lie_in_the_training_domain(self):
        x, put = contracts(256, seed=3)
        p = {name: x[:, k] for k, name in enumerate(STOCKS.fields)}
        weights = np.column_stack([p[f"weight_{i}"] for i in range(1, 8)])
        weights = np.column_stack((weights, 1-weights.sum(1)))
        self.assertGreaterEqual(weights.min(), 0.04-1e-12)
        correlations = np.tile(np.eye(8), (len(x), 1, 1))
        for i, j in STOCKS.pairs:
            correlations[:, i, j] = correlations[:, j, i] = p[f"correlation_{i+1}{j+1}"]
        self.assertGreaterEqual(np.linalg.eigvalsh(correlations)[:, 0].min(), 0.05-1e-12)
        entries = correlations[:, *np.triu_indices(8, 1)]
        self.assertGreaterEqual(entries.min(), -0.5)
        self.assertLessEqual(entries.max(), 0.95)
        self.assertTrue(np.all((p["tau"] >= 0.05) & (p["tau"] <= 2.0)))
        self.assertEqual(int(put.sum()), 128)

    def test_torch_and_numpy_factor_samplers_agree(self):
        numpy_draws = factor_loadings(np.random.default_rng(1), 8, MODEL, 4000)
        R = numpy_draws@np.swapaxes(numpy_draws, 1, 2)
        numpy_entries = R[:, *np.triu_indices(8, 1)].ravel()
        torch_entries = _correlations(STOCKS, torch.Generator().manual_seed(1), 4000, MODEL).numpy().ravel()
        for statistic in (np.mean, np.std, lambda v: np.quantile(v, 0.05), lambda v: np.quantile(v, 0.95)):
            self.assertAlmostEqual(statistic(numpy_entries), statistic(torch_entries), delta=0.01)
        communality = (numpy_draws**2).sum(-1)
        self.assertLessEqual(communality.max(), 0.95)


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
            for column in [f"spot_{i}" for i in range(1, 9)]+["strike"]:
                scaled[:, col[column]] *= 1.7
            torch.testing.assert_close(model.price(scaled, torch.zeros(n, dtype=torch.bool)), 1.7*call)

    def test_reduced_models_with_permutation_symmetric_data_are_not_forced_symmetric(self):
        model = small_model("deeponet")
        features, _ = reduced_coordinates(STOCKS, torch.as_tensor(self.x))
        order = torch.tensor([[1, 0, 3, 2, 5, 4, 7, 6]]).expand(len(features), -1)
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
        rule = dict(kind="sobol", points=256, seed=1)
        cls.monitor = dict(inputs=(x, put), reference=reference_values(STOCKS, x, put, nodes=rule))
        cls.data = labelled_data(STOCKS, DOMAIN, STRIKE, 64, seed=23, nodes=rule)

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
        basket = sum(base[col[f"weight_{i}"]]*base[col[f"spot_{i}"]] for i in range(1, 8))
        basket += (1-sum(base[col[f"weight_{i}"]] for i in range(1, 8)))*base[col["spot_8"]]
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
