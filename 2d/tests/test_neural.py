"""Shared neural-pricer machinery: reference, coordinates, PDE, Monte Carlo, scoring."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.basket import arithmetic_greeks, quadrature_price
from src.neural.coordinates import COLUMN, features_from_log_forwards, levy_call, log_forwards, reduced_coordinates
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.pde import raw_residual, reduced_residual
from src.neural.reference import QUANTITIES, call_pricer, price_and_greeks, reference_call, reference_labels, \
    reference_values
from src.neural.training import score
from tests.neural_helpers import contract, contracts, small_model


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.put = contracts(8, seed=11)
        cls.reference = reference_values(cls.x, cls.put)

    def test_prices_and_greeks_match_adaptive_quadrature(self):
        for i in range(3):
            c = contract(self.x[i], self.put[i])
            self.assertAlmostEqual(self.reference["price"][i], quadrature_price(c).price, delta=1e-10)
            expected = arithmetic_greeks(c)
            for name in ("delta_1", "delta_2", "gamma_11", "gamma_12", "gamma_22"):
                self.assertAlmostEqual(self.reference[name][i], expected[name], delta=1e-9)
            for name in ("vega_1", "vega_2", "theta", "rho"):
                self.assertAlmostEqual(self.reference[name][i], expected[name], delta=5e-5)
            self.assertAlmostEqual(self.reference["correlation_12"][i], expected["correlation"], delta=5e-5)

    def test_reduced_pde_holds_for_reference(self):
        features, _ = reduced_coordinates(torch.as_tensor(self.x))
        residual, _ = reduced_residual(reference_call, features)
        self.assertLess(residual.abs().max().item(), 1e-11)

    def test_raw_pde_holds_for_reference(self):
        price = call_pricer(reference_call)
        residual = raw_residual(lambda x, put: price(x, put)/x[:, COLUMN["strike"]],
                                torch.as_tensor(self.x), torch.as_tensor(self.put))
        self.assertLess(residual.abs().max().item(), 1e-10)

    def test_labels_satisfy_reduced_pde(self):
        features, _ = reduced_coordinates(torch.as_tensor(self.x))
        c, c1, c2, c11, c12, c22, ct = reference_labels(features)[:, :7].T
        v1, v2, rho = features[:, 3:].T
        assert_allclose(ct, 0.5*v1*v1*(c11-c1)+rho*v1*v2*c12+0.5*v2*v2*(c22-c2), atol=1e-11)

    def test_coordinate_round_trip(self):
        features, _ = reduced_coordinates(torch.as_tensor(self.x))
        y1, y2 = log_forwards(features)
        torch.testing.assert_close(features_from_log_forwards(y1, y2, features[:, 2:]), features)

    def test_levy_is_exact_at_expiry_and_close_before(self):
        features, _ = reduced_coordinates(torch.as_tensor(self.x))
        expiring = features.clone()
        expiring[:, 2] = 1e-14
        torch.testing.assert_close(levy_call(expiring), torch.relu(torch.expm1(features[:, 0])), atol=1e-12, rtol=0)
        self.assertLess((levy_call(features)-reference_call(features)).abs().max().item(), 0.02)

    def test_monte_carlo_is_unbiased(self):
        estimates = monte_carlo_estimates(self.x[0], self.put[0], 400_000, seed=7)
        z = [(estimates[q][0]-self.reference[q][0])/estimates[q][1] for q in QUANTITIES]
        self.assertLess(max(abs(v) for v in z), 4.5)

    def test_score_is_zero_for_reference(self):
        self.assertEqual(score(self.reference, self.reference)[0], 0.0)


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        self.x, self.put = contracts(32, seed=9)

    def test_reduced_models_keep_parity_and_homogeneity(self):
        for family in ("pinn", "deeponet"):
            model = small_model(family)
            x = torch.as_tensor(self.x)
            n = len(x)
            call = model.price(x, torch.zeros(n, dtype=torch.bool))
            put = model.price(x, torch.ones(n, dtype=torch.bool))
            w, tau = x[:, COLUMN["weight_1"]], x[:, COLUMN["tau"]]
            forward = (w*x[:, COLUMN["spot_1"]]*torch.exp(-x[:, COLUMN["dividend_1"]]*tau)
                       + (1-w)*x[:, COLUMN["spot_2"]]*torch.exp(-x[:, COLUMN["dividend_2"]]*tau))
            torch.testing.assert_close(call-put, forward-x[:, COLUMN["strike"]]*torch.exp(-x[:, COLUMN["rate"]]*tau))
            scaled = x.clone()
            for column in ("spot_1", "spot_2", "strike"):
                scaled[:, COLUMN[column]] *= 1.7
            torch.testing.assert_close(model.price(scaled, torch.zeros(n, dtype=torch.bool)), 1.7*call)

    def test_levy_output_with_zero_network_is_levy(self):
        for family, output in (("pinn", "levy"), ("deeponet", "levy"), ("pinn", "levy_envelope")):
            model = small_model(family, output=output)
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            features, _ = reduced_coordinates(torch.as_tensor(self.x))
            torch.testing.assert_close(model.call(features), levy_call(features))

    def test_save_and_load_round_trip(self):
        import tempfile
        from pathlib import Path
        for model in (small_model("pinn"), small_model("deeponet"),
                      Ensemble([small_model("deeponet", seed=1), small_model("deeponet", seed=2)])):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"model.pt"
                save_model(model, path)
                loaded = load_model(path)
            x, put = torch.as_tensor(self.x), torch.as_tensor(self.put)
            torch.testing.assert_close(loaded.price(x, put), model.price(x, put), rtol=0, atol=0)

    def test_ensemble_greeks_are_member_averages(self):
        members = [small_model(seed=s) for s in (1, 2)]
        ensemble = price_and_greeks(Ensemble(members).price, self.x, self.put)
        separate = [price_and_greeks(m.price, self.x, self.put) for m in members]
        for name in QUANTITIES:
            assert_allclose(ensemble[name], (separate[0][name]+separate[1][name])/2, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
