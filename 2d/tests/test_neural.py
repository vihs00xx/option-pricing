from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
import torch

from src.basket import BasketContract, arithmetic_greeks, quadrature_price
from src.neural.domain import COLUMN, ContractBox, black_call, moment_matched_vol, sample_contracts
from src.neural.models import BasketPricer, EnsemblePricer, reduced_features
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.reference import QUANTITIES, contract_row, price_and_greeks, reference_values
from src.neural.training import (LossWeights, Schedule, label_scales, loss_terms, pde_residual,
                                 reduced_tensor, reference_labels, train)

torch.set_default_dtype(torch.float64)


def small_model(output="implied_vol", seed=0):
    torch.manual_seed(seed)
    x = sample_contracts(ContractBox(), 256, seed=1)
    features = reduced_features(*reduced_tensor(x).unbind(-1))
    model = BasketPricer("reduced", output, width=16, depth=2, mean=features.mean(0), std=features.std(0))
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.contracts = [BasketContract(dividends=(0.01, 0.03)),
                          BasketContract(spots=(80, 130), weights=(0.3, 0.7), option="put", correlation=-0.7),
                          BasketContract(tau=0.02, correlation=0.95, volatilities=(0.5, 0.65)),
                          BasketContract(spots=(60, 140), weights=(0.8, 0.2), tau=2, correlation=-0.95)]
        rows = [contract_row(c) for c in self.contracts]
        self.x = np.array([r[0] for r in rows])
        self.put = np.array([r[1] for r in rows])
        self.reference = reference_values(self.x, self.put)

    def test_prices_match_adaptive_quadrature(self):
        expected = [quadrature_price(c).price for c in self.contracts]
        assert_allclose(self.reference["price"], expected, rtol=0, atol=1e-10)

    def test_greeks_match_repository_greeks(self):
        for i, contract in enumerate(self.contracts[:2]):
            expected = arithmetic_greeks(contract)
            for name in QUANTITIES[1:6]:
                self.assertAlmostEqual(self.reference[name][i], expected[name], delta=1e-9)
            for name in QUANTITIES[6:]:
                self.assertAlmostEqual(self.reference[name][i], expected[name], delta=5e-5)

    def test_homogeneity_and_parity(self):
        scaled = self.x.copy()
        for column in ("spot_1", "spot_2", "strike"):
            scaled[:, COLUMN[column]] *= 2.5
        values = reference_values(scaled, self.put)
        assert_allclose(values["price"], 2.5*self.reference["price"], rtol=1e-12)
        assert_allclose(values["delta_1"], self.reference["delta_1"], atol=1e-12)
        calls = reference_values(self.x, np.zeros(len(self.x), bool), greeks=False)["price"]
        puts = reference_values(self.x, np.ones(len(self.x), bool), greeks=False)["price"]
        for i, contract in enumerate(self.contracts):
            parity = contract.forward_value()-contract.strike*np.exp(-contract.rate*contract.tau)
            self.assertAlmostEqual(calls[i]-puts[i], parity, places=10)

    def test_labels_satisfy_scale_free_pde(self):
        r = reduced_tensor(sample_contracts(ContractBox(), 128, seed=3))
        labels = reference_labels(r, 2)
        self.assertLess(pde_residual(r, labels["dc"], labels["d2c"]).abs().max().item(), 1e-12)

    def test_monte_carlo_estimators_are_unbiased(self):
        estimates = monte_carlo_estimates(self.x[0], False, 400_000, seed=7)
        z = [(estimates[q][0]-self.reference[q][0])/estimates[q][1] for q in QUANTITIES]
        self.assertLess(max(abs(v) for v in z), 4.5)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.x = sample_contracts(ContractBox(), 64, seed=9)
        self.put = np.arange(64) % 2 == 1

    def test_zero_network_is_levy_approximation(self):
        model = BasketPricer("reduced", "implied_vol", width=8, depth=2)
        r = reduced_tensor(self.x)
        f1, f2, a1, a2, rho = r.unbind(-1)
        expected = black_call(f1+f2, moment_matched_vol(f1, f2, a1, a2, rho))
        torch.testing.assert_close(model.call_from_reduced(f1, f2, a1, a2, rho), expected)

    def test_autograd_greeks_match_finite_differences(self):
        model = small_model()
        greeks = price_and_greeks(model, self.x, self.put)
        for name, field, sign in (("delta_1", "spot_1", 1), ("vega_2", "vol_2", 1), ("theta", "tau", -1),
                                  ("rho", "rate", 1), ("correlation", "correlation", 1)):
            h = 1e-6
            up, down = self.x.copy(), self.x.copy()
            up[:, COLUMN[field]] += h
            down[:, COLUMN[field]] -= h
            fd = sign*(price_and_greeks(model, up, self.put, greeks=False)["price"]
                       - price_and_greeks(model, down, self.put, greeks=False)["price"])/(2*h)
            assert_allclose(greeks[name], fd, rtol=1e-5, atol=1e-7)

    def test_put_call_parity_is_exact(self):
        model = small_model("direct")
        x = torch.as_tensor(self.x)
        call = model(x, torch.zeros(64, dtype=torch.bool))
        put = model(x, torch.ones(64, dtype=torch.bool))
        w1 = x[:, COLUMN["weight_1"]]
        tau, rate = x[:, COLUMN["tau"]], x[:, COLUMN["rate"]]
        forward = (w1*x[:, COLUMN["spot_1"]]*torch.exp(-x[:, COLUMN["dividend_1"]]*tau)
                   + (1-w1)*x[:, COLUMN["spot_2"]]*torch.exp(-x[:, COLUMN["dividend_2"]]*tau))
        torch.testing.assert_close(call-put, forward-x[:, COLUMN["strike"]]*torch.exp(-rate*tau))

    def test_ensemble_greeks_are_member_averages(self):
        members = [small_model(seed=s) for s in (1, 2)]
        ensemble = price_and_greeks(EnsemblePricer(members), self.x, self.put)
        separate = [price_and_greeks(m, self.x, self.put) for m in members]
        for name in QUANTITIES:
            assert_allclose(ensemble[name], (separate[0][name]+separate[1][name])/2, rtol=1e-12, atol=1e-14)

    def test_sobolev_training_reduces_loss(self):
        r = reduced_tensor(sample_contracts(ContractBox(), 512, seed=4))
        data = dict(r=r, **reference_labels(r, 2))
        model = small_model()
        weights = LossWeights(first=1, second=0.5, pde=0.1)
        before = float(loss_terms(model, data, weights, label_scales(data))[0].detach())
        train(model, data, data, weights, Schedule(epochs=60, batch_size=128, learning_rate=3e-3,
                                                    evaluate_every=10), seed=0, log=lambda *_: None)
        after = float(loss_terms(model, data, weights, label_scales(data))[0].detach())
        self.assertLess(after, 0.5*before)

    def test_interrupted_training_resumes_identically(self):
        r = reduced_tensor(sample_contracts(ContractBox(), 256, seed=6))
        data = dict(r=r, **reference_labels(r, 1))
        weights, schedule = LossWeights(first=1), Schedule(epochs=4, batch_size=64, evaluate_every=2)
        straight = small_model()
        train(straight, data, data, weights, schedule, seed=3, log=lambda *_: None)

        def interrupt(message):
            if message.startswith("adam 2:"):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"resume.pt"
            resumed = small_model()
            with self.assertRaises(KeyboardInterrupt):
                train(resumed, data, data, weights, schedule, seed=3, log=interrupt, resume_path=path, save_every=1)
            resumed = small_model()
            train(resumed, data, data, weights, schedule, seed=3, log=lambda *_: None, resume_path=path, save_every=1)
        for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
