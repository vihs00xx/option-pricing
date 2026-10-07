import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.heston import torch_reference as tr
from src.heston.model import QUANTITIES, Contracts, development_contracts
from src.neural.coordinates import (base_call, black_call, price_from_call, reduced_coordinates,
                                    rows_from_contracts)
from src.neural.networks import load_model, save_model
from src.neural.pde import raw_residual, reduced_residual, zero_variance_residual
from src.neural.pinn import PINN
from src.neural.deeponet import DeepONet
from src.neural.reference import LABELS, price_and_greeks, reference_labels, reference_values

torch.set_default_dtype(torch.float64)
BOXES = dict(x=(-3, 2.5), v=(0, 3), tau=(0, 2), kappa=(0.5, 5), theta=(0.01, 0.36), xi=(0.1, 1), rho=(-0.9, 0.3))


def twin_call(features, panels=64):
    """Normalized call of the torch twin as a differentiable function of reduced features."""
    f = features
    S = torch.exp(f[:, 0])
    zeros = torch.zeros_like(S)
    t = dict(S=S, tau=f[:, 2], r=zeros, q=zeros, v0=f[:, 1], kappa=f[:, 3], theta=f[:, 4], xi=f[:, 5],
             rho=f[:, 6], call=torch.ones(len(S), dtype=torch.bool))
    c = Contracts(**{k: v.detach().numpy() for k, v in t.items()})
    U = torch.as_tensor(tr.truncation_point(c, tr.TRUNCATION))
    s, w = tr.composite_rule(panels)
    return tr.price(t, U, s, w)


def sample_features(n, seed, v_low=0.01):
    g = np.random.default_rng(seed)
    return torch.as_tensor(np.stack([g.uniform(-0.4, 0.4, n), g.uniform(v_low, 0.36, n), g.uniform(0.1, 2, n),
                                     g.uniform(0.5, 5, n), g.uniform(0.01, 0.36, n), g.uniform(0.1, 1, n),
                                     g.uniform(-0.9, 0.3, n)], axis=1))


class TestNeural(unittest.TestCase):
    def test_reduced_pde_holds_for_reference(self):
        f = sample_features(16, 1)
        residual, call = reduced_residual(twin_call, f)
        self.assertLess(float(residual.detach().abs().max()), 1e-8)

    def test_zero_variance_equation_holds_for_reference(self):
        f = sample_features(8, 2)
        f[:, 1] = 0.0
        self.assertLess(float(zero_variance_residual(twin_call, f).detach().abs().max()), 1e-8)

    def test_raw_pde_holds_for_reference(self):
        main, _, _ = development_contracts()
        x, put = rows_from_contracts(main.take(np.arange(8)))
        def price(xt, p):
            features, scale = reduced_coordinates(xt)
            return price_from_call(twin_call(features), features, scale, p)
        r = raw_residual(price, torch.as_tensor(x), torch.as_tensor(put))
        self.assertLess(float(r.detach().abs().max()), 1e-8)

    def test_price_and_greeks_match_twin(self):
        main, _, _ = development_contracts()
        c = main.take(np.arange(12))
        x, put = rows_from_contracts(c)
        def price(xt, p):
            features, scale = reduced_coordinates(xt)
            return price_from_call(twin_call(features, 128), features, scale, p)
        mine = price_and_greeks(price, x, put)
        ref = reference_values(x, put)
        for q in QUANTITIES:
            np.testing.assert_allclose(mine[q], ref[q], atol=1e-8, err_msg=q)

    def test_labels_are_call_derivatives(self):
        f = sample_features(6, 3)
        labels = reference_labels(f)
        self.assertEqual(labels.shape, (6, len(LABELS)))
        np.testing.assert_allclose(labels[:, 0], twin_call(f, 128).detach(), atol=1e-10)
        fx = f.clone().requires_grad_(True)
        c = twin_call(fx, 128)
        g, = torch.autograd.grad(c.sum(), fx)
        np.testing.assert_allclose(labels[:, 1], g[:, 0], atol=1e-8)   # c_x
        np.testing.assert_allclose(labels[:, 2], g[:, 1], atol=1e-8)   # c_v
        np.testing.assert_allclose(labels[:, 3], g[:, 2], atol=1e-8)   # c_tau

    def test_base_is_black_scholes_and_exact_at_expiry(self):
        f = sample_features(5, 4)
        f[:, 2] = 1e-12
        np.testing.assert_allclose(base_call(f), torch.relu(torch.expm1(f[:, 0])), atol=1e-6)
        x, tv = torch.tensor([0.1]), torch.tensor([0.2])
        d1 = x / tv + tv / 2
        expected = math.exp(0.1) * 0.5 * math.erfc(-float(d1) / math.sqrt(2)) - 0.5 * math.erfc(-float(d1 - tv) / math.sqrt(2))
        self.assertAlmostEqual(float(black_call(x, tv)), expected, places=12)

    def test_hard_payoff_and_parity(self):
        torch.manual_seed(0)
        for cls in (PINN, DeepONet):
            for output in ("bs", "bs_envelope"):
                m = cls("reduced", output, BOXES, width=8, depth=2, **({"rank": 8} if cls is DeepONet else {}),
                        v_input="sqrt")
                f = sample_features(5, 5, v_low=0.0)
                f[:, 2] = 1e-12          # the base divides by sqrt(w tau); training keeps tau >= 1e-6
                f[:, 0] = torch.linspace(-0.3, 0.3, 5)
                payoff = torch.relu(torch.expm1(f[:, 0]))
                np.testing.assert_allclose(m.call(f).detach(), payoff, atol=1e-6)
        main, _, _ = development_contracts()
        x, put = rows_from_contracts(main.take(np.arange(6)))
        xt = torch.as_tensor(x)
        call = m.price(xt, torch.zeros(6, dtype=torch.bool))
        p = m.price(xt, torch.ones(6, dtype=torch.bool))
        S, tau, r, q = xt[:, 0], xt[:, 1], xt[:, 2], xt[:, 3]
        np.testing.assert_allclose((call - p).detach(), S * torch.exp(-q * tau) - torch.exp(-r * tau), atol=1e-12)

    def test_sqrt_input_has_finite_v_slope_at_zero(self):
        m = PINN("reduced", "bs", BOXES, width=8, depth=2, v_input="sqrt")
        f = sample_features(4, 6)
        f[:, 1] = 0.0
        self.assertTrue(torch.isfinite(zero_variance_residual(m.call, f)).all())

    def test_save_load_roundtrip(self):
        m = DeepONet("reduced", "bs", BOXES, width=8, depth=2, rank=8)
        f = sample_features(4, 7)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            save_model(m, path)
            m2 = load_model(path)
        np.testing.assert_allclose(m.call(f).detach(), m2.call(f).detach(), atol=0)


if __name__ == "__main__":
    unittest.main()
