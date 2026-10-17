"""Shared fixtures for the Heston basket tests; the number of stocks is the protocol's."""

import json

import numpy as np
import torch

from src.heston_basket.model import HestonBasket
from src.neural.coordinates import Stocks, protocol_domain, protocol_path, sample_domain
from src.neural.networks import build_model
from src.neural.reference import price_and_greeks
from src.neural.training import raw_box, reduced_box

torch.set_default_dtype(torch.float64)
D = int(json.loads(protocol_path().read_text(encoding="utf-8"))["stocks"])
STOCKS = Stocks(D)
DOMAIN, STRIKE = protocol_domain()
COMPUTATIONAL = {"z": [-3.0, 2.5], "share_floor": 0.01, "v": [0.0, 1.5]}


def basket(**kwargs):
    """A moderate contract with dispersed stocks."""
    spread = lambda low, high: tuple(np.linspace(low, high, D))
    values = dict(spots=spread(95, 105), strike=100, tau=1.0, rate=0.03, dividends=spread(0.0, 0.02),
                  weights=(1 / D,) * D, v0=spread(0.04, 0.09), kappa=spread(2.0, 1.5), theta=spread(0.05, 0.08),
                  xi=spread(0.5, 0.7), leverage=spread(-0.7, -0.4), correlation=0.4, vol_correlation=0.3)
    values.update(kwargs)
    return HestonBasket(**values)


def contracts(count, seed):
    return sample_domain(STOCKS, DOMAIN, STRIKE, count, seed=seed)


def small_model(family="pinn", layout="reduced", output="levy", seed=0, similarity=None, v_input="plain"):
    torch.manual_seed(seed)
    boxes = reduced_box(STOCKS, DOMAIN, COMPUTATIONAL) if layout == "reduced" else raw_box(STOCKS, DOMAIN)
    sizes = dict(width=12, depth=2) if family == "pinn" else dict(width=12, depth=2, rank=8)
    model = build_model(family, d=D, layout=layout, output=output, boxes=boxes, similarity=similarity,
                        v_input=v_input, **sizes)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


def finite_difference_check(model, x, put):
    """Autograd Greeks of ``model`` against central differences of its price."""
    greeks = price_and_greeks(STOCKS, model.price, x, put)
    for name, field, sign in (("delta_1", "spot_1", 1), (f"vega_{D}", f"v0_{D}", 1), ("theta", "tau", -1),
                              ("rho", "rate", 1), ("d_kappa_1", "kappa_1", 1), (f"d_leverage_{D}", f"leverage_{D}", 1),
                              ("correlation_12", "correlation_12", 1), ("vol_correlation_12", "vol_correlation_12", 1)):
        column = STOCKS.column[field]
        h = 1e-6
        up, down = x.copy(), x.copy()
        up[:, column] += h
        down[:, column] -= h
        fd = sign*(price_and_greeks(STOCKS, model.price, up, put, greeks=False)["price"]
                   - price_and_greeks(STOCKS, model.price, down, put, greeks=False)["price"])/(2*h)
        np.testing.assert_allclose(greeks[name], fd, rtol=1e-5, atol=1e-7, err_msg=f"{model.family} {model.layout} {name}")


SHORT_RUN = dict(
    computational_domain=COMPUTATIONAL,
    sampling={"focus": 0.7, "focus_z": [-0.6, 0.55], "tau_power": 2.0, "v_power": 2.0, "weight": "time",
              "rad": {"size": 256, "pool_factor": 2, "k": 1.0, "c": 1.0, "every": 20}},
    batch={"collocation": 128, "terminal": 32, "boundary": 32, "zero_variance": 32, "variance_cap": 32, "data": 64},
    optimizer={"adam_steps": 40, "learning_rate": 3e-3, "final_learning_rate": 1e-3, "check_every": 20,
               "lbfgs_iterations": 6, "lbfgs_chunk": 3,
               "lbfgs_batch": {"collocation": 128, "terminal": 32, "boundary": 32, "zero_variance": 32,
                               "variance_cap": 32, "data": 64},
               "nncg_steps": 2, "nncg_check_every": 1, "nncg_rank": 4, "nncg_cg_steps": 5,
               "nncg_batch": {"collocation": 64, "terminal": 16, "boundary": 16, "zero_variance": 16,
                              "variance_cap": 16, "data": 32}})
