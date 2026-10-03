"""Shared fixtures for the PINN and DeepONet tests."""

import numpy as np
import torch

from src.basket import BasketContract
from src.neural.coordinates import FIELDS, protocol_domain, sample_domain
from src.neural.networks import build_model
from src.neural.reference import price_and_greeks, reference_values
from src.neural.training import raw_box, reduced_box

torch.set_default_dtype(torch.float64)
DOMAIN, STRIKE = protocol_domain()
COMPUTATIONAL = {"z": [-3.0, 2.5], "s": [0.01, 0.99]}


def contract(row, put):
    r = dict(zip(FIELDS, row))
    return BasketContract(spots=(r["spot_1"], r["spot_2"]), strike=r["strike"], tau=r["tau"], rate=r["rate"],
                          volatilities=(r["vol_1"], r["vol_2"]), dividends=(r["dividend_1"], r["dividend_2"]),
                          weights=(r["weight_1"], 1-r["weight_1"]), correlation=r["correlation"],
                          option="put" if put else "call")


def small_model(family="pinn", layout="reduced", output="levy", seed=0, similarity=None):
    torch.manual_seed(seed)
    boxes = reduced_box(DOMAIN, COMPUTATIONAL) if layout == "reduced" else raw_box(DOMAIN)
    sizes = dict(width=12, depth=2) if family == "pinn" else dict(width=12, depth=2, rank=8)
    model = build_model(family, layout=layout, output=output, boxes=boxes, similarity=similarity, **sizes)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


def contracts(count, seed):
    return sample_domain(DOMAIN, STRIKE, count, seed=seed)


def finite_difference_check(test, model, x, put):
    """Autograd Greeks of ``model`` against central differences of its price."""
    greeks = price_and_greeks(model.price, x, put)
    for name, field, sign in (("delta_1", "spot_1", 1), ("vega_2", "vol_2", 1), ("theta", "tau", -1),
                              ("rho", "rate", 1), ("correlation_12", "correlation", 1)):
        column = FIELDS.index(field)
        h = 1e-6
        up, down = x.copy(), x.copy()
        up[:, column] += h
        down[:, column] -= h
        fd = sign*(price_and_greeks(model.price, up, put, greeks=False)["price"]
                   - price_and_greeks(model.price, down, put, greeks=False)["price"])/(2*h)
        np.testing.assert_allclose(greeks[name], fd, rtol=1e-5, atol=1e-7,
                                   err_msg=f"{model.family} {model.layout} {name}")


def monitor_panel(count=64, seed=21):
    x, put = contracts(count, seed)
    return dict(inputs=(x, put), reference=reference_values(x, put))


SHORT_RUN = dict(
    computational_domain=COMPUTATIONAL,
    sampling={"focus": 0.7, "focus_z": [-0.6, 0.55], "tau_power": 2.0, "weight": "time",
              "rad": {"size": 512, "pool_factor": 2, "k": 1.0, "c": 1.0, "every": 20}},
    batch={"collocation": 256, "terminal": 64, "boundary": 64, "data": 128},
    optimizer={"adam_steps": 60, "learning_rate": 3e-3, "final_learning_rate": 1e-3, "check_every": 30,
               "lbfgs_iterations": 10, "lbfgs_chunk": 5,
               "lbfgs_batch": {"collocation": 256, "terminal": 64, "boundary": 64, "data": 128},
               "nncg_steps": 2, "nncg_check_every": 1, "nncg_rank": 4, "nncg_cg_steps": 5,
               "nncg_batch": {"collocation": 128, "terminal": 32, "boundary": 32, "data": 64}})
