"""Shared fixtures for the neural-pricer tests."""

import numpy as np
import torch

from src.neural.coordinates import Stocks, sample_domain
from src.neural.networks import build_model
from src.neural.reference import price_and_greeks
from src.neural.training import raw_box, reduced_box
from src.pricing import Basket

torch.set_default_dtype(torch.float64)
STOCKS = Stocks(4)
# The training domain of experiments/config/protocol/four_stock_test_v1.json, as
# coordinates.protocol_domain returns it.
DOMAIN = dict({f"moneyness_{i}": [0.7, 1.3] for i in range(1, 5)}, tau=[0.05, 2.0], rate=[-0.02, 0.1],
              **{f"volatility_{i}": [0.1, 0.6] for i in range(1, 5)},
              **{f"dividend_{i}": [0.0, 0.06] for i in range(1, 5)},
              weight_floor=0.1, correlation=[-0.5, 0.9], min_eigenvalue=0.05)
STRIKE = 100.0
COMPUTATIONAL = {"z": [-3.0, 2.5], "share_floor": 0.01}
CORRELATION_STAR = np.array([[1, -0.5, 0.3, 0.6], [-0.5, 1, -0.4, -0.3],
                             [0.3, -0.4, 1, 0.2], [0.6, -0.3, 0.2, 1]])


def c_star(**kwargs):
    values = dict(spots=(100, 95, 105, 100), strike=100, tau=2.0, rate=0.03,
                  volatilities=(0.1, 0.6, 0.3, 0.45), dividends=(0, 0.02, 0.04, 0.01),
                  weights=(0.4, 0.3, 0.2, 0.1), correlation=CORRELATION_STAR)
    values.update(kwargs)
    return Basket(**values)


def row(c):
    """Contract row of an engine ``Basket``."""
    stocks = Stocks(c.dimension)
    values = dict(strike=c.strike, tau=c.tau, rate=c.rate)
    for i in range(c.dimension):
        values.update({f"spot_{i+1}": c.spots[i], f"dividend_{i+1}": c.dividends[i],
                       f"vol_{i+1}": c.volatilities[i], f"weight_{i+1}": c.weights[i]})
    for i, j in stocks.pairs:
        values[f"correlation_{i+1}{j+1}"] = c.correlation[i][j]
    return np.array([values[f] for f in stocks.fields])


def contract(r, put, stocks=STOCKS):
    """Engine ``Basket`` of a contract row."""
    p = dict(zip(stocks.fields, r))
    d = stocks.d
    weights = [p[f"weight_{i}"] for i in range(1, d)]
    correlation = np.eye(d)
    for i, j in stocks.pairs:
        correlation[i, j] = correlation[j, i] = p[f"correlation_{i+1}{j+1}"]
    return Basket(spots=tuple(p[f"spot_{i}"] for i in range(1, d+1)), strike=p["strike"], tau=p["tau"],
                  rate=p["rate"], volatilities=tuple(p[f"vol_{i}"] for i in range(1, d+1)),
                  dividends=tuple(p[f"dividend_{i}"] for i in range(1, d+1)),
                  weights=tuple(weights)+(1-sum(weights),), correlation=correlation,
                  option="put" if put else "call")


def contracts(count, seed):
    return sample_domain(STOCKS, DOMAIN, STRIKE, count, seed=seed)


def small_model(family="pinn", layout="reduced", output="levy", seed=0, similarity=None):
    torch.manual_seed(seed)
    boxes = reduced_box(STOCKS, DOMAIN, COMPUTATIONAL) if layout == "reduced" else raw_box(STOCKS, DOMAIN)
    sizes = dict(width=12, depth=2) if family == "pinn" else dict(width=12, depth=2, rank=8)
    model = build_model(family, d=4, layout=layout, output=output, boxes=boxes, similarity=similarity, **sizes)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return model


def finite_difference_check(test, model, x, put):
    """Autograd Greeks of ``model`` against central differences of its price."""
    greeks = price_and_greeks(STOCKS, model.price, x, put)
    for name, field, sign in (("delta_1", "spot_1", 1), ("vega_4", "vol_4", 1), ("theta", "tau", -1),
                              ("rho", "rate", 1), ("correlation_23", "correlation_23", 1)):
        column = STOCKS.column[field]
        h = 1e-6
        up, down = x.copy(), x.copy()
        up[:, column] += h
        down[:, column] -= h
        fd = sign*(price_and_greeks(STOCKS, model.price, up, put, greeks=False)["price"]
                   - price_and_greeks(STOCKS, model.price, down, put, greeks=False)["price"])/(2*h)
        np.testing.assert_allclose(greeks[name], fd, rtol=1e-5, atol=1e-7,
                                   err_msg=f"{model.family} {model.layout} {name}")


SHORT_RUN = dict(
    computational_domain=COMPUTATIONAL,
    sampling={"focus": 0.7, "focus_z": [-0.6, 0.55], "tau_power": 2.0, "weight": "time",
              "rad": {"size": 256, "pool_factor": 2, "k": 1.0, "c": 1.0, "every": 20}},
    batch={"collocation": 128, "terminal": 32, "boundary": 32, "data": 64},
    optimizer={"adam_steps": 40, "learning_rate": 3e-3, "final_learning_rate": 1e-3, "check_every": 20,
               "lbfgs_iterations": 6, "lbfgs_chunk": 3,
               "lbfgs_batch": {"collocation": 128, "terminal": 32, "boundary": 32, "data": 64},
               "nncg_steps": 2, "nncg_check_every": 1, "nncg_rank": 4, "nncg_cg_steps": 5,
               "nncg_batch": {"collocation": 64, "terminal": 16, "boundary": 16, "data": 32}})
