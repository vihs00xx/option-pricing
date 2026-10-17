"""Contract rows, the training domain, the reduced coordinates and the Levy approximation.

Written for any number d of stocks; ``Stocks(d)`` holds the positions. A contract row
holds the fields of ``Stocks(d).fields``. Under risk-neutral GBM the call price is
homogeneous of degree one in (spots, strike), the weights enter only through w_i S_i,
and rates and dividends only through the forwards and the discount factor:

    C = K exp(-r tau) c(z, s, tau; sigma, R),
    f_i = w_i S_i exp((r - q_i) tau) / K,  z = log(sum_i f_i),  s_i = f_i / sum_j f_j.

In y_i = log f_i the normalized call c solves the driftless d-asset PDE

    c_tau = 1/2 sum_i sigma_i^2 (c_ii - c_i) + sum_{i<j} rho_ij sigma_i sigma_j c_ij

with c(tau = 0) = (exp(z) - 1)^+. Puts follow from exact put-call parity,
p = c - (exp(z) - 1). The features are (z, s_1, ..., s_{d-1}, tau, sigma_1, ...,
sigma_d, rho_12, rho_13, ..., rho_{d-1,d}): for eight stocks 60 contract fields become
8 coordinates, tau and 36 parameters. For d = 2 the fields and features are those of
the two-stock study in the same positions.

Pair names join the two stock numbers (``correlation_38``), so they are unique up to
nine stocks.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
import torch

PROTOCOL_PATH = Path(__file__).resolve().parents[2]/"experiments/config/protocol/eight_stock_test_v1.json"


class Stocks:
    """Field and feature positions for a basket of d stocks."""

    def __init__(self, d):
        if d < 2:
            raise ValueError("a basket needs at least two stocks")
        self.d = d
        self.pairs = tuple((i, j) for i in range(d) for j in range(i+1, d))
        n = range(1, d+1)
        self.fields = (tuple(f"spot_{i}" for i in n)+("strike", "tau", "rate")
                       + tuple(f"dividend_{i}" for i in n)+tuple(f"vol_{i}" for i in n)
                       + tuple(f"correlation_{i+1}{j+1}" for i, j in self.pairs)
                       + tuple(f"weight_{i}" for i in range(1, d)))
        self.column = {name: k for k, name in enumerate(self.fields)}
        self.features = (("z",)+tuple(f"s_{i}" for i in range(1, d))+("tau",)
                         + tuple(f"sigma_{i}" for i in n)+tuple(f"rho_{i+1}{j+1}" for i, j in self.pairs))
        self.tau = d                       # feature index of tau
        self.sigma = slice(d+1, 2*d+1)
        self.rho = slice(2*d+1, None)
        # rho (N, P) -> R (N, d, d) = I + rho @ pattern.
        pattern = torch.zeros(len(self.pairs), d, d, dtype=torch.float64)
        for k, (i, j) in enumerate(self.pairs):
            pattern[k, i, j] = pattern[k, j, i] = 1.0
        self.pattern = pattern

    def take(self, x, prefix):
        return torch.stack([x[:, self.column[f]] for f in self.fields if f.startswith(prefix)], dim=1)

    def weights(self, x):
        w = self.take(x, "weight_")
        return torch.cat((w, 1-w.sum(1, keepdim=True)), dim=1)

    def correlation(self, rho):
        return torch.eye(self.d, dtype=rho.dtype)+torch.einsum("np,pij->nij", rho, self.pattern.to(rho.dtype))

    def split(self, features):
        """(z, s (N, d) including s_d, tau, sigma (N, d), rho (N, P))."""
        z, s = features[:, 0], features[:, 1:self.d]
        s = torch.cat((s, 1-s.sum(1, keepdim=True)), dim=1)
        return z, s, features[:, self.tau], features[:, self.sigma], features[:, self.rho]


# ---------------------------------------------------------------- training domain
def protocol_domain(path=PROTOCOL_PATH):
    """Training domain and strike of the frozen test-set protocol.

    Returns a dict with a [low, high] range per box parameter (moneyness_i, tau,
    volatility_i, rate, dividend_i) and the weight floor, correlation entry range,
    smallest eigenvalue of the correlation matrix and the factor model the correlation
    matrices are drawn from. The protocol stores the weight range as "weight" and the
    eigenvalue floor and factor model under "domain_constraints"; they are translated
    here, so the protocol file itself never changes.
    """
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    training = protocol["training_domain"]
    d = sum(name.startswith("moneyness_") for name in training)
    domain = {name: list(training[name]) for name in box_names(d)}
    domain.update(weight_floor=float(training["weight"][0]), correlation=list(training["correlation"]),
                  min_eigenvalue=float(protocol["domain_constraints"]["min_eigenvalue"]),
                  correlation_model=dict(protocol["domain_constraints"]["correlation_model"]))
    return domain, float(protocol["strike"])


def box_names(d):
    n = range(1, d+1)
    return (tuple(f"moneyness_{i}" for i in n)+("tau",)+tuple(f"volatility_{i}" for i in n)
            + ("rate",)+tuple(f"dividend_{i}" for i in n))


def factor_loadings(generator, d, model, count=None):
    """Loading rows (market, style 1, ..., style k) of the domain's factor model.

    Stock by stock, the market loading is uniform on ``model["market"]`` and each style
    loading on ``model["style"]``; a stock's row is redrawn while its communality
    |b_i|^2 exceeds ``model["max_communality"]``. Returns (d, k + 1), or (count, d, k + 1).
    """
    if count is not None:
        return np.stack([factor_loadings(generator, d, model) for _ in range(count)])
    k = int(model["styles"])
    low = np.array([model["market"][0]]+[model["style"][0]]*k)
    high = np.array([model["market"][1]]+[model["style"][1]]*k)
    rows = []
    for _ in range(d):
        while True:
            b = low+(high-low)*generator.uniform(size=k+1)
            if b @ b <= model["max_communality"]:
                rows.append(b)
                break
    return np.array(rows)


def factor_matrix(loadings):
    """R = B B' + diag(1 - |b_i|^2) for loadings (..., d, k)."""
    R = loadings @ np.swapaxes(loadings, -1, -2)
    d = loadings.shape[-2]
    R[..., np.arange(d), np.arange(d)] = 1.0
    return R


def valid_correlations(stocks, generator, count, model):
    """Correlation entries (count, P) of factor-model matrices, in pair order."""
    R = factor_matrix(factor_loadings(generator, stocks.d, model, count))
    pairs = np.array(stocks.pairs)
    return R[:, pairs[:, 0], pairs[:, 1]]


def rows_from_parameters(stocks, values, weights, correlations, strike):
    """Contract rows from protocol-named arrays, weights (N, d) and correlation entries (N, P)."""
    d = stocks.d
    x = np.empty((len(weights), len(stocks.fields)))
    x[:, stocks.column["strike"]] = strike
    for name in ("tau", "rate"):
        x[:, stocks.column[name]] = values[name]
    for i in range(1, d+1):
        x[:, stocks.column[f"spot_{i}"]] = strike*np.asarray(values[f"moneyness_{i}"])
        x[:, stocks.column[f"dividend_{i}"]] = values[f"dividend_{i}"]
        x[:, stocks.column[f"vol_{i}"]] = values[f"volatility_{i}"]
    for i in range(1, d):
        x[:, stocks.column[f"weight_{i}"]] = weights[:, i-1]
    for k, (i, j) in enumerate(stocks.pairs):
        x[:, stocks.column[f"correlation_{i+1}{j+1}"]] = correlations[:, k]
    return x


def sample_domain(stocks, domain, strike, count, seed, tau=None):
    """Scrambled Sobol contracts in the protocol's training domain; calls and puts alternate.

    Sobol covers the box parameters and d - 1 uniforms whose sorted spacings are
    Dirichlet(1) weights above the floor; correlations are factor-model draws from
    their own stream. ``tau`` optionally overrides the maturity range.
    """
    d, names = stocks.d, box_names(stocks.d)
    unit = qmc.Sobol(d=len(names)+d-1, scramble=True, seed=seed).random(count)
    values = {}
    for k, name in enumerate(names):
        low, high = tau if (name == "tau" and tau is not None) else domain[name]
        values[name] = low+(high-low)*unit[:, k]
    edges = np.sort(unit[:, len(names):], axis=1)
    spacings = np.diff(np.concatenate((np.zeros((count, 1)), edges, np.ones((count, 1))), axis=1), axis=1)
    floor = domain["weight_floor"]
    weights = floor+(1-d*floor)*spacings
    correlations = valid_correlations(stocks, np.random.default_rng(seed+100), count, domain["correlation_model"])
    return rows_from_parameters(stocks, values, weights, correlations, strike), np.arange(count) % 2 == 1


# ---------------------------------------------------------------- reduced coordinates
def reduced_coordinates(stocks, x):
    """Features and the discounted strike for a torch contract tensor."""
    strike, tau, rate = (x[:, stocks.column[n]] for n in ("strike", "tau", "rate"))
    spots, dividends = stocks.take(x, "spot_"), stocks.take(x, "dividend_")
    f = stocks.weights(x)*spots*torch.exp((rate[:, None]-dividends)*tau[:, None])/strike[:, None]
    forward = f.sum(1)
    features = torch.cat((torch.log(forward)[:, None], f[:, :-1]/forward[:, None], tau[:, None],
                          stocks.take(x, "vol_"), stocks.take(x, "correlation_")), dim=1)
    return features, strike*torch.exp(-rate*tau)


def price_from_call(call, features, scale, put):
    """Currency price from the normalized call; ``put`` is a boolean tensor."""
    parity = torch.where(put, torch.exp(features[:, 0])-1, torch.zeros_like(call))
    return scale*(call-parity)


def log_forwards(stocks, features):
    """y_i = z + log s_i, with s_d = 1 - s_1 - ... - s_{d-1}; shape (N, d)."""
    z, s = features[:, 0], features[:, 1:stocks.d]
    return torch.cat((z[:, None]+torch.log(s), (z+torch.log1p(-s.sum(1)))[:, None]), dim=1)


def features_from_log_forwards(stocks, y, rest):
    """Inverse of ``log_forwards``; ``rest`` holds (tau, sigma, rho)."""
    z = torch.logsumexp(y, dim=1)
    return torch.cat((z[:, None], torch.exp(y[:, :-1]-z[:, None]), rest), dim=1)


def permute(stocks, features, order):
    """Features of the same contract with the stocks relabelled: new stock k is old order[k].

    The normalized call is unchanged by a relabelling.
    """
    z, s, tau, sigma, rho = stocks.split(features)
    R = stocks.correlation(rho)
    index = order[:, :, None].expand(-1, -1, stocks.d)
    R = torch.gather(torch.gather(R, 1, index), 2, order[:, None, :].expand(-1, stocks.d, -1))
    pairs = torch.tensor(stocks.pairs)
    return torch.cat((z[:, None], torch.gather(s, 1, order)[:, :-1], tau[:, None], torch.gather(sigma, 1, order),
                      R[:, pairs[:, 0], pairs[:, 1]]), dim=1)


# ---------------------------------------------------------------- Levy approximation
def levy_volatility(stocks, features):
    """Total volatility of the lognormal matching the basket's first two moments (Levy)."""
    _, s, tau, sigma, rho = stocks.split(features)
    a = sigma*torch.sqrt(tau)[:, None]
    covariance = a[:, :, None]*stocks.correlation(rho)*a[:, None, :]
    # E[B^2] / E[B]^2 with B the normalized basket; expm1 keeps short maturities exact.
    ratio = torch.einsum("ni,nij,nj->n", s, torch.expm1(covariance), s)
    return torch.sqrt(torch.log1p(ratio))


def black_call(z, total_vol):
    """Undiscounted Black call with unit strike on forward exp(z)."""
    # Zero total volatility (expiry) is the payoff; the stand-in volatility avoids 0 / 0.
    live = total_vol > 0
    total_vol = torch.where(live, total_vol, torch.ones_like(total_vol))
    d1 = z/total_vol+total_vol/2
    value = torch.exp(z)*torch.special.ndtr(d1)-torch.special.ndtr(d1-total_vol)
    return torch.where(live, value, torch.relu(torch.expm1(z)))


def levy_call(stocks, features):
    return black_call(features[:, 0], levy_volatility(stocks, features))


def basket_row(c):
    """Contract row of an engine ``src.pricing.Basket`` (any number of stocks)."""
    stocks = Stocks(c.dimension)
    values = dict(strike=c.strike, tau=c.tau, rate=c.rate)
    for i in range(c.dimension):
        values.update({f"spot_{i+1}": c.spots[i], f"dividend_{i+1}": c.dividends[i],
                       f"vol_{i+1}": c.volatilities[i], f"weight_{i+1}": c.weights[i]})
    for i, j in stocks.pairs:
        values[f"correlation_{i+1}{j+1}"] = c.correlation[i][j]
    return np.array([values[f] for f in stocks.fields])
