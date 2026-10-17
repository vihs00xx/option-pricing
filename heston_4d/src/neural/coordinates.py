"""Contract rows, the training domain, the reduced coordinates and the moment-matched base.

Written for any number d of stocks (``src/heston_basket/model.py`` defines the model).
A contract row holds the fields of ``Stocks(d).fields``. As under Black-Scholes, the call
price is homogeneous of degree one in (spots, strike), the weights enter only through
w_i S_i, and rates and dividends only through the forwards and the discount factor:

    C = K exp(-r tau) c(z, s, v, tau; kappa, theta, xi, rho, R, C),
    f_i = w_i S_i exp((r - q_i) tau) / K,  z = log(sum_i f_i),  s_i = f_i / sum_j f_j.

In y_i = log f_i and the variances v_i the normalized call solves the driftless PDE

    c_tau = sum_i [1/2 v_i (c_{y_i y_i} - c_{y_i}) + kappa_i (theta_i - v_i) c_{v_i}
                   + 1/2 xi_i^2 v_i c_{v_i v_i} + rho_i xi_i v_i c_{y_i v_i}]
          + sum_{i<j} sqrt(v_i v_j) [R_ij c_{y_i y_j} + xi_i xi_j C_ij c_{v_i v_j}]
          + sum_{i != j} sqrt(v_i v_j) rho_i xi_j C_ij c_{y_i v_j}

with c(tau = 0) = (exp(z) - 1)^+; puts follow from parity, p = c - (exp(z) - 1). The
features are (z, s_1, ..., s_{d-1}, v_1, ..., v_d, tau, kappa_1..d, theta_1..d, xi_1..d,
leverage_1..d, rho_12, ..., eta_12, ...), with rho the stock correlations R and eta the
variance correlations C.

The base the networks correct is the Levy (moment-matched lognormal) call of a basket whose
log returns have the expected integrated variances I_i = theta_i tau + (v_i - theta_i)
(1 - exp(-kappa_i tau)) / kappa_i and covariance R_ij sqrt(I_i I_j); it is exact for one
stock as xi -> 0.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
import torch

PROTOCOL_DIR = Path(__file__).resolve().parents[2]/"experiments/config/protocol"
PER_STOCK = ("v0", "kappa", "theta", "xi", "leverage")


def protocol_path():
    paths = sorted(PROTOCOL_DIR.glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one protocol in {PROTOCOL_DIR}")
    return paths[0]


class Stocks:
    """Field and feature positions for a Heston basket of d stocks."""

    def __init__(self, d):
        if d < 2:
            raise ValueError("a basket needs at least two stocks")
        self.d = d
        self.pairs = tuple((i, j) for i in range(d) for j in range(i+1, d))
        n = range(1, d+1)
        pair_names = tuple(f"{i+1}{j+1}" for i, j in self.pairs)
        self.fields = (tuple(f"spot_{i}" for i in n)+("strike", "tau", "rate")
                       + tuple(f"dividend_{i}" for i in n)
                       + tuple(f"{p}_{i}" for p in PER_STOCK for i in n)
                       + tuple(f"correlation_{k}" for k in pair_names)
                       + tuple(f"vol_correlation_{k}" for k in pair_names)
                       + tuple(f"weight_{i}" for i in range(1, d)))
        self.column = {name: k for k, name in enumerate(self.fields)}
        self.features = (("z",)+tuple(f"s_{i}" for i in range(1, d))+tuple(f"v_{i}" for i in n)+("tau",)
                         + tuple(f"{p}_{i}" for p in ("kappa", "theta", "xi", "leverage") for i in n)
                         + tuple(f"rho_{k}" for k in pair_names)+tuple(f"eta_{k}" for k in pair_names))
        self.v = slice(d, 2*d)
        self.tau = 2*d                     # feature index of tau
        P = len(self.pairs)
        start = 2*d+1
        self.kappa, self.theta = slice(start, start+d), slice(start+d, start+2*d)
        self.xi, self.leverage = slice(start+2*d, start+3*d), slice(start+3*d, start+4*d)
        self.rho, self.eta = slice(start+4*d, start+4*d+P), slice(start+4*d+P, start+4*d+2*P)
        pattern = torch.zeros(P, d, d, dtype=torch.float64)
        for k, (i, j) in enumerate(self.pairs):
            pattern[k, i, j] = pattern[k, j, i] = 1.0
        self.pattern = pattern

    def take(self, x, prefix):
        return torch.stack([x[:, self.column[f]] for f in self.fields if f.startswith(prefix)
                            and not (prefix == "correlation_" and f.startswith("vol_"))], dim=1)

    def weights(self, x):
        w = self.take(x, "weight_")
        return torch.cat((w, 1-w.sum(1, keepdim=True)), dim=1)

    def correlation(self, entries):
        return torch.eye(self.d, dtype=entries.dtype)+torch.einsum("np,pij->nij", entries, self.pattern.to(entries.dtype))

    def split(self, features):
        """dict of z, s (N, d) including s_d, v, tau, kappa, theta, xi, leverage, rho, eta."""
        s = features[:, 1:self.d]
        s = torch.cat((s, 1-s.sum(1, keepdim=True)), dim=1)
        return dict(z=features[:, 0], s=s, v=features[:, self.v], tau=features[:, self.tau],
                    kappa=features[:, self.kappa], theta=features[:, self.theta], xi=features[:, self.xi],
                    leverage=features[:, self.leverage], rho=features[:, self.rho], eta=features[:, self.eta])


# ---------------------------------------------------------------- training domain
def box_names(d):
    n = range(1, d+1)
    return (tuple(f"moneyness_{i}" for i in n)+("tau", "rate")+tuple(f"dividend_{i}" for i in n)
            + tuple(f"{p}_{i}" for p in PER_STOCK for i in n))


def protocol_domain(path=None):
    """Training domain and strike of the test-set protocol.

    Returns a dict with a [low, high] range per box parameter and the weight floor, the
    entry ranges and eigenvalue floors of the variance correlation C and the residual
    correlation P (the domain draws C and P; the stock correlation R follows), and the
    attainable range of R's entries.
    """
    protocol = json.loads(Path(path or protocol_path()).read_text(encoding="utf-8"))
    training = protocol["training_domain"]
    d = int(protocol["stocks"])
    constraints = protocol["domain_constraints"]
    domain = {name: list(training[name]) for name in box_names(d)}
    domain.update(weight_floor=float(training["weight"][0]), correlation=list(training["correlation"]),
                  vol_correlation=list(training["vol_correlation"]),
                  residual_correlation=list(training["residual_correlation"]),
                  min_eigenvalue=float(constraints["min_eigenvalue"]))
    return domain, float(protocol["strike"])


def uniform_correlations(generator, count, d, low, high, min_eigenvalue):
    """Entries (count, P) uniform in [low, high] with rejection on the smallest eigenvalue."""
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    out = np.empty((0, len(pairs)))
    while len(out) < count:
        entries = generator.uniform(low, high, size=(2*(count-len(out))+16, len(pairs)))
        matrices = np.tile(np.eye(d), (len(entries), 1, 1))
        for k, (i, j) in enumerate(pairs):
            matrices[:, i, j] = matrices[:, j, i] = entries[:, k]
        out = np.concatenate((out, entries[np.linalg.eigvalsh(matrices)[:, 0] >= min_eigenvalue]))
    return out[:count]


def matrices(entries, d):
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    M = np.tile(np.eye(d), entries.shape[:-1]+(1, 1))
    for k, (i, j) in enumerate(pairs):
        M[..., i, j] = M[..., j, i] = entries[..., k]
    return M


def stock_correlation(C, P, leverage):
    """R = (rho rho') o C + (rbar rbar') o P (numpy, batched)."""
    rho = np.asarray(leverage)
    rbar = np.sqrt(1-rho**2)
    R = rho[..., :, None]*rho[..., None, :]*C+rbar[..., :, None]*rbar[..., None, :]*P
    d = R.shape[-1]
    R[..., np.arange(d), np.arange(d)] = 1.0
    return R


def domain_correlations(generator, count, d, domain, leverage):
    """(R entries, C entries), each (count, P): C and P drawn uniformly from the domain, R from them."""
    C = uniform_correlations(generator, count, d, *domain["vol_correlation"], domain["min_eigenvalue"])
    P = uniform_correlations(generator, count, d, *domain["residual_correlation"], domain["min_eigenvalue"])
    R = stock_correlation(matrices(C, d), matrices(P, d), leverage)
    i, j = np.triu_indices(d, 1)
    return R[:, i, j], C


def rows_from_parameters(stocks, values, weights, R, C, strike):
    """Contract rows from protocol-named arrays, weights (N, d) and correlation entries (N, P)."""
    d = stocks.d
    x = np.empty((len(weights), len(stocks.fields)))
    x[:, stocks.column["strike"]] = strike
    for name in ("tau", "rate"):
        x[:, stocks.column[name]] = values[name]
    for i in range(1, d+1):
        x[:, stocks.column[f"spot_{i}"]] = strike*np.asarray(values[f"moneyness_{i}"])
        x[:, stocks.column[f"dividend_{i}"]] = values[f"dividend_{i}"]
        for p in PER_STOCK:
            x[:, stocks.column[f"{p}_{i}"]] = values[f"{p}_{i}"]
    for i in range(1, d):
        x[:, stocks.column[f"weight_{i}"]] = weights[:, i-1]
    for k, (i, j) in enumerate(stocks.pairs):
        x[:, stocks.column[f"correlation_{i+1}{j+1}"]] = R[:, k]
        x[:, stocks.column[f"vol_correlation_{i+1}{j+1}"]] = C[:, k]
    return x


def sample_domain(stocks, domain, strike, count, seed, tau=None):
    """Scrambled Sobol contracts in the protocol's training domain; calls and puts alternate.

    Sobol covers the box parameters and d - 1 uniforms whose sorted spacings are
    Dirichlet(1) weights above the floor; C and P are drawn from their own stream.
    ``tau`` optionally overrides the maturity range.
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
    leverage = np.column_stack([values[f"leverage_{i}"] for i in range(1, d+1)])
    R, C = domain_correlations(np.random.default_rng(seed+100), count, d, domain, leverage)
    return rows_from_parameters(stocks, values, weights, R, C, strike), np.arange(count) % 2 == 1


def contract_of(stocks, row, put):
    """Engine ``HestonBasket`` of a contract row."""
    from src.heston_basket.model import HestonBasket
    p = dict(zip(stocks.fields, np.asarray(row, dtype=float)))
    d = stocks.d
    weights = [p[f"weight_{i}"] for i in range(1, d)]
    R, C = np.eye(d), np.eye(d)
    for i, j in stocks.pairs:
        R[i, j] = R[j, i] = p[f"correlation_{i+1}{j+1}"]
        C[i, j] = C[j, i] = p[f"vol_correlation_{i+1}{j+1}"]
    per = lambda name: tuple(p[f"{name}_{i}"] for i in range(1, d+1))
    return HestonBasket(spots=per("spot"), strike=p["strike"], tau=p["tau"], rate=p["rate"],
                        dividends=per("dividend"), weights=tuple(weights)+(1-sum(weights),),
                        v0=per("v0"), kappa=per("kappa"), theta=per("theta"), xi=per("xi"),
                        leverage=per("leverage"), correlation=R, vol_correlation=C,
                        option="put" if put else "call")


def row_of(c):
    """Contract row of an engine ``HestonBasket``."""
    stocks = Stocks(c.dimension)
    values = dict(strike=c.strike, tau=c.tau, rate=c.rate)
    for i in range(c.dimension):
        values.update({f"spot_{i+1}": c.spots[i], f"dividend_{i+1}": c.dividends[i], f"weight_{i+1}": c.weights[i]})
        for p in PER_STOCK:
            values[f"{p}_{i+1}"] = getattr(c, p)[i]
    for i, j in stocks.pairs:
        values[f"correlation_{i+1}{j+1}"] = c.correlation[i][j]
        values[f"vol_correlation_{i+1}{j+1}"] = c.vol_correlation[i][j]
    return np.array([values[f] for f in stocks.fields])


# ---------------------------------------------------------------- reduced coordinates
def reduced_coordinates(stocks, x):
    """Features and the discounted strike for a torch contract tensor."""
    strike, tau, rate = (x[:, stocks.column[n]] for n in ("strike", "tau", "rate"))
    spots, dividends = stocks.take(x, "spot_"), stocks.take(x, "dividend_")
    f = stocks.weights(x)*spots*torch.exp((rate[:, None]-dividends)*tau[:, None])/strike[:, None]
    forward = f.sum(1)
    features = torch.cat((torch.log(forward)[:, None], f[:, :-1]/forward[:, None], stocks.take(x, "v0_"),
                          tau[:, None], stocks.take(x, "kappa_"), stocks.take(x, "theta_"), stocks.take(x, "xi_"),
                          stocks.take(x, "leverage_"), stocks.take(x, "correlation_"),
                          stocks.take(x, "vol_correlation_")), dim=1)
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
    """Inverse of ``log_forwards``; ``rest`` holds (v, tau, parameters)."""
    z = torch.logsumexp(y, dim=1)
    return torch.cat((z[:, None], torch.exp(y[:, :-1]-z[:, None]), rest), dim=1)


def permute(stocks, features, order):
    """Features of the same contract with the stocks relabelled: new stock k is old order[k]."""
    p = stocks.split(features)
    pairs = torch.tensor(stocks.pairs)
    d = stocks.d
    index = order[:, :, None].expand(-1, -1, d)

    def matrix(entries):
        M = stocks.correlation(entries)
        M = torch.gather(torch.gather(M, 1, index), 2, order[:, None, :].expand(-1, d, -1))
        return M[:, pairs[:, 0], pairs[:, 1]]
    take = lambda name: torch.gather(p[name], 1, order)
    return torch.cat((p["z"][:, None], take("s")[:, :-1], take("v"), p["tau"][:, None], take("kappa"),
                      take("theta"), take("xi"), take("leverage"), matrix(p["rho"]), matrix(p["eta"])), dim=1)


# ---------------------------------------------------------------- moment-matched base
def integrated_variance(p):
    """E[int_0^tau v_i dt] for the split features ``p``, shape (N, d)."""
    tau = p["tau"][:, None]
    return p["theta"]*tau-(p["v"]-p["theta"])*torch.expm1(-p["kappa"]*tau)/p["kappa"]


def levy_volatility(stocks, features):
    """Total volatility of the lognormal matching the base basket's first two moments (Levy)."""
    p = stocks.split(features)
    a = torch.sqrt(integrated_variance(p).clamp_min(0.0))
    covariance = a[:, :, None]*stocks.correlation(p["rho"])*a[:, None, :]
    ratio = torch.einsum("ni,nij,nj->n", p["s"], torch.expm1(covariance), p["s"])
    return torch.sqrt(torch.log1p(ratio))


def black_call(z, total_vol):
    """Undiscounted Black call with unit strike on forward exp(z)."""
    live = total_vol > 0
    total_vol = torch.where(live, total_vol, torch.ones_like(total_vol))
    d1 = z/total_vol+total_vol/2
    value = torch.exp(z)*torch.special.ndtr(d1)-torch.special.ndtr(d1-total_vol)
    return torch.where(live, value, torch.relu(torch.expm1(z)))


def levy_call(stocks, features):
    return black_call(features[:, 0], levy_volatility(stocks, features))
