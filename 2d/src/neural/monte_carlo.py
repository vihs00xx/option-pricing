"""Plain Monte Carlo prices and Greeks for one basket contract, the textbook baseline.

Carried over from the earlier two-stock neural study, reordered to the test set's quantities.

Exact GBM terminal draws with pseudo-random normals (no QMC, no conditioning, no
control variate). First-order Greeks use pathwise derivatives of the payoff;
the spot Hessian uses the likelihood-ratio/pathwise mixed estimator
    Gamma_ij = E[h_i (score_j - delta_ij / S_i)],
where h_i is the pathwise Delta_i integrand and score_j = (R^{-1} Z)_j / (sigma_j sqrt(tau) S_j)
is the derivative of the log terminal density in S_j. Gamma_12 averages the two
mixed orders. Units match ``price_and_greeks``.
"""

import numpy as np

from .coordinates import FIELDS
from .reference import QUANTITIES


def monte_carlo_estimates(row, put, n_paths, seed, chunk=1 << 20):
    """Return {quantity: (mean, standard error)} from n_paths independent paths."""
    p = dict(zip(FIELDS, row))
    w = np.array([p["weight_1"], 1-p["weight_1"]])
    spot = np.array([p["spot_1"], p["spot_2"]])
    vol = np.array([p["vol_1"], p["vol_2"]])
    dividend = np.array([p["dividend_1"], p["dividend_2"]])
    tau, rate, rho, strike = p["tau"], p["rate"], p["correlation"], p["strike"]
    root_tau, orthogonal = np.sqrt(tau), np.sqrt(1-rho*rho)
    drift = rate-dividend-0.5*vol*vol
    discount, sign = np.exp(-rate*tau), -1.0 if put else 1.0
    rng = np.random.default_rng(seed)
    sums = np.zeros(len(QUANTITIES))
    squares = np.zeros(len(QUANTITIES))
    for start in range(0, n_paths, chunk):
        n = min(chunk, n_paths-start)
        e1, e2 = rng.standard_normal((2, n))
        z = np.stack((e1, rho*e1+orthogonal*e2))
        terminal = spot[:, None]*np.exp(drift[:, None]*tau+vol[:, None]*root_tau*z)
        basket = w@terminal
        payoff = np.maximum(sign*(basket-strike), 0)
        slope = discount*sign*(sign*(basket-strike) > 0)
        weighted = w[:, None]*terminal
        h = slope*weighted/spot[:, None]
        score = np.stack((z[0]-rho*z[1], z[1]-rho*z[0]))/(1-rho*rho)/(vol*root_tau*spot)[:, None]
        dz2 = e1-rho*e2/orthogonal
        samples = np.stack((
            discount*payoff,
            h[0], h[1],
            h[0]*(score[0]-1/spot[0]),
            0.5*(h[0]*score[1]+h[1]*score[0]),
            h[1]*(score[1]-1/spot[1]),
            slope*weighted[0]*(root_tau*z[0]-vol[0]*tau),
            slope*weighted[1]*(root_tau*z[1]-vol[1]*tau),
            rate*discount*payoff-slope*(weighted*(drift[:, None]+vol[:, None]*z/(2*root_tau))).sum(0),
            -tau*discount*payoff+slope*tau*basket,
            slope*weighted[1]*vol[1]*root_tau*dz2,
        ))
        sums += samples.sum(axis=1)
        squares += (samples*samples).sum(axis=1)
    mean = sums/n_paths
    variance = np.maximum(squares/n_paths-mean*mean, 0)*n_paths/(n_paths-1)
    return {name: (float(mean[i]), float(np.sqrt(variance[i]/n_paths)))
            for i, name in enumerate(QUANTITIES)}
