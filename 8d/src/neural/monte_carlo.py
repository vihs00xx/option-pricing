"""Plain Monte Carlo prices and Greeks for one basket contract of d stocks, the textbook baseline.

The two-stock study's estimator written for d stocks, in the test set's quantity order.

Exact GBM terminal draws Z = L e with R = L L' and pseudo-random normals e (no QMC, no
conditioning, no control variate). First-order Greeks use pathwise derivatives of the
payoff; the correlation sensitivity differentiates L, dL = L Phi(L^-1 dR L^-T) with Phi
the lower triangle and half the diagonal. The spot Hessian uses the
likelihood-ratio/pathwise mixed estimator
    Gamma_ij = E[h_i (score_j - delta_ij / S_i)],
where h_i is the pathwise Delta_i integrand and score_j = (R^{-1} Z)_j / (sigma_j sqrt(tau) S_j)
is the derivative of the log terminal density in S_j; off-diagonal entries average the
two mixed orders. Units match ``price_and_greeks``.
"""

import numpy as np

from .reference import quantities


def cholesky_derivatives(stocks, correlation):
    """dL / d rho_ij for each pair (a symmetric unit change of R), shape (P, d, d)."""
    lower = np.linalg.cholesky(correlation)
    inverse = np.linalg.inv(lower)
    out = []
    for i, j in stocks.pairs:
        change = np.zeros((stocks.d, stocks.d))
        change[i, j] = change[j, i] = 1.0
        inner = inverse@change@inverse.T
        out.append(lower@(np.tril(inner, -1)+0.5*np.diag(np.diag(inner))))
    return np.array(out)


def monte_carlo_estimates(stocks, row, put, n_paths, seed, chunk=1 << 18):
    """Return {quantity: (mean, standard error)} from n_paths independent paths."""
    d = stocks.d
    p = dict(zip(stocks.fields, row))
    take = lambda prefix: np.array([p[f"{prefix}_{i}"] for i in range(1, d+1)])
    spot, vol, dividend = take("spot"), take("vol"), take("dividend")
    w = np.array([p[f"weight_{i}"] for i in range(1, d)])
    w = np.append(w, 1-w.sum())
    correlation = np.eye(d)
    for i, j in stocks.pairs:
        correlation[i, j] = correlation[j, i] = p[f"correlation_{i+1}{j+1}"]
    lower, precision = np.linalg.cholesky(correlation), np.linalg.inv(correlation)
    d_lower = cholesky_derivatives(stocks, correlation)
    tau, rate, strike = p["tau"], p["rate"], p["strike"]
    root_tau = np.sqrt(tau)
    drift = rate-dividend-0.5*vol*vol
    discount, sign = np.exp(-rate*tau), -1.0 if put else 1.0
    upper = [(i, j) for i in range(d) for j in range(i, d)]
    rng = np.random.default_rng(seed)
    names = quantities(stocks)
    sums = np.zeros(len(names))
    squares = np.zeros(len(names))
    for start in range(0, n_paths, chunk):
        n = min(chunk, n_paths-start)
        e = rng.standard_normal((d, n))
        z = lower@e
        terminal = spot[:, None]*np.exp(drift[:, None]*tau+vol[:, None]*root_tau*z)
        basket = w@terminal
        payoff = np.maximum(sign*(basket-strike), 0)
        slope = discount*sign*(sign*(basket-strike) > 0)
        weighted = w[:, None]*terminal
        h = slope*weighted/spot[:, None]
        score = (precision@z)/(vol*root_tau*spot)[:, None]
        gamma = [0.5*(h[i]*score[j]+h[j]*score[i])-(h[i]/spot[i] if i == j else 0) for i, j in upper]
        vega = slope*weighted*(root_tau*z-vol[:, None]*tau)
        theta = rate*discount*payoff-slope*(weighted*(drift[:, None]+vol[:, None]*z/(2*root_tau))).sum(0)
        rho = -tau*discount*payoff+slope*tau*basket
        exposure = slope*weighted*vol[:, None]*root_tau
        correlation_terms = [(exposure*(d_lower[k]@e)).sum(0) for k in range(len(stocks.pairs))]
        samples = np.vstack([discount*payoff, h, *gamma, vega, theta, rho, *correlation_terms])
        sums += samples.sum(axis=1)
        squares += (samples*samples).sum(axis=1)
    mean = sums/n_paths
    variance = np.maximum(squares/n_paths-mean*mean, 0)*n_paths/(n_paths-1)
    return {name: (float(mean[i]), float(np.sqrt(variance[i]/n_paths))) for i, name in enumerate(names)}
