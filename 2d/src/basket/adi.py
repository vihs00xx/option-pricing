from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import splu

from src.black_scholes import black_scholes_price


@dataclass
class ADISolution:
    stock_1: np.ndarray
    stock_2: np.ndarray
    values: np.ndarray
    steps: int
    theta: float
    domain: tuple

    def evaluate(self, spots, dx=0, dy=0):
        if not (0 <= spots[0] <= self.domain[0] and 0 <= spots[1] <= self.domain[1]):
            raise ValueError("Evaluation stocks lie outside the PDE domain")
        spline = RectBivariateSpline(self.stock_1, self.stock_2, self.values, kx=3, ky=3, s=0)
        return float(spline.ev(spots[0], spots[1], dx=dx, dy=dy))

    def spot_greeks(self, spots):
        return {name: self.evaluate(spots, dx, dy) for name, dx, dy in (
            ("delta_1", 1, 0), ("delta_2", 0, 1), ("gamma_11", 2, 0),
            ("gamma_22", 0, 2), ("gamma_12", 1, 1))}


def _operators(axis, rate, dividend, volatility):
    n = len(axis)
    h = axis[1]-axis[0]
    d1 = sparse.diags([-np.ones(n-1), np.ones(n-1)], [-1, 1], shape=(n, n), format="lil")/(2*h)
    d2 = sparse.diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1, 0, 1], format="lil")/h**2
    d1[0, :] = d1[-1, :] = 0
    d2[0, :] = d2[-1, :] = 0
    drift = sparse.diags(axis) @ d1.tocsr()
    operator = 0.5*volatility**2*sparse.diags(axis**2) @ d2.tocsr() + (rate-dividend)*drift
    return operator-0.5*rate*sparse.eye(n), drift


def solve_adi(contract, n_space=80, n_steps=160, s_max=None, theta=1/3, damping_steps=2):
    """Modified Craig-Sneyd splitting with implicit-Euler startup damping.

    Zero-stock faces use exact one-stock prices. Upper faces use call asymptotics;
    both mesh and domain convergence must be checked before treating output as a reference.
    Puts are derived using arithmetic basket put-call parity.
    """
    for name, value, minimum in (("n_space", n_space, 8), ("n_steps", n_steps, 2),
                                 ("damping_steps", damping_steps, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(theta) or theta < 1/3 or theta > 1 or damping_steps > n_steps:
        raise ValueError("Require 1/3 <= theta <= 1 and damping_steps <= n_steps")
    if min(contract.weights) <= 0:
        raise ValueError("ADI requires two positive basket weights")
    if s_max is None:
        s_max = tuple(max(3*s, 2*contract.strike/w) for s, w in zip(contract.spots, contract.weights))
    if len(s_max) != 2 or not np.all(np.isfinite(s_max)) or any(m <= s for m, s in zip(s_max, contract.spots)):
        raise ValueError("s_max must contain two finite values strictly above the initial stocks")
    x, y = [np.linspace(0, maximum, n_space+1) for maximum in s_max]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    call = contract.changed(option="call")
    w1, w2 = call.weights

    def boundary(tau):
        result = np.zeros_like(xx)
        result[-1, :] = w1*x[-1]*np.exp(-call.dividends[0]*tau) + w2*y*np.exp(-call.dividends[1]*tau)-call.strike*np.exp(-call.rate*tau)
        result[:, -1] = w1*x*np.exp(-call.dividends[0]*tau) + w2*y[-1]*np.exp(-call.dividends[1]*tau)-call.strike*np.exp(-call.rate*tau)
        result[0, :] = black_scholes_price(w2*y, call.strike, tau, call.rate, call.volatilities[1], call.dividends[1])
        result[:, 0] = black_scholes_price(w1*x, call.strike, tau, call.rate, call.volatilities[0], call.dividends[0])
        return result

    values = np.maximum(w1*xx+w2*yy-call.strike, 0)
    if call.tau > 0:
        lx, dx = _operators(x, call.rate, call.dividends[0], call.volatilities[0])
        ly, dy = _operators(y, call.rate, call.dividends[1], call.volatilities[1])
        full = [call.correlation*call.volatilities[0]*call.volatilities[1]*sparse.kron(dx, dy, format="csr"),
                sparse.kron(lx, sparse.eye(len(y)), format="csr"),
                sparse.kron(sparse.eye(len(x)), ly, format="csr")]
        interior = np.arange(values.size).reshape(values.shape)[1:-1, 1:-1].ravel()
        rows = [matrix[interior, :].tocsr() for matrix in full]
        operators = [matrix[:, interior].tocsc() for matrix in rows]
        combined = sum(operators)
        dt = call.tau/n_steps
        identity = sparse.eye(len(interior), format="csc")
        factors = [splu(identity-theta*dt*matrix) for matrix in operators[1:]]
        damped = splu(identity-0.5*dt*combined) if damping_steps else None
        u = values.ravel()[interior]

        def forcing(tau):
            b = boundary(tau).ravel()
            return [matrix @ b for matrix in rows]

        for step in range(n_steps):
            start = step*dt
            if step < damping_steps:
                for half in (1, 2):
                    g = sum(forcing(start+half*dt/2))
                    u = damped.solve(u+dt*g/2)
                continue
            old, new = forcing(start), forcing(start+dt)
            y0 = u+dt*(combined @ u+sum(old))

            def sweep(initial):
                result = initial
                for j, factor in enumerate(factors, 1):
                    result = factor.solve(result-theta*dt*(operators[j] @ u)
                                          + theta*dt*(new[j]-old[j]))
                return result

            y2 = sweep(y0)
            correction = y0+theta*dt*(operators[0] @ (y2-u)+new[0]-old[0])
            correction += (0.5-theta)*dt*(combined @ (y2-u)+sum(new)-sum(old))
            u = sweep(correction)
        values = boundary(call.tau)
        values[1:-1, 1:-1] = u.reshape(n_space-1, n_space-1)
    if contract.option == "put":
        values = values-w1*xx*np.exp(-call.dividends[0]*call.tau)-w2*yy*np.exp(-call.dividends[1]*call.tau)+call.strike*np.exp(-call.rate*call.tau)
    return ADISolution(x, y, values, n_steps, theta, tuple(s_max))
