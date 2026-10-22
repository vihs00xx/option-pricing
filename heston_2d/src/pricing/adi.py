"""Two-stock modified Craig-Sneyd ADI on smooth sinh-clustered grids.

Each axis uses x(u) = S + alpha * sinh(c1 + (c2 - c1) u) on a uniform u-grid,
clustered around the evaluation spot, which is placed exactly on a node.
Halving u doubles the grid with every old node retained, so Richardson
extrapolation compares nested solutions. The far boundaries scale with
volatility and maturity, following the 1D Crank-Nicolson rule.

Zero-stock faces use exact one-stock Black-Scholes prices; far faces use the
deep in-the-money call asymptote. Puts follow from basket put-call parity.
Two initial intervals take two Douglas (theta = 1) half-steps each to damp the
payoff kink. Theta comes from the pricing equation at the spot node; vega, rho
and correlation sensitivities use central bumps re-solved on the same grid.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from src.black_scholes import black_scholes_price

from .contract import parameter_bumps


@dataclass(frozen=True)
class AxisLayout:
    spot: float
    alpha: float
    start: float
    stop: float
    intervals: int
    spot_index: int

    def nodes(self, refinement=1):
        n = self.intervals * refinement
        x = self.spot + self.alpha * np.sinh(self.start + (self.stop - self.start) * np.arange(n + 1) / n)
        x[0] = 0.0
        x[self.spot_index * refinement] = self.spot
        return x

    @property
    def upper(self):
        return float(self.spot + self.alpha * np.sinh(self.stop))


def axis_layout(spot, upper, intervals, alpha):
    """Grid with the spot on a node and a far boundary at least `upper`."""
    if not upper > spot > 0 or alpha <= 0 or intervals < 4:
        raise ValueError("Require upper > spot > 0, positive alpha and at least four intervals")
    start = np.arcsinh(-spot / alpha)
    target = np.arcsinh((upper - spot) / alpha)
    # Rounding the spot position down moves the far boundary outwards.
    index = int(np.clip(np.floor(intervals * -start / (target - start)), 1, intervals - 1))
    return AxisLayout(float(spot), float(alpha), float(start), float(-start * (intervals - index) / index),
                      int(intervals), index)


def default_layout(contract, intervals=80, tail_sd=5.0, clustering=0.5):
    """Far boundary: the deep in-the-money asymptote must hold along each face,
    so w_i S_max >= K far beyond the spread of the stock over the life."""
    c = contract
    if c.dimension != 2 or min(c.weights) <= 0 or c.tau <= 0:
        raise ValueError("ADI requires two stocks, positive weights and positive maturity")
    layouts = []
    for i in range(2):
        s, sigma, q = c.spots[i], c.volatilities[i], c.dividends[i]
        width = max(np.log(4.0), abs(c.rate - q - 0.5 * sigma**2) * c.tau + tail_sd * sigma * np.sqrt(c.tau))
        upper = max(s, c.strike / c.weights[i]) * np.exp(width)
        alpha = clustering * s * np.clip(sigma * np.sqrt(c.tau), 0.02, 0.5)
        layouts.append(axis_layout(s, upper, intervals, alpha))
    return tuple(layouts)


def _stencils(x):
    """Three-point first and second derivative weights at interior nodes."""
    hm, hp = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    first = (-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp)))
    second = (2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp)))
    return first, second


def _along(weights, values, axis):
    """Apply interior three-point weights along an axis of a full grid array."""
    lower, centre, upper = weights
    if axis == 0:
        return lower[:, None] * values[:-2] + centre[:, None] * values[1:-1] + upper[:, None] * values[2:]
    return lower * values[:, :-2] + centre * values[:, 1:-1] + upper * values[:, 2:]


@dataclass
class ADISolution:
    stock_1: np.ndarray
    stock_2: np.ndarray
    values: np.ndarray
    spot_index: tuple
    contract: object

    def spot_values(self):
        """Price, spot Greeks and PDE Theta at the spot node, keyed like the RQMC engine."""
        c, V = self.contract, self.values
        x, y = self.stock_1, self.stock_2
        i, j = self.spot_index
        (fx, sx), (fy, sy) = _stencils(x), _stencils(y)
        dx = _along(fx, V, 0)[i - 1, j]
        dy = _along(fy, V, 1)[i, j - 1]
        gxx = _along(sx, V, 0)[i - 1, j]
        gyy = _along(sy, V, 1)[i, j - 1]
        gxy = _along(fy, _along(fx, V, 0), 1)[i - 1, j - 1]
        (s1, s2), (q1, q2), rho = c.volatilities, c.dividends, c.correlation[0][1]
        generator = (0.5 * s1**2 * x[i]**2 * gxx + 0.5 * s2**2 * y[j]**2 * gyy
                     + rho * s1 * s2 * x[i] * y[j] * gxy + (c.rate - q1) * x[i] * dx
                     + (c.rate - q2) * y[j] * dy - c.rate * V[i, j])
        return dict(price=float(V[i, j]), delta_1=float(dx), delta_2=float(dy), gamma_11=float(gxx),
                    gamma_12=float(gxy), gamma_22=float(gyy), theta=float(-generator))


def solve_adi(contract, layout=None, steps=None, refinement=1, theta=1 / 3, damping_steps=2):
    c = contract
    layout = default_layout(c) if layout is None else layout
    x, y = layout[0].nodes(refinement), layout[1].nodes(refinement)
    steps = layout[0].intervals * refinement if steps is None else int(steps) * refinement
    if not 1 / 3 <= theta <= 1 or not 0 <= damping_steps <= steps:
        raise ValueError("Require 1/3 <= theta <= 1 and 0 <= damping_steps <= steps")
    (w1, w2), (q1, q2), (s1, s2) = c.weights, c.dividends, c.volatilities
    r, K, rho = c.rate, c.strike, c.correlation[0][1]
    X, Y = np.meshgrid(x, y, indexing="ij")
    (fx, sx), (fy, sy) = _stencils(x), _stencils(y)

    def directional(first, second, nodes, sigma, q):
        inner = nodes[1:-1]
        diffusion, drift = 0.5 * sigma**2 * inner**2, (r - q) * inner
        return (diffusion * second[0] + drift * first[0], diffusion * second[1] + drift * first[1] - 0.5 * r,
                diffusion * second[2] + drift * first[2])

    ax, ay = directional(fx, sx, x, s1, q1), directional(fy, sy, y, s2, q2)
    mixed = rho * s1 * s2 * np.outer(x[1:-1], y[1:-1])

    def boundary(tau):
        B = np.zeros_like(X)
        B[-1, :] = w1 * x[-1] * np.exp(-q1 * tau) + w2 * y * np.exp(-q2 * tau) - K * np.exp(-r * tau)
        B[:, -1] = w1 * x * np.exp(-q1 * tau) + w2 * y[-1] * np.exp(-q2 * tau) - K * np.exp(-r * tau)
        B[0, :] = black_scholes_price(w2 * y, K, tau, r, s2, q2)
        B[:, 0] = black_scholes_price(w1 * x, K, tau, r, s1, q1)
        return B

    def terms(U, B):
        """Mixed, stock-1 and stock-2 operator values with boundary data B."""
        V = B.copy()
        V[1:-1, 1:-1] = U
        return (mixed * _along(fy, _along(fx, V, 0), 1), _along(ax, V, 0)[:, 1:-1],
                _along(ay, V, 1)[1:-1, :])

    zero = np.zeros((len(x) - 2, len(y) - 2))

    def implicit(weights, rhs, factor, axis):
        lower, centre, upper = weights
        band = np.zeros((3, len(centre)))
        band[0, 1:] = -factor * upper[:-1]
        band[1] = 1 - factor * centre
        band[2, :-1] = -factor * lower[1:]
        if axis == 0:
            return solve_banded((1, 1), band, rhs)
        return solve_banded((1, 1), band, rhs.T).T

    U = np.maximum(w1 * X + w2 * Y - K, 0)[1:-1, 1:-1]
    dt = c.tau / steps
    tau = 0.0
    for step in range(steps):
        damped = step < damping_steps
        for h in ((dt / 2, dt / 2) if damped else (dt,)):
            old_b, new_b = boundary(tau), boundary(tau + h)
            old = terms(U, old_b)
            new_forcing = terms(zero, new_b)
            weight = 1.0 if damped else theta

            def sweep(start):
                result = start
                for j, (ops, axis) in enumerate(((ax, 0), (ay, 1)), 1):
                    result = implicit(ops, result - weight * h * old[j] + weight * h * new_forcing[j],
                                      weight * h, axis)
                return result

            y0 = U + h * sum(old)
            y2 = sweep(y0)
            if not damped:
                new = terms(y2, new_b)
                y0 = y0 + theta * h * (new[0] - old[0]) + (0.5 - theta) * h * (sum(new) - sum(old))
                y2 = sweep(y0)
            U, tau = y2, tau + h
    V = boundary(c.tau)
    V[1:-1, 1:-1] = U
    if c.option == "put":
        V = V - w1 * X * np.exp(-q1 * c.tau) - w2 * Y * np.exp(-q2 * c.tau) + K * np.exp(-r * c.tau)
    return ADISolution(x, y, V, (layout[0].spot_index * refinement, layout[1].spot_index * refinement), c)


def adi_values(contract, intervals=80, steps=None, richardson=False, parameter_greeks=False,
               tail_sd=5.0, clustering=0.5, bump_scale=1.0):
    """Values at the spot; Richardson combines the grid and its nested doubling."""
    layout = default_layout(contract, intervals, tail_sd, clustering)

    def solve(c, refinement):
        result = solve_adi(c, layout, steps, refinement).spot_values()
        if parameter_greeks:
            for name, field, index, step, sign in parameter_bumps(contract, bump_scale):
                if field == "tau":
                    continue
                upper = solve_adi(c.bumped(field, step, index), layout, steps, refinement).spot_values()["price"]
                lower = solve_adi(c.bumped(field, -step, index), layout, steps, refinement).spot_values()["price"]
                result[name] = sign * (upper - lower) / (2 * step)
        return result

    coarse = solve(contract, 1)
    if not richardson:
        return coarse
    fine = solve(contract, 2)
    return {name: (4 * fine[name] - coarse[name]) / 3 for name in fine}
