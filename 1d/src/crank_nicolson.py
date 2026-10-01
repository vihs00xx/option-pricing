from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import diags
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class CrankNicolsonGrid:
    spots: np.ndarray
    values: np.ndarray
    maturity: float
    space_step: float
    time_step: float
    option: str


def _validate(K, tau, r, sigma, q, option, s_max, n_space, n_time,
              rannacher_steps, mesh, concentration):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    for name, value in (("K", K), ("tau", tau), ("r", r), ("sigma", sigma),
                        ("q", q), ("s_max", s_max)):
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar")
    if K <= 0 or tau < 0 or sigma <= 0 or s_max <= 0:
        raise ValueError("K, sigma, and s_max must be positive, and tau must be non-negative")
    for name, value, minimum in (("n_space", n_space, 4), ("n_time", n_time, 1),
                                 ("rannacher_steps", rannacher_steps, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if rannacher_steps > n_time:
        raise ValueError("rannacher_steps cannot exceed n_time")
    if mesh not in ("uniform", "strike_clustered"):
        raise ValueError("mesh must be 'uniform' or 'strike_clustered'")
    if not np.isscalar(concentration) or not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be a positive finite scalar")
    if mesh == "strike_clustered" and s_max <= K:
        raise ValueError("s_max must exceed K for the strike-clustered mesh")


def _boundaries(K, maturity, r, q, s_max, option):
    if option == "call":
        return 0.0, s_max * np.exp(-q * maturity) - K * np.exp(-r * maturity)
    return K * np.exp(-r * maturity), 0.0


def _factor(lower, diagonal, upper):
    return splu(diags((lower, diagonal, upper), (-1, 0, 1), format="csc"))


def _default_s_max(S, K, tau, sigma):
    scale = sigma * np.sqrt(tau) if np.isfinite(tau) and tau >= 0 and np.isfinite(sigma) else 0.0
    tail_factor = max(4.0, 2.0 + 5.0 * scale)
    return max(tail_factor * K, 2.0 * S)


def _clustered_s_max(S, K, tau, r, sigma, q):
    log_width = max(np.log(4.0), abs(r - q - 0.5 * sigma**2) * tau
                    + 5.0 * sigma * np.sqrt(tau))
    return max(S, K) * np.exp(log_width)


def _spots(K, s_max, n_space, mesh, concentration):
    if mesh == "uniform":
        return np.linspace(0.0, s_max, n_space + 1)
    n_left = n_space // 2
    n_right = n_space - n_left
    scale = np.sinh(concentration)
    left_u = np.linspace(0.0, 1.0, n_left + 1)
    right_u = np.linspace(0.0, 1.0, n_right + 1)
    left = K * (1.0 - np.sinh(concentration * (1.0 - left_u)) / scale)
    right = K + (s_max - K) * np.sinh(concentration * right_u) / scale
    return np.concatenate((left[:-1], right))


def crank_nicolson_grid(K, tau, r, sigma, q=0.0, option="call", *,
                        s_max=None, n_space=400, n_time=400, rannacher_steps=2,
                        mesh="uniform", concentration=5.0):
    if s_max is None:
        s_max = (_clustered_s_max(K, K, tau, r, sigma, q)
                 if mesh == "strike_clustered" else _default_s_max(0.0, K, tau, sigma))
    else:
        s_max = float(s_max)
    _validate(K, tau, r, sigma, q, option, s_max, n_space, n_time,
              rannacher_steps, mesh, concentration)
    spots = _spots(K, s_max, n_space, mesh, concentration)
    sign = 1 if option == "call" else -1
    values = np.maximum(sign * (spots - K), 0.0)
    dS = float(np.max(np.diff(spots)))
    if tau == 0:
        return CrankNicolsonGrid(spots, values, tau, dS, 0.0, option)

    dt = tau / n_time
    interior = spots[1:-1]
    h_left = interior - spots[:-2]
    h_right = spots[2:] - interior
    first_lower = -h_right / (h_left * (h_left + h_right))
    first_centre = (h_right - h_left) / (h_left * h_right)
    first_upper = h_left / (h_right * (h_left + h_right))
    second_lower = 2.0 / (h_left * (h_left + h_right))
    second_centre = -2.0 / (h_left * h_right)
    second_upper = 2.0 / (h_right * (h_left + h_right))
    diffusion = 0.5 * sigma**2 * interior**2
    convection = (r - q) * interior
    lower_weight = diffusion * second_lower + convection * first_lower
    centre_weight = diffusion * second_centre + convection * first_centre - r
    upper_weight = diffusion * second_upper + convection * first_upper

    cn = _factor(-0.5 * dt * lower_weight[1:],
                 1 - 0.5 * dt * centre_weight,
                 -0.5 * dt * upper_weight[:-1])
    half_dt = 0.5 * dt
    implicit_half = _factor(-half_dt * lower_weight[1:],
                            1 - half_dt * centre_weight,
                            -half_dt * upper_weight[:-1])

    current_maturity = 0.0
    for step in range(n_time):
        if step < rannacher_steps:
            for _ in range(2):
                next_maturity = current_maturity + half_dt
                left, right = _boundaries(K, next_maturity, r, q, s_max, option)
                rhs = values[1:-1].copy()
                rhs[0] += half_dt * lower_weight[0] * left
                rhs[-1] += half_dt * upper_weight[-1] * right
                values[1:-1] = implicit_half.solve(rhs)
                values[0], values[-1] = left, right
                current_maturity = next_maturity
        else:
            next_maturity = current_maturity + dt
            left, right = _boundaries(K, next_maturity, r, q, s_max, option)
            rhs = values[1:-1] + 0.5 * dt * (
                lower_weight * values[:-2]
                + centre_weight * values[1:-1]
                + upper_weight * values[2:]
            )
            rhs[0] += 0.5 * dt * lower_weight[0] * left
            rhs[-1] += 0.5 * dt * upper_weight[-1] * right
            values[1:-1] = cn.solve(rhs)
            values[0], values[-1] = left, right
            current_maturity = next_maturity
    return CrankNicolsonGrid(spots, values, tau, dS, dt, option)


def _surface(S, K, tau, r, sigma, q, option, s_max, n_space, n_time,
             rannacher_steps, mesh, concentration):
    if s_max is None:
        upper = (_clustered_s_max(S, K, tau, r, sigma, q)
                 if mesh == "strike_clustered" else _default_s_max(S, K, tau, sigma))
    else:
        upper = float(s_max)
    if not np.isfinite(S) or S < 0 or S > upper:
        raise ValueError("S must be finite and lie between zero and s_max")
    grid = crank_nicolson_grid(K, tau, r, sigma, q, option, s_max=upper,
                              n_space=n_space, n_time=n_time,
                              rannacher_steps=rannacher_steps, mesh=mesh,
                              concentration=concentration)
    return grid, CubicSpline(grid.spots, grid.values)


def crank_nicolson_price(S, K, tau, r, sigma, q=0.0, option="call", *,
                         s_max=None, n_space=400, n_time=400, rannacher_steps=2,
                         richardson=False, mesh="uniform", concentration=5.0):
    if not isinstance(richardson, (bool, np.bool_)):
        raise ValueError("richardson must be a boolean")
    if richardson:
        upper = (_clustered_s_max(S, K, tau, r, sigma, q)
                 if s_max is None and mesh == "strike_clustered"
                 else _default_s_max(S, K, tau, sigma)
                 if s_max is None else float(s_max))
        coarse = crank_nicolson_price(
            S, K, tau, r, sigma, q, option, s_max=upper,
            n_space=n_space, n_time=n_time, rannacher_steps=rannacher_steps,
            mesh=mesh, concentration=concentration,
        )
        fine = crank_nicolson_price(
            S, K, tau, r, sigma, q, option, s_max=upper,
            n_space=2 * n_space, n_time=2 * n_time, rannacher_steps=rannacher_steps,
            mesh=mesh, concentration=concentration,
        )
        return float((4 * fine - coarse) / 3)
    _, spline = _surface(S, K, tau, r, sigma, q, option, s_max,
                         n_space, n_time, rannacher_steps, mesh, concentration)
    return float(spline(S))


def crank_nicolson_values(S, K, tau, r, sigma, q=0.0, option="call", *,
                          s_max=None, n_space=400, n_time=400, rannacher_steps=2,
                          volatility_bump=0.005, rate_bump=0.0001,
                          richardson=False, mesh="uniform", concentration=5.0):
    if not isinstance(richardson, (bool, np.bool_)):
        raise ValueError("richardson must be a boolean")
    if richardson:
        upper = (_clustered_s_max(S, K, tau, r, sigma, q)
                 if s_max is None and mesh == "strike_clustered"
                 else _default_s_max(S, K, tau, sigma)
                 if s_max is None else float(s_max))
        arguments = dict(
            s_max=upper, rannacher_steps=rannacher_steps,
            volatility_bump=volatility_bump, rate_bump=rate_bump,
            mesh=mesh, concentration=concentration,
        )
        coarse = crank_nicolson_values(
            S, K, tau, r, sigma, q, option,
            n_space=n_space, n_time=n_time, **arguments,
        )
        fine = crank_nicolson_values(
            S, K, tau, r, sigma, q, option,
            n_space=2 * n_space, n_time=2 * n_time, **arguments,
        )
        return {name: (4 * fine[name] - coarse[name]) / 3 for name in fine}
    if S <= 0 or tau <= 0:
        raise ValueError("Price and Greeks together require positive S and tau")
    if not np.isfinite(volatility_bump) or not 0 < volatility_bump < 1:
        raise ValueError("volatility_bump must be finite and between zero and one")
    if not np.isfinite(rate_bump) or rate_bump <= 0:
        raise ValueError("rate_bump must be positive and finite")
    upper = (_clustered_s_max(S, K, tau, r, sigma, q)
             if s_max is None and mesh == "strike_clustered"
             else _default_s_max(S, K, tau, sigma)
             if s_max is None else float(s_max))
    _, spline = _surface(S, K, tau, r, sigma, q, option, upper,
                         n_space, n_time, rannacher_steps, mesh, concentration)
    price = float(spline(S))
    delta = float(spline(S, 1))
    gamma = float(spline(S, 2))
    vol_step = volatility_bump * sigma
    plus_vol = crank_nicolson_price(S, K, tau, r, sigma + vol_step, q, option,
                                    s_max=upper, n_space=n_space, n_time=n_time,
                                    rannacher_steps=rannacher_steps, mesh=mesh,
                                    concentration=concentration)
    minus_vol = crank_nicolson_price(S, K, tau, r, sigma - vol_step, q, option,
                                     s_max=upper, n_space=n_space, n_time=n_time,
                                     rannacher_steps=rannacher_steps, mesh=mesh,
                                     concentration=concentration)
    plus_rate = crank_nicolson_price(S, K, tau, r + rate_bump, sigma, q, option,
                                     s_max=upper, n_space=n_space, n_time=n_time,
                                     rannacher_steps=rannacher_steps, mesh=mesh,
                                     concentration=concentration)
    minus_rate = crank_nicolson_price(S, K, tau, r - rate_bump, sigma, q, option,
                                      s_max=upper, n_space=n_space, n_time=n_time,
                                      rannacher_steps=rannacher_steps, mesh=mesh,
                                      concentration=concentration)
    vega = (plus_vol - minus_vol) / (2 * vol_step)
    rho = (plus_rate - minus_rate) / (2 * rate_bump)
    theta = -0.5 * sigma**2 * S**2 * gamma - (r - q) * S * delta + r * price
    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


def crank_nicolson_greeks(S, K, tau, r, sigma, q=0.0, option="call", **kwargs):
    values = crank_nicolson_values(S, K, tau, r, sigma, q, option, **kwargs)
    values.pop("price")
    return values
