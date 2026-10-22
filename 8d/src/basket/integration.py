from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

from .analytical import geometric_parameters, lognormal_payoff


@dataclass(frozen=True)
class QuadratureEstimate:
    price: float
    integration_error: float
    tail_bound: float
    evaluations: int


GREEK_NAMES = ("delta_1", "delta_2", "gamma_11", "gamma_22", "gamma_12",
               "vega_1", "vega_2", "theta", "rho", "correlation")


def greek_bumps(contract, scale=1.0):
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("bump scale must lie in (0, 1]")
    return [(name, field, index, step*scale, sign) for name, field, index, step, sign in (
        ("vega_1", "volatilities", 0, min(1e-4, contract.volatilities[0]/4), 1),
        ("vega_2", "volatilities", 1, min(1e-4, contract.volatilities[1]/4), 1),
        ("theta", "tau", None, min(1e-5, contract.tau/1000), -1),
        ("rho", "rate", None, 1e-4, 1),
        ("correlation", "correlation", None, min(1e-4, (1-abs(contract.correlation))/4), 1))]


def _require_regular_greeks(c):
    if c.tau <= 0 or min(c.spots) <= 0 or min(c.weights) <= 0 or min(c.volatilities) <= 0 or abs(c.correlation) >= 1:
        raise ValueError("Arithmetic Greeks require positive spots, weights, volatilities and tau, and |correlation| < 1")


def conditional_spot_samples(c, z, kind="arithmetic"):
    _require_regular_greeks(c)
    z = np.atleast_1d(np.asarray(z, float))
    variance, loading = common_factor(c)
    sd = np.sqrt(variance)
    drift = (c.rate-np.asarray(c.dividends)-0.5*np.asarray(c.volatilities)**2)*c.tau
    discount = np.exp(-c.rate*c.tau)
    if kind == "geometric":
        m = np.dot(c.weights, np.log(c.spots)+drift)+np.dot(c.weights, loading)*z
        d1 = (m-np.log(c.strike)+sd**2)/sd
        sign = 1 if c.option == "call" else -1
        mean = np.exp(m+sd**2/2)
        pm = sign*discount*mean*ndtr(sign*d1)
        pmm = pm+discount*mean*np.exp(-d1*d1/2)/(np.sqrt(2*np.pi)*sd)
        w1, w2 = c.weights
        s1, s2 = c.spots
        return np.column_stack((pm*w1/s1, pm*w2/s2,
                                (pmm*w1*w1-pm*w1)/s1**2,
                                (pmm*w2*w2-pm*w2)/s2**2, pmm*w1*w2/(s1*s2)))
    if kind != "arithmetic":
        raise ValueError("kind must be arithmetic or geometric")
    spot_derivatives = np.asarray(c.weights)*np.exp(drift+z[:, None]*loading)
    basket = spot_derivatives @ np.asarray(c.spots)
    d1 = (np.log(basket/c.strike)+variance)/sd
    sign = 1 if c.option == "call" else -1
    slope = sign*discount*np.exp(variance/2)*ndtr(sign*d1)
    curvature = discount*np.exp(variance/2)*np.exp(-d1*d1/2)/(np.sqrt(2*np.pi)*basket*sd)
    h1, h2 = spot_derivatives.T
    return np.column_stack((slope*h1, slope*h2, curvature*h1*h1,
                            curvature*h2*h2, curvature*h1*h2))


def common_factor(c):
    """Cov(log S_T) = variance * 11' + loading * loading' in two dimensions."""
    a = np.asarray(c.volatilities)*np.sqrt(c.tau)
    cross = c.correlation*a[0]*a[1]
    difference = a[0]**2+a[1]**2-2*cross
    if difference <= np.finfo(float).eps*max(a[0]**2+a[1]**2, 1e-300):
        return float(a[0]**2), np.zeros(2)
    loading = np.array([a[0]**2-cross, cross-a[1]**2])/np.sqrt(difference)
    variance = a[0]**2*a[1]**2*max(1-c.correlation**2, 0)/difference
    return float(variance), loading


def _orientation(contract):
    if contract.volatilities[0] == 0 and contract.volatilities[1] > 0:
        return contract.changed(**{name: getattr(contract, name)[::-1]
                                   for name in ("spots", "weights", "volatilities", "dividends")})
    return contract


def asset_conditional_samples(contract, z, kind="arithmetic"):
    c = _orientation(contract)
    z = np.asarray(z, float)
    a = np.asarray(c.volatilities) * np.sqrt(c.tau)
    drift = (c.rate - np.asarray(c.dividends) - 0.5 * np.asarray(c.volatilities)**2) * c.tau
    independent_variance = a[0]**2 * max(1 - c.correlation**2, 0)
    if kind == "arithmetic":
        with np.errstate(divide="ignore"):
            m = np.log(c.weights[0] * c.spots[0]) + drift[0] + c.correlation * a[0] * z
        other = c.weights[1] * c.spots[1] * np.exp(drift[1] + a[1] * z)
        value = lognormal_payoff(m, independent_variance, c.strike - other, c.option)
    elif kind == "geometric":
        active = np.asarray(c.weights) > 0
        with np.errstate(divide="ignore"):
            base = np.asarray(c.weights)[active] @ np.log(np.asarray(c.spots)[active])
        m = base + np.dot(c.weights, drift) + (c.weights[0] * c.correlation * a[0]
                                              + c.weights[1] * a[1]) * z
        value = lognormal_payoff(m, c.weights[0]**2 * independent_variance, c.strike, c.option)
    else:
        raise ValueError("kind must be arithmetic or geometric")
    return np.exp(-c.rate * c.tau) * value


def conditional_samples(contract, z, kind="arithmetic"):
    c = contract
    if min(c.volatilities) == 0 or abs(c.correlation) == 1 or c.tau == 0:
        return asset_conditional_samples(c, z, kind)
    z = np.asarray(z, float)
    variance, loading = common_factor(c)
    drift = (c.rate-np.asarray(c.dividends)-0.5*np.asarray(c.volatilities)**2)*c.tau
    if kind == "arithmetic":
        basket = np.sum(np.asarray(c.weights)*np.asarray(c.spots)*np.exp(drift+z[..., None]*loading), axis=-1)
        with np.errstate(divide="ignore"):
            m = np.log(basket)
    elif kind == "geometric":
        active = np.asarray(c.weights) > 0
        with np.errstate(divide="ignore"):
            base = np.dot(np.asarray(c.weights)[active], np.log(np.asarray(c.spots)[active]))
        m = base+np.dot(c.weights, drift)+np.dot(c.weights, loading)*z
    else:
        raise ValueError("kind must be arithmetic or geometric")
    return np.exp(-c.rate*c.tau)*lognormal_payoff(m, variance, c.strike, c.option)


def _breakpoints(contract, limit):
    c = _orientation(contract)
    points = [0.0]
    a = np.asarray(c.volatilities) * np.sqrt(c.tau)
    drift = (c.rate - np.asarray(c.dividends) - 0.5 * np.asarray(c.volatilities)**2) * c.tau
    if a[1] > 0 and c.weights[1] * c.spots[1] > 0:
        points.append((np.log(c.strike / (c.weights[1] * c.spots[1])) - drift[1]) / a[1])
    if a[0]**2 * (1 - c.correlation**2) == 0:
        def residual(z):
            return (c.weights[0] * c.spots[0] * np.exp(drift[0] + c.correlation * a[0] * z)
                    + c.weights[1] * c.spots[1] * np.exp(drift[1] + a[1] * z) - c.strike)
        grid = np.linspace(-limit, limit, 257)
        for left, right in zip(grid[:-1], grid[1:]):
            if residual(left) * residual(right) < 0:
                points.append(brentq(residual, left, right))
    return sorted(set([-limit, limit] + [p for p in points if -limit < p < limit]))


def _integrate(contract, function, atol, rtol, limit):
    edges = _breakpoints(contract, limit)
    value = error = 0.0
    evaluations = 0
    for left, right in zip(edges[:-1], edges[1:]):
        output = quad(lambda z: float(function(z)) * np.exp(-z*z/2) / np.sqrt(2*np.pi),
                      left, right, epsabs=atol/(len(edges)-1), epsrel=rtol,
                      limit=300, full_output=True)
        if len(output) != 3:
            raise RuntimeError(f"Quadrature failed to meet tolerance: {output[3]}")
        result, err, info = output
        value += result
        error += err
        evaluations += info["neval"]
    return value, error, evaluations


def quadrature_price(contract, kind="arithmetic", atol=1e-10, rtol=1e-11, limit=12.0, conditioning="common"):
    if kind not in ("arithmetic", "geometric"):
        raise ValueError("kind must be arithmetic or geometric")
    if not np.all(np.isfinite([atol, rtol, limit])) or min(atol, rtol) <= 0 or limit < 6:
        raise ValueError("Positive tolerances and a finite normal limit >= 6 are required")
    if contract.tau == 0:
        return QuadratureEstimate(float(contract.payoff(np.asarray(contract.spots), kind)), 0, 0, 0)
    if conditioning not in ("common", "asset"):
        raise ValueError("conditioning must be common or asset")
    evaluator = conditional_samples if conditioning == "common" else asset_conditional_samples
    value, error, evaluations = _integrate(contract, lambda z: evaluator(contract, z, kind),
                                           atol, rtol, limit)
    c = _orientation(contract)
    if conditioning == "common" and min(c.volatilities) > 0 and abs(c.correlation) < 1:
        _, shifts = common_factor(c)
    else:
        shifts = np.asarray(c.volatilities) * np.sqrt(c.tau) * np.array([c.correlation, 1])
    if c.option == "put":
        tail = 2 * c.strike * np.exp(-c.rate*c.tau) * ndtr(-limit)
    elif kind == "arithmetic":
        means = np.asarray(c.weights) * np.asarray(c.spots) * np.exp(-np.asarray(c.dividends)*c.tau)
        tail = np.dot(means, ndtr(shifts-limit) + ndtr(-limit-shifts))
    else:
        m, v, _ = geometric_parameters(c)
        shift = np.dot(c.weights, shifts)
        tail = np.exp(m+v/2-c.rate*c.tau) * (ndtr(shift-limit)+ndtr(-limit-shift))
    return QuadratureEstimate(float(value), float(error), float(tail), evaluations)


def arithmetic_greeks(contract, bump_scale=1.0):
    """Spot Greeks are integrated analytically; other Greeks use central quadrature bumps.

    Bump Greeks have discretisation bias; compare bump_scale=1 and 0.5 before using labels.
    Vega/rho/correlation use unit decimal changes; theta is calendar time per year.
    """
    _require_regular_greeks(contract)
    result = {name: _integrate(contract, lambda z: conditional_spot_samples(contract, z)[0, i], 1e-10, 1e-11, 12)[0]
              for i, name in enumerate(GREEK_NAMES[:5])}
    for name, field, index, step, sign in greek_bumps(contract, bump_scale):
        upper = quadrature_price(contract.bumped(field, step, index)).price
        lower = quadrature_price(contract.bumped(field, -step, index)).price
        result[name] = sign*(upper-lower)/(2*step)
    return result
