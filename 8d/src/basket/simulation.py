from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t

from .analytical import geometric_price, geometric_greeks
from .integration import (conditional_samples, asset_conditional_samples,
                          conditional_spot_samples, greek_bumps, GREEK_NAMES)


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def normal_points(n_points, dimension, seed, sampler="sobol"):
    _integer("n_points", n_points, 2)
    if sampler == "mc":
        return np.random.default_rng(seed).standard_normal((n_points, dimension))
    if sampler != "sobol":
        raise ValueError("sampler must be mc or sobol")
    if n_points & (n_points-1):
        raise ValueError("Sobol points per scramble must be a power of two")
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    uniform = engine.random_base2(int(n_points).bit_length()-1)
    return ndtri(np.clip(uniform, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))


def terminal_from_normals(contract, normals):
    z = np.asarray(normals, float)
    if z.ndim != 2 or z.shape[1] != 2 or not np.all(np.isfinite(z)):
        raise ValueError("normals must have shape (n, 2) and contain finite values")
    correlated = np.column_stack((z[:, 0], contract.correlation*z[:, 0]
                                  + np.sqrt(max(1-contract.correlation**2, 0))*z[:, 1]))
    s = np.asarray(contract.volatilities)
    drift = (contract.rate-np.asarray(contract.dividends)-0.5*s**2)*contract.tau
    return np.asarray(contract.spots)*np.exp(drift+s*np.sqrt(contract.tau)*correlated)


def simulate_terminal(contract, n_points=4096, seed=42, sampler="sobol"):
    """Exact GBM terminal draws; no time discretisation is needed for European payoffs."""
    return terminal_from_normals(contract, normal_points(n_points, 2, seed, sampler))


@dataclass(frozen=True)
class RandomizedEstimate:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    replicate_means: tuple
    beta: float
    production_points: int
    pilot_points: int
    normal_dimension: int
    sampler: str
    estimator: str


def _replicated(evaluate, known, dimension, n_points, n_replicates, seed,
                sampler, control, pilot_points, pilot_replicates):
    _integer("n_replicates", n_replicates, 4)
    _integer("n_points", n_points, 2)
    if sampler not in ("sobol", "mc"):
        raise ValueError("sampler must be mc or sobol")
    if sampler == "sobol" and n_points & (n_points-1):
        raise ValueError("Sobol points per scramble must be a power of two")
    if control:
        _integer("pilot_points", pilot_points, 2)
        _integer("pilot_replicates", pilot_replicates, 4)
        if sampler == "sobol" and pilot_points & (pilot_points-1):
            raise ValueError("Sobol pilot points must be a power of two")
    production_seed, pilot_seed = np.random.SeedSequence(seed).spawn(2)

    def means(child, count):
        z = normal_points(count, dimension, int(child.generate_state(1)[0]), sampler)
        y, x = evaluate(z)
        return y.mean(axis=0), x.mean(axis=0) if control else np.zeros_like(known)

    beta = np.zeros_like(known)
    if control:
        pilot = [means(child, pilot_points) for child in pilot_seed.spawn(pilot_replicates)]
        y, x = [np.array([row[i] for row in pilot]) for i in (0, 1)]
        y, x = y-y.mean(axis=0), x-x.mean(axis=0)
        variance = np.sum(x*x, axis=0)
        beta = np.divide(np.sum(x*y, axis=0), variance, out=beta,
                         where=variance > np.finfo(float).eps**2)
    values = []
    for child in production_seed.spawn(n_replicates):
        y, x = means(child, n_points)
        values.append(y-beta*(x-known))
    values = np.array(values)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/np.sqrt(n_replicates)
    radius = float(t.ppf(0.975, n_replicates-1))*se
    return mean, se, radius, values, beta


def randomized_price(contract, n_points=4096, n_replicates=16, seed=42,
                     sampler="sobol", estimator="conditional", control=False,
                     pilot_points=256, pilot_replicates=16, kind="arithmetic"):
    """Intervals use independent replicate means; pilot cost is reported separately.

    For Sobol, points within a scramble are not independent observations.
    The interval estimates sampling uncertainty, not quadrature or model error.
    """
    if estimator not in ("raw", "conditional", "asset_conditional"):
        raise ValueError("estimator must be raw, conditional or asset_conditional")
    if kind not in ("arithmetic", "geometric"):
        raise ValueError("kind must be arithmetic or geometric")
    if kind == "geometric" and control:
        raise ValueError("A geometric payoff cannot control itself")
    dimension = 2 if estimator == "raw" else 1
    def evaluate(z):
        if estimator != "raw":
            evaluator = conditional_samples if estimator == "conditional" else asset_conditional_samples
            y = evaluator(contract, z[:, 0], kind)
            x = evaluator(contract, z[:, 0], "geometric") if control else None
        else:
            terminal = terminal_from_normals(contract, z)
            discount = np.exp(-contract.rate*contract.tau)
            y = discount*contract.payoff(terminal, kind)
            x = discount*contract.payoff(terminal, "geometric") if control else None
        return y[:, None], x[:, None] if control else None
    known = np.array([geometric_price(contract) if control else 0.0])
    mean, se, radius, values, beta = _replicated(evaluate, known, dimension, n_points, n_replicates,
                                               seed, sampler, control, pilot_points, pilot_replicates)
    return RandomizedEstimate(float(mean[0]), float(se[0]), float(mean[0]-radius[0]), float(mean[0]+radius[0]),
                              tuple(float(v) for v in values[:, 0]), float(beta[0]),
                              n_points*n_replicates, pilot_points*pilot_replicates if control else 0,
                              dimension, sampler, estimator)


def randomized_greeks(contract, n_points=4096, n_replicates=16, seed=42,
                      control=False, pilot_points=256, pilot_replicates=16, bump_scale=1.0):
    """Conditional RQMC Greeks with common random numbers for parameter bumps.

    Delta/Gamma use exact conditional derivatives; the remaining five are bumped.
    Confidence intervals are marginal and exclude finite-difference bias.
    """
    bumps = greek_bumps(contract, bump_scale)
    def samples(c, z, kind):
        spot = conditional_spot_samples(c, z, kind)
        other = [sign*(conditional_samples(c.bumped(field, step, index), z, kind)
                       - conditional_samples(c.bumped(field, -step, index), z, kind))/(2*step)
                 for _, field, index, step, sign in bumps]
        return np.column_stack((spot, *other))

    def evaluate(z):
        return samples(contract, z[:, 0], "arithmetic"), samples(contract, z[:, 0], "geometric") if control else None

    known = np.zeros(len(GREEK_NAMES))
    if control:
        exact = geometric_greeks(contract)
        known[:5] = [exact[name] for name in GREEK_NAMES[:5]]
        for i, (_, field, index, step, sign) in enumerate(bumps, 5):
            known[i] = sign*(geometric_price(contract.bumped(field, step, index))
                            - geometric_price(contract.bumped(field, -step, index)))/(2*step)
    mean, se, radius, values, beta = _replicated(evaluate, known, 1, n_points, n_replicates,
                                               seed, "sobol", control, pilot_points, pilot_replicates)
    return dict(quantities={name: dict(value=float(mean[i]), standard_error=float(se[i]),
                                      ci_low=float(mean[i]-radius[i]), ci_high=float(mean[i]+radius[i]),
                                      beta=float(beta[i])) for i, name in enumerate(GREEK_NAMES)},
                replicate_means=values.tolist(), production_points=n_points*n_replicates,
                pilot_points=pilot_points*pilot_replicates if control else 0,
                normal_dimension=1, bump_scale=bump_scale)
