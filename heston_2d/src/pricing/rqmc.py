"""Replicated randomized quasi-Monte Carlo over the conditional residual factors."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t

from .conditional import sample_values, value_names
from .geometric import geometric_known_values, geometric_price


@dataclass(frozen=True)
class RQMCEstimate:
    """Intervals use independent replicate means and exclude bump bias.

    Points inside one scrambled Sobol net are not independent observations.
    Intervals are marginal, not simultaneous.
    """

    names: tuple
    values: dict
    standard_errors: dict
    intervals: dict
    replicate_means: np.ndarray
    beta: np.ndarray | None
    production_points: int
    pilot_points: int
    sampled_dimension: int
    sampler: str
    direction: str
    control: bool


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def normal_points(n_points, dimension, seed, sampler="sobol"):
    if sampler == "mc":
        return np.random.default_rng(seed).standard_normal((n_points, dimension))
    if sampler != "sobol":
        raise ValueError("sampler must be sobol or mc")
    if n_points & (n_points - 1):
        raise ValueError("Sobol points per replicate must be a power of two")
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    uniform = engine.random_base2(int(n_points).bit_length() - 1)
    return ndtri(np.clip(uniform, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))


def rqmc_estimate(contract, n_points=4096, n_replicates=16, seed=42, *, direction="volatility",
                  greeks=True, control=False, pilot_points=256, pilot_replicates=16,
                  sampler="sobol", bump_scale=1.0):
    """Price, and optionally every sensitivity, of an arithmetic basket option.

    The geometric control uses the same residual draws and bumps; its exact mean
    comes from the closed form. Control coefficients are fitted on independent
    pilot replicate means, so the production estimate stays unbiased; pilot
    cost is reported separately.
    """
    c = contract
    _integer("n_points", n_points, 2)
    _integer("n_replicates", n_replicates, 4)
    if control:
        _integer("pilot_points", pilot_points, 2)
        _integer("pilot_replicates", pilot_replicates, 4)
    names = tuple(value_names(c, greeks))
    k = c.dimension - 1

    def means(seed_sequence, count):
        U = np.zeros((1, 0)) if k == 0 else normal_points(
            count, k, int(seed_sequence.generate_state(1)[0]), sampler)
        y = sample_values(c, U, "arithmetic", direction, greeks, bump_scale).mean(axis=0)
        x = sample_values(c, U, "geometric", direction, greeks, bump_scale).mean(axis=0) if control else None
        return y, x

    production_seed, pilot_seed = np.random.SeedSequence(seed).spawn(2)
    beta = None
    if control:
        known = geometric_known_values(c, bump_scale) if greeks else np.array([geometric_price(c)])
        pilot = [means(child, pilot_points) for child in pilot_seed.spawn(pilot_replicates)]
        y = np.array([row[0] for row in pilot])
        x = np.array([row[1] for row in pilot])
        scale = np.abs(x.mean(axis=0))
        y, x = y - y.mean(axis=0), x - x.mean(axis=0)
        variance = np.sum(x * x, axis=0)
        # A control that is constant up to rounding (for example the geometric
        # basket of a fully symmetric contract) carries no information.
        informative = np.sqrt(variance / (pilot_replicates - 1)) > 1e-10 * scale + np.finfo(float).tiny
        beta = np.divide(np.sum(x * y, axis=0), variance, out=np.zeros(len(names)), where=informative)
    replicates = []
    for child in production_seed.spawn(n_replicates):
        y, x = means(child, n_points)
        replicates.append(y - beta * (x - known) if control else y)
    replicates = np.array(replicates)
    mean = replicates.mean(axis=0)
    se = replicates.std(axis=0, ddof=1) / np.sqrt(n_replicates)
    radius = float(t.ppf(0.975, n_replicates - 1)) * se
    return RQMCEstimate(
        names, dict(zip(names, map(float, mean))), dict(zip(names, map(float, se))),
        {name: (float(m - r), float(m + r)) for name, m, r in zip(names, mean, radius)},
        replicates, beta, n_points * n_replicates,
        pilot_points * pilot_replicates if control else 0, k, sampler, direction, bool(control))
