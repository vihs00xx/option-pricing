from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, t


METHODS = ("plain", "antithetic", "control", "antithetic_control", "sobol", "sobol_control")


@dataclass
class SampleEstimate:
    mean: np.ndarray
    se: np.ndarray
    critical: float
    observations: int
    pilot_samples: int
    beta: np.ndarray | None


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        size = len(values)
        mean = values.mean(axis=0)
        total = self.count + size
        delta = mean - self.mean
        self.m2 = self.m2 + np.sum((values - mean)**2, axis=0) + delta**2 * self.count * size / total
        self.mean = self.mean + delta * size / total
        self.count = total

    def se(self):
        return np.sqrt(self.m2 / (self.count - 1) / self.count)


class Regression:
    def __init__(self):
        self.count = 0
        self.mx = self.my = self.xx = self.xy = 0.0

    def update(self, x, y):
        size = len(x)
        mx, my = x.mean(axis=0), y.mean(axis=0)
        dx, dy = mx - self.mx, my - self.my
        total = self.count + size
        weight = self.count * size / total
        self.xx = self.xx + np.sum((x - mx)**2, axis=0) + dx**2 * weight
        self.xy = self.xy + np.sum((x - mx) * (y - my), axis=0) + dx * dy * weight
        self.mx = self.mx + dx * size / total
        self.my = self.my + dy * size / total
        self.count = total

    def coefficient(self):
        return np.divide(self.xy, self.xx, out=np.zeros_like(self.xy), where=self.xx > 0)


def integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_sampling(n_samples, method, batch_size, pilot_samples, n_scrambles):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    integer("n_samples", n_samples, 2)
    integer("batch_size", batch_size, 1)
    paired = method.startswith("antithetic")
    controlled = "control" in method
    production = int(n_samples)
    if controlled:
        integer("pilot_samples", pilot_samples, 4 if paired else 2)
        production -= pilot_samples
        if production < (4 if paired else 2):
            raise ValueError("The budget leaves too few production observations after the pilot")
        if paired and pilot_samples % 2:
            raise ValueError("Antithetic pilot samples must be even")
    if paired and (production < 4 or production % 2):
        raise ValueError("Antithetic sampling requires even production samples >= 4")
    if method.startswith("sobol"):
        integer("n_scrambles", n_scrambles, 2)
        for count in ([production, pilot_samples] if controlled else [production]):
            per_scramble = int(count) // int(n_scrambles)
            if count % n_scrambles or per_scramble < 2 or per_scramble & (per_scramble - 1):
                raise ValueError("Each Sobol pilot/production scramble must contain a power of two >= 2 points")


def sobol_normals(seed, size):
    engine = qmc.Sobol(d=1, scramble=True, seed=int(seed.generate_state(1)[0]))
    u = engine.random_base2(int(size).bit_length() - 1)[:, 0]
    return ndtri(np.clip(u, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))


def estimate(evaluate, control_mean, n_samples, method, seed, batch_size, pilot_samples, n_scrambles):
    paired = method.startswith("antithetic")
    controlled = "control" in method
    quasi = method.startswith("sobol")
    pilot = int(pilot_samples) if controlled else 0
    production = int(n_samples) - pilot
    root = np.random.SeedSequence(seed)
    production_seed, pilot_seed = root.spawn(2) if controlled or not quasi else (root, None)
    regression, moments = Regression(), Moments()
    beta = None

    def observations(z):
        x, y = evaluate(z)
        if paired:
            opposite_x, opposite_y = evaluate(-z)
            x, y = (x + opposite_x) / 2, (y + opposite_y) / 2
        return x, y

    if controlled:
        if quasi:
            # Fit on independent scramble means, not correlated points within a net.
            for child in pilot_seed.spawn(int(n_scrambles)):
                x, y = observations(sobol_normals(child, pilot // n_scrambles))
                regression.update(x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True))
        else:
            rng = np.random.default_rng(pilot_seed)
            remaining = pilot // 2 if paired else pilot
            while remaining:
                size = min(int(batch_size), remaining)
                regression.update(*observations(rng.standard_normal(size)))
                remaining -= size
        beta = regression.coefficient()

    if quasi:
        for child in production_seed.spawn(int(n_scrambles)):
            x, y = observations(sobol_normals(child, production // n_scrambles))
            if controlled:
                y = y - beta * (x - control_mean)
            moments.update(y.mean(axis=0, keepdims=True))
        critical = float(t.ppf(0.975, n_scrambles - 1))
    else:
        rng = np.random.default_rng(production_seed)
        remaining = production // 2 if paired else production
        while remaining:
            size = min(int(batch_size), remaining)
            x, y = observations(rng.standard_normal(size))
            if controlled:
                y = y - beta * (x - control_mean)
            moments.update(y)
            remaining -= size
        critical = 1.959963984540054
    return SampleEstimate(moments.mean, moments.se(), critical, moments.count, pilot, beta)
