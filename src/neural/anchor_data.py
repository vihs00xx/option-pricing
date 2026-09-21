"""Reproducible labelled datasets for the anchored PINN ablation.

One dataset is generated once and saved so every model in a study sees
exactly the same points. Distinct splits are kept for:

* training price anchors (with their density-component membership);
* training Greek anchors -- independently sampled near the strike and
  near expiry, so they are *not* a subset of the price anchors;
* training PDE collocation points;
* training terminal payoff points;
* validation coordinates and labels (price + Greeks);
* validation PDE collocation points;
* validation terminal payoff points (sampled independently of training);
* untouched test coordinates and labels.

Sampling covers the full log-moneyness domain with extra density near
``S/K = 1``, near expiry, in high-Gamma regions and symmetrically in both
domain-boundary strips.

Points are ordered by a stratified key, so taking a prefix of length ``n``
yields an approximately proportional, *nested* subset of every density
component. Greek anchors remain independently sampled; their coordinate
and label counts are reported separately from the price-anchor count.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.neural.fixed_pinn import FixedBlackScholesProblem


DATASET_SCHEMA_VERSION = 2
DENSITY_KEYS = ("uniform", "near_strike", "near_expiry", "high_gamma",
                "boundaries")
PROVENANCE_KEYS = (
    "problem", "minimum_tau", "density", "greek_density",
    "near_strike_half_width", "near_expiry_tau_span", "high_gamma_half_width",
    "high_gamma_tau_span", "boundary_fraction_of_domain", "anchor_counts",
    "greek_anchor_divisor", "greek_anchor_max", "validation_points",
    "test_points", "pde_points", "terminal_points", "seed",
)


def problem_from(settings):
    p = settings["problem"]
    return FixedBlackScholesProblem(
        p["K"], p["T"], p["r"], p["sigma"], p["q"], p["x_min"], p["x_max"],
    )


def validate_density(density, name="density"):
    """Fractions must be known keys, non-negative and sum to one."""
    unknown = set(density) - set(DENSITY_KEYS)
    if unknown:
        raise ValueError(f"{name} has unknown components: {sorted(unknown)}")
    for key, value in density.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{key}] must be finite and non-negative")
    total = float(sum(density.values()))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to 1, got {total!r}")


def config_hash(settings):
    """Stable hash of the data-generation configuration."""
    payload = json.dumps({key: settings[key] for key in PROVENANCE_KEYS},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _sobol(count, seed, dtype=torch.float64):
    """Sobol draw cast to the requested dtype, never the process default."""
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    return engine.draw(count).to(dtype=dtype).double().numpy()


def _component(name, n, problem, seed, settings, y_min, y_max):
    """Exactly ``n`` points from one density component."""
    t_min = settings["minimum_tau"]
    span = problem.T - t_min
    u = _sobol(n, seed, torch.float64)
    if name == "uniform":
        y = y_min + (y_max - y_min) * u[:, 0]
        tau = t_min + span * u[:, 1]
    elif name == "near_strike":
        half = settings["near_strike_half_width"]
        y = np.log(np.clip(1.0 + half * (2 * u[:, 0] - 1),
                           problem.x_min, problem.x_max))
        tau = t_min + span * u[:, 1]
    elif name == "near_expiry":
        y = y_min + (y_max - y_min) * u[:, 0]
        span_e = min(settings["near_expiry_tau_span"], span)
        tau = t_min + span_e * u[:, 1] ** 2
    elif name == "high_gamma":
        half = settings["high_gamma_half_width"]
        y = np.log(np.clip(1.0 + half * (2 * u[:, 0] - 1),
                           problem.x_min, problem.x_max))
        span_e = min(settings["high_gamma_tau_span"], span)
        tau = t_min + span_e * u[:, 1] ** 2
    else:  # boundaries: two symmetric strips of equal width
        edge = settings["boundary_fraction_of_domain"] * (y_max - y_min)
        left = u[:, 0] < 0.5
        v = np.where(left, 2.0 * u[:, 0], 2.0 * (u[:, 0] - 0.5))
        y = np.where(left, y_min + edge * v, y_max - edge * v)
        tau = t_min + span * u[:, 1]
    return y.astype(np.float64), tau.astype(np.float64)


def sample_points(problem, count, seed, density, settings):
    """Stratified mixture sample; a prefix of length n is nested and
    approximately proportional across density components.

    Returns ``(y, tau, component)`` with ``component`` indexing
    ``DENSITY_KEYS``.
    """
    validate_density(density)
    y_min = float(np.log(problem.x_min))
    y_max = float(np.log(problem.x_max))
    counts = {name: int(np.floor(count * density.get(name, 0.0)))
              for name in DENSITY_KEYS}
    remainder = count - sum(counts.values())
    if remainder:
        counts["uniform"] += remainder
    ys, taus, keys, comps = [], [], [], []
    for index, name in enumerate(DENSITY_KEYS):
        n = counts[name]
        if n <= 0:
            continue
        y, tau = _component(name, n, problem, seed + 1000 * (index + 1),
                            settings, y_min, y_max)
        ys.append(y)
        taus.append(tau)
        # Stratified key: prefixes interleave components in proportion.
        keys.append((np.arange(n) + 0.5) / n)
        comps.append(np.full(n, index, dtype=np.int64))
    if not ys:  # count == 0
        return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64))
    y = np.concatenate(ys)
    tau = np.concatenate(taus)
    key = np.concatenate(keys)
    component = np.concatenate(comps)
    order = np.argsort(key, kind="stable")
    return y[order], tau[order], component[order]


def component_counts(component):
    """Reportable dict of per-component sample counts."""
    return {DENSITY_KEYS[index]: int(np.sum(component == index))
            for index in range(len(DENSITY_KEYS))}


def _rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def _labels(problem, y, tau, with_greeks):
    """Analytical normalised labels: u = C/K, theta/K, Delta, Gamma."""
    spots = problem.K * np.exp(y)
    price = black_scholes_price(
        spots, problem.K, tau, problem.r, problem.sigma, problem.q, "call")
    out = {"price": np.asarray(price, dtype=np.float64).reshape(-1) / problem.K}
    if with_greeks:
        greeks = black_scholes_greeks(
            spots, problem.K, tau, problem.r, problem.sigma, problem.q, "call")
        out["delta"] = np.asarray(greeks["delta"], dtype=np.float64).reshape(-1)
        out["gamma"] = np.asarray(greeks["gamma"], dtype=np.float64).reshape(-1)
        out["theta"] = (np.asarray(greeks["theta"], dtype=np.float64).reshape(-1)
                        / problem.K)
    return out


def _terminal(y):
    return np.maximum(np.exp(y) - 1.0, 0.0).astype(np.float64)


def build_dataset(settings):
    """Build the full fixed dataset at the largest anchor count.

    Normalisation scales are computed from the *full* training label sets,
    so the anchor-count sweep does not silently change the objective.
    """
    validate_density(settings["density"])
    validate_density(settings["greek_density"], "greek_density")
    problem = problem_from(settings)
    seed = settings["seed"]
    anchor_count = max(settings["anchor_counts"])
    greek_count = min(settings["greek_anchor_max"],
                      max(anchor_count // settings["greek_anchor_divisor"], 1))

    anchor_y, anchor_tau, anchor_component = sample_points(
        problem, anchor_count, seed + 1, settings["density"], settings)
    anchor = _labels(problem, anchor_y, anchor_tau, with_greeks=False)

    greek_y, greek_tau, greek_component = sample_points(
        problem, greek_count, seed + 2, settings["greek_density"], settings)
    greek = _labels(problem, greek_y, greek_tau, with_greeks=True)

    validation_y, validation_tau, _ = sample_points(
        problem, settings["validation_points"], seed + 3, settings["density"],
        settings)
    validation = _labels(problem, validation_y, validation_tau, with_greeks=True)

    validation_pde_y, validation_pde_tau, _ = sample_points(
        problem, settings["pde_points"], seed + 7, settings["density"], settings)

    validation_terminal_y, _, _ = sample_points(
        problem, settings["terminal_points"], seed + 8, settings["density"],
        settings)
    validation_terminal_tau = np.zeros_like(validation_terminal_y)

    test_y, test_tau, _ = sample_points(
        problem, settings["test_points"], seed + 4, settings["density"], settings)
    test = _labels(problem, test_y, test_tau, with_greeks=True)

    pde_y, pde_tau, _ = sample_points(
        problem, settings["pde_points"], seed + 5, settings["density"], settings)

    terminal_y, _, _ = sample_points(
        problem, settings["terminal_points"], seed + 6, settings["density"],
        settings)
    terminal_tau = np.zeros_like(terminal_y)

    dataset = {
        "anchor_y": anchor_y, "anchor_tau": anchor_tau,
        "anchor_price": anchor["price"], "anchor_component": anchor_component,
        "greek_y": greek_y, "greek_tau": greek_tau,
        "greek_delta": greek["delta"], "greek_gamma": greek["gamma"],
        "greek_theta": greek["theta"], "greek_component": greek_component,
        "validation_y": validation_y, "validation_tau": validation_tau,
        "validation_price": validation["price"],
        "validation_delta": validation["delta"],
        "validation_gamma": validation["gamma"],
        "validation_theta": validation["theta"],
        "validation_pde_y": validation_pde_y,
        "validation_pde_tau": validation_pde_tau,
        "validation_terminal_y": validation_terminal_y,
        "validation_terminal_tau": validation_terminal_tau,
        "validation_terminal_payoff": _terminal(validation_terminal_y),
        "pde_y": pde_y, "pde_tau": pde_tau,
        "terminal_y": terminal_y, "terminal_tau": terminal_tau,
        "terminal_payoff": _terminal(terminal_y),
        "test_y": test_y, "test_tau": test_tau,
        "test_price": test["price"], "test_delta": test["delta"],
        "test_gamma": test["gamma"], "test_theta": test["theta"],
    }
    return {name: np.asarray(value) for name, value in dataset.items()}


def reference_scales(dataset):
    """Normalisation scales from the full training labels only."""
    return {
        "price": _rms(dataset["anchor_price"]),
        "delta": _rms(dataset["greek_delta"]),
        "gamma": _rms(dataset["greek_gamma"]),
        "theta": _rms(dataset["greek_theta"]),
    }


def _content_hash(dataset):
    digest = hashlib.sha256()
    for name in sorted(dataset):
        array = np.ascontiguousarray(dataset[name])
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def dataset_metadata(dataset, settings):
    problem = problem_from(settings)
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "config_hash": config_hash(settings),
        "content_hash": _content_hash(dataset),
        "counts": {
            "price_anchors": int(dataset["anchor_y"].size),
            "greek_anchors": int(dataset["greek_y"].size),
            "validation": int(dataset["validation_y"].size),
            "validation_pde": int(dataset["validation_pde_y"].size),
            "validation_terminal": int(dataset["validation_terminal_y"].size),
            "test": int(dataset["test_y"].size),
            "pde": int(dataset["pde_y"].size),
            "terminal": int(dataset["terminal_y"].size),
        },
        "label_counts": {
            "price": int(dataset["anchor_price"].size),
            "theta": int(dataset["greek_theta"].size),
            "delta": int(dataset["greek_delta"].size),
            "gamma": int(dataset["greek_gamma"].size),
            "validation_price": int(dataset["validation_price"].size),
            "validation_theta": int(dataset["validation_theta"].size),
            "validation_delta": int(dataset["validation_delta"].size),
            "validation_gamma": int(dataset["validation_gamma"].size),
        },
        "component_counts": component_counts(dataset["anchor_component"]),
        "greek_component_counts": component_counts(dataset["greek_component"]),
        "ranges": {
            "train_y": [float(dataset["anchor_y"].min()),
                        float(dataset["anchor_y"].max())],
            "train_tau": [float(dataset["anchor_tau"].min()),
                          float(dataset["anchor_tau"].max())],
            "test_y": [float(dataset["test_y"].min()),
                       float(dataset["test_y"].max())],
        },
        "scales": reference_scales(dataset),
        "problem": {"K": problem.K, "T": problem.T, "r": problem.r,
                    "sigma": problem.sigma, "q": problem.q,
                    "x_min": problem.x_min, "x_max": problem.x_max},
    }


def save_dataset(path, dataset, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
    metadata = dataset_metadata(dataset, settings)
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_dataset(path, settings=None):
    """Load a saved dataset, optionally verifying it matches ``settings``."""
    path = Path(path)
    with np.load(path) as handle:
        dataset = {name: handle[name] for name in handle.files}
    if settings is None:
        return dataset
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ValueError(f"dataset {path} has no metadata; regenerate it")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"dataset {path} schema {metadata.get('schema_version')} != "
            f"{DATASET_SCHEMA_VERSION}; regenerate it")
    expected = config_hash(settings)
    if metadata.get("config_hash") != expected:
        raise ValueError(
            f"dataset {path} was built with data config "
            f"{metadata.get('config_hash')} but {expected} was requested; "
            "use a new versioned dataset path or regenerate it")
    return dataset


def subset(dataset, anchor_count, settings, greek_count=None):
    """Nested prefix subset: proportional across density components."""
    n = min(anchor_count, int(dataset["anchor_y"].size))
    if greek_count is None:
        greek_count = min(settings["greek_anchor_max"],
                          max(n // settings["greek_anchor_divisor"], 1))
    g = min(greek_count, int(dataset["greek_y"].size))
    sub = dict(dataset)
    for key in ("anchor_y", "anchor_tau", "anchor_price", "anchor_component"):
        sub[key] = dataset[key][:n]
    for key in ("greek_y", "greek_tau", "greek_delta", "greek_gamma",
                "greek_theta", "greek_component"):
        sub[key] = dataset[key][:g]
    return sub


COORDINATE_GROUPS = (
    ("anchor_train", "anchor_y", "anchor_tau"),
    ("greek_train", "greek_y", "greek_tau"),
    ("pde_train", "pde_y", "pde_tau"),
    ("terminal_train", "terminal_y", "terminal_tau"),
    ("validation", "validation_y", "validation_tau"),
    ("validation_pde", "validation_pde_y", "validation_pde_tau"),
    ("validation_terminal", "validation_terminal_y", "validation_terminal_tau"),
    ("test", "test_y", "test_tau"),
)


def verify_disjoint(dataset):
    """Return ``{group|group: shared_count}`` for accidentally reused points."""
    keys = {}
    for name, y_key, tau_key in COORDINATE_GROUPS:
        keys[name] = set(zip(dataset[y_key].tolist(), dataset[tau_key].tolist()))
    overlaps = {}
    names = [name for name, _, _ in COORDINATE_GROUPS]
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = keys[first] & keys[second]
            if shared:
                overlaps[f"{first}|{second}"] = len(shared)
    return overlaps


def verify_domain(dataset, settings):
    """Return a list of domain violations across all coordinate groups."""
    problem = problem_from(settings)
    t_min = settings["minimum_tau"]
    issues = []
    for name, y_key, tau_key in COORDINATE_GROUPS:
        x = np.exp(dataset[y_key])
        if np.any(x < problem.x_min - 1e-12) or np.any(x > problem.x_max + 1e-12):
            issues.append(f"{name}: x outside [{problem.x_min}, {problem.x_max}]")
        tau = dataset[tau_key]
        if np.any(tau < -1e-12) or np.any(tau > problem.T + 1e-12):
            issues.append(f"{name}: tau outside [0, {problem.T}]")
        positive = tau[tau > 0]
        if positive.size and float(positive.min()) < t_min - 1e-12:
            issues.append(f"{name}: positive tau below minimum_tau")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(tau)):
            issues.append(f"{name}: non-finite coordinates")
    return issues


def nested_prefix_check(dataset, counts, greek_anchor_max=1,
                        greek_anchor_divisor=4):
    """Verify prefixes are nested and proportional across components."""
    report = {}
    previous = None
    nested = True
    settings = {"greek_anchor_max": greek_anchor_max,
                "greek_anchor_divisor": greek_anchor_divisor}
    for count in counts:
        sample = subset(dataset, count, settings)
        current = component_counts(sample["anchor_component"])
        report[str(count)] = current
        if previous is not None and any(current[key] < previous[key]
                                       for key in current):
            nested = False
        previous = current
    return {"component_counts": report, "nested": nested}