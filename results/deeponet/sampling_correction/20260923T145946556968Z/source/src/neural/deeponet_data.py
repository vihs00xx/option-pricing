import numpy as np
from scipy.stats import qmc

from src.black_scholes import black_scholes_greeks, black_scholes_price


TARGETS = ("price", "delta", "gamma", "vega", "theta", "rho")
BRANCH_FEATURES = ("K", "T", "r", "q", "sigma", "option_type")
TRUNK_FEATURES = ("S", "tau")


def _scale(unit, lower, upper):
    return lower + unit * (upper - lower)


def _contract_samples(count, domain, seed):
    unit = qmc.LatinHypercube(d=5, seed=seed).random(count)
    contracts = np.column_stack((
        _scale(unit[:, 0], *domain["K"]),
        _scale(unit[:, 1], *domain["T"]),
        _scale(unit[:, 2], *domain["r"]),
        _scale(unit[:, 3], *domain["q"]),
        _scale(unit[:, 4], *domain["sigma"]),
        np.where(np.arange(count) % 2 == 0, 1.0, -1.0),
    ))
    return contracts


def _targets(branch, trunk):
    output = {name: np.empty(len(branch), dtype=np.float64) for name in TARGETS}
    for code, option in ((1.0, "call"), (-1.0, "put")):
        mask = branch[:, 5] == code
        if not np.any(mask):
            continue
        S = trunk[mask, 0]
        tau = trunk[mask, 1]
        K = branch[mask, 0]
        r = branch[mask, 2]
        q = branch[mask, 3]
        sigma = branch[mask, 4]
        output["price"][mask] = black_scholes_price(
            S, K, tau, r, sigma, q, option
        )
        greeks = black_scholes_greeks(S, K, tau, r, sigma, q, option)
        for name in TARGETS[1:]:
            output[name][mask] = greeks[name]
    return np.column_stack([output[name] for name in TARGETS])


def synthetic_split(contract_count, coordinates_per_contract, domain, seed):
    contracts = _contract_samples(contract_count, domain, seed)
    branch = np.repeat(contracts, coordinates_per_contract, axis=0)
    unit = qmc.LatinHypercube(d=2, seed=seed + 1).random(len(branch))
    moneyness = _scale(unit[:, 0], *domain["moneyness"])
    minimum_tau = float(domain["tau_minimum"])
    tau = minimum_tau + unit[:, 1] * (branch[:, 1] - minimum_tau)
    trunk = np.column_stack((branch[:, 0] * moneyness, tau))
    contract_id = np.repeat(np.arange(contract_count), coordinates_per_contract)
    return {
        "branch": branch,
        "trunk": trunk,
        "targets": _targets(branch, trunk),
        "contract_id": contract_id,
    }


MIXED_REGIONS = ("broad", "near_strike", "near_expiry", "near_strike_and_expiry")
STRIKE_BAND = (0.95, 1.05)
EXPIRY_TAU = 0.05


def contract_coordinate_split(contracts, coordinates_per_contract, domain, seed, mixed=False):
    """Coordinates and targets for fixed contracts sampled uniformly or with
    near-expiry and near-strike concentrations.

    mixed=True keeps total size unchanged: half of each contract's coordinates
    stay broad, one quarter concentrate near the strike, one eighth near expiry
    and the remainder near both. Region codes in ``region`` record the layout
    (MIXED_REGIONS indices; all zero when mixed=False).
    """
    count = len(contracts)
    branch = np.repeat(contracts, coordinates_per_contract, axis=0)
    moneyness = np.empty((count, coordinates_per_contract))
    tau = np.empty((count, coordinates_per_contract))
    region = np.zeros((count, coordinates_per_contract), dtype=np.int64)
    minimum_tau = float(domain["tau_minimum"])
    horizon = contracts[:, 1:2]
    if mixed:
        sizes = [coordinates_per_contract // 2, coordinates_per_contract // 4,
                 coordinates_per_contract // 8]
        sizes.append(coordinates_per_contract - sum(sizes))
        if min(sizes) < 1:
            raise ValueError("mixed sampling needs at least 8 coordinates per contract")
    else:
        sizes = [coordinates_per_contract]
    bands = (domain["moneyness"], STRIKE_BAND, domain["moneyness"], STRIKE_BAND)
    capped = (False, False, True, True)
    offset = 0
    for code, size in enumerate(sizes):
        unit = qmc.LatinHypercube(d=2, seed=seed + 10 * code).random(count * size)
        band = bands[code] if mixed else domain["moneyness"]
        upper = np.minimum(EXPIRY_TAU, horizon) if (mixed and capped[code]) else horizon
        moneyness[:, offset:offset + size] = _scale(unit[:, 0], *band).reshape(count, size)
        tau[:, offset:offset + size] = (
            minimum_tau + unit[:, 1].reshape(count, size) * (upper - minimum_tau))
        region[:, offset:offset + size] = code if mixed else 0
        offset += size
    trunk = np.column_stack((branch[:, 0] * moneyness.ravel(), tau.ravel()))
    contract_id = np.repeat(np.arange(count), coordinates_per_contract)
    return {
        "branch": branch,
        "trunk": trunk,
        "targets": _targets(branch, trunk),
        "contract_id": contract_id,
        "region": region.ravel(),
    }


def benchmark_split(cases):
    option_code = np.array([
        1.0 if case["option"] == "call" else -1.0 for case in cases
    ])
    branch = np.column_stack((
        [case["K"] for case in cases],
        [case["tau"] for case in cases],
        [case["r"] for case in cases],
        [case["q"] for case in cases],
        [case["sigma"] for case in cases],
        option_code,
    )).astype(np.float64)
    trunk = np.column_stack((
        [case["S"] for case in cases],
        [case["tau"] for case in cases],
    )).astype(np.float64)
    return {
        "branch": branch,
        "trunk": trunk,
        "targets": _targets(branch, trunk),
        "case_id": np.asarray([case["case_id"] for case in cases]),
        "benchmark_split": np.asarray([case["split"] for case in cases]),
    }


def build_dataset(settings, benchmark_cases):
    sampling = settings["sampling"]
    domain = settings["training_domain"]
    seed = int(sampling["seed"])
    train = synthetic_split(
        int(sampling["train_contracts"]),
        int(sampling["coordinates_per_contract"]),
        domain,
        seed,
    )
    validation = synthetic_split(
        int(sampling["validation_contracts"]),
        int(sampling["coordinates_per_contract"]),
        domain,
        seed + 1000,
    )
    test = benchmark_split(benchmark_cases)
    return {"train": train, "validation": validation, "test": test}
