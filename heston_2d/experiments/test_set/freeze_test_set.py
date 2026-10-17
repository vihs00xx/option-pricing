"""Generate, reference and freeze the Heston basket test set. No pricing method is evaluated.

Run from this folder: python -m experiments.test_set.freeze_test_set [--workers 8]
The references are conditional QE-RQMC estimates with standard errors (see the protocol's
"reference"); give the build every core of the lab PC. Per-case results go to
artifacts/test_set_v1_build/cases.jsonl as they finish, so an interrupted build resumes;
the sealed folder is written only at the end. The number of stocks is the protocol's.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import Lock, Pool
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from experiments.records import ROOT, digest, pricing_sources, prepare_output, relative, write_csv, write_metadata
from src.heston_basket.engine import heston_estimate
from src.heston_basket.model import HestonBasket, quantity_names
from src.neural.coordinates import matrices, protocol_path, stock_correlation

PROTOCOL = protocol_path()
WORK = ROOT / "artifacts/test_set_v1_build/cases.jsonl"
FLOORS = {"price": 1e-4, "delta": 1e-4, "gamma": 1e-7, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4,
          "d": 1e-4, "correlation": 1e-4, "vol": 1e-4}
PER_STOCK = ("v0", "kappa", "theta", "xi", "leverage")
WORKERS = 8


def pairs(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def smallest_eigenvalue(M):
    return float(np.linalg.eigvalsh(M)[0])


# ---------------------------------------------------------------- cases

class Stream:
    """One split's Latin hypercube rows and correlation draws."""

    def __init__(self, protocol, split):
        seed, offset = protocol["seed"], protocol["sampling"]["split_offsets"][split]
        self.d = int(protocol["stocks"])
        self.domain = protocol["training_domain"]
        self.box = protocol["sampling"]["box_parameters"]
        self.engine = qmc.LatinHypercube(d=len(self.box) + self.d - 1, seed=seed + offset)
        self.rng = np.random.Generator(np.random.PCG64(seed + 100 + offset))
        self.min_eigenvalue = protocol["domain_constraints"]["min_eigenvalue"]

    def rows(self, count):
        return list(self.engine.random(count))

    def values(self, unit):
        values = {p: self.domain[p][0] + u * (self.domain[p][1] - self.domain[p][0])
                  for p, u in zip(self.box, unit)}
        spacings = np.diff(np.concatenate(([0.0], np.sort(unit[len(self.box):]), [1.0])))
        low = self.domain["weight"][0]
        for k, w in enumerate(low + (1 - self.d * low) * spacings):
            values[f"weight_{k + 1}"] = float(w)
        return values

    def matrix(self, name, fixed=None):
        """C or P: entries uniform on the domain range (one pair optionally fixed), eigenvalue floor."""
        low, high = self.domain[name]
        while True:
            entries = self.rng.uniform(low, high, len(pairs(self.d)))
            if fixed is not None:
                entries[pairs(self.d).index(fixed[0])] = fixed[1]
            M = matrices(entries, self.d)
            # A fixed extrapolation entry (0.97) cannot meet the domain floor; it uses 0.01.
            if smallest_eigenvalue(M) >= (self.min_eigenvalue if fixed is None else 0.01):
                return M


def set_correlation(values, C, P, d):
    leverage = np.array([values[f"leverage_{i}"] for i in range(1, d + 1)])
    R = stock_correlation(C, P, leverage)
    for i, j in pairs(d):
        values[f"corr_{i + 1}{j + 1}"] = float(R[i, j])
        values[f"vol_corr_{i + 1}{j + 1}"] = float(C[i, j])
        values[f"residual_corr_{i + 1}{j + 1}"] = float(P[i, j])
    return values


def contract_fields(values, option, strike, d):
    per = lambda name: [values[f"{name}_{k}"] for k in range(1, d + 1)]
    R, C = np.eye(d), np.eye(d)
    for i, j in pairs(d):
        R[i, j] = R[j, i] = values[f"corr_{i + 1}{j + 1}"]
        C[i, j] = C[j, i] = values[f"vol_corr_{i + 1}{j + 1}"]
    return dict(spots=[strike * m for m in per("moneyness")], strike=strike, tau=values["tau"], rate=values["rate"],
                dividends=per("dividend"), weights=per("weight"), v0=per("v0"), kappa=per("kappa"),
                theta=per("theta"), xi=per("xi"), leverage=per("leverage"), correlation=R.tolist(),
                vol_correlation=C.tolist(), option=option)


def make_case(case_id, split, reason, values, option, strike, d, draw=0, slot=None):
    values = {k: float(v) for k, v in values.items()}
    return dict(case_id=case_id, split=split, reason=reason, option=option, draw=draw, slot=slot,
                parameters=values, contract=contract_fields(values, option, strike, d))


def extrapolation_slots(split):
    slots = []
    for group, group_values in split["values"].items():
        for value in group_values:
            for k in range(split["cases_per_value"]):
                slots.append((group, value, k))
    return slots


def extrapolate(stream, split, unit, group, value, k):
    """Domain draw with one parameter moved outside; k picks the stock (or the pair)."""
    d = stream.d
    values = stream.values(unit)
    stock = k % d + 1
    C, P = None, None
    if group in ("moneyness", "dividend") + PER_STOCK:
        values[f"{group}_{stock}"] = value
        target = f"{group}_{stock} = {value}"
    elif group in ("tau", "rate"):
        values[group] = value
        target = f"{group} = {value}"
    elif group == "weight":
        others = [w for w in range(1, d + 1) if w != stock]
        scale = (1 - value) / sum(values[f"weight_{w}"] for w in others)
        for w in others:
            values[f"weight_{w}"] *= scale
        values[f"weight_{stock}"] = value
        target = f"weight_{stock} = {value}"
    elif group in ("vol_correlation", "residual_correlation"):
        all_pairs = pairs(d)
        pair = all_pairs[k % len(all_pairs)]
        fixed = (pair, value)
        C = stream.matrix("vol_correlation", fixed if group == "vol_correlation" else None)
        P = stream.matrix("residual_correlation", fixed if group == "residual_correlation" else None)
        target = f"{group}_{pair[0] + 1}{pair[1] + 1} = {value}"
    if C is None:
        C, P = stream.matrix("vol_correlation"), stream.matrix("residual_correlation")
    return set_correlation(values, C, P, d), f"{target} outside the training domain"


def generated_case(stream, protocol, split_name, slot, unit, draw):
    strike, split, d = protocol["strike"], protocol["splits"][split_name], stream.d
    option = "call" if slot % 2 == 0 else "put"
    if split_name == "extrapolation":
        group, value, k = extrapolation_slots(split)[slot]
        values, reason = extrapolate(stream, split, unit, group, value, k)
        case_id = f"extrapolation_{group}_{slot + 1:03d}"
    else:
        values = stream.values(unit)
        values = set_correlation(values, stream.matrix("vol_correlation"), stream.matrix("residual_correlation"), d)
        reason = split["reason"]
        if split_name == "short_maturity":
            low, high = split["tau"]
            values["tau"] = low + unit[stream.box.index("tau")] * (high - low)
            reason = "remaining life below the training domain"
        case_id = f"{split_name}_{slot + 1:03d}"
    return make_case(case_id, split_name, reason, values, option, strike, d, draw, slot)


def anchor_cases(protocol):
    split, strike, d = protocol["splits"]["anchor"], protocol["strike"], int(protocol["stocks"])
    equi = lambda value: value if isinstance(value, list) else (np.full((d, d), value) + (1 - value) * np.eye(d))
    cases = []
    for i, row in enumerate(split["cases"]):
        spec = {**split["defaults"], **{k: v for k, v in row.items() if k not in ("name", "option", "note")}}
        values = {"tau": spec["tau"], "rate": spec["rate"]}
        for k in range(d):
            each = lambda name: spec[name][k] if isinstance(spec[name], list) else spec[name]
            values[f"moneyness_{k + 1}"] = each("moneyness")
            values[f"dividend_{k + 1}"] = spec["dividends"][k]
            values[f"weight_{k + 1}"] = spec["weights"][k]
            for name in PER_STOCK:
                values[f"{name}_{k + 1}"] = each(name)
        set_correlation(values, np.asarray(equi(spec["vol_correlation"]), float),
                        np.asarray(equi(spec["residual_correlation"]), float), d)
        cases.append(make_case(f"anchor_{i + 1:02d}_{row['name']}", "anchor", row.get("note", split["reason"]),
                               values, row["option"], strike, d, slot=i))
    return cases


def build_cases(protocol):
    """Every case in its final order: interpolation, short maturity, extrapolation, anchors."""
    streams = {s: Stream(protocol, s) for s in ("interpolation", "short_maturity", "extrapolation")}
    counts = dict(interpolation=protocol["splits"]["interpolation"]["count"],
                  short_maturity=protocol["splits"]["short_maturity"]["count"],
                  extrapolation=len(extrapolation_slots(protocol["splits"]["extrapolation"])))
    cases = []
    for split, stream in streams.items():
        cases += [generated_case(stream, protocol, split, i, unit, 0) for i, unit in enumerate(stream.rows(counts[split]))]
    cases += anchor_cases(protocol)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise RuntimeError("Case identifiers are not unique")
    for case in cases:
        HestonBasket(**case["contract"])   # validates every contract
    return cases


# ------------------------------------------------------------ references

def floor(quantity):
    return FLOORS[quantity.split("_")[0]]


def relative_differences(values, reference, names):
    return {q: abs(values[i] - reference[i]) / max(abs(reference[i]), floor(q)) for i, q in enumerate(names)}


_lock = None


def _init(lock):
    global _lock
    _lock = lock


def z_scores(a, se_a, b, se_b):
    """(a - b) / sqrt(se_a^2 + se_b^2), with exact agreement at zero error counted as 0."""
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(np.asarray(se_a) ** 2 + np.asarray(se_b) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, (a - b) / se, np.where(a == b, 0.0, np.inf))


def estimate(c, points, seed, settings, **kwargs):
    kwargs.setdefault("dt", settings["dt"])
    e = heston_estimate(c, points, settings["replicates"], seed, greeks=True, **kwargs)
    return np.array([e.values[q] for q in e.names]), np.array([e.standard_errors[q] for q in e.names])


def reference_job(task):
    """Points ladder to the price-error target, then the recorded cross-checks; one JSON-ready row."""
    case, settings, seed, check_seed = task
    c = HestonBasket(**case["contract"])
    names = quantity_names(c.dimension)
    target = settings["price_standard_error_target"]
    levels, seconds = {}, 0.0
    accepted = None
    for points in settings["points_ladder"]:
        start = perf_counter()
        levels[points] = estimate(c, points, seed, settings)
        seconds += perf_counter() - start
        if levels[points][1][0] <= target:
            accepted = points
            break
    flagged = accepted is None
    if flagged:
        accepted = settings["points_ladder"][-1]
    reference, se = levels[accepted]
    relative_se = se / np.array([max(abs(v), floor(q)) for v, q in zip(reference, names)])
    k = int(np.argmax(relative_se))
    row = dict(key=case_key(case), case_id=case["case_id"], split=case["split"], draw=case["draw"],
               accepted_points=accepted, above_target=flagged, seed=seed,
               price_standard_errors={str(p): float(v[1][0]) for p, v in levels.items()},
               reference_seconds=seconds, values=reference.tolist(), standard_errors=se.tolist(),
               max_relative_se=float(relative_se[k]), max_relative_se_quantity=names[k])
    # The cross-checks run at check_points (at most the accepted level) and compare with the
    # combined standard error, so a cheaper check is still a fair one.
    check = min(accepted, settings["check_points"])
    start = perf_counter()
    second, second_se = estimate(c, check, check_seed, settings)
    z = z_scores(second, second_se, reference, se)
    k = int(np.argmax(np.abs(z)))
    row.update(check_seed=check_seed, check_points=check, check_seconds=perf_counter() - start,
               check_max_abs_z=float(np.abs(z[k])), check_worst_quantity=names[k],
               check_rms_z=float(np.sqrt(np.mean(z ** 2))), check_count_abs_z_over_5=int(np.sum(np.abs(z) > 5)))
    coarse, coarse_se = estimate(c, check, seed, settings, dt=2 * settings["dt"])
    steps = z_scores(coarse, coarse_se, reference, se)
    k = int(np.argmax(np.abs(steps)))
    row.update(time_step_max_abs=float(np.abs(steps[k])), time_step_worst_quantity=names[k],
               time_step_price_difference=float(coarse[0] - reference[0]))
    half, _ = estimate(c, check, seed, settings, bump_scale=0.5)
    base, _ = levels[check] if check in levels else estimate(c, check, seed, settings)
    bumped = [q for q in names if not q.startswith(("price", "delta", "gamma", "rho"))]
    rel = {q: v for q, v in relative_differences(half, base, names).items() if q in bumped}
    worst = max(rel, key=rel.get)
    row.update(bump_halving_max_relative=rel[worst], bump_halving_worst_quantity=worst)
    return save(row)


def save(row):
    if _lock is not None:
        with _lock:
            append(row)
    else:
        append(row)
    return row


def append(row):
    WORK.parent.mkdir(parents=True, exist_ok=True)
    with WORK.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def case_key(case):
    return f"{case['case_id']}#{case['draw']}#{digest_values(case['parameters'])}"


def digest_values(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]


def load_work():
    rows = {}
    if WORK.exists():
        for line in WORK.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["key"]] = row
    return rows


# ------------------------------------------------------------ separation

def parameter_vector(fields):
    d = len(fields["spots"])
    R, C = np.asarray(fields["correlation"], float), np.asarray(fields["vol_correlation"], float)
    spots, strike = np.asarray(fields["spots"], float), float(fields["strike"])
    return np.concatenate([spots / strike, [fields["tau"], fields["rate"]], fields["dividends"],
                           *[fields[p] for p in PER_STOCK], fields["weights"],
                           [R[i, j] for i, j in pairs(d)], [C[i, j] for i, j in pairs(d)]])


def widths(domain, d):
    w = lambda name: domain[name][1] - domain[name][0]
    return np.array([w("moneyness_1")] * d + [w("tau"), w("rate")] + [w("dividend_1")] * d
                    + sum(([w(f"{p}_1")] * d for p in PER_STOCK), []) + [w("weight")] * d
                    + [w("correlation")] * len(pairs(d)) + [w("vol_correlation")] * len(pairs(d)))


def same_contract(a, b):
    keys = ("spots", "strike", "tau", "rate", "dividends", "weights", "correlation", "vol_correlation") + PER_STOCK
    return a["option"] == b["option"] and all(
        np.allclose(np.asarray(a[k], float), np.asarray(b[k], float), rtol=0, atol=1e-9) for k in keys)


def development_contracts(protocol):
    path = ROOT / protocol["separation"]["development_contracts"]
    rows = json.loads(path.read_text(encoding="utf-8"))["contracts"]
    return [(row["name"], row["contract"]) for row in rows]


def separation(cases, development, domain, d):
    scale = widths(domain, d)
    dev = np.array([parameter_vector(f) for _, f in development])
    per_case = []
    for case in cases:
        for name, fields in development:
            if same_contract(case["contract"], fields):
                raise RuntimeError(f"{case['case_id']} duplicates development contract {name}")
        distance = np.max(np.abs(dev - parameter_vector(case["contract"])) / scale, axis=1)
        nearest = int(np.argmin(distance))
        per_case.append(dict(case_id=case["case_id"], split=case["split"], distance=float(distance[nearest]),
                             nearest=development[nearest][0]))
    return per_case


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/test_set_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Output exists; use a new output directory or explicit --overwrite")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    settings = protocol["reference"]
    d = int(protocol["stocks"])
    cases = build_cases(protocol)
    done = load_work()
    start = perf_counter()
    todo = [(c, settings, protocol["seed"] + 10000 + i, protocol["seed"] + 20000 + i)
            for i, c in enumerate(cases) if case_key(c) not in done]
    todo.sort(key=lambda t: -t[0]["contract"]["tau"])   # long maturities are the slow ones
    if todo:
        with Pool(args.workers, initializer=_init, initargs=(Lock(),)) as pool:
            for i, row in enumerate(pool.imap_unordered(reference_job, todo), 1):
                done[row["key"]] = row
                if i % 5 == 0 or i == len(todo):
                    print(f"[{perf_counter() - start:8.1f}s] {i}/{len(todo)} cases referenced", flush=True)
    seconds = perf_counter() - start
    development = development_contracts(protocol)
    distances = separation(cases, development, protocol["training_domain"], d)
    rows = [done[case_key(c)] for c in cases]
    output = prepare_output(output, args.overwrite)
    write_outputs(output, protocol, cases, rows, distances, development, seconds, args.workers)


def write_outputs(output, protocol, cases, rows, distances, development, seconds, workers):
    names = quantity_names(int(protocol["stocks"]))
    public = [{k: v for k, v in c.items() if k not in ("slot",)} for c in cases]
    (output / "cases.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    write_csv(output / "cases.csv", [dict(case_id=c["case_id"], split=c["split"], option=c["option"],
                                          reason=c["reason"], draw=c["draw"], **c["parameters"]) for c in cases])
    write_csv(output / "references.csv", [dict(case_id=r["case_id"], split=r["split"], **dict(zip(names, r["values"])))
                                          for r in rows])
    write_csv(output / "reference_standard_errors.csv",
              [dict(case_id=r["case_id"], split=r["split"], **dict(zip(names, r["standard_errors"]))) for r in rows])
    distance = {d["case_id"]: d for d in distances}
    checks = []
    for r in rows:
        checks.append(dict(
            case_id=r["case_id"], split=r["split"], accepted_points=r["accepted_points"],
            above_target=r["above_target"], price_standard_error=r["standard_errors"][0],
            max_relative_se=r["max_relative_se"], max_relative_se_quantity=r["max_relative_se_quantity"],
            reference_seconds=r["reference_seconds"], seed=r["seed"], check_seed=r["check_seed"],
            check_max_abs_z=r["check_max_abs_z"], check_worst_quantity=r["check_worst_quantity"],
            check_rms_z=r["check_rms_z"], check_count_abs_z_over_5=r["check_count_abs_z_over_5"],
            time_step_max_abs=r["time_step_max_abs"], time_step_worst_quantity=r["time_step_worst_quantity"],
            time_step_price_difference=r["time_step_price_difference"],
            bump_halving_max_relative=r["bump_halving_max_relative"],
            bump_halving_worst_quantity=r["bump_halving_worst_quantity"],
            separation_distance=distance[r["case_id"]]["distance"],
            nearest_development_contract=distance[r["case_id"]]["nearest"]))
    write_csv(output / "reference_checks.csv", checks)
    counts = {s: sum(c["split"] == s for c in cases) for s in dict.fromkeys(c["split"] for c in cases)}
    nearest = min(distances, key=lambda d: d["distance"])
    frozen = dict(protocol, frozen_at=datetime.now(timezone.utc).isoformat(), case_counts=counts,
                  total_cases=len(cases), flagged_above_target=[r["case_id"] for r in rows if r["above_target"]],
                  separation_result=dict(minimum=nearest["distance"], case_id=nearest["case_id"],
                                         development_contract=nearest["nearest"],
                                         by_split={s: min(d["distance"] for d in distances if d["split"] == s)
                                                   for s in counts}),
                  development_contracts_checked=len(development),
                  development_contracts_sha256=digest(ROOT / protocol["separation"]["development_contracts"]),
                  reference_build_seconds=seconds,
                  cases_sha256=digest(output / "cases.json"), references_sha256=digest(output / "references.csv"),
                  reference_standard_errors_sha256=digest(output / "reference_standard_errors.csv"))
    (output / "protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n", encoding="utf-8")
    report(output, frozen, checks, len(names))
    sources = pricing_sources() + [Path(__file__), ROOT / "experiments/records.py", ROOT / "src/neural/coordinates.py"]
    write_metadata(output, PROTOCOL, sources,
                   extra=dict(provenance_sha256={relative(ROOT / protocol["separation"]["development_contracts"]):
                                                 frozen["development_contracts_sha256"]},
                              reference_build_seconds=seconds, workers=f"{workers} processes, BLAS on one thread"))
    print(f"Saved {output}; seal {digest(output / 'protocol.json')}", flush=True)


def report(output, protocol, checks, quantities):
    splits, reference = protocol["splits"], protocol["reference"]
    lines = [f"# Frozen {protocol['stocks']}-stock Heston basket test set", "",
             f"Frozen at {protocol['frozen_at']}. {protocol['purpose']}", "", protocol["model"], "",
             "## Cases", "", "| Split | Cases | Definition |", "|---|---:|---|"]
    for split, count in protocol["case_counts"].items():
        lines.append(f"| {split} | {count} | {splits[split]['reason']} |")
    lines += ["", f"Strike {protocol['strike']:g}; calls and puts alternate within each split. "
              + protocol["sampling"]["latin_hypercube"], "", "Training domain:", "", "| Parameter | Range |", "|---|---|"]
    lines += [f"| {p} | {v[0]:g} to {v[1]:g} |" for p, v in protocol["training_domain"].items()]
    lines += ["", f"Weights: {protocol['domain_constraints']['weights']} Correlation: "
              f"{protocol['domain_constraints']['correlation']}", "",
              "Extrapolation values: " + "; ".join(f"{k} {', '.join(map(str, v))}"
                                                   for k, v in splits["extrapolation"]["values"].items()) + ".",
              "", "## References", "", reference["method"], "", reference["acceptance"], "",
              "| Points per replicate | Cases | interpolation | short_maturity | extrapolation | anchor | median s | max s |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for points in reference["points_ladder"]:
        hit = [c for c in checks if c["accepted_points"] == points and not c["above_target"]]
        if hit:
            per = [sum(c["split"] == s for c in hit) for s in ("interpolation", "short_maturity", "extrapolation",
                                                                 "anchor")]
            secs = [c["reference_seconds"] for c in hit]
            lines.append(f"| {points} | {len(hit)} | " + " | ".join(map(str, per))
                         + f" | {np.median(secs):.1f} | {max(secs):.1f} |")
    flagged = protocol["flagged_above_target"]
    lines += ["", f"Above the price standard-error target of {reference['price_standard_error_target']:g} at "
              f"{reference['points_ladder'][-1]} points: " + (", ".join(flagged) if flagged else "none") + ".", "",
              "Largest relative standard errors (standard error / max(|reference|, floor)):", "",
              "| Case | Quantity | Relative SE | Price SE |", "|---|---|---:|---:|"]
    for c in sorted(checks, key=lambda c: -c["max_relative_se"])[:10]:
        lines.append(f"| {c['case_id']} | {c['max_relative_se_quantity']} | {c['max_relative_se']:.1e} | "
                     f"{c['price_standard_error']:.1e} |")
    total = sum(c["check_count_abs_z_over_5"] for c in checks)
    lines += ["", "## Cross-checks (recorded, never used to change a reference)", "",
              f"Independent run (own seed, check points): RMS z over all cases and quantities "
              f"{np.sqrt(np.mean([c['check_rms_z'] ** 2 for c in checks])):.3f}; {total} of "
              f"{quantities * len(checks)} z-values exceed 5 in magnitude.", "",
              "| Check | Worst value | Case | Quantity |", "|---|---:|---|---|"]
    for label, key, quantity in (("Independent run max abs z", "check_max_abs_z", "check_worst_quantity"),
                                 ("Doubled time step, max abs z", "time_step_max_abs",
                                  "time_step_worst_quantity"),
                                 ("Bump halving relative change", "bump_halving_max_relative",
                                  "bump_halving_worst_quantity")):
        w = max(checks, key=lambda c: abs(c[key]))
        lines.append(f"| {label} | {w[key]:.2e} | {w['case_id']} | {w[quantity]} |")
    steps = np.array([c["time_step_price_difference"] for c in checks])
    lines += ["", f"Doubled time step, price difference: median {np.median(steps):.2e}, largest magnitude "
              f"{np.abs(steps).max():.2e} (currency, strike 100).", "",
              "z = difference / sqrt(se_1^2 + se_2^2). Relative differences divide by max(|reference|, floor), "
              "with floors 1e-7 for Gamma and 1e-4 otherwise.", ""]
    sep = protocol["separation_result"]
    lines += ["## Separation from development contracts", "", protocol["separation"]["metric"], "",
              f"Minimum {sep['minimum']:.3f} ({sep['case_id']} against {sep['development_contract']}); by split: "
              + ", ".join(f"{s} {v:.3f}" for s, v in sep["by_split"].items())
              + f". No case duplicates any of the {protocol['development_contracts_checked']} development contracts.",
              "", "No pricing method has been evaluated on these cases.", "", "## Rules", ""]
    lines += [f"- {rule}" for rule in protocol["rules"]]
    lines += ["", "`protocol.sha256` identifies `protocol.json`, which pins `cases.json`, `references.csv` and "
              "`reference_standard_errors.csv`; `metadata.json` pins every output and the generating sources. "
              "Verify with `python -m experiments.verify results/test_set_v1`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
