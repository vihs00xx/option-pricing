"""Generate, reference and freeze the four-stock test set. No pricing method is evaluated.

Run from 4d/: python -m experiments.test_set.freeze_test_set
Per-case results go to artifacts/test_set_v1_build/cases.jsonl as they finish, so an
interrupted build resumes; the sealed folder is written only at the end.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import json
from multiprocessing import Lock, Pool, Semaphore
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from experiments.records import ROOT, digest, pricing_sources, prepare_output, relative, write_csv, write_metadata
from src.pricing import Basket, gauss_hermite_values, quantity_names, rqmc_estimate

PROTOCOL = ROOT / "experiments/config/protocol/four_stock_test_v1.json"
WORK = ROOT / "artifacts/test_set_v1_build/cases.jsonl"
D = 4
PAIRS = [(i, j) for i in range(D) for j in range(i + 1, D)]
FLOORS = {"price": 1e-4, "delta": 1e-4, "gamma": 1e-7, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4,
          "correlation": 1e-4}
LARGE_NODES, LARGE_SLOTS = 96, 2
WORKERS = 8


# ---------------------------------------------------------------- cases

def matrix_from(entries):
    R = np.eye(D)
    for (i, j), value in zip(PAIRS, entries):
        R[i, j] = R[j, i] = value
    return R


def smallest_eigenvalue(R):
    return float(np.linalg.eigvalsh(R)[0])


class Stream:
    """One split's Latin hypercube rows and correlation draws, continued for spares."""

    def __init__(self, protocol, split):
        seed, offset = protocol["seed"], protocol["sampling"]["split_offsets"][split]
        self.domain = protocol["training_domain"]
        self.box = protocol["sampling"]["box_parameters"]
        self.engine = qmc.LatinHypercube(d=len(self.box) + 3, seed=seed + offset)
        self.rng = np.random.Generator(np.random.PCG64(seed + 100 + offset))
        self.min_eigenvalue = protocol["domain_constraints"]["min_eigenvalue"]

    def rows(self, count):
        return list(self.engine.random(count))

    def spare_row(self):
        return self.engine.random(1)[0]

    def values(self, unit):
        values = {p: self.domain[p][0] + u * (self.domain[p][1] - self.domain[p][0])
                  for p, u in zip(self.box, unit)}
        spacings = np.diff(np.concatenate(([0.0], np.sort(unit[len(self.box):]), [1.0])))
        low = self.domain["weight"][0]
        for k, w in enumerate(low + (1 - D * low) * spacings):   # 0.1 + 0.6 * Dirichlet(1, 1, 1, 1)
            values[f"weight_{k + 1}"] = float(w)
        return values

    def correlation(self, fixed=None, min_eigenvalue=None):
        low, high = self.domain["correlation"]
        threshold = self.min_eigenvalue if min_eigenvalue is None else min_eigenvalue
        while True:
            entries = self.rng.uniform(low, high, len(PAIRS))
            if fixed is not None:
                entries[PAIRS.index(fixed[0])] = fixed[1]
            R = matrix_from(entries)
            if smallest_eigenvalue(R) >= threshold:
                return R


def set_correlation(values, R):
    for i, j in PAIRS:
        values[f"corr_{i + 1}{j + 1}"] = float(R[i, j])
    return values


def contract_fields(values, option, strike):
    return dict(spots=[strike * values[f"moneyness_{k}"] for k in range(1, D + 1)], strike=strike,
                tau=values["tau"], rate=values["rate"],
                volatilities=[values[f"volatility_{k}"] for k in range(1, D + 1)],
                dividends=[values[f"dividend_{k}"] for k in range(1, D + 1)],
                weights=[values[f"weight_{k}"] for k in range(1, D + 1)],
                correlation=matrix_from([values[f"corr_{i + 1}{j + 1}"] for i, j in PAIRS]).tolist(),
                option=option)


def make_case(case_id, split, reason, values, option, strike, draw=0, slot=None):
    values = {k: float(v) for k, v in values.items()}
    return dict(case_id=case_id, split=split, reason=reason, option=option, draw=draw, slot=slot,
                parameters=values, contract=contract_fields(values, option, strike))


def extrapolation_slots(split):
    slots = []
    for group, group_values in split["values"].items():
        for value in group_values:
            for k in range(split["cases_per_value"]):
                slots.append((group, value, k))
    return slots


def extrapolate(stream, split, unit, group, value, k):
    """Domain draw with one parameter moved outside; k picks the stock or pair."""
    values = stream.values(unit)
    stock = k % D + 1
    if group in ("moneyness", "volatility", "dividend"):
        values[f"{group}_{stock}"] = value
        target = f"{group}_{stock} = {value}"
    elif group in ("tau", "rate"):
        values[group] = value
        target = f"{group} = {value}"
    elif group == "weight":
        others = [w for w in range(1, D + 1) if w != stock]
        scale = (1 - value) / sum(values[f"weight_{w}"] for w in others)
        for w in others:
            values[f"weight_{w}"] *= scale
        values[f"weight_{stock}"] = value
        target = f"weight_{stock} = {value}"
    if group == "correlation":
        pair = tuple(p - 1 for p in split["correlation_pairs"][k])
        R = stream.correlation(fixed=(pair, value), min_eigenvalue=split["correlation_min_eigenvalue"])
        target = f"corr_{pair[0] + 1}{pair[1] + 1} = {value}"
    else:
        R = stream.correlation()
    return set_correlation(values, R), f"{target} outside the training domain"


def generated_case(stream, protocol, split_name, slot, unit, draw):
    strike, split = protocol["strike"], protocol["splits"][split_name]
    option = "call" if slot % 2 == 0 else "put"
    if split_name == "extrapolation":
        group, value, k = extrapolation_slots(split)[slot]
        values, reason = extrapolate(stream, split, unit, group, value, k)
        case_id = f"extrapolation_{group}_{slot + 1:03d}"
    else:
        values = set_correlation(stream.values(unit), stream.correlation())
        reason = split["reason"]
        if split_name == "short_maturity":
            low, high = split["tau"]
            values["tau"] = low + unit[stream.box.index("tau")] * (high - low)
            reason = "remaining life below the training domain"
        case_id = f"{split_name}_{slot + 1:03d}"
    return make_case(case_id, split_name, reason, values, option, strike, draw, slot)


def anchor_cases(protocol):
    split, strike = protocol["splits"]["anchor"], protocol["strike"]
    cases = []
    for i, row in enumerate(split["cases"]):
        spec = {**split["defaults"], **{k: v for k, v in row.items() if k not in ("name", "option", "note")}}
        values = {"tau": spec["tau"], "rate": spec["rate"]}
        moneyness = spec["moneyness"]
        for k in range(D):
            values[f"moneyness_{k + 1}"] = moneyness[k] if isinstance(moneyness, list) else moneyness
            values[f"volatility_{k + 1}"] = spec["volatilities"][k]
            values[f"dividend_{k + 1}"] = spec["dividends"][k]
            values[f"weight_{k + 1}"] = spec["weights"][k]
        correlation = spec["correlation"]
        if not isinstance(correlation, list):
            correlation = (np.full((D, D), correlation) + (1 - correlation) * np.eye(D)).tolist()
        set_correlation(values, np.array(correlation))
        cases.append(make_case(f"anchor_{i + 1:02d}_{row['name']}", "anchor",
                               row.get("note", split["reason"]), values, row["option"], strike, slot=i))
    return cases


# ------------------------------------------------------------ references

def floor(quantity):
    return FLOORS[quantity.split("_")[0]]


def relative_differences(values, reference, names):
    return {q: abs(values[i] - reference[i]) / max(abs(reference[i]), floor(q)) for i, q in enumerate(names)}


_lock = _large = None


def _init(lock, large):
    global _lock, _large
    _lock, _large = lock, large


def gauss_hermite(c, nodes, **kwargs):
    gate = _large if _large is not None and nodes >= LARGE_NODES else None
    if gate:
        gate.acquire()
    try:
        return gauss_hermite_values(c, nodes, **kwargs)
    finally:
        if gate:
            gate.release()


def reference_job(task):
    """Ladder, then the recorded cross-checks; returns one JSON-ready row."""
    case, settings, rqmc_seed = task
    c = Basket(**case["contract"])
    names = quantity_names(c)
    ladder, tolerance = settings["node_ladder"], settings["tolerance"]
    levels, changes, seconds = {}, {}, 0.0
    accepted = confirm = None
    failed = []
    previous = None
    for nodes in ladder:
        start = perf_counter()
        levels[nodes] = gauss_hermite(c, nodes, direction=settings["direction"])
        seconds += perf_counter() - start
        if previous is not None:
            changes[nodes] = float(np.max(np.abs(levels[nodes] - levels[previous])))
            if accepted is None:
                if changes[nodes] < tolerance:
                    accepted = nodes
            elif changes[nodes] < tolerance:
                confirm = nodes
                break
            else:
                failed.append(accepted)
                accepted = None
        previous = nodes
    row = dict(key=case_key(case), case_id=case["case_id"], split=case["split"], draw=case["draw"],
               accepted_nodes=accepted, confirm_nodes=confirm,
               confirm_change=changes.get(confirm), failed_confirmations=failed,
               changes={str(k): v for k, v in changes.items()}, reference_seconds=seconds)
    if accepted is None or confirm is None:
        row["accepted_nodes"] = None
        return save(row)
    reference = levels[accepted]
    row["values"] = reference.tolist()
    equal = gauss_hermite(c, accepted, direction="equal")
    rel = relative_differences(equal, reference, names)
    worst = max(rel, key=rel.get)
    row.update(equal_max_relative=rel[worst], equal_worst_quantity=worst,
               equal_worst_absolute=float(equal[names.index(worst)] - reference[names.index(worst)]))
    half = gauss_hermite(c, accepted, direction=settings["direction"], bump_scale=0.5)
    bumped = [q for q in names if not q.startswith(("price", "delta", "gamma"))]
    rel = {q: v for q, v in relative_differences(half, reference, names).items() if q in bumped}
    worst = max(rel, key=rel.get)
    row.update(bump_halving_max_relative=rel[worst], bump_halving_worst_quantity=worst)
    start = perf_counter()
    e = rqmc_estimate(c, 65536, 16, rqmc_seed, direction=settings["direction"], greeks=True, control=True)
    est = np.array([e.values[q] for q in names])
    se = np.array([e.standard_errors[q] for q in names])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (est - reference) / se, np.where(est == reference, 0.0, np.inf))
    inside = [e.intervals[q][0] <= reference[i] <= e.intervals[q][1] for i, q in enumerate(names)]
    worst = int(np.argmax(np.abs(z)))
    row.update(rqmc_seed=rqmc_seed, rqmc_seconds=perf_counter() - start, rqmc_z=z.tolist(),
               rqmc_max_abs_z=float(np.abs(z[worst])), rqmc_worst_quantity=names[worst],
               rqmc_rms_z=float(np.sqrt(np.mean(z ** 2))), rqmc_count_abs_z_over_5=int(np.sum(np.abs(z) > 5)),
               rqmc_inside_95=float(np.mean(inside)), rqmc_price_relative_error=float(
                   abs(est[0] - reference[0]) / max(abs(reference[0]), floor("price"))))
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
    import hashlib
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
    spots, strike = np.asarray(fields["spots"], float), float(fields["strike"])
    R = np.asarray(fields["correlation"], float)
    return np.concatenate([spots / strike, [fields["tau"]], fields["volatilities"], [fields["rate"]],
                           fields["dividends"], fields["weights"], [R[i, j] for i, j in PAIRS]])


def widths(domain):
    w = lambda name: domain[name][1] - domain[name][0]
    return np.array([w("moneyness_1")] * D + [w("tau")] + [w("volatility_1")] * D + [w("rate")]
                    + [w("dividend_1")] * D + [w("weight")] * D + [w("correlation")] * len(PAIRS))


def same_contract(a, b):
    return a["option"] == b["option"] and all(
        np.allclose(np.asarray(a[k], float), np.asarray(b[k], float), rtol=0, atol=1e-9)
        for k in ("spots", "strike", "tau", "rate", "volatilities", "dividends", "weights", "correlation"))


def development_contracts(protocol):
    path = ROOT / protocol["separation"]["development_contracts"]
    rows = json.loads(path.read_text(encoding="utf-8"))["contracts"]
    return [(row["name"], {**row["contract"], "correlation": np.asarray(row["contract"]["correlation"]).tolist()})
            for row in rows]


def separation(cases, development, domain):
    scale = widths(domain)
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

    streams = {s: Stream(protocol, s) for s in ("interpolation", "short_maturity", "extrapolation")}
    counts = dict(interpolation=protocol["splits"]["interpolation"]["count"],
                  short_maturity=protocol["splits"]["short_maturity"]["count"],
                  extrapolation=len(extrapolation_slots(protocol["splits"]["extrapolation"])))
    slots = {}
    for split, stream in streams.items():
        slots[split] = [generated_case(stream, protocol, split, i, unit, 0)
                        for i, unit in enumerate(stream.rows(counts[split]))]
    anchors = anchor_cases(protocol)
    order = ["interpolation", "short_maturity", "extrapolation"]
    index_of = {}
    for split in order:
        for case in slots[split]:
            index_of[(split, case["slot"])] = len(index_of)
    for case in anchors:
        index_of[("anchor", case["slot"])] = len(index_of)

    replacements = []
    done = load_work()
    start = perf_counter()
    with Pool(args.workers, initializer=_init, initargs=(Lock(), Semaphore(LARGE_SLOTS))) as pool:
        while True:
            current = [c for split in order for c in slots[split]] + anchors
            todo = [c for c in current if case_key(c) not in done]
            if todo:
                tasks = [(c, settings, protocol["seed"] + 10000 + index_of[(c["split"], c["slot"])]) for c in todo]
                tasks.sort(key=lambda t: t[0]["split"] != "extrapolation")  # likely slow ones first
                for i, row in enumerate(pool.imap_unordered(reference_job, tasks), 1):
                    done[row["key"]] = row
                    if i % 10 == 0 or i == len(tasks):
                        print(f"[{perf_counter() - start:7.1f}s] {i}/{len(tasks)} cases referenced", flush=True)
            failed = [c for c in current if done[case_key(c)]["accepted_nodes"] is None]
            bad_anchors = [c["case_id"] for c in failed if c["split"] == "anchor"]
            if bad_anchors:
                raise RuntimeError(f"Anchor cases not accepted by {settings['node_ladder'][-1]} nodes: {bad_anchors}")
            if not failed:
                break
            for case in failed:   # slot order within each split keeps the spare draws deterministic
                stream = streams[case["split"]]
                spare = generated_case(stream, protocol, case["split"], case["slot"], stream.spare_row(),
                                       case["draw"] + 1)
                row = done[case_key(case)]
                replacements.append(dict(case_id=case["case_id"], split=case["split"], draw=case["draw"],
                                         replacement_draw=spare["draw"], last_change=row["changes"].get(
                                             str(settings["node_ladder"][-1])), parameters=case["parameters"],
                                         option=case["option"]))
                slots[case["split"]][case["slot"]] = spare
                print(f"Replacing {case['case_id']} (draw {case['draw']}) with draw {spare['draw']}", flush=True)
    seconds = perf_counter() - start

    cases = [c for split in order for c in slots[split]] + anchors
    if len({c["case_id"] for c in cases}) != len(cases):
        raise RuntimeError("Case identifiers are not unique")
    for case in cases:
        Basket(**case["contract"])   # validates every contract
    development = development_contracts(protocol)
    distances = separation(cases, development, protocol["training_domain"])
    rows = [done[case_key(c)] for c in cases]
    output = prepare_output(output, args.overwrite)
    write_outputs(output, protocol, cases, rows, replacements, distances, development, seconds)


def write_outputs(output, protocol, cases, rows, replacements, distances, development, seconds):
    names = quantity_names(Basket(**cases[0]["contract"]))
    public = [{k: v for k, v in c.items() if k not in ("slot",)} for c in cases]
    (output / "cases.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    write_csv(output / "cases.csv", [dict(case_id=c["case_id"], split=c["split"], option=c["option"],
                                          reason=c["reason"], draw=c["draw"], **c["parameters"]) for c in cases])
    write_csv(output / "references.csv", [dict(case_id=r["case_id"], split=r["split"], **dict(zip(names, r["values"])))
                                          for r in rows])
    replaced = {r["case_id"]: r for r in replacements}
    distance = {d["case_id"]: d for d in distances}
    checks = []
    for r in rows:
        checks.append(dict(
            case_id=r["case_id"], split=r["split"], accepted_nodes=r["accepted_nodes"],
            confirm_nodes=r["confirm_nodes"], confirm_change=r["confirm_change"],
            failed_confirmations=" ".join(map(str, r["failed_confirmations"])),
            reference_seconds=r["reference_seconds"],
            equal_max_relative=r["equal_max_relative"], equal_worst_quantity=r["equal_worst_quantity"],
            equal_worst_absolute=r["equal_worst_absolute"],
            bump_halving_max_relative=r["bump_halving_max_relative"],
            bump_halving_worst_quantity=r["bump_halving_worst_quantity"],
            rqmc_seed=r["rqmc_seed"], rqmc_max_abs_z=r["rqmc_max_abs_z"], rqmc_worst_quantity=r["rqmc_worst_quantity"],
            rqmc_rms_z=r["rqmc_rms_z"], rqmc_count_abs_z_over_5=r["rqmc_count_abs_z_over_5"],
            rqmc_inside_95=r["rqmc_inside_95"], rqmc_price_relative_error=r["rqmc_price_relative_error"],
            draw=r["draw"],
            replacement_log="; ".join(f"draw {x['draw']} not accepted by 128 (last change {x['last_change']:.1e})"
                                      for x in replacements if x["case_id"] == r["case_id"]),
            separation_distance=distance[r["case_id"]]["distance"],
            nearest_development_contract=distance[r["case_id"]]["nearest"]))
    write_csv(output / "reference_checks.csv", checks)
    counts = {s: sum(c["split"] == s for c in cases) for s in dict.fromkeys(c["split"] for c in cases)}
    nearest = min(distances, key=lambda d: d["distance"])
    frozen = dict(protocol, frozen_at=datetime.now(timezone.utc).isoformat(), case_counts=counts,
                  total_cases=len(cases), replacements=replacements,
                  separation_result=dict(minimum=nearest["distance"], case_id=nearest["case_id"],
                                         development_contract=nearest["nearest"],
                                         by_split={s: min(d["distance"] for d in distances if d["split"] == s)
                                                   for s in counts}),
                  development_contracts_checked=len(development),
                  development_contracts_sha256=digest(ROOT / protocol["separation"]["development_contracts"]),
                  reference_build_seconds=seconds,
                  cases_sha256=digest(output / "cases.json"), references_sha256=digest(output / "references.csv"))
    (output / "protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n", encoding="utf-8")
    report(output, frozen, checks)
    # The references come from the patched src/pricing engine, so verification pins
    # it, the generator and the protocol. The development contracts are pinned by
    # hash inside protocol.json.
    write_metadata(output, PROTOCOL, pricing_sources() + [Path(__file__), ROOT / "experiments/records.py"],
                   extra=dict(provenance_sha256={relative(ROOT / protocol["separation"]["development_contracts"]):
                                                 frozen["development_contracts_sha256"]},
                              reference_build_seconds=seconds, workers="up to 8 processes, BLAS on one thread"))
    print(f"Saved {output}; seal {digest(output / 'protocol.json')}", flush=True)


def report(output, protocol, checks):
    splits = protocol["splits"]
    lines = ["# Frozen four-stock test set", "", f"Frozen at {protocol['frozen_at']}. {protocol['purpose']}", "",
             "## Cases", "", "| Split | Cases | Definition |", "|---|---:|---|"]
    for split, count in protocol["case_counts"].items():
        lines.append(f"| {split} | {count} | {splits[split]['reason']} |")
    lines += ["", f"Strike {protocol['strike']:g}; calls and puts alternate within each split. Latin hypercube "
              f"seed {protocol['seed']} plus split offset; correlations from a separate generator (seed + 100 + "
              "offset). " + protocol["sampling"]["latin_hypercube"], "", "Training domain:", "",
              "| Parameter | Range |", "|---|---|"]
    lines += [f"| {p} | {v[0]:g} to {v[1]:g} |" for p, v in protocol["training_domain"].items()]
    lines += ["", f"Weights: {protocol['domain_constraints']['weights']} Correlation: "
              f"{protocol['domain_constraints']['correlation']}", "",
              "Extrapolation values (4 cases each, stock or pair rotated): "
              + "; ".join(f"{k} {', '.join(map(str, v))}" for k, v in splits["extrapolation"]["values"].items()) + ".",
              "", "## References", "", protocol["reference"]["method"], "", protocol["reference"]["acceptance"], "",
              "| Accepted nodes | Cases | interpolation | short_maturity | extrapolation | anchor | median s | max s |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for nodes in protocol["reference"]["node_ladder"]:
        hit = [c for c in checks if c["accepted_nodes"] == nodes]
        if hit:
            per = [sum(c["split"] == s for c in hit) for s in ("interpolation", "short_maturity", "extrapolation",
                                                                 "anchor")]
            secs = [c["reference_seconds"] for c in hit]
            lines.append(f"| {nodes} | {len(hit)} | " + " | ".join(map(str, per))
                         + f" | {np.median(secs):.1f} | {max(secs):.1f} |")
    lines += ["", "Replacements (pre-declared rule: " + protocol["reference"]["replacement_rule"] + "):", ""]
    if protocol["replacements"]:
        lines += ["| Slot | Draw replaced | Last change at 128 | Option |", "|---|---:|---:|---|"]
        lines += [f"| {r['case_id']} | {r['draw']} | {r['last_change']:.1e} | {r['option']} |"
                  for r in protocol["replacements"]]
    else:
        lines.append("None.")
    worst = lambda key: max(checks, key=lambda c: abs(c[key]))
    lines += ["", "## Cross-checks (recorded, never used to change a reference)", "",
              "| Check | Worst value | Case | Quantity |", "|---|---:|---|---|"]
    for label, key, quantity in (("Equal-direction relative difference", "equal_max_relative", "equal_worst_quantity"),
                                 ("Bump halving relative change", "bump_halving_max_relative",
                                  "bump_halving_worst_quantity"),
                                 ("RQMC max abs z", "rqmc_max_abs_z", "rqmc_worst_quantity")):
        w = worst(key)
        lines.append(f"| {label} | {w[key]:.2e} | {w['case_id']} | {w[quantity]} |")
    total = sum(c["rqmc_count_abs_z_over_5"] for c in checks)
    lines += ["", f"RQMC (65536 points x 16 replicates, geometric control): RMS z over all cases and quantities "
              f"{np.sqrt(np.mean([c['rqmc_rms_z'] ** 2 for c in checks])):.3f}; share of reported 95% intervals "
              f"containing the reference {np.mean([c['rqmc_inside_95'] for c in checks]):.3f}; {total} of "
              f"{27 * len(checks)} quantity z-values exceed 5 in magnitude.", "",
              "Relative differences divide by max(|reference|, floor), with floors 1e-7 for Gamma and 1e-4 "
              "otherwise. Equal loadings shrink the smoothing factor when volatilities differ strongly, so that "
              "cross-check is expected to be under-resolved at the accepted level in such cases (Phase 0: it "
              "converges to the volatility-direction value with more nodes).", "",
              "Largest equal-direction differences:", "",
              "| Case | Quantity | Relative | Accepted nodes |", "|---|---|---:|---:|"]
    for c in sorted(checks, key=lambda c: -c["equal_max_relative"])[:10]:
        lines.append(f"| {c['case_id']} | {c['equal_worst_quantity']} | {c['equal_max_relative']:.1e} | "
                     f"{c['accepted_nodes']} |")
    lines += ["", "Largest RQMC |z|:", "", "| Case | Quantity | max abs z | RMS z | inside 95% |",
              "|---|---|---:|---:|---:|"]
    for c in sorted(checks, key=lambda c: -c["rqmc_max_abs_z"])[:10]:
        lines.append(f"| {c['case_id']} | {c['rqmc_worst_quantity']} | {c['rqmc_max_abs_z']:.2f} | "
                     f"{c['rqmc_rms_z']:.2f} | {c['rqmc_inside_95']:.2f} |")
    lines += ["", "Largest bump-halving changes:", "", "| Case | Quantity | Relative |", "|---|---|---:|"]
    for c in sorted(checks, key=lambda c: -c["bump_halving_max_relative"])[:5]:
        lines.append(f"| {c['case_id']} | {c['bump_halving_worst_quantity']} | {c['bump_halving_max_relative']:.1e} |")
    sep = protocol["separation_result"]
    lines += ["", "## Separation from development contracts", "", protocol["separation"]["metric"], "",
              f"Minimum {sep['minimum']:.3f} ({sep['case_id']} against {sep['development_contract']}); by split: "
              + ", ".join(f"{s} {v:.3f}" for s, v in sep["by_split"].items())
              + f". No case duplicates any of the {protocol['development_contracts_checked']} development contracts.",
              "", "No pricing method has been evaluated on these cases.", "", "## Rules", ""]
    lines += [f"- {rule}" for rule in protocol["rules"]]
    lines += ["", "`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; "
              "`metadata.json` pins every output and the generating sources. Verify with "
              "`python -m experiments.verify results/test_set_v1`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
