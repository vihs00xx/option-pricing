"""Generate, reference and seal heston_test_v1. No pricing method is evaluated here.

Run from heston/: python -m experiments.test_set.freeze_test_set [--workers 12]
Per-case rows go to artifacts/test_set_v1_build/cases.jsonl as they finish, so an
interrupted build resumes; the sealed folder results/test_set_v1 is written at the end.
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from experiments.records import ROOT, digest, write_csv, write_json, write_metadata
from src.heston.model import FIELDS, QUANTITIES, Contracts, development_contracts, feller_violated, load_domain

PROTOCOL = ROOT / "experiments/config/protocol/heston_test_v1.json"
WORK = ROOT / "artifacts/test_set_v1_build/cases.jsonl"
OUTPUT = ROOT / "results/test_set_v1"
BOX = ("S", "tau", "r", "q", "v0", "theta", "kappa", "xi", "rho")
RANGE_KEY = {"S": "moneyness"}


class Stream:
    """One split's Latin hypercube rows, continued for spare draws."""

    def __init__(self, seed):
        self.engine = qmc.LatinHypercube(d=len(BOX), seed=seed)

    def row(self):
        return self.engine.random(1)[0]


def values_from(unit, ranges, tau_range=None):
    out = {}
    for name, u in zip(BOX, unit):
        lo, hi = tau_range if (name == "tau" and tau_range) else ranges[RANGE_KEY.get(name, name)]
        out[name] = float(lo + u * (hi - lo))
    return out


def extrapolation_slots(split):
    return [(g, v, k) for g, vals in split["values"].items() for v in vals for k in range(split["cases_per_value"])]


def generated_case(protocol, ranges, split_name, slot, unit, draw):
    split = protocol["splits"][split_name]
    call = slot % 2 == 0
    if split_name == "extrapolation":
        group, value, _ = extrapolation_slots(split)[slot]
        vals = values_from(unit, ranges)
        vals[group] = float(value)
        return dict(case_id=f"extrapolation_{group}_{slot + 1:03d}", split=split_name, draw=draw, slot=slot,
                    reason=f"{group} = {value} outside the training domain", call=call, parameters=vals)
    vals = values_from(unit, ranges, split.get("tau"))
    return dict(case_id=f"{split_name}_{slot + 1:03d}", split=split_name, draw=draw, slot=slot,
                reason=split["reason"], call=call, parameters=vals)


def anchor_cases(protocol):
    a = protocol["splits"]["anchor"]
    cases = []
    for i, row in enumerate(a["cases"]):
        vals = {k: float(row.get(k, a["defaults"][k])) for k in BOX}
        cases.append(dict(case_id=f"anchor_{i + 1:02d}_{row['name']}", split="anchor", draw=0, slot=i,
                          reason=row.get("note", a["reason"]), call=bool(row["call"]), parameters=vals))
    return cases


def contracts_of(cases):
    return Contracts.from_records([{**c["parameters"], "call": c["call"]} for c in cases])


def key(case):
    blob = json.dumps([case["case_id"], case["draw"], case["parameters"], case["call"]], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def reference_job(task):
    case, rqmc_seed = task
    import torch
    torch.set_num_threads(1)
    from src.heston import torch_reference as tr
    from src.heston.monte_carlo import rqmc_quantities
    from src.heston.reference import reference
    c = contracts_of([case])
    t = perf_counter()
    ref, info = reference(c)
    ref_seconds = perf_counter() - t
    tw, tinfo = tr.twin(c)
    twin_diff = {q: float(abs(tw[q][0] - ref[q][0])) for q in QUANTITIES}
    row = dict(key=key(case), case_id=case["case_id"], split=case["split"], draw=case["draw"],
               values={q: float(ref[q][0]) for q in QUANTITIES}, level=int(info["level"][0]),
               change=float(info["change"][0]), converged=bool(info["converged"][0]),
               worst_quantity=str(info["worst_quantity"][0]), U=float(info["U"][0]),
               reference_seconds=ref_seconds, twin_nodes=int(tinfo["nodes"][0]),
               twin_max_diff=max(twin_diff.values()), twin_worst=max(twin_diff, key=twin_diff.get))
    row["accepted"] = row["converged"] and row["twin_max_diff"] <= 1e-8
    if row["accepted"]:
        t = perf_counter()
        val, se = rqmc_quantities(c, 12, 1 / 32, rqmc_seed)
        z = {q: (val[q] - row["values"][q]) / se[q] if se[q] > 0 else 0.0 for q in QUANTITIES}
        row.update(rqmc_seed=rqmc_seed, rqmc_seconds=perf_counter() - t, rqmc_z=z,
                   rqmc_max_abs_z=max(abs(v) for v in z.values()),
                   rqmc_worst=max(z, key=lambda q: abs(z[q])), rqmc_values=val, rqmc_se=se)
    return row


def load_work():
    rows = {}
    if WORK.exists():
        for line in WORK.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows[r["key"]] = r
            except json.JSONDecodeError:
                continue
    return rows


def separation(cases, ranges):
    main, short, hard = development_contracts()
    dev = Contracts.concat(main, short, hard)
    widths = np.array([ranges[RANGE_KEY.get(n, n)][1] - ranges[RANGE_KEY.get(n, n)][0] for n in BOX])
    D = np.stack([getattr(dev, n) for n in BOX], axis=1)
    out = []
    for c in cases:
        p = np.array([c["parameters"][n] for n in BOX])
        dist = np.max(np.abs(D - p) / widths, axis=1)
        k = int(np.argmin(dist))
        out.append(dict(case_id=c["case_id"], split=c["split"], distance=float(dist[k]), nearest_index=k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{OUTPUT} exists; it is sealed")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ranges = load_domain()["ranges"]
    seed = protocol["seed"]
    offsets = protocol["sampling"]["split_offsets"]
    streams = {s: Stream(seed + offsets[s]) for s in offsets}
    counts = {"interpolation": protocol["splits"]["interpolation"]["count"],
              "short_maturity": protocol["splits"]["short_maturity"]["count"],
              "extrapolation": len(extrapolation_slots(protocol["splits"]["extrapolation"]))}
    slots = {s: [generated_case(protocol, ranges, s, i, streams[s].row(), 0) for i in range(counts[s])]
             for s in counts}
    anchors = anchor_cases(protocol)
    order = list(counts)
    seed_of = {}
    for s in order + ["anchor"]:
        for case in (slots[s] if s != "anchor" else anchors):
            seed_of[(s, case["slot"])] = seed + 10000 + len(seed_of)
    done = load_work()
    WORK.parent.mkdir(parents=True, exist_ok=True)
    replacements = []
    start = perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        while True:
            current = [c for s in order for c in slots[s]] + anchors
            todo = [c for c in current if key(c) not in done]
            if todo:
                tasks = [(c, seed_of[(c["split"], c["slot"])]) for c in todo]
                for i, row in enumerate(pool.map(reference_job, tasks), 1):
                    done[row["key"]] = row
                    with WORK.open("a", encoding="utf-8", newline="\n") as fh:
                        fh.write(json.dumps(row) + "\n")
                    if i % 20 == 0 or i == len(tasks):
                        print(f"[{perf_counter() - start:7.1f}s] {i}/{len(tasks)} cases referenced", flush=True)
            failed = [c for c in current if not done[key(c)]["accepted"]]
            bad_anchor = [c["case_id"] for c in failed if c["split"] == "anchor"]
            if bad_anchor:
                raise RuntimeError(f"anchor cases failed the reference: {bad_anchor}")
            if not failed:
                break
            for c in failed:
                spare = generated_case(protocol, ranges, c["split"], c["slot"], streams[c["split"]].row(), c["draw"] + 1)
                r = done[key(c)]
                replacements.append(dict(case_id=c["case_id"], draw=c["draw"], change=r["change"],
                                         twin_max_diff=r["twin_max_diff"], parameters=c["parameters"]))
                slots[c["split"]][c["slot"]] = spare
                print(f"replacing {c['case_id']} draw {c['draw']}", flush=True)
    seconds = perf_counter() - start
    cases = [c for s in order for c in slots[s]] + anchors
    rows = [done[key(c)] for c in cases]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    public = [{k: v for k, v in c.items() if k != "slot"} for c in cases]
    write_json(OUTPUT / "cases.json", public)
    write_csv(OUTPUT / "cases.csv", [dict(case_id=c["case_id"], split=c["split"], call=int(c["call"]),
                                          draw=c["draw"], reason=c["reason"], **c["parameters"]) for c in cases])
    write_csv(OUTPUT / "references.csv", [dict(case_id=r["case_id"], split=r["split"],
                                               **{q: repr(r["values"][q]) for q in QUANTITIES}) for r in rows])
    sep = separation(cases, ranges)
    sep_by = {s["case_id"]: s for s in sep}
    fel = feller_violated(contracts_of(cases))
    checks = [dict(case_id=r["case_id"], split=r["split"], level=r["level"], change=r["change"],
                   worst_quantity=r["worst_quantity"], U=r["U"], reference_seconds=r["reference_seconds"],
                   twin_nodes=r["twin_nodes"], twin_max_diff=r["twin_max_diff"], twin_worst=r["twin_worst"],
                   rqmc_max_abs_z=r["rqmc_max_abs_z"], rqmc_worst=r["rqmc_worst"],
                   rqmc_price_z=r["rqmc_z"]["price"], feller_violated=int(f),
                   separation_distance=sep_by[r["case_id"]]["distance"]) for r, f in zip(rows, fel)]
    write_csv(OUTPUT / "reference_checks.csv", checks)
    write_json(OUTPUT / "rqmc_crosscheck.json", {r["case_id"]: dict(z=r["rqmc_z"], values=r["rqmc_values"],
                                                                   se=r["rqmc_se"]) for r in rows})
    counts_out = {s: sum(c["split"] == s for c in cases) for s in order + ["anchor"]}
    allz = np.array([[r["rqmc_z"][q] for q in QUANTITIES] for r in rows])
    frozen = dict(protocol, frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  case_counts=counts_out, total_cases=len(cases), replacements=replacements,
                  feller_violations={s: int(sum(f for f, c in zip(fel, cases) if c["split"] == s)) for s in counts_out},
                  separation_result=dict(minimum=min(x["distance"] for x in sep),
                                         by_split={s: min(x["distance"] for x in sep if x["split"] == s) for s in counts_out}),
                  reference_build_seconds=seconds,
                  rqmc_summary=dict(rms_z=float(np.sqrt(np.mean(allz**2))), inside_1p96=float(np.mean(np.abs(allz) <= 1.96)),
                                    max_abs_z=float(np.abs(allz).max())),
                  cases_sha256=digest(OUTPUT / "cases.json"), references_sha256=digest(OUTPUT / "references.csv"))
    write_json(OUTPUT / "protocol.json", frozen)
    (OUTPUT / "protocol.sha256").write_text(digest(OUTPUT / "protocol.json") + "\n", encoding="utf-8")
    write_report(frozen, checks)
    write_metadata(OUTPUT, [PROTOCOL, Path(__file__), ROOT / "src/heston/reference.py",
                            ROOT / "src/heston/characteristic.py", ROOT / "src/heston/model.py",
                            ROOT / "src/heston/torch_reference.py", ROOT / "src/heston/monte_carlo.py"])
    print(f"sealed {OUTPUT}: {digest(OUTPUT / 'protocol.json')}", flush=True)


def write_report(p, checks):
    L = ["# Frozen one-stock Heston test set (heston_test_v1)", "", f"Frozen at {p['frozen_at']}. {p['purpose']}", "",
         "| Split | Cases | Feller violated | Definition |", "|---|---:|---:|---|"]
    for s, n in p["case_counts"].items():
        L.append(f"| {s} | {n} | {p['feller_violations'][s]} | {p['splits'][s]['reason']} |")
    L += ["", f"Latin hypercube seed {p['seed']} + split offset; calls and puts alternate. Extrapolation values "
          "(2 cases each): " + "; ".join(f"{k} {', '.join(map(str, v))}" for k, v in p['splits']['extrapolation']['values'].items()) + ".",
          "", "## References", "", p["reference"]["method"] + ".", "", "Acceptance: " + p["reference"]["acceptance"] + ".",
          "", "Replacement rule (declared before the build): " + p["reference"]["replacement_rule"] + ".", ""]
    if p["replacements"]:
        L += ["| Slot | Draw | Last change | Twin diff |", "|---|---:|---:|---:|"]
        L += [f"| {r['case_id']} | {r['draw']} | {r['change']:.1e} | {r['twin_max_diff']:.1e} |" for r in p["replacements"]]
    else:
        L.append("Replacements: none.")
    lv = {}
    for c in checks:
        lv[c["level"]] = lv.get(c["level"], 0) + 1
    L += ["", "Levels used: " + ", ".join(f"level {k}: {v}" for k, v in sorted(lv.items())) +
          f". Largest last change {max(c['change'] for c in checks):.1e}; largest torch-twin difference "
          f"{max(c['twin_max_diff'] for c in checks):.1e}; torch nodes up to {max(c['twin_nodes'] for c in checks)}.",
          "", "## Cross-check (recorded only)", "",
          f"QE-RQMC, 2^12 Sobol points x 16 scrambles, Brownian bridge, dt = 1/32, all 12 quantities: RMS z "
          f"{p['rqmc_summary']['rms_z']:.2f}, share |z| <= 1.96 {p['rqmc_summary']['inside_1p96']:.3f}, max |z| "
          f"{p['rqmc_summary']['max_abs_z']:.1f}. The z-values include time-step and bump bias, so they are a "
          "sanity check, not a coverage test.", "", "Largest |z|:", "", "| Case | Quantity | max abs z |", "|---|---|---:|"]
    for c in sorted(checks, key=lambda c: -c["rqmc_max_abs_z"])[:10]:
        L.append(f"| {c['case_id']} | {c['rqmc_worst']} | {c['rqmc_max_abs_z']:.1f} |")
    L += ["", "## Separation from development contracts", "", p["separation"]["metric"] + ".", "",
          f"Minimum {p['separation_result']['minimum']:.3f}; by split " +
          ", ".join(f"{s} {v:.3f}" for s, v in p["separation_result"]["by_split"].items()) + ".", "",
          "No pricing method has been evaluated on these cases.", "", "## Rules", ""]
    L += [f"- {r}" for r in p["rules"]]
    L += ["", "`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; "
          "`metadata.json` pins every file and the generating sources.", ""]
    (OUTPUT / "report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
