"""Gauss-Hermite node convergence of the four-stock conditional reference.

Each contract climbs NODE_LEVELS with greeks (direction "volatility") until the
largest change over all 27 quantities against the previous level falls below
TOLERANCE; that level's values are kept and one more level confirms them. A
failed confirmation resumes the climb. Contracts accepted at 64 nodes or more
are cross-checked with equal loadings at the accepted level.

Raw values go to convergence.jsonl one level at a time, so an interrupted run
resumes where it stopped.
"""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import argparse
from multiprocessing import Lock, Pool, Semaphore
from time import perf_counter

import numpy as np

from src.pricing import gauss_hermite_values, quantity_names

from .common import (NODE_LEVELS, OUTPUT, TOLERANCE, WORKERS, append_jsonl, diagnostics, load_contracts,
                     read_jsonl, write_csv)

RAW = OUTPUT / "convergence.jsonl"
LARGE_NODES = 96        # about 0.6 GB per process at 96 nodes, 1.3 GB at 128
LARGE_SLOTS = 2

_lock = _large = None


def _init(lock, large):
    global _lock, _large
    _lock, _large = lock, large


def assess(values):
    """Acceptance from {nodes: values} on the volatility direction, in NODE_LEVELS order."""
    levels = [n for n in NODE_LEVELS if n in values]
    changes = {n: float(np.max(np.abs(values[n] - values[m]))) for m, n in zip(levels, levels[1:])}
    accepted = confirm = None
    failed = []
    for n in levels[1:]:
        if accepted is None:
            if changes[n] < TOLERANCE:
                accepted = n
        elif confirm is None:
            if changes[n] < TOLERANCE:
                confirm = n
            else:
                failed.append(accepted)
                accepted = None
    return dict(accepted=accepted, confirm=confirm, failed_confirmations=failed, changes=changes,
                finished=confirm is not None or levels[-1:] == [NODE_LEVELS[-1]])


def run_level(name, c, direction, nodes):
    gate = _large if nodes >= LARGE_NODES else None
    if gate:
        gate.acquire()
    try:
        start = perf_counter()
        values = gauss_hermite_values(c, nodes, direction=direction)
        seconds = perf_counter() - start
    finally:
        if gate:
            gate.release()
    row = dict(contract=name, direction=direction, nodes=nodes, seconds=seconds, values=values.tolist())
    with _lock:
        append_jsonl(RAW, row)
    return values


def climb(task):
    name, c, done = task
    values = {n: np.array(v) for (direction, n), v in done.items() if direction == "volatility"}
    for nodes in NODE_LEVELS:
        if assess(values)["finished"]:
            break
        if nodes not in values:
            values[nodes] = run_level(name, c, "volatility", nodes)
    accepted = assess(values)["accepted"]
    if accepted is not None and accepted >= 64 and ("equal", accepted) not in done:
        run_level(name, c, "equal", accepted)
    return name


def summarize(contracts):
    rows = read_jsonl(RAW)
    raw = {}
    for row in rows:
        raw[(row["contract"], row["direction"], row["nodes"])] = row
    long_rows, summary = [], []
    for name, c, info in contracts:
        names = quantity_names(c)
        for direction in ("volatility", "equal"):
            previous = None
            for nodes in NODE_LEVELS:
                row = raw.get((name, direction, nodes))
                if row is None:
                    continue
                values = np.array(row["values"])
                for i, quantity in enumerate(names):
                    change = "" if previous is None else abs(values[i] - previous[i])
                    long_rows.append(dict(contract=name, direction=direction, nodes=nodes, quantity=quantity,
                                          value=repr(float(values[i])), change=change, seconds=row["seconds"]))
                previous = values
        values = {n: np.array(raw[(name, "volatility", n)]["values"]) for n in NODE_LEVELS
                  if (name, "volatility", n) in raw}
        result = assess(values)
        accepted = result["accepted"]
        equal = raw.get((name, "equal", accepted)) if accepted else None
        changes = result["changes"]
        summary.append(dict(
            contract=name, group=info["group"], option=c.option, tau=c.tau, **diagnostics(c),
            accepted_nodes=accepted or "", confirm_nodes=result["confirm"] or "",
            confirm_change=changes.get(result["confirm"], "") if result["confirm"] else "",
            failed_confirmations=" ".join(map(str, result["failed_confirmations"])),
            flag="" if result["confirm"] else ("not accepted by 128" if accepted is None else "accepted at 128, unconfirmed"),
            last_change=changes[max(changes)] if changes else "",
            largest_last_change_quantity=names[int(np.argmax(np.abs(values[max(values)] - values[sorted(values)[-2]])))]
            if len(values) > 1 else "",
            equal_direction_max_diff=float(np.max(np.abs(np.array(equal["values"]) - values[accepted])))
            if equal else "",
            seconds_to_accept=sum(raw[(name, "volatility", n)]["seconds"] for n in values
                                  if accepted and n <= accepted) or "",
            seconds_total=sum(raw[(name, "volatility", n)]["seconds"] for n in values),
            accepted_seconds=raw[(name, "volatility", accepted)]["seconds"] if accepted else "",
            **{f"change_{n}": changes.get(n, "") for n in NODE_LEVELS[1:]}))
    write_csv(OUTPUT / "convergence.csv", long_rows)
    write_csv(OUTPUT / "summary_by_contract.csv", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    contracts = load_contracts()
    if not args.summarize_only:
        done = {}
        for row in read_jsonl(RAW):
            done.setdefault(row["contract"], {})[(row["direction"], row["nodes"])] = row["values"]
        # Hard contracts (large q * variance) first so the slow ones do not finish last.
        order = sorted(contracts, key=lambda x: -x[2]["q"] * x[2]["max_variance"])
        tasks = [(name, c, done.get(name, {})) for name, c, _ in order]
        start = perf_counter()
        with Pool(args.workers, initializer=_init, initargs=(Lock(), Semaphore(LARGE_SLOTS))) as pool:
            for i, name in enumerate(pool.imap_unordered(climb, tasks), 1):
                print(f"[{perf_counter() - start:7.1f}s] {i}/{len(tasks)} {name}", flush=True)
    summary = summarize(contracts)
    levels = [row["accepted_nodes"] for row in summary]
    for n in NODE_LEVELS:
        print(f"accepted at {n:3d}: {levels.count(n)}")
    print("not accepted:", [row["contract"] for row in summary if row["accepted_nodes"] == ""])


if __name__ == "__main__":
    main()
