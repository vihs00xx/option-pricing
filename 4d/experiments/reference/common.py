"""Shared paths, contract I/O and resumable records for the Phase 0 reference check.

Import this module before NumPy: it pins BLAS to one thread so that worker
processes, not BLAS threads, share the cores.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import json
from pathlib import Path

import numpy as np

from src.pricing import Basket

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "experiments/config/reference/dev_contracts_v1.json"
OUTPUT = ROOT / "results/reference_check_v1"
NODE_LEVELS = (24, 32, 40, 48, 64, 80, 96, 128)
TOLERANCE = 1e-8
WORKERS = 8


def contract_to_dict(c):
    return dict(spots=list(c.spots), strike=c.strike, tau=c.tau, rate=c.rate,
                volatilities=list(c.volatilities), dividends=list(c.dividends),
                weights=list(c.weights), correlation=[list(row) for row in c.correlation], option=c.option)


def load_contracts():
    """[(name, Basket, info)] in file order."""
    rows = json.loads(CONTRACTS.read_text(encoding="utf-8"))["contracts"]
    return [(row["name"], Basket(**row["contract"]), row) for row in rows]


def diagnostics(c):
    """q = 1'R^-1 1, the smallest correlation eigenvalue and the largest sigma^2 tau."""
    R = c.correlation_matrix()
    ones = np.ones(c.dimension)
    return dict(q=float(ones @ np.linalg.solve(R, ones)), min_eigenvalue=float(np.linalg.eigvalsh(R)[0]),
                max_variance=float(max(c.volatilities) ** 2 * c.tau))


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut short by an interrupted run
                pass
    return rows


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def peak_memory_gb():
    """Peak working set of this process (Windows) or peak RSS elsewhere."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e9
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
