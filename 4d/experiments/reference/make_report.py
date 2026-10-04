"""Write report.md and metadata.json for the Phase 0 reference check from its result files."""

from . import common  # noqa: F401  (pins BLAS threads before NumPy)

import csv
from datetime import datetime, timezone
import json
import platform
import subprocess
import sys

import numpy as np
import scipy

from .common import CONTRACTS, NODE_LEVELS, OUTPUT, TOLERANCE, read_jsonl

NOTES = OUTPUT / "report_notes.md"


def table(rows, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            fmt = formats.get(col)
            if fmt and value not in ("", None):
                value = format(float(value), fmt)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load(name):
    path = OUTPUT / name
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def node_schedule(summary):
    rows = []
    random = [r for r in summary if r["group"] == "random"]
    for n in NODE_LEVELS:
        hits = [r for r in summary if r["accepted_nodes"] == str(n)]
        if not hits:
            continue
        rows.append(dict(nodes=n, contracts=len(hits),
                         random_share=f"{sum(r['group'] == 'random' for r in hits) / len(random):.0%}",
                         named=", ".join(r["contract"] for r in hits if r["group"] == "named"),
                         median_seconds_to_accept=float(np.median([float(r["seconds_to_accept"]) for r in hits])),
                         max_seconds_to_accept=max(float(r["seconds_to_accept"]) for r in hits)))
    missing = [r for r in summary if not r["accepted_nodes"]]
    if missing:
        rows.append(dict(nodes="not by 128", contracts=len(missing),
                         random_share=f"{sum(r['group'] == 'random' for r in missing) / len(random):.0%}",
                         named=", ".join(r["contract"] for r in missing)))
    return rows


def metadata():
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, cwd=OUTPUT.parents[2]).stdout.strip()
        except OSError:
            return ""
    packages = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}
    for name in ("torch", "psutil", "mpmath", "pytest"):
        try:
            packages[name] = __import__(name).__version__
        except ImportError:
            pass
    cpu = platform.processor()
    memory = None
    try:
        import psutil
        memory = round(psutil.virtual_memory().total / 2**30, 1)
        cores = psutil.cpu_count(logical=False), psutil.cpu_count()
    except ImportError:
        cores = None
    if sys.platform == "win32":
        name = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True).stdout.strip()
        cpu = name or cpu
    stamps = {p.name: datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
              for p in sorted(OUTPUT.iterdir()) if p.suffix in (".jsonl", ".csv", ".json") and p.name != "metadata.json"}
    return dict(study="reference_check_v1", generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_head=git("rev-parse", "HEAD"), git_head_subject=git("log", "-1", "--format=%s"),
                working_tree_note="4d/ is untracked; src/pricing/conditional.py carries the Phase 0 root fix",
                platform=platform.platform(), cpu=cpu, cores_physical_logical=cores, memory_gib=memory,
                blas_threads=1, packages=packages, contracts_file=CONTRACTS.relative_to(OUTPUT.parents[1]).as_posix(),
                seeds="development range 404000-404999 only", file_timestamps_utc=stamps)


def main():
    evidence = json.loads((OUTPUT / "root_fix_evidence.json").read_text(encoding="utf-8"))
    summary = load("summary_by_contract.csv")
    cost = load("dimension_cost.csv")
    coverage, by_group, tails = load("rqmc_coverage.csv"), load("rqmc_coverage_by_group.csv"), load("rqmc_tails.csv")
    star = evidence["cstar_tau0.001"]
    identity = evidence["bitwise_identity"]
    named = [r for r in summary if r["group"] == "named"]
    cross = [r for r in summary if r["equal_direction_max_diff"]]
    out = [
        "# Phase 0: four-stock reference check",
        "",
        "Development contracts only (seeds 404000-404999). Reference = tensor Gauss-Hermite over the three "
        "residual factors of the conditional estimator, direction \"volatility\", 27 quantities "
        "(price, 4 Deltas, 10 Gammas, 4 Vegas, theta, rho, 6 correlation sensitivities).",
        "",
    ]
    if NOTES.exists():
        out += [NOTES.read_text(encoding="utf-8").strip(), ""]
    out += [
        "## 1. Root-solver fix",
        "",
        "`_root` in `src/pricing/conditional.py` now detects the two-point cycle that Newton falls into when "
        "the step is rounding noise of size ulp(log K)/f', and accepts it once every step is below an "
        "8-eps noise floor. Anything the old tolerance test accepted is returned unchanged.",
        "",
        "| check | unpatched | patched |",
        "|---|---|---|",
    ]
    for test, result in evidence["tests"].items():
        if isinstance(result, dict):
            out.append(f"| `{test.split('::')[-1]}` | {result['unpatched']} | {result['patched']} |")
    out += [
        f"| full 4d test suite | - | {evidence['tests']['full 4d suite (patched)']} |",
        "",
        f"C* at tau 0.001: 16 vs 32 nodes differ by at most {star['max_abs_diff_16_vs_32_first15']:.2e} "
        f"(relative {star['max_rel_diff_16_vs_32_first15']:.1e}) on price, Deltas and Gammas; the root residual "
        f"on the 32-node grid is {star['root_residual_32_nodes_mpmath_40_digits']:.2e} = "
        f"{star['residual_in_ulps_of_log_K']:.2f} ulp(log K) ({star['note']}).",
        "",
        f"Bitwise identity: {identity['n_cases']} cases ({identity['cases']}); the old code failed on "
        f"{identity['old_code_failures']}, maximum absolute difference {identity['max_abs_difference']}.",
        "",
        "## 2. Node schedule",
        "",
        f"Accepted at the first level whose largest change over all 27 quantities is below {TOLERANCE:g}; "
        "one more level must confirm. Seconds are single-threaded and include all levels up to acceptance.",
        "",
        table(node_schedule(summary), ["nodes", "contracts", "random_share", "median_seconds_to_accept",
                                        "max_seconds_to_accept", "named"],
              dict(median_seconds_to_accept=".1f", max_seconds_to_accept=".1f")),
        "",
        "Named corners:",
        "",
        table(named, ["contract", "tau", "q", "min_eigenvalue", "max_variance", "accepted_nodes", "confirm_change",
                      "last_change", "largest_last_change_quantity", "flag"],
              dict(tau="g", q=".3g", min_eigenvalue=".3g", max_variance=".3g", confirm_change=".1e", last_change=".1e")),
        "",
    ]
    failed = [r for r in summary if r["failed_confirmations"]]
    if failed:
        out += ["Failed confirmations (accepted, then the next level moved by 1e-8 or more): "
                + ", ".join(f"{r['contract']} at {r['failed_confirmations']}" for r in failed), ""]
    if cross:
        out += ["Equal-loadings cross-check at the accepted level (contracts accepted at 64 or more):", "",
                table(cross, ["contract", "accepted_nodes", "equal_direction_max_diff"],
                      dict(equal_direction_max_diff=".2e")), ""]
    if cost:
        out += ["## 3. Cost by number of stocks", "",
                "Tensor grid of n^(d-1) points, one process, BLAS on one thread; peak = peak working set.", "",
                table(cost, ["family", "d", "greeks", "nodes", "grid_points", "quantities", "seconds", "peak_gb",
                             "max_change_vs_previous", "skipped"],
                      dict(seconds=".2f", peak_gb=".2f", max_change_vs_previous=".1e")), ""]
    if coverage:
        out += ["## 4. RQMC against the reference", "",
                "rqmc_estimate: 16 replicates, direction volatility, greeks, geometric control. z = (estimate - "
                "reference) / reported standard error, pooled over contracts, seeds and quantities.", "",
                table(coverage, ["points", "runs", "price_rmse", "rms_z", "coverage_95", "max_abs_z", "mean_seconds"],
                      dict(price_rmse=".2e", rms_z=".3f", coverage_95=".3f", max_abs_z=".2f", mean_seconds=".2f")), "",
                table(by_group, ["points", "group", "n", "rms_z", "coverage_95", "max_abs_z"],
                      dict(rms_z=".3f", coverage_95=".3f", max_abs_z=".2f")), ""]
    if tails:
        out += ["Replicate-mean tails (price-only runs x 16 replicates at 4096 points: 100 runs with the "
                "geometric control, 50 without):", "",
                table(tails, ["contract", "control", "replicate_means", "excess_kurtosis", "excess_kurtosis_ci95",
                              "excess_kurtosis_standardized", "skewness", "price_coverage_95",
                              "mean_minus_reference_over_se"],
                      dict(excess_kurtosis=".2f", excess_kurtosis_standardized=".2f", skewness=".2f",
                           price_coverage_95=".2f", mean_minus_reference_over_se=".2f")), ""]
    (OUTPUT / "report.md").write_text("\n".join(out), encoding="utf-8")
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata(), indent=1), encoding="utf-8")
    print("wrote", OUTPUT / "report.md")


if __name__ == "__main__":
    main()
