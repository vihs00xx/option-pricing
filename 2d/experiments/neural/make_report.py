"""Write the study report, figures, ladder manifest and hash-pinned metadata.

Reads the ladder records, the frozen selection, the test-set scores and the Monte
Carlo benchmark; evaluates the selected pricers on the validation panels for the
maturity breakdown. Trains and tests nothing.

Usage: python -m experiments.neural.make_report
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.neural.common import LADDER, RESULTS, SETTINGS_FILES, STUDY, evaluation_set, load_settings
from experiments.records import ROOT, digest, relative, write_metadata
from src.neural.coordinates import COLUMN
from src.neural.networks import load_model
from src.neural.reference import QUANTITIES, price_and_greeks

LABEL = {"price": "Price", "delta_1": "Delta 1", "delta_2": "Delta 2", "gamma_11": "Gamma 11",
         "gamma_12": "Gamma 12", "gamma_22": "Gamma 22", "vega_1": "Vega 1", "vega_2": "Vega 2",
         "theta": "Theta", "rho": "Rate rho", "correlation_12": "Correlation"}
SHOWN = ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation_12")
SPLIT_LABEL = {"interpolation": "Interpolation", "short_maturity": "Short maturity",
               "extrapolation": "Extrapolation", "anchor": "Anchor", "all": "All cases"}
MATURITY_EDGES = (0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def g(value, digits=3):
    if value == 0:
        return "0"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:.{digits}g}"


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(header, rows, align=None):
    align = align or ["---"]+["---:"]*(len(header)-1)
    lines = ["| "+" | ".join(header)+" |", "| "+" | ".join(align)+" |"]
    lines += ["| "+" | ".join(str(c) for c in row)+" |" for row in rows]
    return "\n".join(lines)


def ladder_records(settings):
    seed = settings["seeds"]["training"]
    records = {}
    for level in settings["levels"]:
        path = LADDER/f"{level}_seed{seed}.json"
        if path.exists():
            records[level] = json.loads(path.read_text(encoding="utf-8"))
    return records


def ladder_table(settings, records, family, selected):
    rows = []
    for level, record in records.items():
        if settings["levels"][level]["family"] != family:
            continue
        v = record["scores"]["validation"]
        name = f"`{level}`"+(" **(selected)**" if level == selected else "")
        rows.append([name, settings["levels"][level]["technique"], g(record["parameters"]),
                     g(v["score"]), g(record["scores"]["short_validation"]["score"])]
                    + [g(v["rmse"][q]) for q in SHOWN]+[g(round(record["total_seconds"]))])
    header = (["Level", "Technique added", "Parameters", "Validation score", "Short-maturity score"]
              + [f"{LABEL[q]} RMSE" for q in SHOWN]+["Training s"])
    return table(header, rows, ["---", "---"]+["---:"]*(len(header)-2))


def ladder_figure(settings, records, selection, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, family, title in ((axes[0], "pinn", "PINN ladder"), (axes[1], "deeponet", "DeepONet ladder")):
        levels = [l for l in records if settings["levels"][l]["family"] == family]
        scores = [records[l]["scores"]["validation"]["score"] for l in levels]
        short = [records[l]["scores"]["short_validation"]["score"] for l in levels]
        x = np.arange(len(levels))
        axis.plot(x, scores, "o-", color="#2a6fbb", label="validation (training domain)")
        axis.plot(x, short, "s--", color="#c4572a", label="short maturity (below the domain)")
        chosen = selection["selected"][family]["level"]
        if chosen in levels:
            i = levels.index(chosen)
            axis.plot([i], [scores[i]], "o", ms=13, mfc="none", mec="black")
        axis.set_xticks(x, [l.split("_", 1)[0] for l in levels])
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("score: mean of RMSE / RMS over price and 10 Greeks")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def maturity_breakdown(models):
    """Relative RMSE by maturity bucket on the validation and short-maturity panels."""
    rows = []
    for family, model in models.items():
        values, references, taus = {q: [] for q in QUANTITIES}, {q: [] for q in QUANTITIES}, []
        for kind in ("short_validation", "validation"):
            data = evaluation_set(kind)
            x, put = data["inputs"]
            out = price_and_greeks(model.price, x, put)
            taus.append(x[:, COLUMN["tau"]])
            for q in QUANTITIES:
                values[q].append(out[q])
                references[q].append(data["reference"][q])
        tau = np.concatenate(taus)
        for low, high in zip(MATURITY_EDGES[:-1], MATURITY_EDGES[1:]):
            mask = (tau >= low) & (tau < high) if high < MATURITY_EDGES[-1] else (tau >= low) & (tau <= high)
            row = dict(model=family, tau_low=low, tau_high=high, contracts=int(mask.sum()))
            for q in QUANTITIES:
                v, r = np.concatenate(values[q])[mask], np.concatenate(references[q])[mask]
                row[q] = float(np.sqrt(np.mean((v-r)**2))/np.sqrt(np.mean(r**2)))
            rows.append(row)
    return rows


def maturity_figure(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, family in zip(axes, ("pinn", "deeponet")):
        part = [r for r in rows if r["model"] == family]
        centers = [np.sqrt(r["tau_low"]*r["tau_high"]) for r in part]
        for q in SHOWN:
            axis.plot(centers, [r[q] for r in part], "o-", label=LABEL[q])
        axis.axvline(0.05, color="grey", ls=":", lw=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("years to expiry")
        axis.set_title(f"Selected {'PINN' if family == 'pinn' else 'DeepONet'}")
        axis.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMSE / RMS of reference")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def benchmark_figure(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, q in zip(axes, ("price", "gamma_11", "theta")):
        for method, style, label in (("plain_monte_carlo", "o-", "plain Monte Carlo"),
                                     ("sobol_rqmc_engine", "s-", "Sobol RQMC engine"),
                                     ("adi_richardson", "^-", "ADI grid, Richardson")):
            part = sorted((r for r in rows if r["method"] == method and r["quantity"] == q),
                          key=lambda r: float(r["seconds_per_contract"]))
            if part:
                axis.plot([float(r["seconds_per_contract"]) for r in part], [float(r["rmse"]) for r in part],
                          style, label=label)
        for method, marker, label in (("pinn_selected", "*", "PINN"), ("deeponet_selected", "D", "DeepONet")):
            for r in rows:
                if r["method"] == method and r["quantity"] == q:
                    axis.plot([float(r["seconds_per_contract"])], [float(r["rmse"])], marker, ms=11, label=label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("seconds per contract (price and all Greeks)")
        axis.set_title(f"{LABEL[q]} RMSE")
        axis.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_manifest(records, settings):
    """SHA-256 of every ladder file, including the extra seeds behind an ensemble."""
    seed = settings["seeds"]["training"]
    stems = {f"{level}_seed{seed}" for level in records}
    for record in records.values():
        stems |= {Path(member).stem for member in record.get("members", {})}
    manifest = {}
    for stem in sorted(stems):
        for suffix in ("pt", "json", "log"):
            path = LADDER/f"{stem}.{suffix}"
            manifest[relative(path)] = digest(path)
    (RESULTS/"ladder_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def main():
    torch.set_num_threads(4)
    settings = load_settings()
    records = ladder_records(settings)
    selection = json.loads((RESULTS/"selection.json").read_text(encoding="utf-8"))
    models = {family: load_model(ROOT/entry["checkpoint"]) for family, entry in selection["selected"].items()}
    ladder_figure(settings, records, selection, RESULTS/"ladder.png")
    maturity = maturity_breakdown(models)
    from experiments.records import write_csv
    write_csv(RESULTS/"error_by_maturity.csv", maturity)
    maturity_figure(maturity, RESULTS/"error_by_maturity.png")
    panel = "test" if (RESULTS/"monte_carlo_benchmark.csv").exists() else "development"
    prefix = "" if panel == "test" else "development_"
    benchmark = read_csv(RESULTS/f"{prefix}monte_carlo_benchmark.csv")
    if panel == "test" and (RESULTS/"adi_benchmark.csv").exists():
        benchmark += read_csv(RESULTS/"adi_benchmark.csv")
    benchmark_figure(benchmark, RESULTS/"monte_carlo_benchmark.png")
    write_manifest(records, settings)
    from experiments.neural.report_text import report
    (RESULTS/"report.md").write_text(report(settings, records, selection, maturity, panel, prefix),
                                     encoding="utf-8")
    sources = (sorted((ROOT/"src/neural").glob("*.py"))+sorted((ROOT/"experiments/neural").glob("*.py"))
               + [ROOT/"experiments/records.py", ROOT/"experiments/test_set/freeze_test_set.py"]
               + sorted((ROOT/"src/pricing").glob("*.py")))
    write_metadata(RESULTS, STUDY, list(sources)+list(SETTINGS_FILES), dict(torch=torch.__version__, matplotlib=matplotlib.__version__,
                                                    test_set_protocol_sha256=selection["test_set"]["protocol_sha256"]))


if __name__ == "__main__":
    main()
