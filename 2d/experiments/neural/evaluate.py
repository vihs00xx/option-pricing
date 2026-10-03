"""Select the ladder level on validation, test it once, benchmark against Monte Carlo and check Greeks.

Steps, all on saved checkpoints (no training):
1. Rank levels by the pre-declared validation score and pick the lowest.
2. Score every level on the frozen test panels; the selection never sees them.
3. Compare the selected pricer with plain Monte Carlo and the repository's Sobol
   RQMC estimators on the first 256 test contracts, for prices and all ten Greeks.
4. Check Greek identities and no-arbitrage bounds on the test panels.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from src.basket import BasketContract, randomized_greeks, randomized_price
from src.neural.domain import COLUMN, FIELDS
from src.neural.models import load_ensemble, load_pricer
from src.neural.monte_carlo import monte_carlo_estimates
from src.neural.reference import GREEK_NAMES, QUANTITIES, price_and_greeks
from src.neural.training import quantity_metrics, selection_score

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT/"experiments/config/neural/basket_neural_settings.json"
LEVELS = ROOT/"experiments/config/neural/basket_neural_levels.json"
RESULTS = ROOT/"results/neural/two_asset_v1"
LADDER = RESULTS/"ladder"
PANELS = ("test", "short_maturity", "study_cases")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(level):
    if "ensemble_of" in level:
        return load_ensemble([LADDER/f"{level['ensemble_of']}_seed{s}.pt" for s in level["seeds"]])
    return load_pricer(LADDER/f"{level['name']}_seed{level['seeds'][0]}.pt")[0]


def record_path(level):
    name = level["name"] if "ensemble_of" in level else f"{level['name']}_seed{level['seeds'][0]}"
    return LADDER/f"{name}.json"


def validation_record(level):
    return json.loads(record_path(level).read_text(encoding="utf-8"))


def panels():
    with np.load(RESULTS/"test_sets.npz", allow_pickle=False) as arrays:
        out = {}
        for panel in PANELS:
            reference = arrays[f"{panel}_reference"]
            out[panel] = dict(x=arrays[f"{panel}_contracts"], put=arrays[f"{panel}_put"],
                              reference={q: reference[:, i] for i, q in enumerate(QUANTITIES)})
        names = arrays["study_case_names"].tolist()
    return out, names


def contract_object(row, put):
    p = dict(zip(FIELDS, row))
    return BasketContract(spots=(p["spot_1"], p["spot_2"]), strike=p["strike"], tau=p["tau"], rate=p["rate"],
                          volatilities=(p["vol_1"], p["vol_2"]), dividends=(p["dividend_1"], p["dividend_2"]),
                          weights=(p["weight_1"], 1-p["weight_1"]), correlation=p["correlation"],
                          option="put" if put else "call")


def neural_timing(model, x, put, repeats=3):
    """Seconds per contract for a batch (price only, price and Greeks) and one-contract latency."""
    torch.set_num_threads(1)
    timings = {}
    for label, greeks, batch in (("batch_price", False, x), ("batch_greeks", True, x),
                                 ("single_price", False, x[:1]), ("single_greeks", True, x[:1])):
        best = math.inf
        for _ in range(repeats):
            start = perf_counter()
            price_and_greeks(model, batch, put[:len(batch)], chunk=len(batch), greeks=greeks)
            best = min(best, perf_counter()-start)
        timings[label] = best/len(batch)
    torch.set_num_threads(4)
    return timings


def classical_estimates(panel, settings, log):
    """Plain MC and Sobol RQMC estimates on the benchmark subset, cached because they are model-free."""
    spec = settings["benchmark"]
    cache = RESULTS/"classical_estimates.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as arrays:
            meta = json.loads(str(arrays["meta"]))
            return {method: dict(values={q: arrays[f"{method}_values"][:, i] for i, q in enumerate(QUANTITIES)},
                                 standard_errors={q: arrays[f"{method}_errors"][:, i] for i, q in enumerate(QUANTITIES)},
                                 **meta[method]) for method in meta}
    x, put = panel["x"][:spec["contracts"]], panel["put"][:spec["contracts"]]
    methods = {}
    for paths in spec["mc_paths"]:
        values = {q: np.empty(len(x)) for q in QUANTITIES}
        errors = {q: np.empty(len(x)) for q in QUANTITIES}
        start = perf_counter()
        for i in range(len(x)):
            estimate = monte_carlo_estimates(x[i], put[i], paths, seed=spec["mc_seed"]+i)
            for q in QUANTITIES:
                values[q][i], errors[q][i] = estimate[q]
        methods[f"mc_{paths}"] = dict(values=values, standard_errors=errors, budget=paths,
                                      seconds=(perf_counter()-start)/len(x))
        log(f"plain MC {paths} paths: {methods[f'mc_{paths}']['seconds']:.3f}s per contract")
    for points in spec["rqmc_points"]:
        values = {q: np.empty(len(x)) for q in QUANTITIES}
        errors = {q: np.empty(len(x)) for q in QUANTITIES}
        start = perf_counter()
        for i in range(len(x)):
            contract = contract_object(x[i], put[i])
            price = randomized_price(contract, points, spec["rqmc_replicates"], seed=spec["mc_seed"]+i,
                                     estimator="conditional", control=True)
            greeks = randomized_greeks(contract, points, spec["rqmc_replicates"], seed=spec["mc_seed"]+i,
                                       control=True)["quantities"]
            values["price"][i], errors["price"][i] = price.price, price.standard_error
            for q in GREEK_NAMES:
                values[q][i], errors[q][i] = greeks[q]["value"], greeks[q]["standard_error"]
        methods[f"rqmc_{points}"] = dict(values=values, standard_errors=errors,
                                         budget=points*spec["rqmc_replicates"],
                                         seconds=(perf_counter()-start)/len(x))
        log(f"Sobol RQMC {points}x{spec['rqmc_replicates']}: {methods[f'rqmc_{points}']['seconds']:.3f}s per contract")
    arrays = {}
    for method, record in methods.items():
        arrays[f"{method}_values"] = np.column_stack([record["values"][q] for q in QUANTITIES])
        arrays[f"{method}_errors"] = np.column_stack([record["standard_errors"][q] for q in QUANTITIES])
    meta = {m: dict(budget=r["budget"], seconds=r["seconds"]) for m, r in methods.items()}
    np.savez(cache, meta=np.array(json.dumps(meta)), **arrays)
    return methods


def benchmark(model, panel, settings, neural_seconds, log):
    """Errors, interval coverage and seconds per contract for neural, plain MC and Sobol RQMC."""
    spec = settings["benchmark"]
    x, put = panel["x"][:spec["contracts"]], panel["put"][:spec["contracts"]]
    reference = {q: v[:spec["contracts"]] for q, v in panel["reference"].items()}
    neural = price_and_greeks(model, x, put, chunk=len(x))
    methods = {"neural": dict(values=neural, seconds=neural_seconds, budget=0),
               **classical_estimates(panel, settings, log)}
    rows = []
    for method, record in methods.items():
        metrics = quantity_metrics(record["values"], reference)
        for q in QUANTITIES:
            row = dict(method=method, normals=record["budget"], quantity=q, rmse=metrics[q]["rmse"],
                       median_abs=float(np.median(np.abs(record["values"][q]-reference[q]))),
                       max_abs=metrics[q]["max_abs"], seconds_per_contract=record["seconds"])
            if "standard_errors" in record:
                inside = np.abs(record["values"][q]-reference[q]) <= 1.96*record["standard_errors"][q]
                row["reference_inside_95ci"] = float(inside.mean())
                neural_inside = np.abs(neural[q]-record["values"][q]) <= 1.96*record["standard_errors"][q]
                row["neural_inside_95ci"] = float(neural_inside.mean())
            else:
                row["reference_inside_95ci"] = row["neural_inside_95ci"] = math.nan
            rows.append(row)
    return rows


def identity_checks(model, panel):
    """Homogeneity, the two-asset PDE, Delta bounds and Gamma positive semi-definiteness."""
    x, put = panel["x"], panel["put"]
    values = price_and_greeks(model, x, put)
    c = {name: x[:, COLUMN[name]] for name in FIELDS}
    w = (c["weight_1"], 1-c["weight_1"])
    out = {}
    tensor = torch.as_tensor(x).requires_grad_(True)
    price = model(tensor, torch.as_tensor(put))
    strike_derivative = torch.autograd.grad(price.sum(), tensor)[0][:, COLUMN["strike"]].detach().numpy()
    euler = (c["spot_1"]*values["delta_1"]+c["spot_2"]*values["delta_2"]+c["strike"]*strike_derivative)
    out["homogeneity_max_abs"] = float(np.max(np.abs(values["price"]-euler)))
    pde = (values["theta"]+0.5*c["vol_1"]**2*c["spot_1"]**2*values["gamma_11"]
           + 0.5*c["vol_2"]**2*c["spot_2"]**2*values["gamma_22"]
           + c["correlation"]*c["vol_1"]*c["vol_2"]*c["spot_1"]*c["spot_2"]*values["gamma_12"]
           + (c["rate"]-c["dividend_1"])*c["spot_1"]*values["delta_1"]
           + (c["rate"]-c["dividend_2"])*c["spot_2"]*values["delta_2"]-c["rate"]*values["price"])
    out["pde_residual_rmse"] = float(np.sqrt(np.mean(pde**2)))
    out["pde_residual_over_theta_rms"] = out["pde_residual_rmse"]/float(np.sqrt(np.mean(values["theta"]**2)))
    violations, excess = 0, 0.0
    for i, name in enumerate(("delta_1", "delta_2")):
        bound = w[i]*np.exp(-c[f"dividend_{i+1}"]*c["tau"])
        signed = np.where(put, -values[name], values[name])
        violations += int(np.sum((signed < -1e-10) | (signed > bound+1e-10)))
        excess = max(excess, float(np.max(np.maximum(-signed, signed-bound))))
    out["delta_bound_violations"] = violations
    out["delta_bound_worst_excess"] = max(excess, 0.0)
    gamma = np.stack((np.stack((values["gamma_11"], values["gamma_12"]), -1),
                      np.stack((values["gamma_12"], values["gamma_22"]), -1)), -2)
    eigenvalues = np.linalg.eigvalsh(gamma)
    out["gamma_not_psd"] = int(np.sum(eigenvalues[:, 0] < -1e-10))
    # Size of the negative eigenvalue relative to the largest one, worst contract.
    negative = np.maximum(-eigenvalues[:, 0], 0)
    ratio = np.divide(negative, np.abs(eigenvalues[:, 1]), out=np.zeros_like(negative), where=negative > 0)
    out["gamma_worst_negative_ratio"] = float(np.max(ratio))
    discount = np.exp(-c["rate"]*c["tau"])
    forward = (w[0]*c["spot_1"]*np.exp(-c["dividend_1"]*c["tau"])
               + w[1]*c["spot_2"]*np.exp(-c["dividend_2"]*c["tau"]))
    intrinsic = np.maximum(np.where(put, -1, 1)*(forward-c["strike"]*discount), 0)
    out["price_below_intrinsic"] = int(np.sum(values["price"] < intrinsic-1e-10))
    out["contracts"] = len(x)
    return out


def error_by_maturity(model, panels_data, edges):
    rows = []
    x = np.vstack([panels_data[p]["x"] for p in ("test", "short_maturity")])
    put = np.concatenate([panels_data[p]["put"] for p in ("test", "short_maturity")])
    reference = {q: np.concatenate([panels_data[p]["reference"][q] for p in ("test", "short_maturity")])
                 for q in QUANTITIES}
    predicted = price_and_greeks(model, x, put)
    tau = x[:, COLUMN["tau"]]
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (tau >= low) & (tau < high)
        metrics = quantity_metrics({q: v[mask] for q, v in predicted.items()},
                                   {q: v[mask] for q, v in reference.items()})
        rows.append(dict(tau_low=low, tau_high=high, contracts=int(mask.sum()),
                         **{f"{q}_rmse": metrics[q]["rmse"] for q in QUANTITIES},
                         **{f"{q}_reference_rms": float(np.sqrt(np.mean(reference[q][mask]**2)))
                            for q in QUANTITIES}))
    return rows


def figures(ladder_rows, test_rows, bench_rows, maturity_rows, selected):
    plt.rcParams.update({"font.size": 9})
    names = [r["level"] for r in ladder_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    for quantity, marker in (("price", "o"), ("delta_1", "s"), ("gamma_11", "^"), ("vega_1", "D"),
                             ("theta", "v"), ("correlation", "P")):
        ax.plot(names, [r[f"{quantity}_nrmse"] for r in ladder_rows], marker=marker, label=quantity)
    ax.plot(names, [r["score"] for r in ladder_rows], "k-", lw=2.2, label="selection score")
    ax.set_yscale("log")
    ax.set_ylabel("validation RMSE / RMS of reference")
    ax.set_title("Each level adds one technique (validation set)")
    ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS/"ladder.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for ax, quantity in zip(axes.flat, ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation")):
        mc = [r for r in bench_rows if r["method"].startswith("mc_") and r["quantity"] == quantity]
        rq = [r for r in bench_rows if r["method"].startswith("rqmc_") and r["quantity"] == quantity]
        nn = [r for r in bench_rows if r["method"] == "neural" and r["quantity"] == quantity][0]
        ax.loglog([r["normals"] for r in mc], [r["rmse"] for r in mc], "o-", label="plain MC")
        ax.loglog([r["normals"] for r in rq], [r["rmse"] for r in rq], "s-", label="Sobol RQMC (repo)")
        ax.axhline(nn["rmse"], color="k", ls="--", label=f"neural ({selected})")
        ax.set_title(quantity)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[1]:
        ax.set_xlabel("normal draws per contract")
    axes[0, 0].set_ylabel("RMSE vs reference")
    axes[1, 0].set_ylabel("RMSE vs reference")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Neural pricer against Monte Carlo on 256 test contracts")
    fig.tight_layout()
    fig.savefig(RESULTS/"monte_carlo_benchmark.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    centers = [math.sqrt(r["tau_low"]*r["tau_high"]) for r in maturity_rows]
    for quantity in ("price", "delta_1", "gamma_11", "vega_1", "theta", "correlation"):
        ax.loglog(centers, [r[f"{quantity}_rmse"]/r[f"{quantity}_reference_rms"] for r in maturity_rows],
                  "o-", label=quantity)
    ax.set_xlabel("time to expiry (years)")
    ax.set_ylabel("RMSE / RMS of reference")
    ax.set_title(f"Selected pricer ({selected}) error by maturity, test panels")
    ax.legend(ncol=3, fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS/"error_by_maturity.png", dpi=160)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    levels = json.loads(LEVELS.read_text(encoding="utf-8"))["levels"]
    missing = [level["name"] for level in levels if not record_path(level).exists()]
    if missing:
        print(f"Skipping levels without a trained result: {', '.join(missing)}")
    levels = [level for level in levels if level["name"] not in missing]
    protocol = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8"))
    if digest(RESULTS/"test_sets.npz") != protocol["test_sets_sha256"]:
        raise RuntimeError("Frozen test sets changed after the protocol was recorded")

    ladder_rows = []
    for level in levels:
        record = validation_record(level)
        ladder_rows.append(dict(level=level["name"], technique=level["technique"], score=record["score"],
                                training_seconds=record["training_seconds"],
                                **{f"{q}_rmse": record["metrics"][q]["rmse"] for q in QUANTITIES},
                                **{f"{q}_nrmse": record["metrics"][q]["normalized_rmse"] for q in QUANTITIES}))
    write_csv(RESULTS/"ladder_validation.csv", ladder_rows)
    selected = min(ladder_rows, key=lambda r: r["score"])["level"]
    selected_level = next(level for level in levels if level["name"] == selected)
    print(f"Selected {selected} by validation score")

    data, case_names = panels()
    test_rows, predictions = [], {}
    for level in levels:
        model = load_model(level)
        for panel in PANELS:
            predicted = price_and_greeks(model, data[panel]["x"], data[panel]["put"])
            metrics = quantity_metrics(predicted, data[panel]["reference"])
            if level["name"] == selected:
                predictions[panel] = np.column_stack([predicted[q] for q in QUANTITIES])
            test_rows.append(dict(level=level["name"], panel=panel, score=selection_score(metrics),
                                  **{f"{q}_rmse": metrics[q]["rmse"] for q in QUANTITIES},
                                  **{f"{q}_max_abs": metrics[q]["max_abs"] for q in QUANTITIES}))
    write_csv(RESULTS/"test_metrics.csv", test_rows)
    np.savez(RESULTS/"selected_predictions.npz", **{f"{p}_predicted": v for p, v in predictions.items()},
             quantities=np.array(QUANTITIES))

    model = load_model(selected_level)
    case_rows = []
    for i, name in enumerate(case_names):
        for j, q in enumerate(QUANTITIES):
            case_rows.append(dict(case=name, quantity=q, reference=data["study_cases"]["reference"][q][i],
                                  neural=predictions["study_cases"][i, j],
                                  error=predictions["study_cases"][i, j]-data["study_cases"]["reference"][q][i]))
    write_csv(RESULTS/"study_cases.csv", case_rows)
    checks = {panel: identity_checks(model, data[panel]) for panel in PANELS}
    edges = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0001]
    maturity_rows = error_by_maturity(model, data, edges)
    write_csv(RESULTS/"error_by_maturity.csv", maturity_rows)
    timing = neural_timing(model, data["test"]["x"], data["test"]["put"])

    bench_rows = benchmark(model, data["test"], settings, timing["batch_greeks"], print)
    write_csv(RESULTS/"monte_carlo_benchmark.csv", bench_rows)
    figures(ladder_rows, test_rows, bench_rows, maturity_rows, selected)

    summary = dict(
        created_at=datetime.now(timezone.utc).isoformat(), selected=selected,
        selection_rule=settings["selection_rule"], identity_checks=checks, neural_seconds_per_contract=timing,
        checkpoints_sha256={p.name: digest(p) for p in sorted(LADDER.glob("*.pt"))},
        outputs_sha256={name: digest(RESULTS/name) for name in (
            "ladder_validation.csv", "test_metrics.csv", "selected_predictions.npz", "study_cases.csv",
            "error_by_maturity.csv", "monte_carlo_benchmark.csv", "classical_estimates.npz")},
        source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(
            list((ROOT/"src/neural").glob("*.py"))+list((ROOT/"experiments/neural").glob("*.py"))+[SETTINGS, LEVELS])},
        environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                         scipy=scipy.__version__, torch=torch.__version__),
    )
    (RESULTS/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(selected=selected, checks=checks, timing=timing), indent=2))


if __name__ == "__main__":
    main()
