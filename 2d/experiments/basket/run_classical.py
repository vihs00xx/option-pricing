"""Run a small two-stock classical verification study; no neural training or final-test selection."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.basket import (BasketContract, geometric_price, quadrature_price,
                        arithmetic_greeks, randomized_price, randomized_greeks, solve_adi)
from src.basket.integration import GREEK_NAMES

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT/"experiments/config/classical/basket_settings.json"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"results/basket/two_asset_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError("Study output exists; use a new output directory or explicit --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    cases = [(row["name"], BasketContract(**{k: v for k, v in row.items() if k != "name"}))
             for row in settings["cases"]]
    references, greek_references, checks = {}, {}, []
    for name, c in cases:
        estimate = quadrature_price(c)
        g = arithmetic_greeks(c)
        half = arithmetic_greeks(c, bump_scale=0.5)
        references[name], greek_references[name] = estimate.price, g
        geometric = geometric_price(c)
        integrated = quadrature_price(c, kind="geometric")
        checks.append(dict(case=name, arithmetic_price=estimate.price,
                           geometric_price=geometric, geometric_quadrature_error=abs(integrated.price-geometric),
                           independent_conditioning_error=abs(estimate.price-quadrature_price(c, conditioning="asset").price),
                           integration_error_estimate=estimate.integration_error, tail_bound=estimate.tail_bound,
                           parameter_greek_bump_change=max(abs(g[k]-half[k]) for k in GREEK_NAMES[5:])))
        print(f"Reference {name}: {estimate.price:.10f}", flush=True)
    methods = [("mc_raw", "mc", "raw", False), ("sobol_raw", "sobol", "raw", False),
               ("sobol_asset_conditional", "sobol", "asset_conditional", False),
               ("sobol_conditional", "sobol", "conditional", False),
               ("sobol_conditional_control", "sobol", "conditional", True)]
    runs, price_replicates = [], []
    for name, c in cases:
        for points in settings["points_per_replicate"]:
            for method, sampler, estimator, control in methods:
                for seed in settings["seeds"]:
                    start = perf_counter()
                    result = randomized_price(c, points, settings["replicates"], seed, sampler, estimator, control,
                                              settings["pilot_points_per_replicate"], settings["pilot_replicates"])
                    runs.append(dict(case=name, method=method, points_per_replicate=points, seed=seed,
                                     price=result.price, reference=references[name], error=result.price-references[name],
                                     standard_error=result.standard_error, ci_low=result.ci_low, ci_high=result.ci_high,
                                     covered=result.ci_low <= references[name] <= result.ci_high,
                                     production_points=result.production_points, pilot_points=result.pilot_points,
                                     total_points=result.production_points+result.pilot_points,
                                     normal_dimension=result.normal_dimension, beta=result.beta,
                                     seconds=perf_counter()-start))
                    price_replicates.append(result.replicate_means)
        print(f"Price estimators complete: {name}", flush=True)
    summary = []
    for method, _, _, _ in methods:
        for points in settings["points_per_replicate"]:
            rows = [r for r in runs if r["method"] == method and r["points_per_replicate"] == points]
            errors = np.array([r["error"] for r in rows])
            summary.append(dict(method=method, points_per_replicate=points, total_points=rows[0]["total_points"],
                                price_rmse=float(np.sqrt(np.mean(errors**2))), max_absolute_error=float(np.max(abs(errors))),
                                coverage=float(np.mean([r["covered"] for r in rows])),
                                median_seconds=float(np.median([r["seconds"] for r in rows]))))
    greek_rows, greek_replicates, greek_groups = [], [], []
    for name, c in cases:
        for control in (False, True):
            for seed in settings["seeds"]:
                result = randomized_greeks(c, n_points=4096, n_replicates=settings["replicates"], seed=seed, control=control,
                                           pilot_points=settings["pilot_points_per_replicate"],
                                           pilot_replicates=settings["pilot_replicates"])
                greek_replicates.append(result["replicate_means"])
                greek_groups.append(f"{name}|{'conditional_control' if control else 'conditional'}|{seed}")
                for quantity, entry in result["quantities"].items():
                    reference = greek_references[name][quantity]
                    greek_rows.append(dict(case=name, method="conditional_control" if control else "conditional",
                                           seed=seed, quantity=quantity, reference=reference, value=entry["value"],
                                           error=entry["value"]-reference, standard_error=entry["standard_error"],
                                           ci_low=entry["ci_low"], ci_high=entry["ci_high"],
                                           covered=entry["ci_low"] <= reference <= entry["ci_high"],
                                           production_points=result["production_points"], pilot_points=result["pilot_points"],
                                           beta=entry["beta"]))
        print(f"Greek estimators complete: {name}", flush=True)
    adi_rows = []
    for name in ("standard", "negative_correlation", "near_expiry"):
        c = dict(cases)[name]
        for grid in settings["adi_grids"]:
            start = perf_counter()
            solution = solve_adi(c, n_space=grid, n_steps=2*grid, s_max=(400, 400))
            quantities = {"price": solution.evaluate(c.spots), **solution.spot_greeks(c.spots)}
            elapsed = perf_counter()-start
            for quantity, value in quantities.items():
                reference = references[name] if quantity == "price" else greek_references[name][quantity]
                adi_rows.append(dict(case=name, study="mesh", grid=grid, steps=2*grid, s_max=400,
                                     quantity=quantity, value=value, reference=reference, error=value-reference, seconds=elapsed))
        # Keep spatial spacing and time steps fixed while moving the far boundary.
        start = perf_counter()
        larger = solve_adi(c, n_space=240, n_steps=320, s_max=(600, 600))
        elapsed = perf_counter()-start
        for quantity, value in {"price": larger.evaluate(c.spots), **larger.spot_greeks(c.spots)}.items():
            reference = references[name] if quantity == "price" else greek_references[name][quantity]
            adi_rows.append(dict(case=name, study="domain", grid=240, steps=320, s_max=600,
                                 quantity=quantity, value=value, reference=reference, error=value-reference, seconds=elapsed))
        print(f"ADI convergence complete: {name}", flush=True)
    for filename, rows in (("reference_checks.csv", checks), ("price_runs.csv", runs),
                           ("price_summary.csv", summary), ("greek_runs.csv", greek_rows), ("adi_convergence.csv", adi_rows)):
        write_csv(output/filename, rows)
    np.savez_compressed(output/"replicate_means.npz", price=np.asarray(price_replicates),
                        greeks=np.asarray(greek_replicates), greek_groups=np.asarray(greek_groups),
                        greek_quantities=np.asarray(GREEK_NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = ["#64748b", "#0284c7", "#f59e0b", "#0891b2", "#7c3aed"]
    for (method, _, _, _), color in zip(methods, colors):
        rows = [r for r in summary if r["method"] == method]
        axes[0].loglog([r["total_points"] for r in rows], [r["price_rmse"] for r in rows], "o-", color=color, label=method.replace("_", " "))
    axes[0].set(xlabel="Production + pilot Gaussian points", ylabel="Price RMSE", title="Eight contracts, three repeated studies")
    axes[0].legend(fontsize=7)
    for name, color in zip(("standard", "negative_correlation", "near_expiry"), colors):
        rows = [r for r in adi_rows if r["case"] == name and r["quantity"] == "price" and r["study"] == "mesh"]
        axes[1].loglog([r["grid"] for r in rows], [abs(r["error"]) for r in rows], "o-", color=color, label=name.replace("_", " "))
    axes[1].set(xlabel="Spatial intervals per stock", ylabel="Absolute price error", title="MCS ADI mesh refinement")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    fig.savefig(output/"convergence.pdf")
    plt.close(fig)
    lines = ["# Two-stock classical verification", "", "This is an exploratory baseline study; these cases are now observed and are not a future neural final test.",
             "", "Arithmetic and geometric basket calls/puts use correlated GBM, constant dividends and positive weights summing to one.",
             "The primary conditional estimator decomposes correlated log returns into a shared Gaussian factor with identical stock loadings and a residual factor. The shared factor is integrated analytically; only the residual is sampled or numerically integrated.",
             "For log-return covariance C, the shared-factor variance is det(C)/(C11+C22-2C12), and the residual loadings are (C11-C12, C12-C22)/sqrt(C11+C22-2C12). Their outer product plus the shared variance times an all-ones matrix reconstructs C.",
             "A separate estimator conditions on the second stock directly and analytically integrates the first stock. Agreement between these distinct quadratures provides an independent representation check. Singular correlations and zero volatilities use the asset-conditional price path; regular Greek estimation requires positive stock volatilities and |correlation| < 1.",
             "The +/-12 normal truncation has a separately computed price-tail bound. The quadrature error is a numerical estimate, not a rigorous total-error guarantee.",
             "", "## Price estimates at the largest budget", "", "| Estimator | Total points incl. pilot | Price RMSE | Max absolute error | Marginal CI coverage |", "|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["points_per_replicate"] == max(settings["points_per_replicate"]):
            lines.append(f"| {row['method']} | {row['total_points']} | {row['price_rmse']:.6g} | {row['max_absolute_error']:.6g} | {row['coverage']:.3f} |")
    lines.extend(["", "Pilot control coefficients are fitted from independent replicate means. Sobol uncertainty is estimated across independent scrambles, never across points within a net.",
                  "`replicate_means.npz` preserves the independent means used for uncertainty calculations; its price rows follow `price_runs.csv`, and Greek groups and quantity order are stored in the archive.",
                  "Gaussian-point counts do not equal CPU work: conditional formulas, geometric controls and Greek bumps have different costs. Recorded timings are exploratory, without a controlled cross-method timing protocol.",
                  "", f"## Greek errors at {4096*settings['replicates']:,} production points", "", "| Quantity | Conditional RMSE | Conditional + control RMSE |", "|---|---:|---:|"])
    for quantity in GREEK_NAMES:
        rmses = [np.sqrt(np.mean([r["error"]**2 for r in greek_rows if r["quantity"] == quantity and r["method"] == method]))
                 for method in ("conditional", "conditional_control")]
        lines.append(f"| {quantity} | {rmses[0]:.6g} | {rmses[1]:.6g} |")
    lines.extend(["", "Delta and the full spot Hessian use analytical conditional derivatives. Vega, Theta, rate Rho and correlation sensitivity use central bumps with common random numbers.",
                  "Their intervals exclude bump bias. `reference_checks.csv` records the effect of halving the reference bumps. Greek intervals are marginal, not simultaneous.",
                  "", "## ADI scope", "", "Modified Craig-Sneyd (theta=1/3), central differences and two initial implicit-Euler-damped intervals. The zero-stock faces are exact vanilla prices; the far faces use asymptotic values.",
                  "`adi_convergence.csv` separates mesh refinement from a domain extension at constant spatial spacing. Near-expiry Hessians require substantially finer grids than regular cases. This implementation is an independent baseline, not yet a production reference across all parameter ranges.",
                  "", "## Sources", "", "- Bayer, Siebenmorgen and Tempone: https://arxiv.org/abs/1607.05572", "- In 't Hout and Foulon (ADI with mixed derivatives): https://arxiv.org/abs/0811.3427", "- Glau and Wunderlich: https://doi.org/10.1016/j.amc.2022.127355", ""])
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    source_files = list((ROOT/"src/basket").glob("*.py"))+[ROOT/"src/black_scholes.py", Path(__file__), SETTINGS]
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), settings=settings, settings_sha256=digest(SETTINGS),
                    contracts={name: asdict(c) for name, c in cases},
                    source_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in source_files},
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
                    outputs_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "metadata.json"})
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
