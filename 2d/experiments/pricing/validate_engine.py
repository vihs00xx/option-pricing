"""Validate the d-stock pricing engine on development cases; no test-set cases are used."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.records import ROOT, pricing_sources, prepare_output, write_csv, write_metadata
from src.basket import BasketContract, arithmetic_greeks, quadrature_price, randomized_price
from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.pricing import Basket, adi_values, gauss_hermite_values, quantity_names, rqmc_estimate

SETTINGS = ROOT / "experiments/config/classical/engine_validation_settings.json"
OLD_ADI = ROOT / "results/basket/two_asset_v1/adi_convergence.csv"
SPOT_AND_THETA = ("price", "delta_1", "delta_2", "gamma_11", "gamma_12", "gamma_22", "theta")


def two_asset_cases(settings):
    cases = []
    for row in settings["two_asset_cases"]:
        values = {**settings["two_asset_defaults"], **{k: v for k, v in row.items() if k != "name"}}
        cases.append((row["name"], BasketContract(**values), Basket(**values)))
    return cases


def quadrature_reference(v1, settings, bump_scale=1.0):
    tolerance = settings["reference"]
    reference = {"price": quadrature_price(v1, atol=tolerance["quadrature_atol"],
                                           rtol=tolerance["quadrature_rtol"]).price,
                 **arithmetic_greeks(v1, bump_scale)}
    reference["correlation_12"] = reference.pop("correlation")
    return reference


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2)))


def summarize(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [dict(**dict(zip(keys, key)), runs=len(group), rmse=rms([r["error"] for r in group]),
                 max_absolute_error=float(max(abs(r["error"]) for r in group)),
                 rms_z=rms([r["z"] for r in group]),
                 coverage=float(np.mean([r["covered"] for r in group])),
                 median_seconds=float(np.median([r["seconds"] for r in group])) if "seconds" in group[0] else "")
            for key, group in groups.items()]


def estimate_row(case, method, points, seed, name, value, se, low, high, reference, seconds=None):
    row = dict(case=case, method=method, points_per_replicate=points, seed=seed, quantity=name,
               value=value, reference=reference, error=value - reference, standard_error=se,
               z=(value - reference) / se if se > 0 else 0.0, covered=bool(low <= reference <= high))
    if seconds is not None:
        row["seconds"] = seconds
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/engine_validation_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    args = parser.parse_args()
    output = prepare_output(args.output, args.overwrite)
    settings_path = args.settings.resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    cases = two_asset_cases(settings)

    # 1. The new conditional formulas against the independent two-asset quadrature.
    references, agreement = {}, []
    for name, v1, c in cases:
        reference = quadrature_reference(v1, settings)
        half = quadrature_reference(v1, settings, 0.5)
        references[name] = reference
        names = quantity_names(c)
        nodes = settings["reference"]["gauss_hermite_nodes"]
        engines = {direction: dict(zip(names, gauss_hermite_values(c, nodes, direction=direction)))
                   for direction in ("equal", "volatility")}
        for quantity in names:
            agreement.append(dict(case=name, quantity=quantity, quadrature=reference[quantity],
                                  gauss_hermite_equal=engines["equal"][quantity],
                                  gauss_hermite_volatility=engines["volatility"][quantity],
                                  difference_equal=engines["equal"][quantity] - reference[quantity],
                                  difference_volatility=engines["volatility"][quantity] - reference[quantity],
                                  reference_bump_change=half[quantity] - reference[quantity]))
        print(f"Reference agreement: {name}", flush=True)

    # 2. Price accuracy and calibration, new variants beside the earlier two-asset estimator.
    config = settings["rqmc"]
    common = dict(pilot_points=config["pilot_points_per_replicate"], pilot_replicates=config["pilot_replicates"])
    price_rows = []
    for name, v1, c in cases:
        reference = references[name]["price"]
        for points in config["price_points_per_replicate"]:
            for seed in config["seeds"]:
                for variant in config["variants"]:
                    start = perf_counter()
                    est = rqmc_estimate(c, points, config["replicates"], seed, direction=variant["direction"],
                                        control=variant["control"], greeks=False, **common)
                    price_rows.append(estimate_row(name, variant["name"], points, seed, "price",
                                                   est.values["price"], est.standard_errors["price"],
                                                   *est.intervals["price"], reference, perf_counter() - start))
                for control in (False, True):
                    start = perf_counter()
                    old = randomized_price(v1, points, config["replicates"], seed, "sobol", "conditional",
                                           control, **common)
                    price_rows.append(estimate_row(name, "two_asset_v1_conditional" + ("_control" if control else ""),
                                                   points, seed, "price", old.price, old.standard_error,
                                                   old.ci_low, old.ci_high, reference, perf_counter() - start))
        print(f"Price estimators: {name}", flush=True)
    price_summary = summarize(price_rows, ("method", "points_per_replicate"))

    # 3. All sensitivities at one budget.
    greek_rows = []
    for name, _, c in cases:
        for seed in config["seeds"]:
            for variant in config["variants"]:
                start = perf_counter()
                est = rqmc_estimate(c, config["greek_points_per_replicate"], config["replicates"], seed,
                                    direction=variant["direction"], control=variant["control"], **common)
                seconds = perf_counter() - start
                for quantity in est.names:
                    greek_rows.append(estimate_row(name, variant["name"], config["greek_points_per_replicate"],
                                                   seed, quantity, est.values[quantity],
                                                   est.standard_errors[quantity], *est.intervals[quantity],
                                                   references[name][quantity], seconds))
        print(f"Greek estimators: {name}", flush=True)
    greek_summary = summarize(greek_rows, ("method", "quantity"))

    # 4. One, three and four stocks: Gauss-Hermite convergence and RQMC calibration.
    dimension_rows, dimension_runs = [], []
    high = settings["higher_dimension"]
    for row in settings["higher_dimension_cases"]:
        c = Basket(**{k: v for k, v in row.items() if k != "name"})
        names = quantity_names(c)
        runs = {}
        for nodes in high["gauss_hermite_nodes"]:
            start = perf_counter()
            runs[nodes] = gauss_hermite_values(c, nodes)
            runs[nodes, "seconds"] = perf_counter() - start
        finest = runs[high["gauss_hermite_nodes"][-1]]
        exact = None
        if c.dimension == 1:
            args = (c.spots[0], c.strike, c.tau, c.rate, c.volatilities[0], c.dividends[0], c.option)
            greeks = black_scholes_greeks(*args)
            exact = dict(price=black_scholes_price(*args), delta_1=greeks["delta"], gamma_11=greeks["gamma"],
                         vega_1=greeks["vega"], theta=greeks["theta"], rho=greeks["rho"])
        for nodes in high["gauss_hermite_nodes"]:
            for quantity, value, best in zip(names, runs[nodes], finest):
                dimension_rows.append(dict(case=row["name"], dimension=c.dimension, nodes=nodes,
                                           evaluations=nodes ** (c.dimension - 1), quantity=quantity, value=value,
                                           change_from_finest=value - best,
                                           black_scholes_error=value - exact[quantity] if exact else "",
                                           seconds=runs[nodes, "seconds"]))
        if c.dimension > 1:
            for seed in config["seeds"]:
                for variant in config["variants"]:
                    start = perf_counter()
                    est = rqmc_estimate(c, high["points_per_replicate"], high["replicates"], seed,
                                        direction=variant["direction"], control=variant["control"], **common)
                    seconds = perf_counter() - start
                    for quantity, best in zip(names, finest):
                        dimension_runs.append(estimate_row(row["name"], variant["name"], high["points_per_replicate"],
                                                           seed, quantity, est.values[quantity],
                                                           est.standard_errors[quantity], *est.intervals[quantity],
                                                           best, seconds))
        print(f"Dimension check: {row['name']}", flush=True)

    # 5. Improved two-stock ADI against the quadrature reference.
    adi = settings["adi"]
    adi_rows = []
    lookup = {name: c for name, _, c in cases}
    for name in adi["cases"]:
        c = lookup[name]
        for intervals in adi["intervals"]:
            for richardson in (False, True):
                start = perf_counter()
                values = adi_values(c, intervals, richardson=richardson, tail_sd=adi["tail_sd"],
                                    clustering=adi["clustering"])
                seconds = perf_counter() - start
                for quantity in SPOT_AND_THETA:
                    adi_rows.append(dict(case=name, intervals=intervals, richardson=richardson,
                                         parameter_greeks=False, quantity=quantity, value=values[quantity],
                                         reference=references[name][quantity],
                                         error=values[quantity] - references[name][quantity], seconds=seconds))
        start = perf_counter()
        values = adi_values(c, adi["parameter_greek_intervals"], richardson=True, parameter_greeks=True,
                            tail_sd=adi["tail_sd"], clustering=adi["clustering"])
        seconds = perf_counter() - start
        for quantity in ("vega_1", "vega_2", "rho", "correlation_12"):
            adi_rows.append(dict(case=name, intervals=adi["parameter_greek_intervals"], richardson=True,
                                 parameter_greeks=True, quantity=quantity, value=values[quantity],
                                 reference=references[name][quantity],
                                 error=values[quantity] - references[name][quantity], seconds=seconds))
        print(f"ADI: {name}", flush=True)

    # 6. Uncertainty calibration over many independent seeds.
    calibration = settings["calibration"]
    calibration_rows = []
    seed = calibration["seed_start"]
    two = calibration["two_stock"]
    v1, c = next((v, c) for name, v, c in cases if name == two["case"])
    for variant in config["variants"]:
        for _ in range(two["seeds"]):
            est = rqmc_estimate(c, two["points_per_replicate"], two["replicates"], seed, greeks=False,
                                direction=variant["direction"], control=variant["control"], **common)
            calibration_rows.append(estimate_row(two["case"], variant["name"], two["points_per_replicate"], seed,
                                                 "price", est.values["price"], est.standard_errors["price"],
                                                 *est.intervals["price"], references[two["case"]]["price"]))
            seed += 1
    three = calibration["three_stock"]
    row = next(r for r in settings["higher_dimension_cases"] if r["name"] == three["case"])
    c = Basket(**{k: v for k, v in row.items() if k != "name"})
    exact = dict(zip(quantity_names(c), gauss_hermite_values(c, max(high["gauss_hermite_nodes"]))))
    for variant in config["variants"]:
        for _ in range(three["seeds"]):
            est = rqmc_estimate(c, three["points_per_replicate"], three["replicates"], seed,
                                direction=variant["direction"], control=variant["control"], **common)
            for quantity in est.names:
                calibration_rows.append(estimate_row(three["case"], variant["name"], three["points_per_replicate"],
                                                     seed, quantity, est.values[quantity],
                                                     est.standard_errors[quantity], *est.intervals[quantity],
                                                     exact[quantity]))
            seed += 1
    print("Calibration complete", flush=True)

    files = (("reference_agreement.csv", agreement), ("price_runs.csv", price_rows),
             ("price_summary.csv", price_summary), ("greek_runs.csv", greek_rows),
             ("greek_summary.csv", greek_summary), ("dimension_quadrature.csv", dimension_rows),
             ("dimension_rqmc.csv", dimension_runs), ("adi_convergence.csv", adi_rows),
             ("calibration_runs.csv", calibration_rows))
    for filename, rows in files:
        write_csv(output / filename, rows)
    with OLD_ADI.open(encoding="utf-8") as handle:
        old_adi = [r for r in csv.DictReader(handle) if r["study"] == "mesh" and r["quantity"] == "price"]
    figure(output, price_summary, adi_rows, old_adi, adi["cases"])
    report(output, settings, agreement, price_summary, greek_summary, dimension_rows, dimension_runs, adi_rows, old_adi,
           calibration_rows)
    write_metadata(output, settings_path, pricing_sources() + list((ROOT / "src/basket").glob("*.py"))
                   + [Path(__file__), ROOT / "experiments/records.py", OLD_ADI])
    print(f"Saved {output}", flush=True)


def figure(output, price_summary, adi_rows, old_adi, adi_cases):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    methods = sorted({r["method"] for r in price_summary})
    colors = dict(zip(methods, ["#64748b", "#94a3b8", "#0284c7", "#0891b2", "#7c3aed", "#a78bfa"]))
    for method in methods:
        rows = sorted((r for r in price_summary if r["method"] == method), key=lambda r: r["points_per_replicate"])
        style = "--" if method.startswith("two_asset") else "-"
        axes[0].loglog([64 * r["points_per_replicate"] for r in rows], [r["rmse"] for r in rows], "o" + style,
                       color=colors[method], label=method.replace("_", " "))
    axes[0].set(xlabel="Production points (64 replicates)", ylabel="Price RMSE", title="Two-stock price, eight cases")
    axes[0].legend(fontsize=7)
    palette = ["#0284c7", "#f59e0b", "#7c3aed", "#0891b2", "#dc2626", "#16a34a"]
    for name, color in zip(adi_cases, palette):
        rows = [r for r in adi_rows if r["case"] == name and r["quantity"] == "price"
                and not r["parameter_greeks"] and not r["richardson"]]
        axes[1].loglog([r["intervals"] for r in rows], [abs(r["error"]) for r in rows], "o-", color=color,
                       label=name.replace("_", " "))
        old = [r for r in old_adi if r["case"] == name]
        if old:
            axes[1].loglog([float(r["grid"]) for r in old], [abs(float(r["error"])) for r in old], "x--",
                           color=color, alpha=0.6)
    axes[1].set(xlabel="Intervals per stock", ylabel="Absolute price error",
                title="ADI: new (solid) vs two_asset_v1 (dashed)")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    fig.savefig(output / "validation.pdf")
    plt.close(fig)


def report(output, settings, agreement, price_summary, greek_summary, dimension_rows, dimension_runs, adi_rows, old_adi,
           calibration_rows):
    largest = max(settings["rqmc"]["price_points_per_replicate"])
    lines = ["# d-stock pricing engine validation", "",
             "Development cases only: the eight observed two_asset_v1 contracts and four one-, three- and "
             "four-stock contracts. No frozen test-set case is used.", "",
             "## Agreement with the independent two-asset quadrature", "",
             "The new engine integrates its conditional values with 128-node Gauss-Hermite; the reference is "
             "the adaptive quadrature of `src/basket`, which uses a different factor decomposition.", "",
             "| Quantity | Max abs difference (equal) | Max abs difference (volatility) | Reference bump change |",
             "|---|---:|---:|---:|"]
    for quantity in dict.fromkeys(r["quantity"] for r in agreement):
        rows = [r for r in agreement if r["quantity"] == quantity]
        lines.append(f"| {quantity} | {max(abs(r['difference_equal']) for r in rows):.2e} | "
                     f"{max(abs(r['difference_volatility']) for r in rows):.2e} | "
                     f"{max(abs(r['reference_bump_change']) for r in rows):.2e} |")
    lines += ["", f"## Price estimates, 64 replicates of {largest:,} points", "",
              "| Method | Price RMSE | Max abs error | RMS z | Coverage | Median seconds |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted((r for r in price_summary if r["points_per_replicate"] == largest), key=lambda r: r["rmse"]):
        lines.append(f"| {r['method']} | {r['rmse']:.3e} | {r['max_absolute_error']:.3e} | {r['rms_z']:.2f} | "
                     f"{r['coverage']:.3f} | {r['median_seconds']:.2f} |")
    lines += ["", "RMS z is the root-mean-square of error divided by the reported standard error; about one "
              "indicates calibrated uncertainty. Each row has 24 runs, but all cases share the same three seeds, "
              "so the runs are strongly correlated and their coverage is not a calibration test; see the "
              "calibration section below.", "",
              "## Uncertainty calibration over independent seeds", "",
              "| Case | Variant | Quantities | Runs | RMS z | Outside 95% interval |", "|---|---|---|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in calibration_rows):
        for variant in dict.fromkeys(r["method"] for r in calibration_rows if r["case"] == case):
            rows = [r for r in calibration_rows if r["case"] == case and r["method"] == variant]
            quantities = len({r["quantity"] for r in rows})
            lines.append(f"| {case} | {variant} | {'price' if quantities == 1 else f'all {quantities}'} | "
                         f"{len({r['seed'] for r in rows})} | {rms([r['z'] for r in rows]):.2f} | "
                         f"{1 - np.mean([r['covered'] for r in rows]):.3f} |")
    lines += ["", "Every seed is distinct. Three-stock references are the finest Gauss-Hermite values. Intervals use "
              "Student-t critical values over independent replicates.", "",
              f"## Sensitivities, 64 replicates of {settings['rqmc']['greek_points_per_replicate']:,} points", "",
              "| Quantity | " + " | ".join(v["name"] for v in settings["rqmc"]["variants"]) + " | RMS z (all) |",
              "|---|" + "---:|" * (len(settings["rqmc"]["variants"]) + 1)]
    for quantity in dict.fromkeys(r["quantity"] for r in greek_summary):
        rows = {r["method"]: r for r in greek_summary if r["quantity"] == quantity}
        z = rms([r["rms_z"] for r in rows.values()])
        lines.append(f"| {quantity} | " + " | ".join(f"{rows[v['name']]['rmse']:.2e}"
                                                    for v in settings["rqmc"]["variants"]) + f" | {z:.2f} |")
    lines += ["", "Deltas and the spot Hessian are exact conditional derivatives; vega, theta, rho and correlation "
              "sensitivities are central bumps with common draws, and their intervals exclude bump bias.", "",
              "## One, three and four stocks", "",
              "| Case | d | Nodes | Evaluations | Max change from finest | Max Black-Scholes error | Seconds |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in dimension_rows):
        for nodes in dict.fromkeys(r["nodes"] for r in dimension_rows if r["case"] == case):
            rows = [r for r in dimension_rows if r["case"] == case and r["nodes"] == nodes]
            bs = [abs(r["black_scholes_error"]) for r in rows if r["black_scholes_error"] != ""]
            lines.append(f"| {case} | {rows[0]['dimension']} | {nodes} | {rows[0]['evaluations']:,} | "
                         f"{max(abs(r['change_from_finest']) for r in rows):.1e} | "
                         f"{max(bs):.1e} | {rows[0]['seconds']:.2f} |" if bs else
                         f"| {case} | {rows[0]['dimension']} | {nodes} | {rows[0]['evaluations']:,} | "
                         f"{max(abs(r['change_from_finest']) for r in rows):.1e} | | {rows[0]['seconds']:.2f} |")
    lines += ["", "The one-stock Black-Scholes error of vega, theta and rho is bump bias of the central differences.", "",
              "| Case | Variant | Price RMSE vs finest Gauss-Hermite | RMS z (all quantities) | Median seconds |",
              "|---|---|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in dimension_runs):
        for variant in dict.fromkeys(r["method"] for r in dimension_runs if r["case"] == case):
            rows = [r for r in dimension_runs if r["case"] == case and r["method"] == variant]
            lines.append(f"| {case} | {variant} | {rms([r['error'] for r in rows if r['quantity'] == 'price']):.2e} | "
                         f"{rms([r['z'] for r in rows]):.2f} | {np.median([r['seconds'] for r in rows]):.2f} |")
    lines += ["", "## Two-stock ADI", "",
              "Sinh-clustered grids with the spot on a node, a far boundary scaled with volatility and maturity, "
              "and steps equal to intervals. Richardson combines the grid with its nested doubling.", "",
              "| Case | Intervals | Price error | Richardson price error | two_asset_v1 ADI price error | Max spot-Greek/theta rel. error (Richardson) | Seconds (Richardson) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in adi_rows):
        for intervals in settings["adi"]["intervals"]:
            plain = [r for r in adi_rows if r["case"] == case and r["intervals"] == intervals
                     and not r["richardson"] and not r["parameter_greeks"]]
            rich = [r for r in adi_rows if r["case"] == case and r["intervals"] == intervals
                    and r["richardson"] and not r["parameter_greeks"]]
            old = [r for r in old_adi if r["case"] == case and int(r["grid"]) == intervals]
            relative_error = max(abs(r["error"]) / max(abs(r["reference"]), 1e-12) for r in rich if r["quantity"] != "price")
            lines.append(f"| {case} | {intervals} | {plain[0]['error']:.2e} | {rich[0]['error']:.2e} | "
                         f"{float(old[0]['error']):.2e} | {relative_error:.1e} | {rich[0]['seconds']:.2f} |" if old else
                         f"| {case} | {intervals} | {plain[0]['error']:.2e} | {rich[0]['error']:.2e} | | "
                         f"{relative_error:.1e} | {rich[0]['seconds']:.2f} |")
    lines += ["", "Parameter sensitivities (Richardson, "
              f"{settings['adi']['parameter_greek_intervals']} intervals, bumped re-solves on a fixed grid):", "",
              "| Case | vega_1 | vega_2 | rho | correlation_12 |", "|---|---:|---:|---:|---:|"]
    for case in dict.fromkeys(r["case"] for r in adi_rows):
        rows = {r["quantity"]: r for r in adi_rows if r["case"] == case and r["parameter_greeks"]}
        lines.append(f"| {case} | " + " | ".join(f"{rows[q]['error']:.1e}"
                                                for q in ("vega_1", "vega_2", "rho", "correlation_12")) + " |")
    lines += ["", "Errors are signed differences from the quadrature reference. Puts are the call solution plus the "
              "exact parity term, so a put shares its call's errors (unequal and unequal_put). Grid solves are a "
              "two-stock baseline; their cost grows exponentially with the number of stocks.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
