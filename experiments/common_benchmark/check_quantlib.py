import csv
import json
from pathlib import Path

import QuantLib as ql

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.common_benchmark import QUANTITIES, inputs_from_case
from src.crank_nicolson import crank_nicolson_values


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def quantlib_values(inputs, settings):
    evaluation_date = ql.Date(2, ql.January, 2025)
    ql.Settings.instance().evaluationDate = evaluation_date
    day_count = ql.Actual365Fixed()
    maturity_days = max(1, round(inputs["tau"] * 365))
    maturity_date = evaluation_date + maturity_days
    adjusted_tau = day_count.yearFraction(evaluation_date, maturity_date)
    spot = ql.SimpleQuote(inputs["S"])
    rate = ql.SimpleQuote(inputs["r"])
    dividend = ql.SimpleQuote(inputs["q"])
    volatility = ql.SimpleQuote(inputs["sigma"])
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, ql.QuoteHandle(rate), day_count)
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(evaluation_date, ql.QuoteHandle(dividend), day_count)
    )
    volatility_curve = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(evaluation_date, ql.NullCalendar(),
                            ql.QuoteHandle(volatility), day_count)
    )
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot), dividend_curve, risk_free_curve, volatility_curve
    )
    option_type = ql.Option.Call if inputs["option"] == "call" else ql.Option.Put
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, inputs["K"]),
        ql.EuropeanExercise(maturity_date),
    )
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(
        process, settings["time_grid"], settings["space_grid"],
        settings["damping_steps"], ql.FdmSchemeDesc.CrankNicolson(),
    ))
    values = {
        "price": option.NPV(), "delta": option.delta(),
        "gamma": option.gamma(),
    }
    values["theta"] = (-0.5 * inputs["sigma"]**2 * inputs["S"]**2 * values["gamma"]
                       - (inputs["r"] - inputs["q"]) * inputs["S"] * values["delta"]
                       + inputs["r"] * values["price"])
    vol_step = 0.005 * inputs["sigma"]
    volatility.setValue(inputs["sigma"] + vol_step)
    plus = option.NPV()
    volatility.setValue(inputs["sigma"] - vol_step)
    minus = option.NPV()
    volatility.setValue(inputs["sigma"])
    values["vega"] = (plus - minus) / (2 * vol_step)
    rate.setValue(inputs["r"] + 0.0001)
    plus = option.NPV()
    rate.setValue(inputs["r"] - 0.0001)
    minus = option.NPV()
    rate.setValue(inputs["r"])
    values["rho"] = (plus - minus) / 0.0002
    return values, adjusted_tau


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "common_benchmark_settings.json").read_text())
    table_dir = ROOT / "results" / "common_benchmark" / "tables"
    with (table_dir / "benchmark_cases.csv").open(newline="") as handle:
        anchors = [row for row in csv.DictReader(handle) if row["split"] == "anchor"]
    positions = [0, 1, 6, 8, 9, 10, 11, 15][:settings["quantlib"]["case_count"]]
    selected = [anchors[index] for index in positions]
    rows = []
    failures = []
    config = settings["crank_nicolson"]
    tolerances = {name: settings["quantlib"][f"{name}_tolerance"] for name in QUANTITIES}
    for case in selected:
        inputs = inputs_from_case(case)
        quantlib, adjusted_tau = quantlib_values(inputs, settings["quantlib"])
        inputs["tau"] = adjusted_tau
        ours = crank_nicolson_values(
            **inputs, n_space=config["base_grid_size"], n_time=config["base_grid_size"],
            rannacher_steps=config["rannacher_steps"], mesh=config["mesh"],
            concentration=config["mesh_concentration"],
            volatility_bump=config["volatility_bump"], rate_bump=config["rate_bump"],
            richardson=config["richardson"],
        )
        analytical = {"price": black_scholes_price(**inputs), **black_scholes_greeks(**inputs)}
        for quantity in QUANTITIES:
            difference = ours[quantity] - quantlib[quantity]
            row = {
                "case_id": case["case_id"], "quantity": quantity,
                "adjusted_tau": adjusted_tau, "ours": ours[quantity],
                "quantlib": quantlib[quantity], "analytical": analytical[quantity],
                "ours_minus_quantlib": difference,
                "absolute_difference": abs(difference),
                "ours_absolute_error": abs(ours[quantity] - analytical[quantity]),
                "quantlib_absolute_error": abs(quantlib[quantity] - analytical[quantity]),
                "tolerance": tolerances[quantity],
            }
            rows.append(row)
            if row["absolute_difference"] > tolerances[quantity]:
                failures.append(row)
    write_csv(table_dir / "quantlib_crosscheck.csv", rows)
    summary = {
        "passed": not failures,
        "quantlib_version": ql.__version__,
        "cases": len(selected),
        "grid": {
            "time": settings["quantlib"]["time_grid"],
            "space": settings["quantlib"]["space_grid"],
            "damping_steps": settings["quantlib"]["damping_steps"],
        },
        "maximum_absolute_differences": {
            quantity: max(row["absolute_difference"] for row in rows if row["quantity"] == quantity)
            for quantity in QUANTITIES
        },
        "failures": failures,
    }
    (table_dir / "quantlib_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise AssertionError("QuantLib cross-check failed")


if __name__ == "__main__":
    main()
