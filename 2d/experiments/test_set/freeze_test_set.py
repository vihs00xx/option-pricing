"""Generate, reference and freeze the two-stock test set. No pricing method is evaluated."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from experiments.records import ROOT, digest, pricing_sources, prepare_output, relative, write_csv, write_metadata
from src.basket import BasketContract, arithmetic_greeks, quadrature_price
from src.pricing import Basket, gauss_hermite_values, quantity_names

PROTOCOL = ROOT / "experiments/config/protocol/test_set_v1.json"
PARAMETERS = ("moneyness_1", "moneyness_2", "tau", "volatility_1", "volatility_2", "correlation",
              "rate", "dividend_1", "dividend_2", "weight_1")
FLOORS = {"price": 1e-4, "delta": 1e-4, "gamma": 1e-7, "vega": 1e-4, "theta": 1e-4, "rho": 1e-4,
          "correlation": 1e-4}


def contract_fields(values, option, strike):
    return dict(spots=[strike * values["moneyness_1"], strike * values["moneyness_2"]], strike=strike,
                tau=values["tau"], rate=values["rate"],
                volatilities=[values["volatility_1"], values["volatility_2"]],
                dividends=[values["dividend_1"], values["dividend_2"]],
                weights=[values["weight_1"], 1.0 - values["weight_1"]],
                correlation=values["correlation"], option=option)


def build_cases(protocol):
    domain, splits, strike, seed = (protocol[k] for k in ("training_domain", "splits", "strike", "seed"))
    lower = np.array([domain[p][0] for p in PARAMETERS])
    upper = np.array([domain[p][1] for p in PARAMETERS])
    cases = []

    def add(case_id, split, reason, values, option):
        values = {p: float(values[p]) for p in PARAMETERS}
        cases.append(dict(case_id=case_id, split=split, reason=reason, option=option, parameters=values,
                          contract=contract_fields(values, option, strike)))

    def sample(count, offset):
        return qmc.LatinHypercube(d=len(PARAMETERS), seed=seed + offset).random(count)

    split = splits["interpolation"]
    for i, row in enumerate(qmc.scale(sample(split["count"], 0), lower, upper)):
        add(f"interpolation_{i + 1:03d}", "interpolation", split["reason"], dict(zip(PARAMETERS, row)),
            "call" if i % 2 == 0 else "put")
    split = splits["short_maturity"]
    unit = sample(split["count"], 1)
    tau_column = PARAMETERS.index("tau")
    for i, (row, u) in enumerate(zip(qmc.scale(unit, lower, upper), unit[:, tau_column])):
        values = dict(zip(PARAMETERS, row))
        values["tau"] = split["tau"][0] + u * (split["tau"][1] - split["tau"][0])
        add(f"short_maturity_{i + 1:03d}", "short_maturity", split["reason"], values,
            "call" if i % 2 == 0 else "put")
    split = splits["extrapolation"]
    outside = [(p, v) for p in PARAMETERS for v in split["values"].get(p, [])]
    rows = qmc.scale(sample(len(outside) * split["cases_per_value"], 2), lower, upper)
    index = 0
    for parameter, value in outside:
        for _ in range(split["cases_per_value"]):
            values = dict(zip(PARAMETERS, rows[index]))
            values[parameter] = value
            add(f"extrapolation_{parameter}_{index + 1:03d}", "extrapolation",
                f"{parameter} = {value} outside the training domain", values,
                "call" if index % 2 == 0 else "put")
            index += 1
    split = splits["anchor"]
    for i, row in enumerate(split["cases"]):
        values = {**split["defaults"], **{k: v for k, v in row.items() if k not in ("name", "option")}}
        add(f"anchor_{i + 1:02d}_{row['name']}", "anchor", split["reason"], values, row["option"])
    return cases


def development_contracts(protocol):
    contracts = []
    for relative in protocol["development_case_settings"]:
        settings = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        defaults = settings.get("two_asset_defaults", {})
        for row in settings.get("cases", []) + settings.get("two_asset_cases", []):
            fields = {**defaults, **{k: v for k, v in row.items() if k != "name"}}
            contracts.append((row["name"], asdict(BasketContract(**fields))))
    return contracts


def same_contract(a, b):
    return a["option"] == b["option"] and all(
        np.allclose(np.asarray(a[k], float), np.asarray(b[k], float), rtol=0, atol=1e-9)
        for k in ("spots", "strike", "tau", "rate", "volatilities", "dividends", "weights", "correlation"))


def floor(quantity):
    return FLOORS[quantity.split("_")[0]]


def reference_rows(case, protocol):
    settings = protocol["reference"]
    v1 = BasketContract(**case["contract"])
    tolerance = dict(atol=settings["quadrature_atol"], rtol=settings["quadrature_rtol"])
    estimate = quadrature_price(v1, **tolerance)
    reference = {"price": estimate.price, **arithmetic_greeks(v1)}
    reference["correlation_12"] = reference.pop("correlation")
    half = arithmetic_greeks(v1, 0.5)
    half["correlation_12"] = half.pop("correlation")
    asset = quadrature_price(v1, conditioning="asset", **tolerance).price
    c = Basket(**case["contract"])
    names = quantity_names(c)
    values = dict(case_id=case["case_id"], split=case["split"], **{q: reference[q] for q in names})
    bumped = [q for q in names if not q.startswith(("price", "delta", "gamma"))]
    checks = dict(case_id=case["case_id"], split=case["split"], integration_error=estimate.integration_error,
                  tail_bound=estimate.tail_bound, asset_conditioning_difference=asset - estimate.price,
                  max_bump_halving_relative_change=max(abs(half[q] - reference[q]) / max(abs(reference[q]), floor(q))
                                                       for q in bumped))
    for direction in ("volatility", "equal"):
        engine = dict(zip(names, gauss_hermite_values(c, settings["gauss_hermite_nodes"], direction=direction)))
        relative = {q: abs(engine[q] - reference[q]) / max(abs(reference[q]), floor(q)) for q in names}
        worst = max(relative, key=relative.get)
        checks[f"max_engine_relative_difference_{direction}"] = relative[worst]
        checks[f"worst_engine_quantity_{direction}"] = worst
        checks[f"worst_engine_absolute_difference_{direction}"] = engine[worst] - reference[worst]
    return values, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/test_set_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = prepare_output(args.output, args.overwrite)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cases = build_cases(protocol)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise RuntimeError("Case identifiers are not unique")
    development = development_contracts(protocol)
    for case in cases:
        fields = asdict(BasketContract(**case["contract"]))
        for name, other in development:
            if same_contract(fields, other):
                raise RuntimeError(f"{case['case_id']} duplicates observed development case {name}")
    start = perf_counter()
    references, checks = [], []
    for i, case in enumerate(cases, 1):
        values, check = reference_rows(case, protocol)
        references.append(values)
        checks.append(check)
        if i % 25 == 0 or i == len(cases):
            print(f"References: {i}/{len(cases)}", flush=True)
    seconds = perf_counter() - start
    (output / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    write_csv(output / "cases.csv", [dict(case_id=c["case_id"], split=c["split"], option=c["option"],
                                          reason=c["reason"], **c["parameters"]) for c in cases])
    write_csv(output / "references.csv", references)
    write_csv(output / "reference_checks.csv", checks)
    counts = {split: sum(c["split"] == split for c in cases) for split in dict.fromkeys(c["split"] for c in cases)}
    frozen = dict(protocol, frozen_at=datetime.now(timezone.utc).isoformat(), case_counts=counts,
                  total_cases=len(cases), development_cases_checked=len(development),
                  development_contracts=[dict(name=name, **fields) for name, fields in development],
                  reference_seconds=seconds,
                  cases_sha256=digest(output / "cases.json"), references_sha256=digest(output / "references.csv"))
    (output / "protocol.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (output / "protocol.sha256").write_text(digest(output / "protocol.json") + "\n", encoding="utf-8")
    report(output, frozen, checks, references)
    # Labels come only from the frozen src/basket quadrature, so verification pins
    # it and the generator. The cross-checking engine and the development settings
    # are recorded but not enforced: the excluded contracts are stored in
    # protocol.json, and later engine or study changes must not invalidate the test set.
    provenance = pricing_sources() + [ROOT / p for p in protocol["development_case_settings"]]
    write_metadata(output, PROTOCOL, list((ROOT / "src/basket").glob("*.py"))
                   + [ROOT / "src/black_scholes.py", Path(__file__)],
                   extra=dict(provenance_sha256={relative(p): digest(p) for p in provenance}))
    print(f"Saved {output}", flush=True)


def report(output, protocol, checks, references):
    lines = ["# Frozen two-stock test set", "", f"Frozen at {protocol['frozen_at']}. {protocol['purpose']}", "",
             "## Cases", "", "| Split | Cases | Definition |", "|---|---:|---|"]
    splits = protocol["splits"]
    for split, count in protocol["case_counts"].items():
        lines.append(f"| {split} | {count} | {splits[split]['reason']} |")
    lines += ["", f"Strike {protocol['strike']:g}; calls and puts alternate within each split. "
              "Latin hypercube seed " + str(protocol["seed"]) + ".", "", "Training domain:", "",
              "| Parameter | Range |", "|---|---|"]
    lines += [f"| {p} | {v[0]:g} to {v[1]:g} |" for p, v in protocol["training_domain"].items()]
    lines += ["", f"No case duplicates any of the {protocol['development_cases_checked']} observed development "
              "contracts.", "", "## References", "", protocol["reference"]["method"], "",
              "| Check | Worst case |", "|---|---:|"]
    worst = lambda key: max(abs(r[key]) for r in checks)
    lines += [f"| Quadrature error estimate | {worst('integration_error'):.1e} |",
              f"| Truncation tail bound | {worst('tail_bound'):.1e} |",
              f"| Asset-conditioning price difference | {worst('asset_conditioning_difference'):.1e} |",
              f"| Gauss-Hermite engine, volatility direction, all quantities | {worst('max_engine_relative_difference_volatility'):.1e} |",
              f"| Gauss-Hermite engine, equal direction, all quantities | {worst('max_engine_relative_difference_equal'):.1e} |",
              f"| Relative change from halving reference bumps | {worst('max_bump_halving_relative_change'):.1e} |",
              "", "Relative differences divide by max(|reference|, floor), with floors 1e-7 for Gamma and 1e-4 "
              "otherwise. The engine cross-check is an independent factorization of the same model. With equal "
              "loadings and strongly unequal volatilities its shared factor is small and 128 nodes under-resolve "
              "some Gammas and correlation sensitivities; finite differences of the reference price confirm the "
              "reference in those cases.", "",
              "Volatility-direction differences above 1e-6 relative:", "",
              "| Case | Quantity | Reference | Engine minus reference | Relative |", "|---|---|---:|---:|---:|"]
    lookup = {r["case_id"]: r for r in references}
    for r in sorted(checks, key=lambda r: -r["max_engine_relative_difference_volatility"]):
        if r["max_engine_relative_difference_volatility"] > 1e-6:
            q = r["worst_engine_quantity_volatility"]
            lines.append(f"| {r['case_id']} | {q} | {lookup[r['case_id']][q]:.6g} | "
                         f"{r['worst_engine_absolute_difference_volatility']:.1e} | "
                         f"{r['max_engine_relative_difference_volatility']:.1e} |")
    lines += ["", "These are either under-resolved engine integrals at extreme correlation (more nodes converge to "
              "the reference) or sensitivities that are zero to round-off. No pricing method has been evaluated "
              "on these cases.", "", "## Rules", ""]
    lines += [f"- {rule}" for rule in protocol["rules"]]
    lines += ["", "`protocol.sha256` identifies `protocol.json`, which pins `cases.json` and `references.csv`; "
              "`metadata.json` pins every output and the generating sources. Verify with "
              "`python -m experiments.verify results/test_set_v1`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
