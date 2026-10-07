"""Cost at equal accuracy and break-even of the selected networks against COS, ADI, MC and RQMC.

Inputs: results/classical_v1/{benchmark.csv, timing.json} and results/neural/pinn_deeponet_v1/
{test_metrics.csv, neural_timing.json, selection.json, load_factor.json, ladder/*.json}.

* Cost at equal accuracy: for each classical method and quantity, the cheapest setting whose
  test RMSE (all 252 cases) is at most the network's, with its quiet single-thread seconds.
* Break-even: N* = (training seconds + label seconds) / (classical seconds - network seconds)
  per contract, for the classical setting that matches the network on the price and on all
  12 quantities at once. Batched networks are compared with COS priced as a batch (its
  vectorised time) and with ADI, MC and RQMC per contract; one-at-a-time networks with every
  classical method one contract at a time. Training seconds are the ladder wall-clock of the selected level and
  every level it was warm-started from, times the family's load factor (load_factor.json);
  process CPU seconds are listed alongside where recorded.

Usage (from heston/): python -m experiments.neural.compare
"""

import csv
import json

from experiments.neural.common import LADDER, RESULTS, level_settings, load_settings
from experiments.records import ROOT, write_csv, write_json
from src.heston.model import QUANTITIES

CLASSICAL = ROOT / "results/classical_v1"
FAMILIES = {"cos": "COS", "adi": "ADI", "mc_dt8": "QE Monte Carlo, dt 1/8", "mc_dt32": "QE Monte Carlo, dt 1/32",
            "rqmc": "QE-RQMC, dt 1/32"}


def family_of(setting):
    if setting.startswith("mc_"):
        return "mc_dt8" if setting.endswith("dt8") else "mc_dt32"
    return setting.split("_")[0]


def training_cost(study, level, factor):
    """Wall-clock (scaled) and CPU seconds of a level plus its warm-start chain, and its label seconds."""
    wall = cpu = 0.0
    labels = 0.0
    chain = []
    name = level
    while name:
        r = json.loads((LADDER / f"{name}_seed0.json").read_text(encoding="utf-8"))
        wall += r["total_seconds"]
        cpu += r.get("process_cpu_seconds") or float("nan")
        labels = max(labels, r.get("label_seconds", 0.0))
        chain.append(name)
        name = level_settings(study, name).get("warm_start")
    return dict(chain=chain, wall_seconds=wall, scaled_seconds=wall * factor, cpu_seconds=cpu, label_seconds=labels)


def main():
    study = load_settings()
    sel = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    factor = json.loads((RESULTS / "load_factor.json").read_text(encoding="utf-8"))
    timing = json.loads((CLASSICAL / "timing.json").read_text(encoding="utf-8"))["seconds"]
    nt = json.loads((RESULTS / "neural_timing.json").read_text(encoding="utf-8"))
    bench = [r for r in csv.DictReader((CLASSICAL / "benchmark.csv").open(encoding="utf-8"))]
    tm = {r["pricer"]: r for r in csv.DictReader((RESULTS / "test_metrics.csv").open(encoding="utf-8")) if r["split"] == "all"}
    rows_equal, rows_break, nets = [], [], {}
    for fam, s in sel["selected"].items():
        net = tm[f"{fam}_selected"]
        cost = training_cost(study, s["level"], factor[fam]["factor"])
        total = cost["scaled_seconds"] + cost["label_seconds"]
        t_net = {"batch": nt[f"{fam}_selected"]["batch_seconds"], "single": nt[f"{fam}_selected"]["single_seconds"]}
        nets[fam] = dict(level=s["level"], **cost, total_seconds=total, seconds_per_contract=t_net)
        for method, label in FAMILIES.items():
            settings = sorted([r for r in bench if family_of(r["setting"]) == method],
                              key=lambda r: timing[r["setting"]])
            for q in QUANTITIES:
                target = float(net[f"rmse_{q}"])
                hit = next((r for r in settings if float(r[f"rmse_{q}"]) <= target), None)
                rows_equal.append(dict(network=fam, level=s["level"], method=label, quantity=q, network_rmse=target,
                                       setting=hit["setting"] if hit else "none reaches it",
                                       setting_rmse=float(hit[f"rmse_{q}"]) if hit else float("nan"),
                                       seconds_per_contract=timing[hit["setting"]] if hit else float("nan")))
            for scope, qs in (("price", ("price",)), ("all 12", QUANTITIES)):
                hit = next((r for r in settings if all(float(r[f"rmse_{q}"]) <= float(net[f"rmse_{q}"]) for q in qs)), None)
                if hit is None:
                    rows_break.append(dict(network=fam, level=s["level"], method=label, matched_on=scope,
                                           setting="none reaches it", classical_seconds=float("nan"),
                                           break_even_batch="classical never matches", break_even_single="classical never matches"))
                    continue
                tc = timing[hit["setting"]]
                # COS is vectorised too: batched networks are compared with batched COS
                tc_batch = timing.get(hit["setting"] + "_batch", tc)
                be = {}
                for mode in ("batch", "single"):
                    gain = (tc_batch if mode == "batch" else tc) - t_net[mode]
                    be[mode] = f"{total / gain:,.0f}" if gain > 0 else "never (classical is faster)"
                rows_break.append(dict(network=fam, level=s["level"], method=label, matched_on=scope,
                                       setting=hit["setting"], classical_seconds=tc, classical_batch_seconds=tc_batch,
                                       break_even_batch=be["batch"], break_even_single=be["single"]))
    write_csv(RESULTS / "equal_accuracy.csv", rows_equal)
    write_csv(RESULTS / "break_even.csv", rows_break)
    write_json(RESULTS / "training_cost.json", dict(networks=nets, load_factor=factor,
                                                    note="scaled_seconds = ladder wall-clock x load factor; "
                                                         "label_seconds = torch-twin labelling of the training data"))
    for r in rows_break:
        print(r)


if __name__ == "__main__":
    main()
