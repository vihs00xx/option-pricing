"""Run shape-violation and accuracy diagnostics for a PINN checkpoint.

Usage:
    python -m experiments.pinn.run_pinn_diagnostics \
        --checkpoint checkpoints/pinn/primary/anchored/fixed_black_scholes_pinn.pt \
        --name original_fixed

Writes ``diagnostics.json``, ``diagnostics.csv`` and a violation figure
under ``results/pinn/research_variants/diagnostics/<name>/tables`` and
``.../figures``. Unlike the
fixed-PINN evaluation this script never raises on failed acceptance;
it reports measurements.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.neural.fixed_pinn import load_fixed_pinn
from src.neural.pinn_diagnostics import (
    full_diagnostics,
    predict_grid,
    reference_values,
    violation_summary,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "neural" / "pinn_study_settings.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True, help="Output folder name")
    arguments = parser.parse_args()

    settings = json.loads(SETTINGS_PATH.read_text())
    evaluation = settings["evaluation"]
    problem = settings["problem"]
    model = load_fixed_pinn(arguments.checkpoint)
    model.eval()

    x_axis = np.linspace(problem["x_min"], problem["x_max"], evaluation["x_points"])
    tau_axis = np.linspace(0, problem["T"], evaluation["tau_points"])
    x_grid, tau_grid = np.meshgrid(x_axis, tau_axis, indexing="xy")
    spots = (problem["K"] * x_grid).ravel()
    tau = tau_grid.ravel()
    y = np.log(spots / problem["K"])
    predicted = predict_grid(model, y, tau, evaluation["batch_size"])
    reference = reference_values(
        spots, tau, problem["K"], problem["r"], problem["sigma"], problem["q"]
    )
    diagnostics = full_diagnostics(
        predicted, reference, spots, tau,
        problem["K"], problem["r"], problem["q"], evaluation,
    )
    diagnostics.pop("regions")
    diagnostics.pop("violation_magnitudes")

    output = ROOT / "results" / "pinn" / "research_variants" / "diagnostics" / arguments.name
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    rows = []
    for region, entries in diagnostics["violations"].items():
        for violation, summary in entries.items():
            rows.append({"region": region, "violation": violation, **summary})
    with (table_dir / "violations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = (
        "price_below_intrinsic", "price_above_discounted_spot",
        "delta_below_zero", "delta_above_discount_factor", "gamma_negative",
    )
    regions = ("near_expiry", "near_strike", "low_tail", "high_tail", "core")
    width = 0.16
    positions = np.arange(len(regions))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, name in enumerate(names):
        proportions = [
            diagnostics["violations"][region][name]["proportion"]
            for region in regions
        ]
        maxima = [
            diagnostics["violations"][region][name]["maximum"]
            for region in regions
        ]
        axes[0].bar(positions + (index - 2) * width, proportions, width, label=name)
        axes[1].bar(positions + (index - 2) * width, maxima, width, label=name)
    for ax, title in zip(axes, ("Violation proportion", "Violation maximum size")):
        ax.set_xticks(positions)
        ax.set_xticklabels(regions)
        ax.set_title(f"{title} — {arguments.name}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Proportion of points")
    axes[1].set_ylabel("Maximum violation magnitude")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "violations.png", dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostics["violations"], indent=2))


if __name__ == "__main__":
    main()
