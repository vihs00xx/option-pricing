"""Train one PINN study variant under the shared matched budget.

Usage:
    python -m experiments.pinn.train_pinn_variant --variant baseline_matched
    python -m experiments.pinn.train_pinn_variant --variant constrained
    python -m experiments.pinn.train_pinn_variant --variant causal
    python -m experiments.pinn.train_pinn_variant --variant constrained \
        --seed-offset 30000 --epochs 120 --tag short_s2

All variants share ``experiments/config/pinn_study_settings.json`` so
the epoch budget, batch sizes, and seeds are identical by construction.
"""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.neural.fixed_pinn import FixedBlackScholesPINN, FixedBlackScholesProblem
from src.neural.pinn_training import VariantTrainer


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "pinn_study_settings.json"
HISTORY_FIELDS = (
    "phase", "step", "loss", "pde_loss", "terminal_loss",
    "category_weight_price_bounds", "category_weight_delta_bounds",
    "category_weight_gamma", "causal_epsilon", "causal_effective_bins",
    "constraint_source", "constraint_loss_price_bounds",
    "constraint_loss_delta_bounds", "constraint_loss_gamma",
    "constraint_total", "elapsed_seconds",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True,
                        choices=("baseline_matched", "constrained", "causal"))
    parser.add_argument("--seed", type=int, default=None,
                        help="Global seed shared by all variants (default: settings seed)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the Adam epoch count (short repeats)")
    parser.add_argument("--tag", default=None,
                        help="Suffix for output paths of repeat runs")
    parser.add_argument("--skip-lbfgs", action="store_true",
                        help="Stop after Adam (save the Adam checkpoint only)")
    arguments = parser.parse_args()

    settings = json.loads(SETTINGS_PATH.read_text())
    budget = dict(settings["matched_budget"])
    if arguments.epochs is not None:
        budget["adam_epochs"] = arguments.epochs
    seed = arguments.seed if arguments.seed is not None else settings["seed"]

    network = settings["network"]
    dtype = torch.float64 if network["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    problem = FixedBlackScholesProblem(**{
        name: settings["problem"][name]
        for name in ("K", "T", "r", "sigma", "q", "x_min", "x_max")
    })
    model = FixedBlackScholesPINN(
        problem, hidden_layers=network["hidden_layers"],
        hidden_width=network["hidden_width"],
    )
    trainer = VariantTrainer(
        model, budget, arguments.variant,
        settings.get("constrained", {}),
        seed,
        causal_settings=settings.get("causal", {}),
        dtype=dtype,
    )
    name = f"{arguments.variant}_{arguments.tag}" if arguments.tag else arguments.variant
    table_dir = ROOT / "results" / "pinn" / f"study_{name}" / "tables"
    figure_dir = ROOT / "results" / "pinn" / f"study_{name}" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training variant '{name}' with seed {seed}, "
          f"{budget['adam_epochs']} epochs")
    history, metadata = trainer.train(
        wall_clock_cap=budget["wall_clock_cap_seconds"],
        log_every=budget["log_every"],
    )
    metadata["training_run_id"] = f"{name}_seed_{seed}"
    metadata["settings_sha256"] = hashlib.sha256(
        SETTINGS_PATH.read_bytes()
    ).hexdigest()[:16]

    # Save the Adam checkpoint before L-BFGS so L-BFGS and NysNewton-CG
    # both start from exactly the same state (fair optimiser comparison).
    adam_checkpoint_path = ROOT / "checkpoints" / "pinn" / f"study_{name}_adam.pt"
    adam_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "problem": asdict(problem),
        "network": {"hidden_layers": network["hidden_layers"],
                    "hidden_width": network["hidden_width"]},
        "variant": name,
        "seed": seed,
    }, adam_checkpoint_path)
    metadata["adam_checkpoint"] = str(adam_checkpoint_path.relative_to(ROOT))

    refinement_history = []
    if not arguments.skip_lbfgs:
        print(f"Refining variant '{name}' with L-BFGS "
              f"(same schedule as all variants)")
        refinement_history, refinement_metadata = trainer.refine_lbfgs(
            settings["refinement"], log=print,
        )
        refinement_metadata["training_run_id"] = f"{name}_seed_{seed}_lbfgs"
        print(json.dumps(refinement_metadata, indent=2))
        metadata["lbfgs"] = refinement_metadata

    checkpoint_path = ROOT / "checkpoints" / "pinn" / f"study_{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "problem": asdict(problem),
        "network": {"hidden_layers": network["hidden_layers"],
                    "hidden_width": network["hidden_width"]},
        "variant": name,
        "seed": seed,
    }, checkpoint_path)
    metadata["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
    with (table_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})
    if refinement_history:
        (table_dir / "refinement_history.json").write_text(
            json.dumps(refinement_history, indent=2) + "\n"
        )
    (table_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy([row["step"] for row in history], [row["loss"] for row in history],
                label="total loss")
    ax.semilogy([row["step"] for row in history], [row["pde_loss"] for row in history],
                label="PDE loss")
    ax.semilogy([row["step"] for row in history],
                [row["terminal_loss"] for row in history], label="terminal loss")
    ax.set_xlabel("Adam epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{name} training (matched budget)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
