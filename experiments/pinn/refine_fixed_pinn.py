import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import torch

from src.neural.fixed_pinn import load_fixed_pinn, sample_collocation, sample_terminal

from experiments.pinn import frozen_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=(
        "Refine a fixed-parity PINN candidate with an extra L-BFGS phase. "
        "Writes a NEW candidate; never touches the frozen primary checkpoint."))
    parser.add_argument("--source", required=True,
                        help="Candidate checkpoint (.pt) to refine, under "
                             "checkpoints/pinn/research_variants/fixed_pinn_candidates/")
    parser.add_argument("--tag", required=True,
                        help="Name of the new refined candidate (no extension)")
    arguments = parser.parse_args()

    settings = json.loads((ROOT / "experiments" / "config" / "neural" / "fixed_pinn_settings.json").read_text())
    training = settings["training"]
    refuse_primary_overwrite = frozen_checkpoint.refuse_primary_overwrite(ROOT)
    source = Path(arguments.source)
    if not source.is_absolute():
        source = ROOT / source
    if source == frozen_checkpoint.primary_checkpoint_path(ROOT):
        raise ValueError("refuse: refining the frozen primary directly is not "
                         "allowed; copy it to a candidate first")
    if not source.exists():
        raise FileNotFoundError(f"refine: source checkpoint not found: {source}")

    model = load_fixed_pinn(source)
    dtype = next(model.parameters()).dtype
    problem = model.problem
    x, tau = sample_collocation(
        problem, training["lbfgs_points"], training["seed"] + 100000,
        training["near_strike_fraction"], training["near_strike_half_width"],
        training["minimum_tau"], dtype,
    )
    terminal_x, terminal_tau, payoff = sample_terminal(
        problem, training["lbfgs_terminal_points"], training["seed"] + 150000,
        training["near_strike_fraction"], training["near_strike_half_width"], dtype,
    )
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=training["refinement_max_iterations"],
        history_size=training["lbfgs_history_size"], line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-14,
    )
    history = []
    calls = 0
    start = perf_counter()

    def closure():
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        residual_loss = torch.mean(model.pde_residual(x, tau)**2)
        terminal_loss = torch.mean((model(terminal_x, terminal_tau) - payoff)**2)
        loss = residual_loss + training["terminal_loss_weight"] * terminal_loss
        loss.backward()
        calls += 1
        if calls == 1 or calls % training["log_every"] == 0:
            history.append({
                "phase": "lbfgs_refine", "step": calls, "loss": float(loss.detach()),
                "pde_loss": float(residual_loss.detach()),
                "terminal_loss": float(terminal_loss.detach()), "learning_rate": "",
                "elapsed_seconds": perf_counter() - start,
            })
        return loss

    optimizer.step(closure)
    refinement_seconds = perf_counter() - start
    residual_loss = float(torch.mean(model.pde_residual(x, tau)**2).detach())
    terminal_loss = float(torch.mean((model(terminal_x, terminal_tau) - payoff)**2).detach())
    final_loss = residual_loss + training["terminal_loss_weight"] * terminal_loss

    checkpoint_path = frozen_checkpoint.candidate_path(ROOT, arguments.tag)
    torch.save({
        "state_dict": model.state_dict(), "problem": vars(problem),
        "network": {
            "hidden_layers": settings["network"]["hidden_layers"],
            "hidden_width": settings["network"]["hidden_width"],
        },
    }, checkpoint_path)

    table_dir = (ROOT / "results" / "pinn" / "research_variants"
                 / "fixed_pinn_candidates" / arguments.tag / "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "refinement_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "model": "fixed_black_scholes_pinn_refined",
        "source_checkpoint": str(source.relative_to(ROOT)),
        "source_sha256": frozen_checkpoint.sha256_of(source),
        "refinement_seed": training["seed"],
        "lbfgs_closure_calls": calls,
        "refinement_wall_seconds": refinement_seconds,
        "final_pde_loss": residual_loss,
        "final_terminal_loss": terminal_loss,
        "final_training_loss": final_loss,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }
    (table_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()