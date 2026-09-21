import csv
import json
from pathlib import Path
from time import perf_counter

import torch

from src.neural.fixed_pinn import load_fixed_pinn, sample_collocation, sample_terminal


ROOT = Path(__file__).resolve().parents[2]


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "fixed_pinn_settings.json").read_text())
    training = settings["training"]
    table_dir = ROOT / "results" / "pinn" / "fixed" / "tables"
    checkpoint_path = ROOT / "checkpoints" / "pinn" / "fixed_black_scholes_pinn.pt"
    model = load_fixed_pinn(checkpoint_path)
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
    history_path = table_dir / "training_history.csv"
    history = list(csv.DictReader(history_path.open()))
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
    torch.save({
        "state_dict": model.state_dict(), "problem": vars(problem),
        "network": {
            "hidden_layers": settings["network"]["hidden_layers"],
            "hidden_width": settings["network"]["hidden_width"],
        },
    }, checkpoint_path)
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    metadata_path = table_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["optimizer_steps"] += calls
    metadata["lbfgs_refinement_closure_calls"] = (
        metadata.get("lbfgs_refinement_closure_calls", 0) + calls
    )
    metadata["refinement_wall_seconds"] = (
        metadata.get("refinement_wall_seconds", 0) + refinement_seconds
    )
    metadata["wall_seconds"] += refinement_seconds
    metadata["final_pde_loss"] = residual_loss
    metadata["final_terminal_loss"] = terminal_loss
    metadata["final_training_loss"] = residual_loss + training["terminal_loss_weight"] * terminal_loss
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
