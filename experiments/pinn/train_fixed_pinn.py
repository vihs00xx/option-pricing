import csv
import json
import platform
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, sleep

import matplotlib.pyplot as plt
import psutil
import torch

from src.neural.fixed_pinn import (
    FixedBlackScholesPINN,
    FixedBlackScholesProblem,
    sample_collocation,
    sample_terminal,
)


ROOT = Path(__file__).resolve().parents[2]


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "fixed_pinn_settings.json").read_text())
    training = settings["training"]
    network = settings["network"]
    dtype = torch.float64 if network["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(training["seed"])
    problem = FixedBlackScholesProblem(**{
        name: settings["problem"][name]
        for name in ("K", "T", "r", "sigma", "q", "x_min", "x_max")
    })
    model = FixedBlackScholesPINN(
        problem, hidden_layers=network["hidden_layers"], hidden_width=network["hidden_width"]
    )
    table_dir = ROOT / "results" / "pinn" / "fixed" / "tables"
    figure_dir = ROOT / "results" / "pinn" / "fixed" / "figures"
    checkpoint_dir = ROOT / "checkpoints" / "pinn"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample_memory():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            sleep(0.01)

    memory_thread = threading.Thread(target=sample_memory, daemon=True)
    memory_thread.start()
    history = []
    start = perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["adam_learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["adam_epochs"], eta_min=training["adam_learning_rate"] * 0.01
    )
    for epoch in range(1, training["adam_epochs"] + 1):
        x, tau = sample_collocation(
            problem, training["adam_batch_size"], training["seed"] + epoch,
            training["near_strike_fraction"], training["near_strike_half_width"],
            training["minimum_tau"], dtype,
        )
        terminal_x, terminal_tau, payoff = sample_terminal(
            problem, training["terminal_batch_size"], training["seed"] + 50000 + epoch,
            training["near_strike_fraction"], training["near_strike_half_width"], dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        residual_loss = torch.mean(model.pde_residual(x, tau)**2)
        terminal_loss = torch.mean((model(terminal_x, terminal_tau) - payoff)**2)
        loss = residual_loss + training["terminal_loss_weight"] * terminal_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % training["log_every"] == 0:
            history.append({
                "phase": "adam", "step": epoch, "loss": float(loss.detach()),
                "pde_loss": float(residual_loss.detach()),
                "terminal_loss": float(terminal_loss.detach()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": perf_counter() - start,
            })

    fixed_x, fixed_tau = sample_collocation(
        problem, training["lbfgs_points"], training["seed"] + 100000,
        training["near_strike_fraction"], training["near_strike_half_width"],
        training["minimum_tau"], dtype,
    )
    terminal_x, terminal_tau, payoff = sample_terminal(
        problem, training["lbfgs_terminal_points"], training["seed"] + 150000,
        training["near_strike_fraction"], training["near_strike_half_width"], dtype,
    )
    optimizer = torch.optim.LBFGS(
        model.parameters(), max_iter=training["lbfgs_max_iterations"],
        history_size=training["lbfgs_history_size"], line_search_fn="strong_wolfe",
        tolerance_grad=1e-10, tolerance_change=1e-12,
    )
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        residual_loss = torch.mean(model.pde_residual(fixed_x, fixed_tau)**2)
        terminal_loss = torch.mean((model(terminal_x, terminal_tau) - payoff)**2)
        loss = residual_loss + training["terminal_loss_weight"] * terminal_loss
        loss.backward()
        closure_calls += 1
        if closure_calls == 1 or closure_calls % training["log_every"] == 0:
            history.append({
                "phase": "lbfgs", "step": closure_calls, "loss": float(loss.detach()),
                "pde_loss": float(residual_loss.detach()),
                "terminal_loss": float(terminal_loss.detach()),
                "learning_rate": "", "elapsed_seconds": perf_counter() - start,
            })
        return loss

    optimizer.step(closure)
    final_residual_loss = float(torch.mean(model.pde_residual(fixed_x, fixed_tau)**2).detach())
    final_terminal_loss = float(torch.mean((model(terminal_x, terminal_tau) - payoff)**2).detach())
    final_loss = final_residual_loss + training["terminal_loss_weight"] * final_terminal_loss
    wall_seconds = perf_counter() - start
    stop.set()
    memory_thread.join()
    peak[0] = max(peak[0], process.memory_info().rss)

    checkpoint_path = checkpoint_dir / "fixed_black_scholes_pinn.pt"
    torch.save({
        "state_dict": model.state_dict(), "problem": asdict(problem),
        "network": {"hidden_layers": network["hidden_layers"],
                    "hidden_width": network["hidden_width"]},
    }, checkpoint_path)
    with (table_dir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "model": "fixed_black_scholes_pinn",
        "training_run_id": f"fixed_seed_{training['seed']}",
        "seed": training["seed"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": training["adam_epochs"],
        "optimizer_steps": training["adam_epochs"] + closure_calls,
        "adam_steps": training["adam_epochs"],
        "lbfgs_closure_calls": closure_calls,
        "wall_seconds": wall_seconds,
        "baseline_rss_mb": baseline / 1024**2,
        "peak_rss_mb": peak[0] / 1024**2,
        "incremental_peak_mb": (peak[0] - baseline) / 1024**2,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "final_training_loss": final_loss,
        "final_pde_loss": final_residual_loss,
        "final_terminal_loss": final_terminal_loss,
        "final_validation_loss": None,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    (table_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(len(history)), [row["loss"] for row in history])
    ax.set_xlabel("Logged optimization step")
    ax.set_ylabel("PDE residual MSE")
    ax.set_title("Fixed-parameter PINN training")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
