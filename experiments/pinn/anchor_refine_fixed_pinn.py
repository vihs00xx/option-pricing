import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.neural.fixed_pinn import load_fixed_pinn, sample_collocation, sample_terminal


ROOT = Path(__file__).resolve().parents[2]


def anchor_data(model, settings, seed, dtype):
    count = settings["anchor_points"]
    core_count = round(count * settings["core_anchor_fraction"])
    global_count = count - core_count
    core_engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    core_unit = core_engine.draw(core_count).to(dtype=dtype)
    low, high = settings["core_x_range"]
    core_x = low + (high - low) * core_unit[:, :1]
    core_tau = settings["minimum_tau"] + (
        model.problem.T - settings["minimum_tau"]
    ) * core_unit[:, 1:]
    global_engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed + 1)
    global_unit = global_engine.draw(global_count).to(dtype=dtype)
    log_low = np.log(model.problem.x_min)
    log_high = np.log(model.problem.x_max)
    global_y = log_low + (log_high - log_low) * global_unit[:, :1]
    global_x = torch.exp(global_y)
    global_tau = settings["minimum_tau"] + (
        model.problem.T - settings["minimum_tau"]
    ) * global_unit[:, 1:]
    x = torch.cat((core_x, global_x))
    tau = torch.cat((core_tau, global_tau))
    price = black_scholes_price(
        (model.problem.K * x.numpy()[:, 0]), model.problem.K,
        tau.numpy()[:, 0], model.problem.r, model.problem.sigma, model.problem.q,
        "call",
    )
    target = torch.tensor(price[:, None] / model.problem.K, dtype=dtype)
    theta = black_scholes_greeks(
        (model.problem.K * x.numpy()[:, 0]), model.problem.K,
        tau.numpy()[:, 0], model.problem.r, model.problem.sigma, model.problem.q,
        "call",
    )["theta"]
    theta_target = torch.tensor(theta[:, None] / model.problem.K, dtype=dtype)
    return torch.log(x), tau, target, theta_target


def save_checkpoint(model, settings, path):
    torch.save({
        "state_dict": model.state_dict(),
        "problem": vars(model.problem),
        "network": {
            "hidden_layers": settings["network"]["hidden_layers"],
            "hidden_width": settings["network"]["hidden_width"],
        },
    }, path)


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "neural" / "fixed_pinn_settings.json").read_text())
    refinement = settings["anchored_refinement"]
    checkpoint = ROOT / "checkpoints" / "pinn" / "primary" / "anchored" / "fixed_black_scholes_pinn.pt"
    physics_checkpoint = checkpoint.with_name("fixed_black_scholes_pinn_physics_only.pt")
    if not physics_checkpoint.exists():
        physics_checkpoint.write_bytes(checkpoint.read_bytes())
    model = load_fixed_pinn(checkpoint)
    dtype = next(model.parameters()).dtype
    torch.manual_seed(refinement["seed"])
    anchor_y, anchor_tau, anchor_target, theta_target = anchor_data(
        model, refinement, refinement["seed"], dtype
    )
    pde_y, pde_tau = sample_collocation(
        model.problem, refinement["pde_points"], refinement["seed"] + 2,
        near_strike_fraction=0.5, near_strike_half_width=0.5,
        minimum_tau=1e-4, dtype=dtype,
    )
    terminal_y, terminal_tau, payoff = sample_terminal(
        model.problem, refinement["terminal_points"], refinement["seed"] + 3,
        near_strike_fraction=0.5, near_strike_half_width=0.5, dtype=dtype,
    )
    history = []
    start = perf_counter()
    total_adam_steps = 0
    total_lbfgs_calls = 0
    for stage_index, stage in enumerate(refinement["stages"]):
        generator = torch.Generator().manual_seed(refinement["seed"] + 4)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=refinement["adam_learning_rate"]
        )
        for epoch in range(1, stage["adam_epochs"] + 1):
            indices = torch.randint(
                len(anchor_y), (refinement["adam_batch_size"],), generator=generator
            )
            batch_y = anchor_y[indices].clone().requires_grad_(True)
            batch_tau = anchor_tau[indices].clone().requires_grad_(True)
            batch_target = anchor_target[indices]
            batch_theta_target = theta_target[indices]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_y, batch_tau)
            price_loss = torch.mean((prediction - batch_target) ** 2)
            value_tau = torch.autograd.grad(
                prediction, batch_tau, torch.ones_like(prediction), create_graph=True
            )[0]
            theta_loss = torch.mean((-value_tau - batch_theta_target) ** 2)
            pde_loss = torch.mean(model.pde_residual(pde_y, pde_tau) ** 2)
            terminal_loss = torch.mean((model(terminal_y, terminal_tau) - payoff) ** 2)
            loss = (
                stage["pde_loss_weight"] * pde_loss
                + stage["price_loss_weight"] * price_loss
                + stage["theta_loss_weight"] * theta_loss
                + refinement["terminal_loss_weight"] * terminal_loss
            )
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % refinement["log_every"] == 0:
                history.append({
                    "stage": stage["name"], "phase": "adam", "step": epoch,
                    "loss": float(loss.detach()), "pde_loss": float(pde_loss.detach()),
                    "price_loss": float(price_loss.detach()),
                    "theta_loss": float(theta_loss.detach()),
                    "terminal_loss": float(terminal_loss.detach()),
                    "elapsed_seconds": perf_counter() - start,
                })
        total_adam_steps += stage["adam_epochs"]
        fixed_anchor_y = anchor_y.detach().clone().requires_grad_(True)
        fixed_anchor_tau = anchor_tau.detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            model.parameters(), max_iter=stage["lbfgs_max_iterations"],
            history_size=refinement["lbfgs_history_size"], line_search_fn="strong_wolfe",
            tolerance_grad=1e-12, tolerance_change=1e-14,
        )
        stage_calls = 0

        def closure():
            nonlocal stage_calls
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fixed_anchor_y, fixed_anchor_tau)
            price_loss = torch.mean((prediction - anchor_target) ** 2)
            value_tau = torch.autograd.grad(
                prediction, fixed_anchor_tau, torch.ones_like(prediction), create_graph=True
            )[0]
            theta_loss = torch.mean((-value_tau - theta_target) ** 2)
            pde_loss = torch.mean(model.pde_residual(pde_y, pde_tau) ** 2)
            terminal_loss = torch.mean((model(terminal_y, terminal_tau) - payoff) ** 2)
            loss = (
                stage["pde_loss_weight"] * pde_loss
                + stage["price_loss_weight"] * price_loss
                + stage["theta_loss_weight"] * theta_loss
                + refinement["terminal_loss_weight"] * terminal_loss
            )
            loss.backward()
            stage_calls += 1
            if stage_calls == 1 or stage_calls % refinement["log_every"] == 0:
                history.append({
                    "stage": stage["name"], "phase": "lbfgs", "step": stage_calls,
                    "loss": float(loss.detach()), "pde_loss": float(pde_loss.detach()),
                    "price_loss": float(price_loss.detach()),
                    "theta_loss": float(theta_loss.detach()),
                    "terminal_loss": float(terminal_loss.detach()),
                    "elapsed_seconds": perf_counter() - start,
                })
            return loss

        optimizer.step(closure)
        total_lbfgs_calls += stage_calls
    elapsed = perf_counter() - start
    save_checkpoint(model, settings, checkpoint)
    table_dir = ROOT / "results" / "pinn" / "primary" / "anchored" / "tables"
    with (table_dir / "anchored_refinement_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    metadata_path = table_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model"] = "fixed_black_scholes_anchored_pinn"
    metadata["anchored_refinement"] = {
        "physics_informed": True,
        "labelled_price_anchors_used": True,
        "labelled_greeks_used": True,
        "labelled_greeks": ["theta"],
        "anchor_source": "analytical Black-Scholes prices and Theta",
        "wall_seconds": elapsed,
        "adam_steps": total_adam_steps,
        "lbfgs_closure_calls": total_lbfgs_calls,
        "settings": refinement,
    }
    metadata["wall_seconds"] += elapsed
    metadata["optimizer_steps"] += total_adam_steps + total_lbfgs_calls
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["anchored_refinement"], indent=2))


if __name__ == "__main__":
    main()
