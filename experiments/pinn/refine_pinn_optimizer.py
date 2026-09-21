"""Compare L-BFGS and NysNewton-CG refinement from one shared checkpoint.

Usage:
    python -m experiments.pinn.refine_pinn_optimizer --optimizer lbfgs
    python -m experiments.pinn.refine_pinn_optimizer --optimizer nys_newton_cg

Both refinements start from the same Adam checkpoint, optimise the same
fixed full-batch physics-only objective, and stop at the same wall-clock
budget, so the comparison isolates the optimiser. The starting
checkpoint is never modified; refined models are written to
``checkpoints/pinn/study_refined_<optimizer>.pt``.
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from src.neural.fixed_pinn import (
    FixedBlackScholesPINN,
    FixedBlackScholesProblem,
    load_fixed_pinn,
    sample_collocation,
    sample_terminal,
)
from src.neural.nys_newton_cg import NysNewtonCG
from src.neural.memory import MemoryTracker


ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "experiments" / "config" / "pinn_study_settings.json"


class WallClockExceeded(Exception):
    pass


def build_objective(model, settings):
    refinement = settings["refinement"]
    problem = model.problem
    dtype = next(model.parameters()).dtype
    x, tau = sample_collocation(
        problem, refinement["pde_points"], settings["seed"] + 100000,
        settings["matched_budget"]["near_strike_fraction"],
        settings["matched_budget"]["near_strike_half_width"],
        settings["matched_budget"]["minimum_tau"], dtype,
    )
    terminal_x, terminal_tau, payoff = sample_terminal(
        problem, refinement["terminal_points"], settings["seed"] + 150000,
        settings["matched_budget"]["near_strike_fraction"],
        settings["matched_budget"]["near_strike_half_width"], dtype,
    )
    weight = settings["matched_budget"]["terminal_loss_weight"]

    def loss_fn():
        residual = model.pde_residual(x, tau)
        terminal = model(terminal_x, terminal_tau) - payoff
        return torch.mean(residual**2) + weight * torch.mean(terminal**2)

    params = [p for p in model.parameters() if p.requires_grad]
    return loss_fn, params


def refine_lbfgs(model, loss_fn, params, settings, log):
    config = settings["refinement"]["lbfgs"]
    cap = settings["refinement"]["wall_clock_seconds"]
    optimizer = torch.optim.LBFGS(
        params, max_iter=config["max_iter"], history_size=config["history_size"],
        line_search_fn="strong_wolfe",
        tolerance_grad=config["tolerance_grad"],
        tolerance_change=config["tolerance_change"],
    )
    start = perf_counter()
    calls = 0

    def closure():
        nonlocal calls
        if perf_counter() - start >= cap:
            raise WallClockExceeded
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        calls += 1
        if calls == 1 or calls % 50 == 0:
            log(f"[lbfgs] closure {calls} loss={float(loss):.6e} "
                f"elapsed={perf_counter() - start:.1f}s")
        return loss

    try:
        optimizer.step(closure)
    except WallClockExceeded:
        pass
    return {
        "optimizer": "lbfgs",
        "closure_calls": calls,
        "wall_seconds": perf_counter() - start,
        "final_loss": float(loss_fn()),
    }


def refine_nysncg(model, loss_fn, params, settings, log):
    config = settings["refinement"]["nys_newton_cg"]
    cap = settings["refinement"]["wall_clock_seconds"]
    optimizer = NysNewtonCG(
        params, loss_fn,
        sketch_rank=config["sketch_rank"], cg_max_steps=config["cg_max_steps"],
        cg_tolerance=config["cg_tolerance"], damping=config["damping"],
        seed=config["seed"],
    )
    start = perf_counter()
    steps = []
    while perf_counter() - start < cap and len(steps) < config["max_steps"]:
        diagnostics = optimizer.step()
        diagnostics["elapsed_seconds"] = perf_counter() - start
        steps.append(diagnostics)
        if diagnostics.get("accepted"):
            log(f"[nysncg] step {len(steps)} "
                f"loss {diagnostics['loss_before']:.6e} -> "
                f"{diagnostics['loss_after']:.6e} "
                f"alpha={diagnostics['step_size']:.3g} "
                f"elapsed={diagnostics['elapsed_seconds']:.1f}s")
        else:
            log(f"[nysncg] step {len(steps)} rejected "
                f"({diagnostics.get('reason')})")
    accepted = [s for s in steps if s.get("accepted")]
    return {
        "optimizer": "nys_newton_cg",
        "steps_attempted": len(steps),
        "steps_accepted": len(accepted),
        "wall_seconds": perf_counter() - start,
        "final_loss": float(loss_fn()),
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", required=True,
                        choices=("lbfgs", "nys_newton_cg"))
    parser.add_argument("--start",
                        default="checkpoints/pinn/study_baseline_matched_adam.pt",
                        help="Starting Adam checkpoint (never modified)")
    arguments = parser.parse_args()

    settings = json.loads(SETTINGS_PATH.read_text())
    network = settings["network"]
    dtype = torch.float64 if network["dtype"] == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    model = load_fixed_pinn(arguments.start)
    loss_fn, params = build_objective(model, settings)
    table_dir = ROOT / "results" / "pinn" / "study_refinement" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    initial_loss = float(loss_fn())
    memory = MemoryTracker(interval=0.1).start()
    if arguments.optimizer == "lbfgs":
        summary = refine_lbfgs(model, loss_fn, params, settings, print)
    else:
        summary = refine_nysncg(model, loss_fn, params, settings, print)
    memory.stop()
    summary["initial_loss"] = initial_loss
    summary["baseline_rss_mb"] = memory.baseline_mb
    summary["peak_rss_mb"] = memory.peak_mb
    summary["incremental_peak_mb"] = memory.incremental_peak_mb
    summary["start_checkpoint"] = arguments.start
    summary["model"] = "fixed_black_scholes_pinn"
    summary["device"] = "cpu"
    summary["torch_version"] = torch.__version__
    summary["objective_points"] = {
        "pde": settings["refinement"]["pde_points"],
        "terminal": settings["refinement"]["terminal_points"],
    }
    checkpoint_path = ROOT / "checkpoints" / "pinn" / (
        f"study_refined_{arguments.optimizer}.pt"
    )
    torch.save({
        "state_dict": model.state_dict(),
        "problem": asdict(model.problem),
        "network": {"hidden_layers": network["hidden_layers"],
                    "hidden_width": network["hidden_width"]},
        "variant": f"refined_{arguments.optimizer}",
        "start_checkpoint": arguments.start,
    }, checkpoint_path)
    summary["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
    (table_dir / f"refinement_{arguments.optimizer}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    steps = summary.pop("steps", [])
    if steps:
        with (table_dir / f"refinement_{arguments.optimizer}_steps.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(steps[0]))
            writer.writeheader()
            writer.writerows(steps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
