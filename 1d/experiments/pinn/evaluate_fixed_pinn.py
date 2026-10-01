import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.neural.fixed_pinn import load_fixed_pinn, sample_collocation

from experiments.pinn import frozen_checkpoint


ROOT = Path(__file__).resolve().parents[2]
QUANTITIES = ("price", "delta", "gamma", "theta")


def errors(estimate, reference):
    difference = estimate - reference
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "bias": float(np.mean(difference)),
    }


def predict(model, y, tau, batch_size):
    output = {name: [] for name in QUANTITIES}
    for start in range(0, len(y), batch_size):
        y_tensor = torch.tensor(y[start:start + batch_size, None], requires_grad=True)
        tau_tensor = torch.tensor(tau[start:start + batch_size, None], requires_grad=True)
        values = model.price_and_greeks(y_tensor, tau_tensor)
        for name in QUANTITIES:
            output[name].append(values[name].detach().numpy()[:, 0])
    return {name: np.concatenate(parts) for name, parts in output.items()}


def main():
    settings = json.loads((ROOT / "experiments" / "config" / "neural" / "fixed_pinn_settings.json").read_text())
    evaluation = settings["evaluation"]
    problem = settings["problem"]
    table_dir = ROOT / "results" / "pinn" / "primary" / "anchored" / "tables"
    figure_dir = ROOT / "results" / "pinn" / "primary" / "anchored" / "figures"
    checkpoint = ROOT / "checkpoints" / "pinn" / "primary" / "anchored" / "fixed_black_scholes_pinn.pt"
    digest = frozen_checkpoint.primary_digest(ROOT)
    if digest["status"] != "frozen":
        raise RuntimeError(
            f"refuse to evaluate a non-baseline primary: status "
            f"'{digest['status']}' (registry pin {digest['frozen_sha256']}); "
            "the frozen anchored baseline must be intact before evaluation")
    checkpoint_sha256 = digest["actual_sha256"]
    model = load_fixed_pinn(checkpoint)
    model.eval()
    x_axis = np.linspace(problem["x_min"], problem["x_max"], evaluation["x_points"])
    tau_axis = np.linspace(0, problem["T"], evaluation["tau_points"])
    x_grid, tau_grid = np.meshgrid(x_axis, tau_axis, indexing="xy")
    x_flat, tau_flat = x_grid.ravel(), tau_grid.ravel()
    y_flat = np.log(x_flat)
    predicted = predict(model, y_flat, tau_flat, evaluation["batch_size"])
    spots = problem["K"] * x_flat
    exact_price = black_scholes_price(
        spots, problem["K"], tau_flat, problem["r"], problem["sigma"],
        problem["q"], problem["option"],
    )
    regular = tau_flat > 0
    exact_greeks = black_scholes_greeks(
        spots[regular], problem["K"], tau_flat[regular], problem["r"],
        problem["sigma"], problem["q"], problem["option"],
    )
    exact_full = {}
    for quantity in ("delta", "gamma", "theta"):
        exact_full[quantity] = np.full(len(x_flat), np.nan)
        exact_full[quantity][regular] = exact_greeks[quantity]
    core = (x_flat >= evaluation["core_x_range"][0]) \
        & (x_flat <= evaluation["core_x_range"][1]) \
        & (tau_flat >= evaluation["core_minimum_tau"])
    metrics = []
    for region, mask in (("full", np.ones_like(regular, dtype=bool)), ("core", core)):
        price_metrics = errors(predicted["price"][mask], exact_price[mask])
        metrics.append({"region": region, "quantity": "price", "points": int(np.sum(mask)), **price_metrics})
        greek_mask = mask & regular
        for quantity in ("delta", "gamma", "theta"):
            quantity_metrics = errors(predicted[quantity][greek_mask], exact_full[quantity][greek_mask])
            metrics.append({
                "region": region, "quantity": quantity,
                "points": int(np.sum(greek_mask)), **quantity_metrics,
            })
    with (table_dir / "evaluation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)

    prediction_rows = []
    for index in range(len(x_flat)):
        row = {
            "x": x_flat[index], "S": spots[index], "tau": tau_flat[index],
            "price": predicted["price"][index], "price_reference": exact_price[index],
            "price_error": predicted["price"][index] - exact_price[index],
        }
        if regular[index]:
            for quantity in ("delta", "gamma", "theta"):
                row[quantity] = predicted[quantity][index]
                row[f"{quantity}_reference"] = exact_full[quantity][index]
                row[f"{quantity}_error"] = predicted[quantity][index] - exact_full[quantity][index]
        else:
            for quantity in ("delta", "gamma", "theta"):
                row[quantity] = row[f"{quantity}_reference"] = row[f"{quantity}_error"] = ""
        prediction_rows.append(row)
    with (table_dir / "surface_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_rows[0])
        writer.writeheader()
        writer.writerows(prediction_rows)

    residual_x, residual_tau = sample_collocation(
        model.problem, evaluation["residual_points"], settings["training"]["seed"] + 200000,
        settings["training"]["near_strike_fraction"],
        settings["training"]["near_strike_half_width"],
        settings["training"]["minimum_tau"], torch.float64,
    )
    residual = model.pde_residual(residual_x, residual_tau).detach().numpy()[:, 0]
    residual_metrics = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
    }
    tolerance = evaluation["shape_tolerance"]
    shape = {
        "negative_price_count": int(np.sum(predicted["price"] < -tolerance)),
        "delta_below_zero_count": int(np.sum(predicted["delta"][regular] < -tolerance)),
        "delta_above_bound_count": int(np.sum(
            predicted["delta"][regular] > np.exp(-problem["q"] * tau_flat[regular]) + tolerance
        )),
        "negative_gamma_count": int(np.sum(predicted["gamma"][regular] < -tolerance)),
    }
    core_regular = core & regular
    core_shape = {
        "negative_price_count": int(np.sum(predicted["price"][core] < -tolerance)),
        "delta_below_zero_count": int(np.sum(predicted["delta"][core_regular] < -tolerance)),
        "delta_above_bound_count": int(np.sum(
            predicted["delta"][core_regular]
            > np.exp(-problem["q"] * tau_flat[core_regular]) + tolerance
        )),
        "negative_gamma_count": int(np.sum(predicted["gamma"][core_regular] < -tolerance)),
    }

    model_dtype = next(model.parameters()).dtype
    timing_x = torch.linspace(
        np.log(0.5), np.log(1.5), evaluation["batch_size"], dtype=model_dtype
    ).reshape(-1, 1)
    timing_tau = torch.linspace(
        0.05, 1.0, evaluation["batch_size"], dtype=model_dtype
    ).reshape(-1, 1)
    for _ in range(2):
        timing_x.requires_grad_(True)
        timing_tau.requires_grad_(True)
        model.price_and_greeks(timing_x, timing_tau)
    elapsed = []
    for _ in range(10):
        x_value = timing_x.detach().clone().requires_grad_(True)
        tau_value = timing_tau.detach().clone().requires_grad_(True)
        start = perf_counter()
        model.price_and_greeks(x_value, tau_value)
        elapsed.append(perf_counter() - start)
    median = float(np.median(elapsed))
    inference = {
        "batch_size": evaluation["batch_size"], "warmup_batches": 2, "repetitions": 10,
        "median_batch_seconds": median,
        "q25_batch_seconds": float(np.quantile(elapsed, 0.25)),
        "q75_batch_seconds": float(np.quantile(elapsed, 0.75)),
        "median_seconds_per_point": median / evaluation["batch_size"],
        "median_points_per_second": evaluation["batch_size"] / median,
    }
    core_metrics = {row["quantity"]: row for row in metrics if row["region"] == "core"}
    acceptance = settings["acceptance"]
    checks = {
        "core_price_rmse": core_metrics["price"]["rmse"] <= acceptance["core_price_rmse"],
        "core_delta_rmse": core_metrics["delta"]["rmse"] <= acceptance["core_delta_rmse"],
        "core_gamma_rmse": core_metrics["gamma"]["rmse"] <= acceptance["core_gamma_rmse"],
        "core_theta_rmse": core_metrics["theta"]["rmse"] <= acceptance["core_theta_rmse"],
        "pde_residual_rmse": residual_metrics["rmse"] <= acceptance["pde_residual_rmse"],
        "shape_checks": not any(shape.values()),
    }
    accuracy_passed = all(value for name, value in checks.items() if name != "shape_checks")
    summary = {
        "passed": all(checks.values()), "accuracy_passed": accuracy_passed,
        "shape_passed": checks["shape_checks"], "checks": checks,
        "core_metrics": core_metrics, "pde_residual": residual_metrics,
        "shape_checks": {"full": shape, "core": core_shape}, "inference": inference,
    }
    (table_dir / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Evaluation provenance lives in its own record and NEVER mutates the
    # immutable training_metadata.json. The record pins the exact checkpoint
    # SHA-256 that produced these metrics.
    import platform
    evaluation_metrics = {
        **summary,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "registry_pin_sha256": digest["frozen_sha256"],
        "python": platform.python_version(),
    }
    (table_dir / "evaluation_metrics.json").write_text(
        json.dumps(evaluation_metrics, indent=2) + "\n")

    price_error = np.abs(predicted["price"] - exact_price).reshape(tau_grid.shape)
    final = tau_flat == problem["T"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(spots[final], exact_price[final], label="Black--Scholes")
    axes[0, 0].plot(spots[final], predicted["price"][final], "--", label="PINN")
    axes[0, 0].set_title("Price at one year to maturity")
    axes[0, 0].set_xlabel("Spot")
    axes[0, 0].set_ylabel("Call price")
    axes[0, 0].legend()
    image = axes[0, 1].pcolormesh(x_axis, tau_axis, price_error, shading="auto")
    axes[0, 1].set_title("Absolute price error")
    axes[0, 1].set_xlabel("Moneyness S/K")
    axes[0, 1].set_ylabel("Time to maturity")
    fig.colorbar(image, ax=axes[0, 1])
    axes[1, 0].plot(spots[final], predicted["delta"][final], label="PINN")
    axes[1, 0].plot(spots[final], exact_full["delta"][final], "--", label="Reference")
    axes[1, 0].set_title("Delta at one year")
    axes[1, 0].set_xlabel("Spot")
    axes[1, 0].legend()
    axes[1, 1].plot(spots[final], predicted["gamma"][final], label="PINN")
    axes[1, 1].plot(spots[final], exact_full["gamma"][final], "--", label="Reference")
    axes[1, 1].set_title("Gamma at one year")
    axes[1, 1].set_xlabel("Spot")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "fixed_pinn_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise AssertionError("Fixed PINN validation failed")


if __name__ == "__main__":
    main()
