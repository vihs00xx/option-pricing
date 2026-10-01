import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from src.neural.deeponet import BlackScholesDeepONet


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "experiments/config/neural/deeponet_training.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_model(model, train, validation, settings, loss_function=None):
    generator = torch.Generator().manual_seed(settings["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    history = []
    selected = None
    best = float("inf")
    best_epoch = 0
    steps = 0
    started = perf_counter()
    peak_rss = 0
    last_progress = 0

    def validate(epoch):
        nonlocal best, selected, best_epoch, peak_rss, last_progress
        model.eval()
        with torch.no_grad():
            prediction = model(validation[0], validation[1])
            error = prediction - validation[2]
            score = float(((error / validation[0][:, 0]) ** 2).mean())
            rmse = float(error.square().mean().sqrt())
        if not np.isfinite(score):
            raise FloatingPointError("Non-finite validation loss")
        if score < best:
            best, best_epoch = score, epoch
            selected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        history.append(dict(epoch=epoch, steps=steps, validation_mse=score,
                            validation_price_rmse=rmse,
                            wall_seconds=perf_counter() - started))
        elapsed = perf_counter() - started
        interval = settings.get('progress_every_seconds', 0)
        if interval and elapsed - last_progress >= interval:
            print(json.dumps(dict(epoch=epoch, steps=steps, seconds=round(elapsed, 1),
                                  current_price_rmse=rmse, best_normalized_price_mse=best)), flush=True)
            last_progress = elapsed

    validate(0)
    stop_reason = "epochs_completed"
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train[0]), generator=generator)
        for indices in order.split(settings["batch_size"]):
            if perf_counter() - started >= settings["wall_seconds"]:
                stop_reason = "wall_budget"
                break
            optimizer.zero_grad(set_to_none=True)
            if loss_function is None:
                prediction = model(train[0][indices], train[1][indices])
                loss = ((prediction - train[2][indices]) / train[0][indices, 0]).square().mean()
            else:
                loss = loss_function(model, train[0][indices], train[1][indices], train[2][indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            steps += 1
        if epoch % settings["validation_every"] == 0 or stop_reason == "wall_budget":
            validate(epoch)
        if stop_reason == "wall_budget":
            break
    if history[-1]["steps"] != steps:
        validate(epoch)
    last = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(selected)
    return history, last, {
        "selected_epoch": best_epoch, "selected_validation_mse": best,
        "optimizer_steps": steps, "stop_reason": stop_reason,
        "training_wall_seconds": perf_counter() - started,
        "peak_sampled_rss_mb": peak_rss / 1024 ** 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    dataset_path = ROOT / settings["dataset"]
    manifest = json.loads((ROOT / settings["dataset_manifest"]).read_text())
    dataset_hash = digest(dataset_path)
    if dataset_hash != manifest["dataset_sha256"]:
        raise ValueError("Dataset does not match the frozen manifest")
    run_name = args.run_name or datetime.now(timezone.utc).strftime("pilot_%Y%m%dT%H%M%S%fZ")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a single directory name")
    output = ROOT / settings["output_root"] / run_name
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings["threads"])
    torch.manual_seed(settings["seed"])
    with np.load(dataset_path, allow_pickle=False) as data:
        splits = []
        for split in ("train", "validation"):
            splits.append(tuple(torch.tensor(data[f"{split}_{key}"], dtype=torch.float64)
                                for key in ("branch", "trunk")))
            splits[-1] += (torch.tensor(data[f"{split}_targets"][:, 0], dtype=torch.float64),)
    model = BlackScholesDeepONet(**settings["architecture"])
    history, last, summary = train_model(model, *splits, settings)
    source_paths = [Path(__file__), ROOT / "src/neural/deeponet.py"]
    metadata = {
        "settings": settings, "dataset_sha256": dataset_hash,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        "seed": settings["seed"], "parameter_count": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": "cpu", "dtype": "float64", "final_test_accessed": False,
        "label_columns_used": ["price"], **summary,
    }
    for kind, state in (("selected", model.state_dict()), ("last", last)):
        torch.save({"state_dict": state, "architecture": model.architecture,
                    "metadata": metadata}, output / f"{kind}.pt")
    metadata["checkpoint_sha256"] = {kind: digest(output / f"{kind}.pt")
                                      for kind in ("selected", "last")}
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    predicted = model.price_and_greeks(splits[1][0], splits[1][1])
    with np.load(dataset_path, allow_pickle=False) as data:
        reference = data["validation_targets"]
    metrics = {}
    for index, name in enumerate(("price", "delta", "gamma", "vega", "theta", "rho")):
        error = predicted[name].numpy() - reference[:, index]
        metrics[name] = {"rmse": float(np.sqrt(np.mean(error ** 2))),
                         "max_absolute_error": float(np.max(np.abs(error)))}
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"run": str(output), **summary, "validation": metrics}, indent=2))


if __name__ == "__main__":
    main()
