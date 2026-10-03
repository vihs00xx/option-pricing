"""Train one ladder level for one seed and score it on the validation set.

Each level adds one technique to the previous one (see the settings file).
Checkpoints and validation metrics go to results/neural/two_asset_v1/ladder.
The test sets are never loaded here.
"""

import argparse
import json
from pathlib import Path

import torch

from src.neural.domain import SAMPLED, box_from_dict
from src.neural.models import BasketPricer, load_ensemble, reduced_features, save_pricer
from src.neural.training import (LossWeights, Schedule, evaluate_pricer, selection_score,
                                 subset, train)

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT/"experiments/config/neural/basket_neural_settings.json"
LEVELS = ROOT/"experiments/config/neural/basket_neural_levels.json"
DATA = ROOT/"artifacts/neural"
LADDER = ROOT/"results/neural/two_asset_v1/ladder"


def load_level(name):
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    levels = {level["name"]: level for level in json.loads(LEVELS.read_text(encoding="utf-8"))["levels"]}
    if name not in levels:
        raise KeyError(f"Unknown level {name}; choose from {sorted(levels)}")
    return settings, levels[name]


def build_model(level, settings, train_data):
    spec = level["model"]
    if spec["features"] == "raw":
        bounds = box_from_dict(settings["train_box"]).bounds()
        low = torch.tensor([bounds[n][0] for n in SAMPLED], dtype=torch.float64)
        high = torch.tensor([bounds[n][1] for n in SAMPLED], dtype=torch.float64)
        return BasketPricer(**spec, low=low, high=high)
    features = reduced_features(*train_data["r"].unbind(-1))
    return BasketPricer(**spec, mean=features.mean(0), std=features.std(0))


def validation_metrics(model, validation):
    _, metrics, seconds = evaluate_pricer(model, validation["x"].numpy(), validation["put"].numpy(),
                                          {k: v.numpy() for k, v in validation["reference"].items()})
    return dict(score=selection_score(metrics), metrics=metrics, evaluation_seconds=seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    settings, level = load_level(args.level)
    LADDER.mkdir(parents=True, exist_ok=True)
    validation = torch.load(DATA/"validation.pt", weights_only=True)

    if "ensemble_of" in level:
        paths = [LADDER/f"{level['ensemble_of']}_seed{s}.pt" for s in level["seeds"]]
        model = load_ensemble(paths)
        record = dict(level=level, members=[p.name for p in paths], training_seconds=sum(
            json.loads((LADDER/f"{level['ensemble_of']}_seed{s}.json").read_text())["training_seconds"]
            for s in level["seeds"]), **validation_metrics(model, validation))
        (LADDER/f"{args.level}.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(f"{args.level}: validation score {record['score']:.5f}")
        return

    train_data = torch.load(DATA/"train.pt", weights_only=True)
    train_data = subset(train_data, slice(0, level["train_count"]))
    raw = level["model"]["features"] == "raw"
    keys = ("x", "c") if raw else ("r", "c", "dc", "d2c")
    data = {k: train_data[k] for k in keys}
    validation_loss_data = {k: validation[k] for k in keys}
    weights = LossWeights(**level["weights"])
    schedule = Schedule(**level["schedule"])
    torch.manual_seed(args.seed)
    model = build_model(level, settings, train_data).to(torch.float64)
    if "init_from" in level:
        model.load_state_dict(torch.load(LADDER/f"{level['init_from']}_seed{args.seed}.pt",
                                         weights_only=True)["state_dict"])
    resume = DATA/f"resume_{args.level}_seed{args.seed}.pt"
    history, seconds = train(model, data, validation_loss_data, weights, schedule, seed=args.seed,
                             resume_path=resume)
    path = LADDER/f"{args.level}_seed{args.seed}.pt"
    save_pricer(model, path, level=args.level, seed=args.seed)
    record = dict(level=level, seed=args.seed, training_seconds=seconds, history=history,
                  **validation_metrics(model, validation))
    (LADDER/f"{args.level}_seed{args.seed}.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    resume.unlink(missing_ok=True)
    print(f"{args.level} seed {args.seed}: validation score {record['score']:.5f} in {seconds:.0f}s")


if __name__ == "__main__":
    main()
