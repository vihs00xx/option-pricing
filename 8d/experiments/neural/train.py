"""Train one PINN or DeepONet ladder level and score it on the validation contracts.

Usage: python -m experiments.neural.train P2_hard_payoff [--seed 0] [--threads 1]

Levels are defined in ``experiments/config/neural/pinn_ladder.json`` (P...) and
``deeponet_ladder.json`` (D...). An interrupted run resumes from its last check, provided the
settings and code are unchanged; otherwise it stops and asks for the resume file to be deleted.

For training times that can be compared, run one level at a time on an otherwise idle
machine with a fixed --threads. The summary records other training runs and the CPU busy
share at the start and end (``machine_load``).
"""

import argparse
import json
import platform
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import (CACHE, LADDER, SETTINGS_FILES, STOCKS, checkpoint_path, domain,
                                       evaluation_set, keep_full_speed, level_settings, load_settings,
                                       machine_load, training_data)
from experiments.records import ROOT, digest, relative
from src.neural.networks import build_model, load_model, save_model
from src.neural.training import Trainer, evaluate, raw_box, reduced_box, score


def new_model(settings, box):
    model = dict(settings["model"])
    boxes = (reduced_box(STOCKS, box, settings["computational_domain"]) if model["layout"] == "reduced"
             else raw_box(STOCKS, box))
    return build_model(settings["family"], d=STOCKS.d, boxes=boxes, **model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    keep_full_speed()
    torch.set_num_threads(args.threads)
    study = load_settings()
    settings = level_settings(study, args.level)
    # Pin the code and settings this run starts from.
    sources = sorted((ROOT/"src/neural").glob("*.py"))+[ROOT/"experiments/neural/train.py",
                                                         ROOT/"experiments/neural/common.py", *SETTINGS_FILES]
    source_sha256 = {relative(path): digest(path) for path in sources}
    seed = study["seeds"]["training"] if args.seed is None else args.seed
    name = f"{args.level}_seed{seed}"
    LADDER.mkdir(parents=True, exist_ok=True)
    if checkpoint_path(args.level, seed).exists() and not args.overwrite:
        raise FileExistsError(f"{name} exists; pass --overwrite to retrain")
    box, strike = domain()
    torch.manual_seed(seed)
    if settings.get("warm_start"):
        model = load_model(checkpoint_path(settings["warm_start"], seed)).train()
    else:
        model = new_model(settings, box)
    data = None
    if {"price", "derivatives"} & {k for k, v in settings["loss"].items() if v > 0}:
        data = training_data(settings["data_contracts"], settings.get("data_seed", "data"))
    resume = CACHE/"resume"/f"{name}.pt"
    resume.parent.mkdir(parents=True, exist_ok=True)
    # A resume file continues only the run it was written by: same settings, same code.
    pin = dict(settings=settings, source_sha256=source_sha256)
    if settings.get("warm_start"):
        pin["warm_start_sha256"] = digest(checkpoint_path(settings["warm_start"], seed))
    pin = json.loads(json.dumps(pin))
    pin_file = resume.with_suffix(".json")
    if resume.exists():
        if not pin_file.exists() or json.loads(pin_file.read_text(encoding="utf-8")) != pin:
            raise RuntimeError(f"{resume} was written under different settings or code; "
                               "delete it (and its .json) to start this level afresh")
    else:
        pin_file.write_text(json.dumps(pin, indent=2), encoding="utf-8")
    log_file = (LADDER/f"{name}.log").open("a" if resume.exists() else "w", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message+"\n")
        log_file.flush()

    load_at_start = machine_load()
    log(f"{name}: {settings['technique']}")
    if load_at_start and load_at_start["other_training_runs"]:
        log(f"note: {load_at_start['other_training_runs']} other training run(s) active; "
            "this run's seconds are not an idle-machine training time")
    log(f"parameters: {sum(p.numel() for p in model.parameters())}")
    trainer = Trainer(model, settings, box, strike, evaluation_set("monitor"), data, log)
    record = trainer.run(seed, resume)
    model.eval()
    scores = {}
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        start = perf_counter()
        values = evaluate(model, *panel["inputs"])
        seconds = perf_counter()-start
        value, ratios = score(values, panel["reference"])
        rmse = {q: float(np.sqrt(np.mean((values[q]-panel["reference"][q])**2))) for q in values}
        scores[kind] = dict(score=value, ratios=ratios, rmse=rmse, seconds=seconds)
        log(f"{kind}: score {value:.4g} price RMSE {rmse['price']:.3g}")
    save_model(model, checkpoint_path(args.level, seed), dict(level=args.level, seed=seed))
    summary = dict(level=args.level, family=settings["family"], seed=seed, technique=settings["technique"],
                   settings=settings, source_sha256=source_sha256,
                   parameters=sum(p.numel() for p in model.parameters()), scores=scores,
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    numpy=np.__version__, threads=args.threads,
                                    processor=platform.processor()),
                   machine_load=dict(start=load_at_start, end=machine_load()),
                   **record)
    (LADDER/f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_file.close()
    resume.unlink(missing_ok=True)
    pin_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
