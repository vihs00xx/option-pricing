"""Train one PINN or DeepONet ladder level and score it on the validation panels.

Usage (from heston/): python -m experiments.neural.train P2_hard_payoff [--seed 0] [--threads 4]

Levels are defined in experiments/config/neural/{pinn,deeponet}_ladder.json. An interrupted
run resumes its Adam stage from the last check. Every level stops at its wall-clock cap
(study.json); the log says so if it does.
"""

import argparse
import json
import platform
from time import perf_counter, process_time

import numpy as np
import torch

from experiments.neural.common import (CACHE, LADDER, SETTINGS_FILES, checkpoint_path, evaluation_set,
                                       keep_full_speed, level_settings, load_settings, ranges, training_data)
from experiments.records import ROOT, digest
from src.neural.networks import build_model, load_model, save_model
from src.neural.training import Trainer, evaluate, raw_box, reduced_box, score


def new_model(settings):
    model = dict(settings["model"])
    boxes = reduced_box(ranges(), settings["computational_domain"]) if model["layout"] == "reduced" else raw_box(ranges())
    return build_model(settings["family"], boxes=boxes, **model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="pilot: override Adam steps (results not saved)")
    args = parser.parse_args()
    keep_full_speed()
    torch.set_num_threads(args.threads)
    study = load_settings()
    settings = level_settings(study, args.level)
    pilot = args.steps is not None
    if pilot:
        settings["optimizer"].update(adam_steps=args.steps, check_every=max(1, args.steps // 2),
                                     lbfgs_iterations=0, nncg_steps=0)
    sources = sorted((ROOT / "src/neural").glob("*.py")) + [ROOT / "experiments/neural/train.py",
                                                            ROOT / "experiments/neural/common.py", *SETTINGS_FILES]
    source_sha256 = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sources}
    seed = study["seeds"]["training"] if args.seed is None else args.seed
    name = f"{args.level}_seed{seed}" + ("_pilot" if pilot else "")
    LADDER.mkdir(parents=True, exist_ok=True)
    if checkpoint_path(args.level, seed).exists() and not args.overwrite and not pilot:
        raise FileExistsError(f"{name} exists; pass --overwrite to retrain")
    torch.manual_seed(seed)
    if settings.get("warm_start"):
        model = load_model(checkpoint_path(settings["warm_start"], seed)).train()
    else:
        model = new_model(settings)
    data = None
    label_seconds = 0.0
    if {"price", "derivatives"} & {k for k, v in settings["loss"].items() if float(v) > 0}:
        data = training_data(settings["data_contracts"], settings.get("data_seed", "data"),
                             settings.get("data_short_share", 0.0))
        label_seconds = float(data.get("label_seconds", 0.0))
    resume = CACHE / "resume" / f"{name}.pt"
    resume.parent.mkdir(parents=True, exist_ok=True)
    log_path = LADDER / f"{name}.log"
    log_file = log_path.open("a" if resume.exists() else "w", encoding="utf-8", newline="\n")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(f"{name}: {settings['technique']}")
    log(f"parameters: {sum(p.numel() for p in model.parameters())}; threads {args.threads}")
    cap = study["wall_clock_cap_seconds"][settings["cap"] if args.seed in (None, 0) else "ensemble_member"]
    monitor = evaluation_set("monitor")
    trainer = Trainer(model, settings, ranges(), monitor, data, log)
    start, cpu_start = perf_counter(), process_time()
    record = trainer.run(seed, None if pilot else resume, cap)
    cpu_seconds = process_time() - cpu_start
    if pilot:
        steps = settings["optimizer"]["adam_steps"]
        log(f"pilot: {record['adam_seconds'] / steps:.4f} s per Adam step, {cpu_seconds / steps:.4f} CPU s per step "
            f"({steps} steps, {args.threads} threads)")
        log_file.close()
        return
    model.eval()
    scores = {}
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        t = perf_counter()
        values = evaluate(model, *panel["inputs"])
        seconds = perf_counter() - t
        value, ratios = score(values, panel["reference"])
        rmse = {q: float(np.sqrt(np.mean((values[q] - panel["reference"][q]) ** 2))) for q in values}
        scores[kind] = dict(score=value, ratios=ratios, rmse=rmse, seconds=seconds)
        log(f"{kind}: score {value:.4g} price RMSE {rmse['price']:.3g}")
    save_model(model, checkpoint_path(args.level, seed), dict(level=args.level, seed=seed))
    summary = dict(level=args.level, family=settings["family"], seed=seed, technique=settings["technique"],
                   settings=settings, source_sha256=source_sha256,
                   parameters=sum(p.numel() for p in model.parameters()), scores=scores,
                   label_seconds=label_seconds, wall_seconds=perf_counter() - start,
                   process_cpu_seconds=cpu_seconds,
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    numpy=np.__version__, threads=args.threads, processor=platform.processor()),
                   **record)
    (LADDER / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    log_file.close()
    resume.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
