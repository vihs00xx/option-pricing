"""Build an ensemble level: average independently seeded trainings of one ladder level.

Usage: python -m experiments.neural.ensemble D9_ensemble

The level's settings name the member level (``members_of``) and the seeds. Each member
is trained with ``experiments.neural.train <level> --seed <seed>``. The ensemble is
scored on the validation panels like any other level and saved as one checkpoint.
"""

import argparse
import json
import platform
from time import perf_counter

import numpy as np
import torch

from experiments.neural.common import LADDER, checkpoint_path, evaluation_set, level_settings, load_settings
from experiments.records import digest, relative
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.training import evaluate, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    study = load_settings()
    settings = level_settings(study, args.level)
    seed = study["seeds"]["training"]
    member_level = settings["members_of"]
    members, records, member_files = [], [], {}
    for member_seed in settings["seeds"]:
        path = checkpoint_path(member_level, member_seed)
        members.append(load_model(path))
        records.append(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))
        member_files[relative(path)] = digest(path)
    model = Ensemble(members)
    scores = {}
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        start = perf_counter()
        values = evaluate(model, *panel["inputs"])
        seconds = perf_counter()-start
        value, ratios = score(values, panel["reference"])
        rmse = {q: float(np.sqrt(np.mean((values[q]-panel["reference"][q])**2))) for q in values}
        scores[kind] = dict(score=value, ratios=ratios, rmse=rmse, seconds=seconds)
        print(f"{kind}: score {value:.4g} price RMSE {rmse['price']:.3g}")
    save_model(model, checkpoint_path(args.level, seed), dict(level=args.level, seed=seed))
    summary = dict(level=args.level, family=settings["family"], seed=seed, technique=settings["technique"],
                   settings=settings, members=member_files,
                   parameters=sum(p.numel() for p in model.parameters()), scores=scores,
                   member_scores=[r["scores"]["validation"]["score"] for r in records],
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    numpy=np.__version__, threads=args.threads, processor=platform.processor()),
                   total_seconds=sum(r["total_seconds"] for r in records), history=[])
    (LADDER/f"{args.level}_seed{seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (LADDER/f"{args.level}_seed{seed}.log").write_text(
        "".join(f"{kind}: score {s['score']:.4g}\n" for kind, s in scores.items()), encoding="utf-8")


if __name__ == "__main__":
    main()
