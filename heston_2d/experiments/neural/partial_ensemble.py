"""Average whichever seeds of a level have finished, outside the ladder (for interim results).

Usage: python -m experiments.neural.partial_ensemble --level D8_envelope --seeds 0 1 2 [--name D9_partial]

The ladder's D9_ensemble waits for all four D8 seeds. This builds the same kind of
ensemble from the seeds named, saves it as <name>_seed0.pt in the ladder folder (so
evaluate_test scores it, as "<name>") with the member hashes in <name>_seed0.json, and
scores it on the validation panels. It changes no setting, so running levels are not
disturbed; the ladder's own D9_ensemble is unaffected and still runs later.
"""

import argparse
import json

import numpy as np
import torch

from experiments.neural.common import checkpoint_path, evaluation_set
from experiments.records import digest, relative
from src.neural.networks import Ensemble, load_model, save_model
from src.neural.training import evaluate, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="D8_envelope")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    name = args.name or f"D9_partial_{''.join(map(str, args.seeds))}"
    paths = [checkpoint_path(args.level, seed) for seed in args.seeds]
    missing = [relative(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"not finished yet: {missing}")
    model = Ensemble([load_model(p) for p in paths])
    scores = {}
    for kind in ("validation", "short_validation"):
        panel = evaluation_set(kind)
        value, ratios = score(evaluate(model, *panel["inputs"]), panel["reference"])
        scores[kind] = dict(score=value, ratios=ratios)
        print(f"{kind}: score {value:.4g}")
    out = checkpoint_path(name, 0)
    save_model(model, out, dict(level=name, seed=0))
    summary = dict(level=name, members_of=args.level, seeds=args.seeds,
                   members={relative(p): digest(p) for p in paths}, scores=scores,
                   note="Partial ensemble built outside the ladder from the seeds finished so far; the ladder's D9_ensemble uses all four seeds.")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {relative(out)}")


if __name__ == "__main__":
    main()
