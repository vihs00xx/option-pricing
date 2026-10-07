"""Measure how much the shared CPU slowed training: pilot steps alone vs the same level under load.

The ladders ran 3 levels at a time (4 threads each) next to the classical benchmark, so their
wall-clock seconds overstate the cost of training one network on an otherwise idle machine.
This runs short pilots of P4 and D4 alone (4 threads, nothing else running) and divides
their seconds per Adam step by the same levels' seconds per step in the ladder run. The
break-even uses each family's ladder wall-clock seconds times this factor.

Usage (from heston/, with nothing else running): python -m experiments.neural.load_factor
"""

import json
import re
import subprocess
import sys

from experiments.neural.common import LADDER, RESULTS
from experiments.records import ROOT, write_json

PILOTS = {"pinn": "P4_variance_boundary", "deeponet": "D4_physics"}
STEPS = 300


def main():
    out = {}
    for family, level in PILOTS.items():
        record = json.loads((LADDER / f"{level}_seed0.json").read_text(encoding="utf-8"))
        steps = max(h["step"] for h in record["history"] if h["stage"] == "adam")
        loaded = record["adam_seconds"] / steps
        result = subprocess.run([sys.executable, "-m", "experiments.neural.train", level, "--steps", str(STEPS),
                                 "--threads", str(record["environment"]["threads"])],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        alone = float(re.search(r"pilot: ([0-9.]+) s per Adam step", result.stdout).group(1))
        out[family] = dict(level=level, loaded_seconds_per_step=loaded, alone_seconds_per_step=alone,
                           factor=alone / loaded, pilot_steps=STEPS)
        print(family, out[family], flush=True)
        (LADDER / f"{level}_seed0_pilot.log").unlink(missing_ok=True)
    write_json(RESULTS / "load_factor.json", out)


if __name__ == "__main__":
    main()
