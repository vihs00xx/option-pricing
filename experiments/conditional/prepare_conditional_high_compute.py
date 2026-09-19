import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "config" / "conditional_high_compute_settings.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace sealed settings: {OUTPUT}")
    case_seed = 9182026
    points = qmc.LatinHypercube(d=5, seed=case_seed).random(12)
    cases = []
    for index, point in enumerate(points):
        cases.append({
            "name": f"high_blind_{index + 1:02d}",
            "inputs": {
                "S": float(80 + 45 * point[0]),
                "K": 100.0,
                "tau": float(np.exp(np.log(0.08) + point[1] * (np.log(2.0) - np.log(0.08)))),
                "r": float(-0.02 + 0.10 * point[2]),
                "sigma": float(np.exp(np.log(0.10) + point[3] * (np.log(0.60) - np.log(0.10)))),
                "q": float(0.04 * point[4]),
                "option": "put" if index % 2 == 0 else "call",
            },
        })
    settings = {
        "created_before_evaluation": True,
        "case_generation": {
            "method": "LatinHypercube", "seed": case_seed,
            "ranges": {"S": [80, 125], "K": [100, 100], "tau": [0.08, 2.0],
                       "r": [-0.02, 0.08], "sigma": [0.10, 0.60], "q": [0.0, 0.04]},
        },
        "conditional_fraction": 0.25,
        "points_per_scramble": 4096,
        "pilot_points_per_scramble": 2048,
        "configurations": [
            {"name": "efficient_r8", "n_scrambles": 8},
            {"name": "high_compute_r64", "n_scrambles": 64},
        ],
        "evaluation": {
            "seed_root": 7400000, "repetitions": 128,
            "nominal_coverage": 0.95, "minimum_case_coverage_screen": 0.85,
        },
        "final_cases": cases,
    }
    content = json.dumps(settings, indent=2) + "\n"
    OUTPUT.write_text(content)
    print(OUTPUT)
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
