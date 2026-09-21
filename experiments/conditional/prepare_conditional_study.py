import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "config" / "classical" / "conditional_study_settings.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace frozen settings: {OUTPUT}")
    seed = 9172026
    count = 12
    points = qmc.LatinHypercube(d=5, seed=seed).random(count)
    cases = []
    for index, point in enumerate(points):
        cases.append({
            "name": f"blind_{index + 1:02d}",
            "inputs": {
                "S": float(80 + 45 * point[0]),
                "K": 100.0,
                "tau": float(np.exp(np.log(0.08) + point[1] * (np.log(2.0) - np.log(0.08)))),
                "r": float(-0.02 + 0.10 * point[2]),
                "sigma": float(np.exp(np.log(0.10) + point[3] * (np.log(0.60) - np.log(0.10)))),
                "q": float(0.04 * point[4]),
                "option": "call" if index % 2 == 0 else "put",
            },
        })
    settings = {
        "created_before_tuning": True,
        "case_generation": {
            "method": "LatinHypercube",
            "seed": seed,
            "ranges": {"S": [80, 125], "K": [100, 100], "tau": [0.08, 2.0],
                       "r": [-0.02, 0.08], "sigma": [0.10, 0.60], "q": [0.0, 0.04]},
        },
        "tuning": {
            "seed_root": 4100000,
            "production_points": 32768,
            "repetitions": 16,
            "conditional_fractions": [0.25, 0.5, 0.75],
            "scramble_counts": [8, 16, 32],
            "pilot_ratios": [0.0, 0.125, 0.25, 0.5, 1.0],
            "minimum_aggregate_coverage": 0.85,
        },
        "convergence": {
            "seed_root": 5200000,
            "production_points": [1024, 4096, 16384, 65536, 262144, 1048576],
            "repetitions": 16,
            "case_names": ["tune_atm", "tune_short", "tune_high_vol", "tune_negative_rate"],
        },
        "final_validation": {
            "seed_root": 6300000,
            "production_points": 32768,
            "repetitions": 128,
            "minimum_case_coverage": 0.85,
        },
        "final_cases": cases,
    }
    content = json.dumps(settings, indent=2) + "\n"
    OUTPUT.write_text(content)
    print(OUTPUT)
    print(hashlib.sha256(content.encode()).hexdigest())


if __name__ == "__main__":
    main()
