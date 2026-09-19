import csv
import json
import platform
from itertools import product
from pathlib import Path

import numpy as np
import scipy

from src.black_scholes import black_scholes_greeks, black_scholes_price


def main():
    project = Path(__file__).resolve().parents[2]
    settings = json.loads((project / "experiments" / "config" / "analytical_settings.json").read_text())
    base = settings["base_case"]
    grid = settings["reference_grid"]
    rows = []

    for S, tau, sigma, option in product(
        grid["S"], grid["tau"], grid["sigma"], grid["option"]
    ):
        inputs = {**base, "S": S, "tau": tau, "sigma": sigma, "option": option}
        rows.append({
            **inputs,
            "price": black_scholes_price(**inputs),
            **black_scholes_greeks(**inputs),
        })

    output = project / "results" / "analytical" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "black_scholes_reference.csv"
    with table_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "method": "Analytical Black-Scholes with continuous dividends",
        "settings": settings,
        "random_sampling_used": False,
        "row_count": len(rows),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output / "black_scholes_reference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print("Base case:", base)
    for option in ("call", "put"):
        print(f"\n{option.title()} price: {black_scholes_price(**base, option=option):.8f}")
        for name, value in black_scholes_greeks(**base, option=option).items():
            print(f"  {name:5s}: {value:.8f}")
    print(f"\nSaved {len(rows)} reference cases to {table_path}")


if __name__ == "__main__":
    main()
