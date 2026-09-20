import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    tables = ROOT / "results" / "finite_difference" / "tables"
    settings = json.loads((ROOT / "experiments" / "config" / "finite_difference_settings.json").read_text())
    with (tables / "fd_convergence.csv").open(newline="") as handle:
        convergence = list(csv.DictReader(handle))
    with (tables / "fd_greeks.csv").open(newline="") as handle:
        greeks = list(csv.DictReader(handle))
    with (tables / "fd_richardson_convergence.csv").open(newline="") as handle:
        richardson = list(csv.DictReader(handle))
    with (tables / "fd_richardson_greeks.csv").open(newline="") as handle:
        richardson_greeks = list(csv.DictReader(handle))
    with (tables / "fd_boundary_sensitivity.csv").open(newline="") as handle:
        boundaries = list(csv.DictReader(handle))
    summary = json.loads((tables / "fd_validation.json").read_text())
    assert len(convergence) == len(settings["cases"]) * len(settings["grid_sizes"])
    assert len(greeks) == len(settings["cases"]) * 6
    assert len(richardson) == len(settings["cases"]) * (len(settings["grid_sizes"]) - 1)
    assert len(richardson_greeks) == len(settings["cases"]) * 6
    assert len(boundaries) == len(settings["boundary_multipliers"])
    assert summary["passed_error_limits"]
    assert summary["passed_shape_checks"]
    assert summary["richardson"]["passed_shape_checks"]
    assert summary["richardson"]["median_convergence_alpha"] >= settings["richardson_min_median_alpha"]
    for quantity, error in summary["richardson"]["maximum_errors"].items():
        assert error <= settings["error_limits"][quantity]
    assert summary["boundary_sensitivity"]["price_range"] <= settings["boundary_price_range_limit"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
