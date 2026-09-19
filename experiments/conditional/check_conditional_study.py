import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(name):
    with (ROOT / "results" / "conditional" / "tables" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    settings_path = ROOT / "experiments" / "config" / "conditional_study_settings.json"
    frozen_path = ROOT / "results" / "conditional" / "tables" / "mc_conditional_frozen_configuration.json"
    settings = json.loads(settings_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    assert hashlib.sha256(settings_path.read_bytes()).hexdigest() == frozen["settings_sha256"]
    assert frozen["eligible"]
    final_names = [case["name"] for case in settings["final_cases"]]
    assert len(final_names) == len(set(final_names)) == 12
    assert not set(final_names) & set(settings["convergence"]["case_names"])

    final_rows = read_csv("mc_conditional_final_runs.csv")
    final_counts = Counter((row["case"], row["quantity"]) for row in final_rows)
    expected_repetitions = settings["final_validation"]["repetitions"]
    assert len(final_rows) == 12 * 6 * expected_repetitions
    assert set(final_counts.values()) == {expected_repetitions}
    assert all(int(row["production_points"]) + int(row["pilot_points"]) == int(row["total_points"])
               for row in final_rows)

    convergence_rows = read_csv("mc_conditional_convergence_runs.csv")
    configurations = 1 if frozen["selected"]["pilot_ratio_to_production"] == 0 else 2
    expected = len(settings["convergence"]["case_names"]) \
        * len(settings["convergence"]["production_points"]) \
        * settings["convergence"]["repetitions"] * configurations * 6
    assert len(convergence_rows) == expected
    status = json.loads((ROOT / "results" / "conditional" / "tables" / "mc_conditional_final_validation.json").read_text())
    assert status["passed_pre_registered_screen"]
    assert status["maximum_zero_standard_error_rate"] == 0
    print(json.dumps({"settings_hash_matches": True, "final_rows": len(final_rows),
                      "convergence_rows": len(convergence_rows), "validation": status}, indent=2))


if __name__ == "__main__":
    main()
