import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    settings_path = ROOT / "experiments" / "config" / "conditional_high_compute_settings.json"
    table_dir = ROOT / "results" / "conditional" / "tables"
    settings = json.loads(settings_path.read_text())
    metadata = json.loads((table_dir / "mc_conditional_high_compute_metadata.json").read_text())
    assert hashlib.sha256(settings_path.read_bytes()).hexdigest() == metadata["settings_sha256"]
    with (table_dir / "mc_conditional_high_compute_runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    repetitions = settings["evaluation"]["repetitions"]
    expected = len(settings["final_cases"]) * len(settings["configurations"]) * repetitions * 6
    assert len(rows) == expected == 18432
    counts = Counter((row["configuration"], row["case"], row["quantity"]) for row in rows)
    assert set(counts.values()) == {repetitions}
    assert all(int(row["production_points"]) + int(row["pilot_points"]) == int(row["total_points"])
               for row in rows)
    totals = {row["configuration"]: int(row["total_points"]) for row in rows}
    assert totals["high_compute_r64"] == 8 * totals["efficient_r8"]
    validation = json.loads((table_dir / "mc_conditional_high_compute_validation.json").read_text())
    assert all(result["passed_pre_registered_screen"] for result in validation.values())
    assert all(result["maximum_zero_standard_error_rate"] == 0 for result in validation.values())
    print(json.dumps({"settings_hash_matches": True, "rows": len(rows),
                      "sample_totals": totals, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
