import csv
import hashlib
import json
from pathlib import Path

from src.classical_benchmark import METHODS
from src.common_benchmark import QUANTITIES


ROOT = Path(__file__).resolve().parents[2]


def load_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    table_dir = ROOT / "results" / "common_benchmark" / "tables"
    settings_path = ROOT / "experiments" / "config" / "common_benchmark_settings.json"
    settings = json.loads(settings_path.read_text())
    benchmark_metadata = json.loads((table_dir / "benchmark_metadata.json").read_text())
    classical_metadata = json.loads((table_dir / "classical_metadata.json").read_text())
    classical_validation = json.loads((table_dir / "classical_validation.json").read_text())
    quantlib_validation = json.loads((table_dir / "quantlib_validation.json").read_text())
    cases_path = table_dir / "benchmark_cases.csv"
    cases = load_csv(cases_path)
    outputs = load_csv(table_dir / "classical_outputs.csv")
    metrics = load_csv(table_dir / "classical_metrics.csv")
    timing = load_csv(table_dir / "classical_timing.csv")
    memory = load_csv(table_dir / "classical_memory.csv")
    settings_hash = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    cases_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    assert benchmark_metadata["created_before_neural_training"]
    assert benchmark_metadata["settings_sha256"] == settings_hash
    assert classical_metadata["settings_sha256"] == settings_hash
    assert benchmark_metadata["cases_sha256"] == cases_hash
    assert classical_metadata["cases_sha256"] == cases_hash
    assert len(cases) == benchmark_metadata["case_count"] == 68
    assert len(outputs) == len(cases) * len(METHODS) * len(QUANTITIES)
    assert len(metrics) == len(METHODS) * len(QUANTITIES) * 4
    assert {row["method"] for row in timing} == set(METHODS)
    assert {row["method"] for row in memory} == set(METHODS)
    assert classical_validation["passed"]
    assert quantlib_validation["passed"]
    assert len(quantlib_validation["failures"]) == 0
    print(json.dumps({
        "passed": True,
        "cases": len(cases),
        "classical_output_rows": len(outputs),
        "quantlib_version": quantlib_validation["quantlib_version"],
        "settings_sha256": settings_hash,
        "cases_sha256": cases_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
