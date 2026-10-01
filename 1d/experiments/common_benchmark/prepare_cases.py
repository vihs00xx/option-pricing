import csv
import hashlib
import json
from pathlib import Path

from src.common_benchmark import QUANTITIES, build_cases, reference_values


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    settings_path = ROOT / "experiments" / "config" / "classical" / "common_benchmark_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    output = ROOT / "results" / "common_benchmark" / "tables"
    output.mkdir(parents=True, exist_ok=True)
    cases = build_cases(settings)
    references = []
    for case in cases:
        values = reference_values(case)
        references.append({"case_id": case["case_id"], **values})
    write_csv(output / "benchmark_cases.csv", cases)
    write_csv(output / "benchmark_references.csv", references)
    case_bytes = (output / "benchmark_cases.csv").read_bytes()
    reference_bytes = (output / "benchmark_references.csv").read_bytes()
    metadata = {
        "protocol_id": settings["protocol_id"],
        "protocol_version": settings["protocol_version"],
        "status": settings["status"],
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "references_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "case_count": len(cases),
        "split_counts": {
            split: sum(case["split"] == split for case in cases)
            for split in ("anchor", "interpolation", "extrapolation")
        },
        "quantities": list(QUANTITIES),
        "option_types": settings["option_types"],
        "created_before_neural_training": settings["created_before_neural_training"],
    }
    (output / "benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
