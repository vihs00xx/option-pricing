import csv
import json
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[2]
    tables = project / "results" / "monte_carlo" / "tables"
    frozen = json.loads((tables / "mc_frozen_configurations.json").read_text())
    with (tables / "mc_greek_evaluation_summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    checks = []
    for row in rows:
        issues = []
        if float(row["coverage"]) < 0.8:
            issues.append("observed coverage below 80%")
        if float(row["zero_se_rate"]) > 0:
            issues.append("some runs reported zero sampling uncertainty")
        if row["config"].startswith("fd_"):
            issues.append("interval excludes finite-difference bias")
        checks.append(dict(case=row["case"], config=row["config"], quantity=row["quantity"],
                           coverage=row["coverage"], zero_se_rate=row["zero_se_rate"],
                           flagged=bool(issues), reason="; ".join(issues)))
    with (tables / "mc_greek_interval_checks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    with (tables / "mc_target_accuracy.csv").open() as file:
        targets = list(csv.DictReader(file))
    price_config = frozen["preferred_by_tuning"]["id"]
    greek_config = frozen["greek_preferred_by_tuning"]
    greek_id = greek_config["id"] if greek_config else None
    failed = [row for row in checks if row["config"] == greek_id and row["flagged"]]
    status = dict(
        price_configuration_selected_on_tuning=price_config,
        price_targets_attained=sum(row["attained"] == "True" for row in targets if row["config"] == price_config),
        price_targets_tested=sum(row["config"] == price_config for row in targets),
        greek_configuration_selected_on_tuning=greek_id,
        greek_interval_screen_passed=greek_id is not None and not failed,
        greek_interval_failures=failed,
        interpretation="Empirical screen only, not a guarantee of nominal coverage. Accuracy comparisons remain valid; flagged intervals must not be presented as calibrated. The failed selection is retained, not silently replaced by a winner chosen on evaluation data.",
    )
    (tables / "mc_validation_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Price targets: {status['price_targets_attained']}/{status['price_targets_tested']}")
    print(f"Selected Greek interval screen passed: {status['greek_interval_screen_passed']}")
    print(f"Flagged rows for selected Greek configuration: {len(failed)}")


if __name__ == "__main__":
    main()
