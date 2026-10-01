"""Deliberate freeze/promote step for the anchored-PINN baseline.

Ordinary training and refinement never touch the frozen primary checkpoint at
``checkpoints/pinn/primary/anchored/fixed_black_scholes_pinn.pt``. This script
is the single deliberate place that may replace it: it copies a candidate from
``checkpoints/pinn/research_variants/fixed_pinn_candidates/`` onto the primary
path and, with ``--update-registry``, re-pins the SHA-256 in
``experiments/config/neural/pinn_model_registry.json``.

The previous frozen checkpoint is preserved (under the results tree) so a
promotion is always reversible.
"""

import argparse
import json
import shutil
from pathlib import Path

from experiments.pinn import frozen_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def _backup_path(root, old_sha):
    directory = ROOT / "results" / "pinn" / "legacy" / "promoted_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"fixed_black_scholes_pinn_{old_sha}.pt"


def main():
    parser = argparse.ArgumentParser(description=(
        "Promote a candidate checkpoint to the frozen primary baseline."))
    parser.add_argument("--source", required=True,
                        help="Candidate checkpoint (.pt) under "
                             "checkpoints/pinn/research_variants/fixed_pinn_candidates/")
    parser.add_argument("--update-registry", action="store_true",
                        help="Also re-pin primary_model.checkpoint_sha256 in the "
                             "registry to the promoted candidate's SHA-256. "
                             "Required to leave the baseline consistent.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be promoted without changing files")
    arguments = parser.parse_args()

    source = Path(arguments.source)
    if not source.is_absolute():
        source = ROOT / source
    if source == frozen_checkpoint.primary_checkpoint_path(ROOT):
        raise ValueError("promote: source cannot be the primary checkpoint itself")
    if not source.exists():
        raise FileNotFoundError(f"promote: source not found: {source}")
    source.sha = frozen_checkpoint.sha256_of(source)

    current = frozen_checkpoint.primary_digest(ROOT)
    if current["status"] not in ("frozen", "missing"):
        raise RuntimeError(
            f"promote aborted: primary is '{current['status']}' (expected "
            "'frozen' or 'missing'); resolve before promoting")
    new_sha = frozen_checkpoint.sha256_of(source)
    print(f"current primary : {current['path']}")
    print(f"  status        : {current['status']}")
    print(f"  sha256        : {current['actual_sha256']}")
    print(f"source candidate: {source}")
    print(f"  sha256        : new_sha={new_sha}")
    if new_sha == current.get("actual_sha256"):
        raise SystemExit("promote: candidate is identical to the current "
                         "primary; nothing to promote")

    if arguments.dry_run:
        print("\nDRY-RUN: no files changed.")
        return

    if current["actual_sha256"]:
        backup = _backup_path(ROOT, current["actual_sha256"])
        shutil.copyfile(current["path"], backup)
        print(f"backed up previous primary -> {backup}")

    shutil.copyfile(source, current["path"])
    print(f"promoted candidate -> {current['path']}")

    if arguments.update_registry:
        registry_path = frozen_checkpoint.registry_path(ROOT)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry.setdefault("primary_model", {})["checkpoint_sha256"] = new_sha
        temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
        temporary.replace(registry_path)
        print(f"registry checkpoint_sha256 -> {new_sha}")
    else:
        print(("\nNOTE: registry was NOT updated; the primary now differs from "
               "the pinned SHA-256, so ordinary runs will fail fast until you "
               "re-run with --update-registry or restore the original."))
    print("\npromote complete.")


if __name__ == "__main__":
    main()