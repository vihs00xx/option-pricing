"""Guards for the frozen primary anchored-PINN checkpoint.

The frozen reference baseline lives at
``checkpoints/pinn/primary/anchored/fixed_black_scholes_pinn.pt`` and its
SHA-256 is pinned in ``experiments/config/neural/pinn_model_registry.json``
under ``primary_model.checkpoint_sha256``.

Ordinary training and refinement MUST NEVER overwrite it. They write
candidates under ``checkpoints/pinn/research_variants/fixed_pinn_candidates/``
(and their history/metadata under ``results/.../fixed_pinn_candidates/``).
The single deliberate way to replace the frozen reference is the explicit
freeze/promote step in ``experiments/pinn/promote_fixed_pinn.py``.

Every primary-pipeline script calls :func:`refuse_primary_overwrite` before any
local run, so a missing or mismatched baseline fails fast instead of silently
training against a corrupted reference.
"""

import hashlib
import json
from pathlib import Path


def registry_path(root):
    return Path(root) / "experiments" / "config" / "neural" / "pinn_model_registry.json"


def primary_checkpoint_path(root):
    return Path(root) / "checkpoints" / "pinn" / "primary" / "anchored" / "fixed_black_scholes_pinn.pt"


def candidates_dir(root):
    return Path(root) / "checkpoints" / "pinn" / "research_variants" / "fixed_pinn_candidates"


def candidate_path(root, name):
    """A candidate checkpoint path under the research candidates directory."""
    return candidates_dir(root) / f"{name}.pt"


def frozen_sha256(root):
    """Pinned SHA-256 of the primary checkpoint, or None if unpinned."""
    try:
        registry = json.loads(registry_path(root).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    primary = registry.get("primary_model") or {}
    return primary.get("checkpoint_sha256")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def primary_digest(root):
    """Report the primary checkpoint's integrity state.

    status is one of:
      "frozen"   present and matches the registry pinned hash (the reference)
      "missing"  file absent (nothing to protect yet)
      "mismatch" present but hash != registry pin (corrupted or replaced)
      "unpinned" present but no registry pin to compare against
    """
    path = primary_checkpoint_path(root)
    frozen = frozen_sha256(root)
    if not path.exists():
        return {"path": path, "status": "missing", "actual_sha256": None,
                "frozen_sha256": frozen}
    actual = sha256_of(path)
    if frozen is None:
        status = "unpinned"
    elif actual == frozen:
        status = "frozen"
    else:
        status = "mismatch"
    return {"path": path, "status": status, "actual_sha256": actual,
            "frozen_sha256": frozen}


def refuse_primary_overwrite(root):
    """Entry guard for ordinary training/refinement scripts.

    Verifies the frozen reference is present and matches the registry pin, and
    returns the candidate directory. Writes must go to the candidates
    directory, never to the primary path.
    """
    status = primary_digest(root)
    if status["status"] == "mismatch":
        raise RuntimeError(
            f"primary checkpoint {status['path']} has sha256 "
            f"{status['actual_sha256']} but the registry pins "
            f"{status['frozen_sha256']}; refusing to run against a replaced "
            "or corrupted reference. Investigate before any local run.")
    if status["status"] == "unpinned":
        raise RuntimeError(
            f"primary checkpoint {status['path']} exists but the registry has "
            "no pinned checkpoint_sha256; refusing to continue unpinned.")
    candidate_dir = candidates_dir(root)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    return candidate_dir