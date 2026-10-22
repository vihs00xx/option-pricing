"""Hash-pinned study records: CSV output, metadata and verification."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def prepare_output(output, overwrite):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output exists; use a new output directory or explicit --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_metadata(output, settings_path, source_paths, extra=None):
    """Pin sources, settings and every saved output by SHA-256."""
    output = Path(output)
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        settings=json.loads(Path(settings_path).read_text(encoding="utf-8")),
        source_sha256={relative(p): digest(p) for p in sorted(set(map(Path, source_paths)) | {Path(settings_path)})},
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, platform=platform.platform()),
        outputs_sha256={p.name: digest(p) for p in sorted(output.iterdir())
                        if p.is_file() and p.name != "metadata.json"},
        **(extra or {}))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def verify_metadata(output):
    """Check pinned sources and outputs; returns the metadata."""
    output = Path(output).resolve()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in metadata["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise RuntimeError(f"Source integrity mismatch: {name}")
    for name, expected in metadata["outputs_sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or digest(path) != expected:
            raise RuntimeError(f"Output integrity mismatch: {name}")
    return metadata


def pricing_sources():
    return list((ROOT / "src/pricing").glob("*.py")) + [ROOT / "src/black_scholes.py"]
