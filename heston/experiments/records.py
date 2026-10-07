"""Small helpers for result folders: hashing, CSV, metadata."""

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        fh.write("\n")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def environment():
    import numpy, scipy, torch
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "platform": platform.platform(), "processor": platform.processor()}


def write_metadata(folder, sources=(), extra=None):
    """metadata.json pinning every file in `folder` and the generating sources by sha256."""
    folder = Path(folder)
    files = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "metadata.json"}
    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": git("rev-parse", "HEAD"), "environment": environment(), "files_sha256": files,
            "sources_sha256": {str(Path(s).resolve().relative_to(ROOT)).replace("\\", "/"): digest(s) for s in sources},
            **(extra or {})}
    write_json(folder / "metadata.json", meta)
    return meta


def verify_folder(folder):
    """Recompute the hashes pinned by metadata.json; returns a list of mismatches."""
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    bad = []
    for name, h in meta["files_sha256"].items():
        p = folder / name
        if not p.exists() or digest(p) != h:
            bad.append(name)
    for name, h in meta.get("sources_sha256", {}).items():
        p = ROOT / name
        if not p.exists() or digest(p) != h:
            bad.append(name)
    return bad
