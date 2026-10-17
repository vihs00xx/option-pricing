"""Verify a hash-pinned results folder without evaluating any pricing method.

Usage: python -m experiments.verify results/<study>
"""

import argparse
import json
from pathlib import Path

from experiments.records import ROOT, digest, verify_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    folder = (ROOT / parser.parse_args().folder).resolve()
    metadata = verify_metadata(folder)
    protocol = folder / "protocol.json"
    if protocol.exists():
        if digest(protocol) != (folder / "protocol.sha256").read_text(encoding="utf-8").strip():
            raise RuntimeError("Protocol seal mismatch")
        frozen = json.loads(protocol.read_text(encoding="utf-8"))
        for name in ("cases", "references"):
            path = folder / ("cases.json" if name == "cases" else "references.csv")
            if digest(path) != frozen[f"{name}_sha256"]:
                raise RuntimeError(f"{path.name} does not match the frozen protocol")
    print(f"Verified {len(metadata['source_sha256'])} source/configuration files and "
          f"{len(metadata['outputs_sha256'])} saved outputs in {folder.name}"
          + (" including the protocol seal." if protocol.exists() else "."))


if __name__ == "__main__":
    main()
