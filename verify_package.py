#!/usr/bin/env python3
"""Verify the integrity and minimum contents of the public artifact package."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256"
REQUIRED = (
    "requirements.txt",
    "ARTIFACT_INDEX.csv",
    "source/experiments/e1_full_scale_correctness.py",
    "source/experiments/e3_pyg_aligned_benchmark.py",
    "source/experiments/e2e_ogb_stream.py",
    "source/theory/incremental_rough_adjacency.py",
    "source/shared/theory/incremental_nrs.py",
    "results/canonical/e1_full_scale_correctness.json",
    "results/canonical/e3_pyg_aligned_results.json",
    "results/canonical/e2e_ogb_stream_results.json",
    "results/locality_stress/raw_results.csv",
    "results/ogb_robustness/raw_results.csv",
    "results/full_pipeline_controls/summary.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    if not MANIFEST.is_file():
        errors.append("missing MANIFEST.sha256")
    else:
        for line_number, line in enumerate(MANIFEST.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append(f"invalid manifest line {line_number}")
                continue
            path = ROOT / relative
            if not path.is_file():
                errors.append(f"manifest file missing: {relative}")
            elif digest(path) != expected:
                errors.append(f"checksum mismatch: {relative}")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    count = sum(1 for line in MANIFEST.read_text().splitlines() if line.strip())
    print(f"[PASS] {count} files match MANIFEST.sha256; required artifacts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())

