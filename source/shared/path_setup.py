"""Add shared/ and a paper directory to sys.path for experiment scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def setup_paths(paper_dir: str | Path | None = None) -> tuple[Path, Path]:
    """
    Call from experiment entry points:
        from path_setup import setup_paths
        setup_paths(__file__)
    """
    workspace = Path(__file__).resolve().parent.parent
    shared = workspace / "shared"
    if paper_dir is not None:
        paper = Path(paper_dir).resolve()
        if paper.is_file():
            paper = paper.parent.parent  # experiments/foo.py -> paper root
    else:
        paper = None

    for p in (shared, paper):
        if p is not None:
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
    return shared, paper or workspace
