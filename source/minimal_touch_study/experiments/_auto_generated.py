"""Shared header for auto-generated table/figure artifacts.

Vendored copy of ~/Documents/research-commons/scripts/_auto_generated.py so
that the artifact-generation sequence in README.md runs from a fresh clone of
this repository alone (review 2026-09-10 round 2, finding r5). Keep in sync
with the upstream helper; the format below is the frozen contract.

Any file produced by E3 scripts MUST start with this header.
Do not edit the generated body by hand; change the source CSV/script and
regenerate instead.
"""

from __future__ import annotations

from datetime import datetime, timezone


HEADER_MARKER = "Auto-generated"


def latex_header(script_name: str, source_path: str, command: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"% {HEADER_MARKER} by {script_name} — do not edit by hand\n"
        f"% Source: {source_path}\n"
        f"% Command: {command}\n"
        f"% Generated: {ts}\n"
    )
