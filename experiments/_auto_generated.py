"""Shared header for generated table and figure files.

Every generated .tex file starts with this header. Do not edit a generated
file by hand; change the source CSV or script and regenerate it.
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
