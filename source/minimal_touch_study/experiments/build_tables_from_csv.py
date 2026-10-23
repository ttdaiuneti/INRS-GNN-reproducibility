#!/usr/bin/env python3
"""Build a booktabs/threeparttable LaTeX table from CSV.

Vendored from ~/Documents/research-commons/scripts/build_tables_from_csv.py
so that the artifact-generation sequence in README.md runs from a fresh
clone of this repository alone (review 2026-09-10 round 2, finding r5).
Keep in sync with the upstream script.

Generated .tex files are artifacts: do not edit by hand. Change the CSV (or
flags) and regenerate.

Example:
  python3 scripts/build_tables_from_csv.py \\
    --csv results_summary.csv \\
    --output tables/tab_accuracy.tex \\
    --caption "Classification accuracy (mean$\\pm$std, 5-fold CV). Bold: best per row." \\
    --label tab:accuracy \\
    --highlight-best max \\
    --note "Numbers computed from raw_results.csv via summarize_5fold_rows."
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _auto_generated import latex_header


LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def escape_cell(text: str) -> str:
    """Escape LaTeX specials unless the cell already looks like LaTeX math."""
    s = text.strip()
    if not s:
        return "---"
    if "$" in s or s.startswith("\\"):
        return s
    out = []
    for ch in s:
        out.append(LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def parse_numeric(text: str) -> float | None:
    s = text.strip()
    if not s or s in {"---", "n/a", "NA"}:
        return None
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def format_cell(text: str, decimals: int | None) -> str:
    s = text.strip()
    if decimals is None:
        return s
    compact = s.replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", compact):
        return s
    val = parse_numeric(s)
    if val is None:
        return s
    return f"{val:.{decimals}f}"


def highlight_row(cells: list[str], mode: str | None) -> list[str]:
    """Bold best numeric cell in the row, skipping column 0 (row label)."""
    if mode not in {"min", "max"}:
        return cells
    parsed = [(i, parse_numeric(c)) for i, c in enumerate(cells) if i > 0]
    valid = [(i, v) for i, v in parsed if v is not None]
    if not valid:
        return cells
    target = min(v for _, v in valid) if mode == "min" else max(v for _, v in valid)
    out = list(cells)
    for i, v in valid:
        if abs(v - target) < 1e-12:
            out[i] = r"\textbf{" + out[i] + "}"
    return out


def col_spec(n_cols: int) -> str:
    if n_cols < 1:
        raise ValueError("Need at least one column.")
    # tabularx: fixed-width label column, remaining columns share the rest of
    # \linewidth equally (centered) -- extracolsep{\fill} stretched every gap
    # by the same ABSOLUTE amount regardless of each column's natural content
    # width, which bunched short numeric columns next to a wide header and
    # left a lopsided gap elsewhere (user feedback 2026-09-10).
    return "l" + (r">{\centering\arraybackslash}X" * (n_cols - 1))


def build_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    notes: list[str],
    star: bool,
    highlight: str | None,
    decimals: int | None,
) -> str:
    env = "table*" if star else "table"
    n_cols = len(headers)
    body_lines = []
    for raw in rows:
        padded = (raw + [""] * n_cols)[:n_cols]
        formatted = [format_cell(c, decimals) for c in padded]
        formatted = highlight_row(formatted, highlight)
        latex_cells = []
        for cell in formatted:
            if cell.startswith(r"\textbf{") and cell.endswith("}"):
                inner = cell[len(r"\textbf{") : -1]
                latex_cells.append(r"\textbf{" + escape_cell(inner) + "}")
            else:
                latex_cells.append(escape_cell(cell))
        body_lines.append(" & ".join(latex_cells) + r" \\")

    header_row = " & ".join(escape_cell(h) for h in headers) + r" \\"
    notes_block = ""
    if notes:
        items = "\n".join(r"\item " + n for n in notes)
        notes_block = (
            "\\begin{tablenotes}[flushleft]\n"
            "\\footnotesize\n"
            f"{items}\n"
            "\\end{tablenotes}\n"
        )

    spec = col_spec(n_cols)
    # tabularx fills \linewidth and gives the non-label columns an equal,
    # content-aware share of the remaining width (instead of extracolsep's
    # flat stretch per gap), so the table matches the width of its own
    # caption/notes (threeparttable already forces those to \linewidth)
    # without the lopsided column spacing extracolsep produced.
    return "\n".join(
        [
            f"\\begin{{{env}}}[pos=!htbp]",
            "\\begin{threeparttable}",
            "\\centering",
            "\\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}",
            "\\toprule",
            header_row,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabularx}",
            notes_block.rstrip(),
            "\\end{threeparttable}",
            f"\\end{{{env}}}",
            "",
        ]
    )


def load_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("CSV is empty.")
    headers = [c.strip() for c in rows[0]]
    data = [[c.strip() for c in row] for row in rows[1:]]
    return headers, data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a booktabs/threeparttable .tex table from CSV."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output .tex path")
    parser.add_argument("--caption", required=True, help="LaTeX caption (may contain math)")
    parser.add_argument("--label", required=True, help="LaTeX label, e.g. tab:accuracy")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Tablenote item (repeatable)",
    )
    parser.add_argument("--star", action="store_true", help="Use table* (two-column span)")
    parser.add_argument(
        "--highlight-best",
        choices=["min", "max"],
        default=None,
        help="Bold the best numeric value in each row",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="Force this many decimal places on numeric cells",
    )
    args = parser.parse_args()

    headers, data = load_csv(args.csv)
    body = build_table(
        headers=headers,
        rows=data,
        caption=args.caption,
        label=args.label,
        notes=args.note,
        star=args.star,
        highlight=args.highlight_best,
        decimals=args.decimals,
    )
    command = " ".join(sys.argv)
    header = latex_header("build_tables_from_csv.py", args.csv, command)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n" + body, encoding="utf-8")
    print(f"[PASS] Wrote {out}")


if __name__ == "__main__":
    main()
