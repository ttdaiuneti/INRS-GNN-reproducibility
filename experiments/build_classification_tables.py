#!/usr/bin/env python3
"""Regenerate tab_accuracy.tex and tab_wilcoxon.tex from results_summary.csv
and wilcoxon_summary.csv.

Pass --output-dir to write elsewhere.

    python3 experiments/build_classification_tables.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "tables"

TABLES = [
    {
        "csv": "results_summary.csv",
        "output": "tab_accuracy.tex",
        "label": "tab:accuracy",
        "star": True,
        "caption": "Node classification test accuracy (\\%, mean $\\pm$ SD over ten seeds) on the public "
                   "Cora and CiteSeer splits, with the label-safe operator of Section~\\ref{sec:label-safe}.",
        "notes": ["RG-GCN-static is a simplified operator in the style of Ding et al.~\\cite{ding2022rggcn}, "
                  "not a reproduction of their pipeline.",
                  "Published vanilla GCN results on the same splits: Cora $80.7\\pm1.0$, CiteSeer $71.0\\pm0.7$."],
    },
    {
        "csv": "wilcoxon_summary.csv",
        "output": "tab_wilcoxon.tex",
        "label": "tab:accuracy-stats",
        "caption": "Paired comparisons of hybrid and plain GCN accuracy over ten seeds. "
                   "Wilcoxon tests are two-sided; TOST uses a $\\pm 1$ point margin at level $0.05$.",
        "notes": ["Wilcoxon tests use the number of correct test predictions. Confidence intervals and "
                  "Cohen's $d_z$ use the sample SD; Table~\\ref{tab:accuracy} reports the population SD.",
                  "The TOST margin is exploratory (Appendix~\\ref{app:protocol})."],
        # Dataset, N, Mean, 95% CI, d_z, Wilcoxon p, TOST p
        "widths": [0.95, 0.45, 0.9, 1.4, 1.05, 1.1, 1.15],
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUT))
    ap.add_argument("--no-widths", action="store_true")
    a = ap.parse_args()
    for t in TABLES:
        argv = [sys.executable, "build_tables_from_csv.py", "--csv", t["csv"],
                "--output", str(Path(a.output_dir) / t["output"]),
                "--label", t["label"], "--caption", t["caption"]]
        if t.get("star"):
            argv.append("--star")
        for n in t["notes"]:
            argv += ["--note", n]
        if "widths" in t and not a.no_widths:
            argv += ["--col-widths", ",".join(str(w) for w in t["widths"])]
        subprocess.run(argv, cwd=HERE, check=True)


if __name__ == "__main__":
    main()
