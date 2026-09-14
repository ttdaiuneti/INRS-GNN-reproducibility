#!/usr/bin/env python3
"""Regenerate every CSV-driven LaTeX table in the manuscript.

Tables 3-6 and 9 are built by build_tables_from_csv.py from the CSVs that
build_raw_results.py writes. Their caption/note/label text belongs to the
manuscript rather than to the CSV, so it lives here -- otherwise the only
record of it is the "% Command:" header of the generated .tex, which a reader
has to reassemble by hand (review 2026-09-10 round 3, finding R3-r3).

Run after experiments/build_raw_results.py:

    python3 experiments/build_accuracy_tables.py

Pass --output-dir to write elsewhere (e.g. a scratch directory) without
touching the manuscript -- useful for checking that a rebuild reproduces the
committed tables.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BUILDER = os.path.join(HERE, "build_tables_from_csv.py")

TABLES = [
    {
        "csv": 'experiments/results_summary.csv',
        "output": 'tab_accuracy.tex',
        "label": 'tab:accuracy',
        "caption": (
            'Node classification test accuracy (\\%, mean $\\pm$ SD over ten seeds) on the public Cora and CiteSeer splits, using the label-safe protocol of Section~\\ref{sec:label-safe}. Statistical comparisons are reported in Table~\\ref{tab:accuracy-stats}.'
        ),
        "notes": [
            (
                'From raw\\_results.csv via the label-safe re-run, ten seeds (2026-09-01).'
            ),
            (
                'RG-GCN-static: simplified reproduction (Section~\\ref{sec:downstream}), not a faithful '
                'upper bound for Ding et al.'
            ),
            (
                'External reference (not ours): published vanilla GCN, same split, Cora $80.7\\pm1.0$, '
                'CiteSeer $71.0\\pm0.7$.'
            ),
        ],
        "flags": ['--star'],
    },
    {
        "csv": 'experiments/wilcoxon_summary.csv',
        "output": 'tab_wilcoxon.tex',
        "label": 'tab:accuracy-stats',
        "caption": (
            'Paired comparisons of hybrid and plain GCN test accuracy over ten seeds. Wilcoxon tests are two-sided; exploratory TOST uses a $\\pm 1$ percentage-point equivalence margin and significance level $0.05$.'
        ),
        "notes": ['Wilcoxon differences use integer correct-prediction counts. The equivalence margin was chosen during revision, after initial results were examined; the analysis was not preregistered.', "From raw\\_results.csv, per-seed test\\_acc paired by seed order $[42, 0, 1, 2, 3, 4, 5, 6, 7, 8]$, via scipy.stats.wilcoxon and a manual paired-$t$/TOST implementation (experiments/build\\_raw\\_results.py). CI and Cohen's $d_z$ use the sample SD (ddof$=1$); the mean$\\pm$std columns of Table~\\ref{tab:accuracy} use the population SD (ddof$=0$).", 'All four TOST tests reject non-equivalence at $\\pm 1$pp: the hybrid variants achieve statistically equivalent accuracy to plain GCN within this margin on both datasets, conditional on the single rough graph built per dataset (Section~\\ref{sec:experiments}).'],
        "flags": [],
    },
    {
        "csv": 'experiments/ablation_summary.csv',
        "output": 'tab_ablation.tex',
        "label": 'tab:ablation',
        "caption": (
            'Ablation of the structural adjacency: test accuracy (\\%, mean $\\pm$ SD over ten seeds) for rough-only, hybrid and plain GCN variants under the label-safe protocol of Table~\\ref{tab:accuracy}.'
        ),
        "notes": [
            (
                'Psi-only uses the same canonical protocol as GCN-plain/hybrid.'
            ),
        ],
        "flags": [],
    },
    {
        "csv": 'experiments/correctness_summary.csv',
        "output": 'tab_correctness.tex',
        "label": 'tab:correctness',
        "caption": (
            'Incremental versus batch structural correctness under oracle labels. Rows 1--3 use 400-node subgraphs; rows 4--5 compare every node pair on the full active graphs with all features.'
        ),
        "notes": [
            (
                'All differences measured in NumPy double-precision arithmetic; a max of exactly 0 is '
                'reported verbatim, not rounded from a nonzero value.'
            ),
        ],
        "flags": [],
    },
    {
        "csv": 'experiments/timing_summary.csv',
        "output": 'tab_timing.tex',
        "label": 'tab:timing',
        "caption": (
            'Per-insertion maintenance time for safety radii and rough weights at matched graph states. Results average five seeds, each evaluated at five sampled stream steps and the final state; fair and naive-loop batch comparators are defined in Section~\\ref{sec:experiments}.'
        ),
        "notes": [
            (
                'Speedup columns are the mean of the per-seed end-to-end ratios ($\\delta$+$W$ batch total '
                '$/$ incremental total), not the quotient of the printed mean-time columns.'
            ),
            (
                'Breakdown (mean, Cora / CiteSeer, ms): Inc. $W$ 0.90 / 0.80, Inc. $\\delta$ 1.46 / 8.53, '
                'Batch $W$ vec. 24.4 / 90.1, Batch $\\delta$ vec. 36.5 / 80.0, Batch $W$ naive 156.3 / '
                '246.7, Batch $\\delta$ naive 2638.4 / 21050.4. Incremental $\\delta$ uses the same direct-'
                'difference distance kernel as the batch correctness reference (Corollary~\\ref{cor:adj-'
                'correct}), so a Gram-based incremental $\\delta$ would lower its constant factor further at'
                ' the cost of bit-exact agreement.'
            ),
            (
                'The rough-weight step alone is 174$\\times$ / 310$\\times$ faster than the naive-loop $W$ '
                'rebuild and 27$\\times$ / 113$\\times$ faster than the vectorized $W$ rebuild (Cora / '
                'CiteSeer).'
            ),
        ],
        "flags": [],
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",
                    default=os.path.join(ROOT, "docs", "manuscript", "tables"),
                    help="where to write the .tex files")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for t in TABLES:
        argv = [sys.executable, BUILDER,
                "--csv", os.path.join(ROOT, t["csv"]),
                "--output", os.path.join(args.output_dir, t["output"]),
                "--caption", t["caption"],
                "--label", t["label"]]
        for note in t["notes"]:
            argv += ["--note", note]
        argv += t["flags"]
        r = subprocess.run(argv, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
