#!/usr/bin/env python3
"""
Dot plot with error bars (mean +/- std over 5 seeds): GCN-plain vs hybrid
(grid / learned alpha), per dataset. Visualizes the parity finding directly
(overlapping error bars = not a real effect), rather than only reporting it
in a table. PDF via Matplotlib, per table_figure_skill.md.

Usage: python3 experiments/plot_accuracy_parity.py
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _auto_generated import latex_header  # noqa: E402  (vendored, see r5)

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "accuracy_parity_plot.csv")
FIG_DIR = os.path.join(HERE, "..", "docs", "manuscript", "figures")
PDF_PATH = os.path.join(FIG_DIR, "fig_accuracy_parity.pdf")
TEX_PATH = os.path.join(FIG_DIR, "fig_accuracy_parity.tex")

METHOD_ORDER = ["GCN-plain", "INRS-hybrid-grid", "INRS-hybrid-learned"]
METHOD_COLORS = {"GCN-plain": "#4c72b0", "INRS-hybrid-grid": "#dd8452", "INRS-hybrid-learned": "#55a868"}
METHOD_MARKERS = {"GCN-plain": "o", "INRS-hybrid-grid": "s", "INRS-hybrid-learned": "^"}


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    by_dataset = defaultdict(dict)
    for r in rows:
        by_dataset[r["dataset"]][r["method"]] = (float(r["test_mean"]), float(r["test_std"]))

    # Fixed order (not alphabetical — "CiteSeer" < "Cora" would reverse it),
    # matching every table in the manuscript.
    preferred_order = ["Cora", "CiteSeer"]
    datasets = [d for d in preferred_order if d in by_dataset] + \
               [d for d in by_dataset if d not in preferred_order]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    offsets = np.linspace(-0.15, 0.15, len(METHOD_ORDER))
    for j, method in enumerate(METHOD_ORDER):
        xs, ys, es = [], [], []
        for i, ds in enumerate(datasets):
            if method not in by_dataset[ds]:
                continue
            mean, std = by_dataset[ds][method]
            xs.append(i + offsets[j])
            ys.append(mean)
            es.append(std)
        ax.errorbar(
            xs, ys, yerr=es, fmt=METHOD_MARKERS[method], color=METHOD_COLORS[method],
            label=method, markersize=7, capsize=4, linewidth=1.5, elinewidth=1.3,
        )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Dataset")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=1)

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Saved PDF figure: {PDF_PATH}")

    caption = (
        'Test accuracy of plain and hybrid GCNs on Cora and CiteSeer under the label-safe protocol. Points and error bars show the mean and one standard deviation over ten downstream seeds. Paired tests and exploratory equivalence analysis are reported in Table~\\ref{tab:accuracy-stats}.'
    )
    command = " ".join(sys.argv)
    body = "\n".join([
        r"\begin{figure}[pos=!htbp]",
        r"\centering",
        r"\includegraphics[width=0.8\linewidth]{figures/fig_accuracy_parity.pdf}",
        rf"\caption{{{caption}}}",
        r"\label{fig:accuracy-parity}",
        r"\end{figure}",
        "",
    ])
    with open(TEX_PATH, "w") as f:
        f.write(latex_header("plot_accuracy_parity.py", "experiments/accuracy_parity_plot.csv", command) + "\n" + body)
    print(f"[PASS] Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
