#!/usr/bin/env python3
"""
Grouped bar chart: incremental delta+W update vs full batch delta+W rebuild
(matched vectorized batch), per dataset. Log-scale y-axis. PDF via
Matplotlib, per table_figure_skill.md tooling standard.

review_opus_round2.md MAJOR-E2-R2: the batch bar is now the fair vectorized
comparator (BLAS-GEMM delta + edge-list NumPy W), not the naive Python-loop
one, and the annotated multiplier is the mean of per-seed end-to-end ratios.

Usage: python3 experiments/plot_speedup_bar.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _auto_generated import latex_header  # noqa: E402  (vendored, see r5)

HERE = os.path.dirname(__file__)
CSV_PATH = os.path.join(HERE, "timing_summary.csv")
FIG_DIR = os.path.join(HERE, "..", "docs", "manuscript", "figures")
PDF_PATH = os.path.join(FIG_DIR, "fig_speedup.pdf")
TEX_PATH = os.path.join(FIG_DIR, "fig_speedup.tex")


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    datasets = [r["Dataset"] for r in rows]
    inc_ms = [float(r["Inc. (ms)"]) for r in rows]
    batch_ms = [float(r["Batch, fair (ms)"]) for r in rows]
    speedups = [float(r["Speedup (fair)"]) for r in rows]

    x = np.arange(len(datasets))
    width = 0.32

    # Flat fills, no hatch, and a single-level y-grid: matches the visual
    # language of fig_scale_speedup / fig_accuracy_parity (same palette, same
    # grid alpha/linewidth) rather than the "clip-art" bar-chart look of
    # hatching + dense log-minor gridlines striping through solid bars
    # (user feedback 2026-09-10: figure read as noticeably rougher than 4/5).
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.bar(x - width / 2 - 0.02, inc_ms, width, label="Incremental $\\delta$+$W$ update",
           color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2 + 0.02, batch_ms, width,
           label="Batch $\\delta$+$W$ rebuild (vectorized)",
           color="#dd8452", edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylim(top=max(batch_ms) * 4)  # headroom so annotations clear the bars
    ax.set_ylabel("Per-insertion time (ms, log scale)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlim(-0.65, len(datasets) - 1 + 0.65)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=2)

    for i in range(len(datasets)):
        ax.annotate(
            f"{speedups[i]:.0f}$\\times$ faster",
            xy=(i, batch_ms[i] * 1.6),
            ha="center", fontsize=8.5, color="#333333",
        )

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Saved PDF figure: {PDF_PATH}")

    caption = (
        'Per-insertion maintenance time for safety radii and rough weights, comparing incremental updates with matched vectorized batch rebuilds (logarithmic time axis). Bars average five seeds, each evaluated at five sampled stream steps and the final state. Annotated speedups are means of per-seed ratios; see Table~\\ref{tab:timing}.'
    )
    command = " ".join(sys.argv)
    body = "\n".join([
        r"\begin{figure}[pos=!htbp]",
        r"\centering",
        r"\includegraphics[width=0.8\linewidth]{figures/fig_speedup.pdf}",
        rf"\caption{{{caption}}}",
        r"\label{fig:speedup}",
        r"\end{figure}",
        "",
    ])
    with open(TEX_PATH, "w") as f:
        f.write(latex_header("plot_speedup_bar.py", "experiments/timing_summary.csv", command) + "\n" + body)
    print(f"[PASS] Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
