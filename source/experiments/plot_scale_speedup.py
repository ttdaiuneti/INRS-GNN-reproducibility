#!/usr/bin/env python3
"""
Scale figure: fair end-to-end incremental-vs-batch speedup as a function of
|V|.  Demonstrates Proposition 4's prediction that the per-insertion speedup
grows ~linearly with |V| (incremental delta O(|V|.|B|) vs batch O(|V|^2.|B|)),
addressing review_opus_round2.md's standing concern that the |V|^2 regime was
never evaluated.

Reads experiments/results/e2e_scale_timing_results.json (Cora, CiteSeer,
PubMed real graphs + synthetic SBM at |V| in {6k,12k,25k}, |B|=100 fixed).

Usage: python3 experiments/plot_scale_speedup.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from _auto_generated import latex_header  # noqa: E402  (vendored, see r5)

HERE = os.path.dirname(__file__)
JSON_PATH = os.path.join(HERE, "results", "e2e_scale_timing_results.json")
FIG_DIR = os.path.join(HERE, "..", "docs", "manuscript", "figures")
PDF_PATH = os.path.join(FIG_DIR, "fig_scale_speedup.pdf")
TEX_PATH = os.path.join(FIG_DIR, "fig_scale_speedup.tex")


OGB_JSON = os.path.join(HERE, "results", "e2e_ogb_stream_results.json")


def main():
    with open(JSON_PATH) as f:
        res = json.load(f)["results"]

    def get(name):
        return next(r for r in res if r["graph"] == name)

    ogb = None
    if os.path.exists(OGB_JSON):
        ogb = json.load(open(OGB_JSON))

    # --- on-trend series (line): every graph EXCEPT CiteSeer, in |V| order ---
    pts = []
    for r in res:
        if r["graph"] == "citeseer":
            continue
        label = {"cora": "Cora", "pubmed": "PubMed"}.get(
            r["graph"], r["graph"].replace("SBM-", "SBM-"))
        pts.append((r["n_active_start"], r["speedup_end_to_end_fair"],
                    r["speedup_std"], label,
                    r["graph"].startswith("SBM-")))
    if ogb:
        pts.append((ogb["batch_anchors"][-1]["n_active"],
                    ogb["speedup_end_to_end_stateful"], 0.0,
                    "ogbn-arxiv", False))
    pts.sort()
    Vt = np.array([p[0] for p in pts])
    St = np.array([p[1] for p in pts])
    Et = np.array([p[2] for p in pts])
    is_sbm = np.array([p[4] for p in pts])

    cs = get("citeseer")
    Vc, Sc, Ec = cs["n_active_start"], cs["speedup_end_to_end_fair"], cs["speedup_std"]

    # Power-law fits. |B| varies across the real graphs and the batch bound is
    # O(|V|^2 |B|), so the SBM sweep (|B| fixed at 100) is the only series that
    # isolates |V|-scaling; it is the fit we plot. The other two are reported in
    # the caption so the reader can see what excluding a point does. All three
    # are descriptive summaries of seven measured configurations, not evidence
    # of a universal law (review 2026-09-10, m4).
    def fit(V, S):
        sl, ic = np.polyfit(np.log(V), np.log(S), 1)
        return float(sl), float(ic)

    slope_sbm, ic_sbm = fit(Vt[is_sbm], St[is_sbm])
    slope, intercept = fit(Vt, St)                       # on-trend (no CiteSeer)
    V_all = np.append(Vt, Vc)
    S_all = np.append(St, Sc)
    slope_all, _ = fit(V_all, S_all)                     # every measured point
    xf = np.array([Vt.min() * 0.85, Vt.max() * 1.25])
    yf = np.exp(ic_sbm) * xf ** slope_sbm

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xf, yf, ls="--", lw=1.0, color="#999999", zorder=1,
            label=rf"SBM fit ($|B|$ fixed), slope ${slope_sbm:.2f}$")
    ax.plot(Vt, St, "-", lw=1.4, color="#4c72b0", zorder=2)
    ax.errorbar(Vt[is_sbm], St[is_sbm], yerr=Et[is_sbm], fmt="o", ms=7,
                color="#4c72b0", capsize=3, zorder=3,
                label=r"SBM synthetic ($|B|=100$)")
    ax.errorbar(Vt[~is_sbm], St[~is_sbm], yerr=Et[~is_sbm], fmt="s", ms=7,
                color="#dd8452", capsize=3, zorder=3,
                label="Real graphs (Cora, PubMed, ogbn-arxiv)")
    ax.scatter([Vc], [Sc], marker="X", s=95, color="#c44e52", zorder=4,
               edgecolor="black", linewidth=0.5,
               label=r"CiteSeer (off-trend: larger $|B|$)")
    ax.errorbar([Vc], [Sc], yerr=[Ec], fmt="none", ecolor="#c44e52", capsize=3, zorder=4)

    for v, s, _, name, _ in pts:
        dy = 12 if name != "PubMed" else -14
        ax.annotate(name, (v, s), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8)
    ax.annotate("CiteSeer", (Vc, Sc), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color="#c44e52")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Active node count $|V|$")
    ax.set_ylabel("Fair end-to-end speedup (batch $/$ incremental)")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Saved {PDF_PATH}  (slopes: SBM-only {slope_sbm:.2f}, "
          f"on-trend {slope:.2f}, all points {slope_all:.2f})")

    ogb_txt = ""
    if ogb:
        nt = f"{ogb['n_total'] / 1e3:.0f}{{,}}{ogb['n_total'] % 1000:03d}"
        anch = ogb["batch_anchors"][-1]
        pi = ogb["per_insertion_ms"]
        spd = f"{ogb['speedup_end_to_end_stateful']:,.0f}".replace(",", "{,}")
        dense_gb = anch["n_active"] ** 2 * 8 / 1e9
        n_init_s = f"{ogb['n_init']:,}".replace(",", "{,}")
        n_end_s = f"{anch['n_active']:,}".replace(",", "{,}")
        ogb_txt = (
            f" The ogbn-arxiv point is plotted at its \\emph{{active}} size, "
            f"$|V| = {n_init_s}$ at the start of the stream and ${n_end_s}$ at "
            f"the end ($|B| = 128$; the full dataset has ${nt}$ nodes and "
            "$\\sim$1.2M edges). It is a stateful stream of 200 consecutive "
            "insertions whose maintained state is re-verified against full "
            "rebuilds at three checkpoints: "
            f"${pi['total_mean']:.1f}$\\,ms per insertion including bookkeeping "
            f"(median ${pi['total_median']:.1f}$) against a single batch "
            f"recomputation of ${anch['batch_total_s']:.0f}$\\,s at the same "
            f"active size, a ratio of ${spd}\\times$. This point "
            "alone uses a different estimator from the others: it is the "
            "end-of-stream batch anchor divided by the mean incremental time "
            "over the 200 steps, not a mean of paired per-step ratios, since "
            "a rebuild at every step is infeasible here. A dense "
            f"$|V| \\times |V|$ distance matrix (${dense_gb:.0f}$\\,GB, fp64) is "
            "infeasible there, so this point uses the sparse-adjacency variant "
            "(Proposition~\\ref{prop:complexity}), not the dense implementation "
            "of the smaller points."
        )
    caption = (
        'End-to-end maintenance speedup versus active node count (log--log axes). Non-OGB points show mean $\\pm$ SD across three seeds, with eight insertions per seed near the 80\\% initial state. The dashed line fits only the fixed-feature SBM sweep; the solid line connects the displayed points except CiteSeer ($\\times$). The ogbn-arxiv point uses one 200-insertion stateful stream without a seed-level SD; its speedup is the final batch-anchor time divided by mean incremental time, including bookkeeping. Estimators and timing components are detailed in Table~\\ref{tab:scale-breakdown}; fits and limitations are discussed in Section~\\ref{sec:experiments}.'
    )
    command = " ".join(sys.argv)
    body = "\n".join([
        r"\begin{figure}[pos=!htbp]",
        r"\centering",
        r"\includegraphics[width=0.72\linewidth]{figures/fig_scale_speedup.pdf}",
        rf"\caption{{{caption}}}",
        r"\label{fig:scale-speedup}",
        r"\end{figure}",
        "",
    ])
    with open(TEX_PATH, "w") as f:
        f.write(latex_header("plot_scale_speedup.py",
                             "experiments/results/e2e_scale_timing_results.json",
                             command) + "\n" + body)
    print(f"[PASS] Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
