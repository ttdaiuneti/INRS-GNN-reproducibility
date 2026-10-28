#!/usr/bin/env python3
"""Draw Figure 3 (speedup against the number of active nodes).

Reads results/scale_minimal.json and results/ogb_stream200_minimal.json and
writes figures/fig_scale_speedup.pdf next to the tables that summarize.py
writes. summarize.py writes the LaTeX wrapper of the figure and the slope
macro ScaleSBMSlope. Usage:
  python3 experiments/minimal_touch/plot_scale_speedup.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
OUT = ROOT / "paper" / "figures" / "fig_scale_speedup.pdf"


def use_manuscript_fonts():
    """Charis SIL text and STIX math, as in the manuscript, if Charis is installed."""
    found = False
    if shutil.which("kpsewhich"):
        for name in ("CharisSIL-Regular.ttf", "CharisSIL-Bold.ttf", "CharisSIL-Italic.ttf", "CharisSIL-BoldItalic.ttf"):
            p = subprocess.run(["kpsewhich", name], capture_output=True, text=True).stdout.strip()
            if p:
                font_manager.fontManager.addfont(p)
                found = True
    if not found:
        print("Charis SIL not found; the figure uses the default font.")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": ["Charis SIL", "DejaVu Sans"],
                                "mathtext.fontset": "stix", "axes.unicode_minus": True})


def main():
    use_manuscript_fonts()
    res = json.load(open(RES / "scale_minimal.json"))["results"]
    ogb = json.load(open(RES / "ogb_stream200_minimal.json"))

    pts = []
    for r in res:
        if r["graph"] == "citeseer":
            continue
        label = {"cora": "Cora", "pubmed": "PubMed"}.get(r["graph"], r["graph"])
        pts.append((r["n_active_start"], r["speedup_end_to_end_fair"], r["speedup_std"], label,
                    r["graph"].startswith("SBM-")))
    pts.append((ogb["batch_anchors"][-1]["n_active"], ogb["speedup_end_to_end_stateful"], 0.0,
                "ogbn-arxiv", False))
    pts.sort()
    Vt = np.array([p[0] for p in pts])
    St = np.array([p[1] for p in pts])
    Et = np.array([p[2] for p in pts])
    is_sbm = np.array([p[4] for p in pts])

    cs = next(r for r in res if r["graph"] == "citeseer")
    Vc, Sc, Ec = cs["n_active_start"], cs["speedup_end_to_end_fair"], cs["speedup_std"]

    # The SBM graphs share |B| = 100, so they are the only series that isolates
    # the growth with |V|; the fitted line uses them only.
    slope_sbm, ic_sbm = np.polyfit(np.log(Vt[is_sbm]), np.log(St[is_sbm]), 1)
    xf = np.array([Vt.min() * 0.85, Vt.max() * 1.25])
    yf = np.exp(ic_sbm) * xf ** slope_sbm

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(xf, yf, ls="--", lw=1.0, color="#999999", zorder=1,
            label=rf"SBM fit ($|B|$ fixed), slope ${slope_sbm:.2f}$")
    ax.plot(Vt, St, "-", lw=1.4, color="#4c72b0", zorder=2)
    ax.errorbar(Vt[is_sbm], St[is_sbm], yerr=Et[is_sbm], fmt="o", ms=7,
                color="#4c72b0", capsize=3, zorder=3, label=r"SBM synthetic ($|B|=100$)")
    ax.errorbar(Vt[~is_sbm], St[~is_sbm], yerr=Et[~is_sbm], fmt="s", ms=7,
                color="#dd8452", capsize=3, zorder=3, label="Real graphs (Cora, PubMed, ogbn-arxiv)")
    ax.scatter([Vc], [Sc], marker="X", s=95, color="#c44e52", zorder=4,
               edgecolor="black", linewidth=0.5, label=r"CiteSeer (off-trend: larger $|B|$)")
    ax.errorbar([Vc], [Sc], yerr=[Ec], fmt="none", ecolor="#c44e52", capsize=3, zorder=4)

    for v, s, _, name, _ in pts:
        dy = 12 if name != "PubMed" else -14
        ax.annotate(name, (v, s), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8)
    ax.annotate("CiteSeer", (Vc, Sc), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#c44e52")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Active node count $|V|$")
    ax.set_ylabel("Fair end-to-end speedup (batch $/$ incremental)")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT} (SBM slope {slope_sbm:.2f})")


if __name__ == "__main__":
    main()
