"""
Mechanism figure for INRS-GNN: the incremental update replaces a full batch
rebuild with two cheap local passes.

  1. Batch rebuild (avoided)  -- after v_new, a from-scratch pipeline recomputes
                                 every delta_B and every rough weight: O(|V|^2 |B|).
  2. Incremental, delta pass   -- Theorem 1: v_new (class 0) is a new enemy only
                                 for class-1 nodes; a single scan shrinks delta_B
                                 on the small set S_delta, friends are untouched.
  3. Incremental, W pass        -- Theorem 3 / Corollary 1: rough weights are
                                 recomputed only on edges meeting
                                 T_adj = T_str u S_delta; everywhere else the
                                 result is bit-identical to the batch rebuild.

A footer strip shows the downstream role (hybrid adjacency into a 2-layer GCN)
and the measured per-insertion cost contrast.

Outputs:  docs/manuscript/figures/fig_mechanism.pdf   (for \input)
          experiments/results/fig_mechanism.png        (for review)

Run: python3 experiments/plot_incremental_mechanism.py
"""
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "..", "docs", "manuscript", "figures")
RESDIR = os.path.join(HERE, "results")

# The footer strip quotes measured quantities. Read them from the artifacts so
# they cannot drift when an experiment is re-run (review 2026-09-10 round 3:
# this figure still carried |T_adj| 16.6 and ~1700x after the stateful OGB
# re-run replaced both).
with open(os.path.join(RESDIR, "e2e_scale_timing_results.json")) as _f:
    _SCALE = json.load(_f)
with open(os.path.join(RESDIR, "e2e_ogb_stream_results.json")) as _f:
    _OGB = json.load(_f)
_by_graph = {g["graph"].lower(): g for g in _SCALE["results"]}
_T_CORA = _by_graph["cora"]["touch_mean"]
_T_CITE = _by_graph["citeseer"]["touch_mean"]
_T_OGB = _OGB["touch"]["t_adj_mean"]
_SPD_OGB = _OGB["speedup_end_to_end_stateful"]
with open(os.path.join(HERE, "timing_summary.csv")) as _f:
    _TIM = {r["Dataset"]: r for r in csv.DictReader(_f)}
_SPD_CITE = int(_TIM["CiteSeer"]["Speedup (fair)"])
_SPD_CORA = int(_TIM["Cora"]["Speedup (fair)"])

# ----------------------------------------------------------------------------
BLUE = "#4C72B0"     # class 0
SAND = "#DD8452"     # class 1
CRIMSON = "#C44E52"  # v_new / affected / recomputed
GREY_N = "#CBCFD6"   # untouched node
GREY_E = "#D7DBDF"   # untouched edge
TOUCH = "#F3DDDC"    # touch-set fill
HALO = "#FBEEEC"     # soft halo / batch tint
INK = "#2F3337"
MUTE = "#55595E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
})

# ----------------------------------------------------------------------------
# fixed schematic graph: 17 existing nodes (0-7 class 0, 8-16 class 1),
# then v_new (id 17, class 0) arrives just inside the class-0 region.
# ----------------------------------------------------------------------------
POS = {
    0: (0.7, 4.3), 1: (1.8, 4.65), 2: (2.7, 3.9), 3: (1.0, 3.2), 4: (2.05, 2.9),
    5: (0.6, 2.0), 6: (1.6, 1.5), 7: (2.8, 2.0),
    8: (3.6, 4.35), 9: (4.65, 4.75), 10: (3.9, 3.25), 11: (4.9, 3.5),
    12: (5.6, 4.15), 13: (4.3, 2.2), 14: (5.35, 2.5), 15: (3.5, 1.3),
    16: (4.7, 1.15),
}
CLS = {i: (0 if i < 8 else 1) for i in range(17)}

EDGES = [
    (0, 1), (0, 3), (1, 2), (2, 4), (3, 4), (3, 5), (4, 7), (5, 6), (6, 7),
    (8, 9), (8, 10), (9, 12), (10, 11), (11, 12), (11, 14), (13, 14),
    (13, 16), (15, 16), (10, 13), (2, 8), (4, 10), (7, 15),
]

V_NEW = 17
POS[V_NEW] = (2.75, 3.35)
CLS[V_NEW] = 0
NEW_EDGES = [(V_NEW, 2), (V_NEW, 4), (V_NEW, 10)]

T_STR = {V_NEW, 2, 4, 10}          # v_new + its structural neighbours
S_DELTA = {10, 11}                 # class-1 nodes whose delta_B shrinks
T_ADJ = T_STR | S_DELTA            # = {17, 2, 4, 10, 11}

XLIM = (-0.55, 6.5)
YLIM = (0.15, 5.75)


def _dist(a, b):
    return math.hypot(POS[a][0] - POS[b][0], POS[a][1] - POS[b][1])


def _draw_nodes(ax, active=frozenset(), plain=False, ring_all=False):
    for i, (x, y) in POS.items():
        base = BLUE if CLS[i] == 0 else SAND
        if i == V_NEW:
            fc, ec, lw, s = CRIMSON, "#7A2B2E", 1.8, 430
        elif ring_all:
            fc, ec, lw, s = base, CRIMSON, 2.0, 320
        elif i in active:
            fc, ec, lw, s = base, CRIMSON, 2.4, 340
        elif plain:
            fc, ec, lw, s = base, "#3A3F45", 1.0, 300
        else:
            fc, ec, lw, s = GREY_N, "white", 1.0, 300
        ax.scatter([x], [y], s=s, marker="o" if CLS[i] == 0 else "s",
                   facecolor=fc, edgecolor=ec, linewidths=lw,
                   zorder=5 if (i == V_NEW or i in active or ring_all) else 3)


def _draw_edges(ax, hot, new_style="solid"):
    for u, v in EDGES:
        h = hot(u, v)
        ax.plot(*zip(POS[u], POS[v]), color=CRIMSON if h else GREY_E,
                lw=2.1 if h else 1.15, zorder=2 if h else 1,
                solid_capstyle="round")
    for u, v in NEW_EDGES:
        ax.plot(*zip(POS[u], POS[v]), color=CRIMSON, lw=2.1, zorder=4,
                ls=(0, (4, 2)) if new_style == "dashed" else "-",
                solid_capstyle="round")


def _frame(ax, num, title, subtitle, badge=INK):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#D6D9DD")
    ax.scatter([0.045], [0.945], s=330, marker="o", color=badge,
               transform=ax.transAxes, clip_on=False, zorder=10)
    ax.text(0.045, 0.945, str(num), color="white", fontsize=11,
            fontweight="bold", ha="center", va="center",
            transform=ax.transAxes, zorder=11)
    ax.text(0.088, 0.945, title, fontsize=11.5, fontweight="bold", ha="left",
            va="center", transform=ax.transAxes)
    ax.text(0.5, -0.052, subtitle, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.9, style="italic", color=MUTE)


# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(13.0, 6.15))
gs = fig.add_gridspec(2, 3, height_ratios=[2.6, 0.95], hspace=0.20, wspace=0.08,
                      left=0.026, right=0.974, top=0.905, bottom=0.165)
ax1, ax2, ax3 = (fig.add_subplot(gs[0, k]) for k in range(3))
foot = fig.add_subplot(gs[1, :])

# ---- Panel 1: batch rebuild (avoided) --------------------------------
ax1.set_facecolor(HALO)
_draw_edges(ax1, lambda u, v: True, new_style="solid")
_draw_nodes(ax1, ring_all=True)
ax1.text(0.5, 0.015, "every node, every edge recomputed from scratch",
         transform=ax1.transAxes, ha="center", va="bottom", fontsize=8.6,
         color="#8A3B3D")
_frame(ax1, 1, "Batch rebuild (avoided)",
       r"recompute all $\delta_B$ and all $W^{\mathrm{rough}}$: $O(|V|^2\,|B|)$ per insertion",
       badge=CRIMSON)

# ---- Panel 2: incremental delta pass --------------------------------
_draw_edges(ax2, lambda u, v: False, new_style="solid")
# safety radius of node 10: v_new becomes its nearest enemy, so delta_B shrinks
c10 = POS[10]
r_old = _dist(10, 2)          # old nearest enemy = node 2
r_new = _dist(10, V_NEW)      # v_new is the new nearest enemy
ax2.add_patch(mpatches.Circle(c10, r_old, fill=False, ec="#A7ACB2", lw=0.9,
                              ls=(0, (2, 2.5)), zorder=1))
ax2.add_patch(mpatches.Circle(c10, r_new, fill=False, ec=CRIMSON, lw=1.6, zorder=2))
ax2.plot([c10[0], POS[V_NEW][0]], [c10[1], POS[V_NEW][1]], color=CRIMSON,
         lw=1.0, ls=(0, (1, 1.5)), zorder=2)
_draw_nodes(ax2, active=S_DELTA | {V_NEW})
ax2.annotate(r"$v_{\mathrm{new}}$ = new nearest enemy", POS[V_NEW],
             (POS[V_NEW][0] - 1.0, POS[V_NEW][1] - 1.15), fontsize=8.2,
             color="#7A2B2E", ha="left", va="top",
             arrowprops=dict(arrowstyle="-", color="#7A2B2E", lw=0.8))
ax2.text(c10[0] + r_new + 0.10, c10[1] - 0.05, r"$\delta_B$ shrinks",
         fontsize=8.6, color=CRIMSON, fontweight="bold", ha="left", va="center")
ax2.text(c10[0] + 0.15, c10[1] - r_old - 0.06, r"old $\delta_B$", fontsize=7.8,
         color="#8A8F95", ha="left", va="top")
ax2.annotate(r"$\in S_\delta$", POS[11], (POS[11][0] + 0.42, POS[11][1] + 0.02),
             fontsize=9, color=CRIMSON, fontweight="bold", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=CRIMSON, lw=0.8))
ax2.annotate(r"friend: $\delta_B$ kept", POS[2],
             (POS[2][0] - 1.35, POS[2][1] + 0.55), fontsize=8, color=MUTE,
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", color="#A7ACB2", lw=0.8))
_frame(ax2, 2, "Incremental · $\\delta_B$ pass",
       r"one scan $O(|V|\,|B|)$; only enemies of $v_{\mathrm{new}}$ shrink $\to S_\delta$ (Theorem 1)")

# ---- Panel 3: incremental W pass ----------------------------------
for i in T_ADJ:
    x, y = POS[i]
    ax3.add_patch(mpatches.Circle((x, y), 0.58, facecolor=HALO, ec="none", zorder=0))
    ax3.add_patch(mpatches.Circle((x, y), 0.40, facecolor=TOUCH, ec="none", zorder=0))
_draw_edges(ax3, lambda u, v: (u in T_ADJ) or (v in T_ADJ), new_style="solid")
_draw_nodes(ax3, active=T_ADJ)
ax3.annotate(r"$\mathcal{T}_{str}$", POS[4], (POS[4][0] - 0.55, POS[4][1] - 0.15),
             fontsize=10, fontweight="bold", ha="right", va="center", color=CRIMSON)
ax3.annotate(r"$S_\delta$", POS[11], (POS[11][0] + 0.42, POS[11][1] + 0.28),
             fontsize=10, fontweight="bold", ha="left", va="center", color=CRIMSON)
ax3.text(0.5, 0.015,
         r"outside $\mathcal{T}_{adj}$: $W^{\mathrm{rough}}$ unchanged $=$ batch result (Corollary 1)",
         transform=ax3.transAxes, ha="center", va="bottom", fontsize=8.4,
         color=MUTE, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec="#D6D9DD", lw=0.8))
_frame(ax3, 3, r"Incremental · $W^{\mathrm{rough}}$ pass",
       r"only edges meeting $\mathcal{T}_{adj}=\mathcal{T}_{str}\cup S_\delta$ (Theorem 3)")

# ---- connectors between panels --------------------------------------
fig.text(0.338, 0.585, "▶", fontsize=15, color="#B9BEC6", ha="center", va="center")
fig.text(0.338, 0.66, "replaced by", fontsize=8.5, color=MUTE, ha="center", va="center")
fig.text(0.662, 0.585, "▶", fontsize=15, color="#B9BEC6", ha="center", va="center")
fig.text(0.662, 0.66, "then", fontsize=8.5, color=MUTE, ha="center", va="center")

# ---- footer: downstream role + cost contrast ----------------------
foot.set_xlim(0, 1)
foot.set_ylim(0, 1)
foot.axis("off")

foot.text(0.010, 0.94, "Downstream use", fontsize=10.5, fontweight="bold",
          ha="left", va="top")
boxes = [(r"$\tilde{A}=A+\alpha W^{\mathrm{rough}}$", 0.040, 0.150),
         ("GCN layer 1", 0.243, 0.108), ("GCN layer 2", 0.383, 0.108),
         (r"$\hat{y}$", 0.523, 0.055)]
for label, x, w in boxes:
    fc = TOUCH if "tilde" in label else "white"
    foot.add_patch(FancyBboxPatch((x, 0.36), w, 0.30,
                                  boxstyle="round,pad=0.010,rounding_size=0.03",
                                  fc=fc, ec="#C3C7CC", lw=1.0))
    foot.text(x + w / 2, 0.51, label, fontsize=8.6, ha="center", va="center")
for x0, x1 in [(0.190, 0.243), (0.351, 0.383), (0.491, 0.523)]:
    foot.add_patch(FancyArrowPatch((x0, 0.51), (x1, 0.51), arrowstyle="-|>",
                                   mutation_scale=12, color="#8A8F95", lw=1.2))
foot.text(0.010, 0.16,
          "Changed normalized rows seed multi-layer propagation; see Remark 7.",
          fontsize=8.6, ha="left", va="center", color=MUTE)

foot.plot([0.590, 0.590], [0.06, 0.98], color="#D6D9DD", lw=1.0)

foot.text(0.610, 0.94, "Sparse-kernel cost (excluding bookkeeping)", fontsize=9.5, fontweight="bold",
          ha="left", va="top")
foot.text(0.610, 0.62,
          r"batch rebuild  $O(|V|^2 |B|)$    vs.    incremental  "
          r"$O(|V||B| + Q_{\mathcal{T}}|B|)$",
          fontsize=8.9, ha="left", va="center")
foot.text(0.610, 0.37,
          rf"measured $|\mathcal{{T}}_{{adj}}|$: {_T_CORA:.1f} Cora, "
          rf"{_T_CITE:.1f} CiteSeer, {_T_OGB:.1f} ogbn-arxiv",
          fontsize=8.5, ha="left", va="center", color=MUTE)
foot.text(0.610, 0.15,
          f"end-to-end speedup: {_SPD_CITE}–{_SPD_CORA}$\\times$ (citation) $\\rightarrow$ "
          rf"{_SPD_OGB:,.0f}$\times$ (ogbn-arxiv stream)",
          fontsize=8.5, ha="left", va="center", color=MUTE)

# ---- shared legend ----------------------------------------------
handles = [
    plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec="k", ms=10, label="class-0 node"),
    plt.Line2D([], [], marker="s", ls="", mfc=SAND, mec="k", ms=9, label="class-1 node"),
    plt.Line2D([], [], marker="o", ls="", mfc=CRIMSON, mec="#7A2B2E", ms=11,
               label=r"arriving node $v_{\mathrm{new}}$"),
    plt.Line2D([], [], marker="o", ls="", mfc=GREY_N, mec="white", ms=10,
               label="untouched (not recomputed)"),
    plt.Line2D([], [], marker="o", ls="", mfc="white", mec=CRIMSON, mew=2.2, ms=11,
               label=r"affected node ($\in\mathcal{T}_{adj}$)"),
    plt.Line2D([], [], color=CRIMSON, lw=2.4, label="rough weight recomputed"),
    mpatches.Patch(fc=TOUCH, ec="none", label=r"touch set $\mathcal{T}_{adj}$"),
]
fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8.8,
           frameon=False, bbox_to_anchor=(0.5, 0.028), columnspacing=1.1,
           handletextpad=0.5)

fig.suptitle("Incremental maintenance of NRS structure under a node insertion",
             fontsize=13.5, fontweight="bold", y=0.968)

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_mechanism.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(RESDIR, "fig_mechanism.png"), dpi=200, bbox_inches="tight")

WRAPPER = r"""% Auto-generated by experiments/plot_incremental_mechanism.py -- do not edit by hand
\begin{figure}[pos=!htbp]
\centering
\includegraphics[width=\linewidth]{figures/fig_mechanism.pdf}
\caption{Incremental NRS maintenance under node insertion: batch rebuilding (left), the safety-radius update (centre), and rough-weight updates on edges incident to $\mathcal{T}_{adj}$ (right). Both symmetric weight entries are updated. The maintenance result assumes oracle labels; downstream classification is evaluated separately on label-safe snapshots. See Proposition~\ref{prop:complexity} for costs and Remark~\ref{rem:gnn-zone} for downstream propagation.}
\label{fig:mechanism}
\end{figure}
"""
open(os.path.join(FIGDIR, "fig_mechanism.tex"), "w").write(WRAPPER)
print("[PASS] wrote figures/fig_mechanism.{pdf,tex} and results/fig_mechanism.png")
