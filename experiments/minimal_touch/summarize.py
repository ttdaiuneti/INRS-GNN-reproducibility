"""Build the manuscript tables, macros and the Figure 3 wrapper from the stored results.

Outputs (paper/):
  tables/tab_timing.tex            citation streams (minimal touch set)
  tables/tab_scale_breakdown.tex   fixed-state scaling + ogbn-arxiv 200-arrival stream
  tables/tab_ogb_robustness.tex    3 x 1000-arrival ogbn-arxiv streams, both policies
  tables/tab_locality_stress.tex   ring/star stress, both policies (paired)
  tables/experiment_numbers.tex    prose macros
  figures/fig_scale_speedup.tex   figure wrapper (plot_scale_speedup.py draws the PDF)
Every number comes from experiments/minimal_touch/results/. Usage:
  python3 experiments/minimal_touch/summarize.py
"""
import csv
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
MAN = ROOT / "paper"
TAB = MAN / "tables"
BUILD = ROOT / "experiments" / "build_tables_from_csv.py"
CSV = RES / "tables"
CSV.mkdir(exist_ok=True)
FEATS = {"cora": 1433, "citeseer": 3703, "pubmed": 500}
LABEL = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def table(name, header, rows, caption, label, notes=(), star=False, widths=None):
    path = CSV / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    cmd = [sys.executable, str(BUILD), "--csv", str(path), "--output", str(TAB / f"{name}.tex"),
           "--caption", caption, "--label", label]
    for n in notes:
        cmd += ["--note", n]
    if star:
        cmd.append("--star")
    if widths:
        cmd += ["--col-widths", ",".join(str(w) for w in widths)]
    subprocess.run(cmd, check=True)


def comma(x):
    return f"{x:,.0f}".replace(",", "{,}")


def sci(x):
    """1.1e-14 -> 1.1\\times10^{-14} (math mode content)."""
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def cell(x):
    """Thousands-separated number as a math cell (the table builder escapes plain braces)."""
    return "$" + comma(x) + "$"


macros = {}

# ---- citation streams (Table: tab_timing) --------------------------------
cit = json.load(open(RES / "citation_fair_minimal_pyg.json"))["results"]
rows = []
for r in cit:
    inc = r["inc_w_ms"][0] + r["inc_delta_ms"][0]
    bf = r["batch_w_vectorized_ms"][0] + r["batch_delta_vectorized_ms"][0]
    bn = r["batch_w_naive_ms"][0] + r["batch_delta_naive_ms"][0]
    rows.append([LABEL[r["dataset"]], f"{r['touch_mean']:.2f}", f"{inc:.2f}", f"{bf:.1f}",
                 f"{r['speedup_end_to_end_fair']:.0f}", f"{bn:.1f}", f"{r['speedup_end_to_end_naive_reference']:.0f}"])
    key = LABEL[r["dataset"]]
    macros[f"Cit{key}Speedup"] = f"{r['speedup_end_to_end_fair']:.0f}"
    macros[f"Cit{key}IncMs"] = f"{inc:.1f}"
    macros[f"Cit{key}Touch"] = f"{r['touch_mean']:.2f}"
    macros[f"Cit{key}NaiveSpeedup"] = comma(r["speedup_end_to_end_naive_reference"])
table("tab_timing",
      ["Dataset", "$|\\mathcal{T}|$", "Inc. (ms)", "Batch, vect. (ms)", "Speedup", "Batch, loop (ms)", "Speedup (loop)"],
      rows,
      "Time per arrival on the citation streams. Means over five seeds; batch times at six sampled steps per seed.",
      "tab:timing",
      ["Speedups are means of per-seed ratios of batch to incremental time, not ratios of the printed means. "
       "$|\\mathcal{T}|$ is the mean touch-set size per arrival."])

# ---- fixed-state scaling + OGB 200-arrival stream (tab_scale_breakdown) --
sc = json.load(open(RES / "scale_minimal.json"))["results"]
o2 = json.load(open(RES / "ogb_stream200_minimal.json"))
rows, pts = [], []
for r in sc:
    g = r["graph"]
    lab = LABEL.get(g, g.replace("000", "k"))
    B = FEATS.get(g, 100)
    rows.append([lab, cell(r["n_active_start"]), B, f"{r['touch_mean']:.2f}", f"{r['inc_delta_ms']:.2f}",
                 f"{r['inc_w_ms']:.2f}", f"{r['batch_delta_vec_ms']:.1f}", f"{r['batch_w_vec_ms']:.1f}",
                 f"${r['speedup_end_to_end_fair']:.0f} \\pm {r['speedup_std']:.0f}$"])
    macros["Scale" + {"SBM-6000": "SBMSixK", "SBM-12000": "SBMTwelveK", "SBM-25000": "SBMTwentyFiveK"}.get(g, lab)] = f"{r['speedup_end_to_end_fair']:.0f}"
a2 = o2["batch_anchors"][-1]
pm = o2["per_insertion_ms"]
rows.append(["ogbn-arxiv$^{a}$", cell(a2["n_active"]), 128, f"{o2['touch']['t_adj_mean']:.2f}",
             f"{pm['delta_mean']:.2f}", f"{pm['w_mean']:.2f}", cell(a2["batch_delta_s"] * 1000),
             f"{a2['batch_w_s'] * 1000:.1f}", f"${comma(o2['speedup_end_to_end_stateful'])}$"])
macros.update(OgbShortSpeedup=comma(o2["speedup_end_to_end_stateful"]), OgbShortIncMs=f"{pm['total_mean']:.1f}",
              OgbShortBatchS=f"{a2['batch_total_s']:.1f}", OgbShortActive=comma(o2["n_init"]), OgbShortActiveEnd=comma(a2["n_active"]),
              OgbShortPeakGB=f"{o2['peak_rss_gb']:.1f}")
table("tab_scale_breakdown",
      ["Graph", "$|V|$", "$|B|$", "$|\\mathcal{T}|$", "Inc. $\\delta$", "Inc. $W$", "Batch $\\delta$", "Batch $W$", "Speedup"],
      rows,
      "Cost per arrival at fixed state (ms) for Figure~\\ref{fig:scale-speedup}. $|V|$: active nodes; $|\\mathcal{T}|$: mean touch-set size.",
      "tab:scale-breakdown",
      ["Citation and SBM graphs: eight arrivals per seed, three seeds; speedup is the mean $\\pm$ SD of per-seed ratios.",
       "$^{a}$ One stream of 200 arrivals; incremental columns are means per arrival and batch columns the final rebuild."],
      star=True)

# ---- OGB 3 x 1000 streams, both policies (tab_ogb_robustness) ------------
rows, means, ratios, allS, allQ, worst = [], {}, {}, [], [], {}
for pol in ["conservative", "minimal"]:
    means[pol], ratios[pol] = [], []
    worst[pol] = None
    for s in [0, 1, 2]:
        d = json.load(open(RES / f"ogb_{pol}" / f"seed_{s}.json"))
        rec = d["per_insertion_records"]
        lat = np.array([x["total_ms"] for x in rec])
        S = np.array([x["s_delta"] for x in rec])
        T = np.array([x["t_adj"] for x in rec])
        assert all(c["passed"] for c in d["correctness_checks"])
        b = d["batch_anchors"][-1]["batch_total_s"]
        ratio = b * 1000 / lat.mean()
        means[pol].append(lat.mean())
        ratios[pol].append(ratio)
        w = max(rec, key=lambda x: x["total_ms"])
        if worst[pol] is None or w["total_ms"] > worst[pol]["total_ms"]:
            worst[pol] = w
        if pol == "minimal":
            allS += S.tolist()
            allQ += (np.array([x["n_updates"] for x in rec]) / 2).tolist()
            dbar = 2 * d["correctness_checks"][-1]["n_edges_checked"] / d["correctness_checks"][-1]["n_active"]
            macros["OgbDbarSeed" + "ABC"[s]] = f"{dbar:.1f}"
        rows.append(["Extended" if pol == "conservative" else "Minimal", s, f"{lat.mean():.1f}", f"{np.median(lat):.1f}",
                     f"{np.percentile(lat, 90):.1f}", f"{lat.max():.1f}", f"{S.mean():.2f}", f"{T.mean():.1f}",
                     f"{b:.1f}", cell(ratio)])
table("tab_ogb_robustness",
      ["Touch set", "Seed", "Mean", "Median", "p90", "Max", "$|S_\\delta|$", "$|\\mathcal{T}|$", "Batch (s)", "Ratio"],
      rows,
      "Three ogbn-arxiv streams of 1,000 arrivals with the extended touch set $\\{v_{\\mathrm{new}}\\} \\cup N(v_{\\mathrm{new}}) \\cup S_\\delta$ and the minimal set of Definition~\\ref{def:touch}. Latencies in ms.",
      "tab:ogb-robustness",
      ["Each seed fixes the 80\\% initial set and the arrival order; both touch sets replay the same stream. "
       "Ratio divides the final rebuild time by the mean latency. $|S_\\delta|$ and $|\\mathcal{T}|$ are means per arrival."],
      star=True)
allS, allQ = np.array(allS), np.array(allQ)
mm, rm = np.array(means["minimal"]), np.array(ratios["minimal"])
mc, rc = np.array(means["conservative"]), np.array(ratios["conservative"])
macros.update(
    OgbRobustMean=f"{mm.mean():.1f}", OgbRobustSD=f"{mm.std():.1f}", OgbRobustRatio=comma(rm.mean()),
    OgbRobustRatioSD=comma(rm.std()), OgbExtMean=f"{mc.mean():.1f}", OgbExtRatio=comma(rc.mean()),
    OgbRobustWorstMs=f"{worst['minimal']['total_ms']:.0f}", OgbExtWorstSec=f"{worst['conservative']['total_ms'] / 1000:.2f}",
    OgbExtWorstTouch=comma(worst["conservative"]["t_adj"]), OgbExtWorstShrink=str(worst["conservative"]["s_delta"]),
    OgbShrinkMean=f"{allS.mean():.2f}", OgbShrinkSE=f"{allS.std(ddof=1) / np.sqrt(len(allS)):.2f}",
    OgbShrinkMax=str(int(allS.max())), OgbEdgeWorkMean=f"{allQ.mean():.1f}",
    OgbEdgeWorkSE=f"{allQ.std(ddof=1) / np.sqrt(len(allQ)):.1f}", OgbEdgeWorkMedian=f"{np.median(allQ):.0f}",
    OgbEdgeWorkMax=comma(allQ.max()), OgbArrivals=comma(len(allS)))

# edge visits Q_j and the per-arrival bound 2*dbar_j of Theorem 4 (q_visits.py)
qv = json.load(open(RES / "q_visits.json"))
assert all(qv[s]["distinct_mismatches"] == 0 for s in qv)
qQ = np.concatenate([[r["Q"] for r in qv[s]["records"]] for s in qv])
qB = np.concatenate([[2 * r["dbar"] for r in qv[s]["records"]] for s in qv])
assert len(qQ) == len(allS)
macros.update(
    OgbQMean=f"{qQ.mean():.1f}", OgbQSE=f"{qQ.std(ddof=1) / np.sqrt(len(qQ)):.1f}",
    OgbQMedian=f"{np.median(qQ):.0f}", OgbQMax=comma(qQ.max()),
    OgbQMeanNoMax=f"{np.delete(qQ, qQ.argmax()).mean():.1f}",
    OgbTwoDbarMean=f"{qB.mean():.1f}", OgbTwoDbarMin=f"{qB.min():.1f}", OgbTwoDbarMax=f"{qB.max():.1f}")

# downstream numbers quoted in the text of Section 4.4 (raw sources: e3_label_safe_diagnostics.json,
# results_summary.csv, wilcoxon_summary.csv, full_pipeline_controls/results/summary.json, label_safe_stream/results)
EXPD = ROOT / "experiments"
dg = {d["dataset"]: d for d in json.load(open(EXPD / "results" / "e3_label_safe_diagnostics.json"))["datasets"]}
cdrop = {}
with open(EXPD / "results_summary.csv") as f:
    rs = {r["Method"]: r for r in csv.DictReader(f)}
for ds, col in (("cora", "Cora"), ("citeseer", "CiteSeer")):
    plain = float(rs["GCN-plain"][col].split("$")[0]); psi = float(rs["INRS-psi-only"][col].split("$")[0])
    cdrop[ds] = plain - psi
with open(EXPD / "wilcoxon_summary.csv") as f:
    wr = list(csv.DictReader(f))
fp = json.load(open(EXPD / "full_pipeline_controls" / "results" / "summary.json"))["comparisons"]
assert len(fp) == 8 and all(c["ci95_low"] <= 0 <= c["ci95_high"] for c in fp)
fpd = {c["dataset"]: c["difference_pp"] for c in fp if c["control"] == "plain"}
ctrl = max(abs(c["difference_pp"]) for c in fp if c["control"] != "plain")
fs = [json.load(open(q)) for q in sorted((EXPD / "label_safe_stream" / "results").glob("*_[0-9].json"))]
fs_checks = sum(len(j["records"]) for j in fs)
fs_wmax = max(x["weight_error"] for j in fs for x in j["records"])
macros.update(
    UntrustedCora=f"{100 * (1 - dg['cora']['trusted_fraction']):.1f}", UntrustedCiteSeer=f"{100 * (1 - dg['citeseer']['trusted_fraction']):.1f}",
    PsiDropCora=f"{cdrop['cora']:.1f}", PsiDropCiteSeer=f"{cdrop['citeseer']:.1f}",
    HybridMaxChange=f"{max(abs(float(r[chr(77) + 'ean $' + chr(92) + 'Delta$ (pp)'])) for r in wr):.2f}",
    FullNrsCora=f"{fpd['cora']:+.2f}".replace("-", "$-$"), FullNrsCiteSeer=f"{fpd['citeseer']:+.2f}".replace("-", "$-$"),
    FullCtrlMax=f"{ctrl:.2f}", FixedChecks=comma(fs_checks), FixedWeightMax=sci(fs_wmax))

old_cit = json.load(open(RES / "citation_fair_minimal.json"))["results"]
macros.update(CitCoraSpeedupRawLoader=f"{next(r for r in old_cit if r['dataset'] == 'cora')['speedup_end_to_end_fair']:.0f}")
zr = json.load(open(RES / "zero_radius.json"))
macros.update(ZeroRadiusCora=str(zr["cora"]["zero"]), ZeroRadiusCiteSeer=str(zr["citeseer"]["zero"]),
              CiteSeerNodes=comma(zr["citeseer"]["n"]))
# edge-case suite (experiments/test_edge_cases.py)
ec = json.load(open(RES / "edge_cases.json"))
macros.update(EdgeNamedCases=str(ec["named_cases"]), EdgeRandomStreams=str(ec["random_streams"]),
              EdgeSteps=comma(ec["totals"]["steps"]), EdgeTies=comma(ec["totals"]["tie"]),
              EdgeZeroDist=comma(ec["totals"]["zero_dist"]), EdgeInfFinite=comma(ec["totals"]["inf_to_finite"]),
              EdgeBothInT=comma(ec["totals"]["both_in_T"]), EdgeShrunk=comma(ec["totals"]["shrink"]))

# per-stream check of Theorem 4 with Q split into own degree and shrink part (q_split.py)
qs = json.load(open(RES / "q_split.json"))
mn = lambda s: f"{s['mean']:.2f}"
rows = []
for s in sorted(qs, key=int):
    r = qs[s]
    d = r["dbar_mean"]
    rows.append([str(int(s)), mn(r["shrink"]), mn(r["own"]), mn(r["shrink_part"]), f"{d:.2f}", mn(r["Q"]),
                 f"{r['Q']['median']:.1f}", f"{int(r['Q']['max']):,}", f"{2 * d:.2f}"])
table("tab_random_order",
      ["Seed", "$|S_j|$", "$\\deg_j(z)$", "Shrink part", "$\\bar d_j$", "$Q_j$", "Median $Q_j$", "Max. $Q_j$", "$2\\bar d_j$"],
      rows,
      "Shrink sets and edge visits on the three ogbn-arxiv streams, means per arrival unless stated.",
      "tab:random-order",
      ["$Q_j = \\deg_j(z) + \\sum_{v\\in S_j}\\deg_j(v)$, where the sum is the shrink part. Theorem~\\ref{thm:random-order} gives, in expectation, $|S_j| \\le 1$, $\\deg_j(z) = \\bar d_j$ and a second term of at most $\\bar d_j$. "
       "$\\bar d_j$ is averaged over the arrivals of the stream. The arrivals of one stream are not independent, so the table reports no standard errors."])
macros.update(OgbShrinkStreamMin=f"{min(qs[s]['shrink']['mean'] for s in qs):.2f}",
              OgbShrinkStreamMax=f"{max(qs[s]['shrink']['mean'] for s in qs):.2f}")

# ---- stress study, both policies (tab_locality_stress) -------------------
P = json.loads((ROOT / "experiments" / "locality_stress" / "protocol.json").read_text())


def stress(pol):
    rows_ = list(csv.DictReader((RES / f"stress_{pol}" / "raw_results.csv").open()))
    assert all(r["correctness_pass"] == "True" for r in rows_)
    out = {}
    for n in P["sizes"]:
        for c in P["scenarios"]:
            g = [r for r in rows_ if r["scenario"] == c and int(r["n_old"]) == n]
            inc, rat = [], []
            for s in P["seeds"]:
                a = [r for r in g if int(r["seed"]) == s]
                i = st.median(float(r["inc_total_ms"]) for r in a)
                b = st.median(float(r["batch_total_ms"]) for r in a)
                inc.append(i)
                rat.append(b / i)
            out[c, n] = dict(touch=int(g[0]["touch"]), q=int(g[0]["q_touch"]), shrink=int(g[0]["shrink"]),
                             inc=st.mean(inc), ratio=st.mean(rat), ratio_sd=st.pstdev(rat), cases=len(g))
    return out


so, sm = stress("conservative"), stress("minimal")
names = {"ring_control": "Ring, control", "ring_enemy": "Ring, enemy", "star_leaf": "Star, leaf", "star_hub": "Star, hub"}
rows = []
for n in P["sizes"]:
    for c in P["scenarios"]:
        a, b = so[c, n], sm[c, n]
        rows.append([names[c], n, a["shrink"], f"{a['touch']} / {b['touch']}", f"{a['q']} / {b['q']}",
                     f"{a['inc']:.2f} / {b['inc']:.2f}", f"{a['ratio']:.2f} / {b['ratio']:.2f}"])
table("tab_locality_stress",
      ["Case", "$n$", "$|S_\\delta|$", "$|\\mathcal{T}|$", "$Q_{\\mathcal{T}}$", "Inc. (ms)", "Batch/inc."],
      rows,
      "Single adverse arrivals on ring and star graphs with $n$ nodes. Each cell gives extended / minimal touch set; ratios below one mean the rebuild was faster.",
      "tab:locality-stress",
      ["Five seeds and five repetitions per case; each seed uses the median over repetitions, and the table gives means over seeds. One CPU thread."],
      star=True)
macros.update(StressCases=str(len(P["sizes"]) * len(P["scenarios"]) * len(P["seeds"])), StressRepeats=str(P["repetitions"]),
              StressSeeds=str(len(P["seeds"])), StressMinN=comma(P["sizes"][0]), StressMaxN=comma(P["sizes"][-1]))
for tag, d in (("Ext", so), ("Min", sm)):
    for n, nm in zip(P["sizes"], ["Small", "Mid", "Large"]):
        macros[f"Stress{tag}Enemy{nm}"] = f"{d['ring_enemy', n]['ratio']:.2f}"
        macros[f"Stress{tag}Hub{nm}"] = f"{d['star_hub', n]['ratio']:.2f}"
        macros[f"Stress{tag}HubMs{nm}"] = f"{d['star_hub', n]['inc']:.2f}"
        macros[f"Stress{tag}LeafMs{nm}"] = f"{d['star_leaf', n]['inc']:.2f}"

# ---- granule membership check (if present) --------------------------------
gpath = RES / "granule_check_pyg.json"
if gpath.exists():
    g = json.load(open(gpath))
    cit_steps = sum(r["steps"] for r in g["citation"])
    macros.update(GranArrivals=comma(cit_steps),
                  GranEvicted=comma(sum(r["members_evicted"] for r in g["citation"])),
                  GranGains=comma(sum(r["granule_gains"] for r in g["citation"])),
                  GranRingEvicted=comma(max(r["members_evicted"] for r in g["ring_enemy"])),
                  GranMismatch=str(sum(r["granule_mismatches"] + r["radius_mismatches"] for v in g.values() for r in v)))

# ---- datasets (tab_datasets), written directly because of \cite cells ----
(TAB / "tab_datasets.tex").write_text(r"""% Auto-generated by experiments/minimal_touch/summarize.py -- do not edit by hand
\begin{table*}[pos=htbp]
\centering
\small
\caption{Graphs used in the experiments. $|B|$ is the feature dimension; edges are undirected.}
\label{tab:datasets}
\begin{tabular}{lrrrlll}
\toprule
Graph & $|V|$ & $|B|$ & Classes & Edges & Source & Used in \\
\midrule
Cora & 2{,}708 & 1433 & 7 & 5{,}278 & Planetoid~\cite{fey2019fast} & Sections~4.2--4.4 \\
CiteSeer & 3{,}327 & 3703 & 6 & 4{,}552 & Planetoid~\cite{fey2019fast} & Sections~4.2--4.4 \\
PubMed & 19{,}717 & 500 & 3 & 44{,}324 & Planetoid~\cite{fey2019fast} & Section~4.3 \\
SBM & 6k, 12k, 25k & 100 & 5 & $\bar d \approx 10$ & synthetic & Section~4.3 \\
Ring, star & 257--2{,}049 & 16 & 2 & $n$ or $n-1$ & synthetic & Sections~4.2, 4.3 \\
ogbn-arxiv & 169{,}343 & 128 & 40 & 1{,}157{,}799 & OGB~\cite{hu2020ogb} & Sections~4.2, 4.3 \\
\bottomrule
\end{tabular}
\end{table*}
""")

# ---- exactness summary (tab_correctness) ---------------------------------
rows = []
fpath = RES / "fullcheck_minimal_pyg.json"
if fpath.exists():
    fc = json.load(open(fpath))
    for ds in ["cora", "citeseer"]:
        g = [r for r in fc if r["dataset"] == ds]
        rows.append([f"{LABEL[ds]}, full stream", f"{len(g)} seeds, all pairs", "radii, $W$ (all pairs)",
                     f"${max(r['max_delta_diff_inc_vs_batch'] for r in g):.0f}$ / ${max(r['max_W_diff_inc_vs_batch_full_matrix'] for r in g):.0f}$", "0"])
    macros["FullCheckSeeds"] = str(len(fc) // 2)
gpath = RES / "granule_check_pyg.json"
if gpath.exists():
    gc = json.load(open(gpath))
    for ds in ["cora", "citeseer"]:
        g = [r for r in gc["citation"] if r["dataset"] == ds]
        rows.append([f"{LABEL[ds]}, full stream", f"{len(g)} seeds, every arrival", "radii, granule members", "$0$ / --",
                     str(sum(r["granule_mismatches"] + r["radius_mismatches"] for r in g))])
    rows.append(["Ring, enemy arrival", f"{len(gc['ring_enemy'])} cases", "radii, granule members", "$0$ / --",
                 str(sum(r["granule_mismatches"] + r["radius_mismatches"] for r in gc["ring_enemy"]))])
dd = ww = 0.0
nck = ninf = 0
for s_ in [0, 1, 2]:
    for c in json.load(open(RES / "ogb_minimal" / f"seed_{s_}.json"))["correctness_checks"]:
        dd, ww = max(dd, c["max_abs_delta_diff"]), max(ww, c["max_abs_w_diff_full_support"])
        nck += 1
        ninf += c["n_inf_mismatch"]
rows.append(["ogbn-arxiv, 3 streams", f"{nck} checkpoints", "radii, $W$ (all edges)", f"${sci(dd)}$ / ${sci(ww)}$", str(ninf)])
macros.update(OgbChecks=str(nck), OgbMaxDelta=sci(dd), OgbMaxW=sci(ww))
srows = list(csv.DictReader((RES / "stress_minimal" / "raw_results.csv").open()))
sd_ = max(max(float(r["delta_error"]), float(r["batch_delta_error"])) for r in srows)
sw_ = max(max(float(r["weight_error"]), float(r["batch_weight_error"])) for r in srows)
rows.append(["Ring and star", f"{len(srows) // P['repetitions']} cases", "radii, $W$ (all edges)", f"${sci(sd_)}$ / ${sci(sw_)}$", "0"])
table("tab_correctness", ["Graph", "Checks", "Compared", "Max. difference (radius / weight)", "Mismatches"],
      rows,
      "Incremental versus batch results with the minimal touch set.",
      "tab:correctness",
      ["Mismatches count nodes whose radius is finite in one result and infinite in the other, or whose granule member set differs. "
       "Cora and CiteSeer use direct-difference distances on both sides; the ogbn-arxiv and stress rebuilds use Gram-matrix distances."],
      star=True, widths=[1.08, 1.32, 0.98, 0.62])

# ---- break-even estimate (Discussion) -------------------------------------
# cost of one edge evaluation in the sparse incremental kernel, from the ring
# enemy cases (edge work Q_T = n+2 dominates there), and the number of edge
# evaluations that would cost as much as one ogbn-arxiv rebuild.
_wm = [json.load(open(RES / "ogb_minimal" / f"seed_{k}.json")) for k in range(3)]
ce_ogb = sum(sum(x["w_ms"] for x in d_["per_insertion_records"]) for d_ in _wm) / sum(sum(x["n_updates"] / 2 for x in d_["per_insertion_records"]) for d_ in _wm)
macros["OgbEdgeEvalUs"] = f"{ce_ogb * 1000:.1f}"
ce = st.median(float(r["inc_w_ms"]) / int(r["q_touch"]) for r in srows if r["scenario"] == "ring_enemy")
ogb_batch_ms = st.mean(json.load(open(RES / "ogb_minimal" / f"seed_{k}.json"))["batch_anchors"][-1]["batch_total_s"] for k in range(3)) * 1000
two_e = 2 * json.load(open(RES / "ogb_minimal" / "seed_0.json"))["correctness_checks"][-1]["n_edges_checked"]
_re = [r for r in srows if r["scenario"] == "ring_enemy" and int(r["n_old"]) == P_min_n] if (P_min_n := min(int(r["n_old"]) for r in srows)) else []
macros.update(StressRingRebuildEvals=f"{st.median(float(r['batch_total_ms']) for r in _re) / ce:.0f}", StressRingRebuildMs=f"{st.median(float(r['batch_total_ms']) for r in _re):.2f}")
macros.update(EdgeEvalUs=f"{ce * 1000:.1f}", OgbBreakEvenM=f"{ogb_batch_ms / ce_ogb / 1e6:.0f}", OgbBreakEvenRatio=f"{ogb_batch_ms / ce_ogb / two_e:.0f}", OgbTwoEM=f"{two_e / 1e6:.1f}")

# ---- trusted-label stream table (tab_fixed_trust): label_safe_stream/summarize.py with this caption and path ----
_src = (ROOT / "experiments" / "label_safe_stream" / "summarize.py").read_text()
for _old, _new in [
    ("M=R/'docs/manuscript'", "M=R/'paper'"),
    ("\\begin{table}[!htbp]", "\\begin{table}[pos=htbp]"),
    ("\\caption{Label-safe insertion with immutable trusted assignments. Each seed runs 100 consecutive arrivals; all active radii and the full rough-edge support are checked after every arrival. The final column compares pre-softmax outputs of the same fixed GCN under incremental and batch hybrid operators ($\\alpha=1$). No new accuracy or timing claim is made.}",
     "\\caption{Streams of 100 arrivals with trusted labels fixed at arrival, checked against a rebuild after every arrival. $\\Delta W$: largest weight difference; $\\Delta Z$: largest logit difference of a fixed GCN.}"),
    ("{s['dataset'].capitalize()}", "{dict(cora='Cora', citeseer='CiteSeer')[s['dataset']]}"),
]:
    assert _src.count(_old) == 1, _old[:40]
    _src = _src.replace(_old, _new)
exec(compile(_src, str(ROOT / "experiments" / "label_safe_stream" / "summarize.py"), "exec"),
     {"__file__": str(ROOT / "experiments" / "label_safe_stream" / "summarize.py"), "__name__": "lss_summarize"})

# ---- Figure 3 wrapper (the PDF is drawn by plot_scale_speedup.py on these results) ----
sbm = [r for r in sc if r["graph"].startswith("SBM")]
slope = np.polyfit(np.log([r["n_active_start"] for r in sbm]), np.log([r["speedup_end_to_end_fair"] for r in sbm]), 1)[0]
macros["ScaleSBMSlope"] = f"{slope:.2f}"
(MAN / "figures" / "fig_scale_speedup.tex").write_text(
    "% Auto-generated by experiments/minimal_touch/summarize.py -- do not edit by hand\n"
    "% Source: experiments/minimal_touch/results/scale_minimal.json, ogb_stream200_minimal.json\n"
    "\\begin{figure}[pos=htbp]\n\\centering\n\\includegraphics[width=0.72\\linewidth]{figures/fig_scale_speedup.pdf}\n"
    "\\caption{Speedup over a vectorized rebuild against the number of active nodes (log--log axes). "
    "Points are means over three seeds, except ogbn-arxiv (one stream); the dashed line is a fit to the SBM graphs. Details are in Table~\\ref{tab:scale-breakdown}.}\n"
    "\\label{fig:scale-speedup}\n\\end{figure}\n")

text = "% Auto-generated by experiments/minimal_touch/summarize.py -- do not edit by hand\n"
text += "% Source: experiments/minimal_touch/results/\n"
for k, v in macros.items():
    assert k.isalpha(), k
    text += "\\newcommand{\\" + k + "}{" + v + "}\n"
(TAB / "experiment_numbers.tex").write_text(text)
print(text)
