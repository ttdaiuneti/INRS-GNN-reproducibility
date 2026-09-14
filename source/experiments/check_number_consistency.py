"""R3 number-consistency sweep (run from anywhere)

    python3 experiments/check_number_consistency.py

Extracts the text of docs/manuscript/main.pdf and checks that every headline
number in it traces back to a value in an experiment artifact -- not to another
number in the paper. Exits nonzero on any mismatch. Requires pdftotext.

Original docstring: every headline number in the PDF text must
trace to an artifact value. Prints CHECK lines; exits nonzero on mismatch."""
import json, csv, re, sys

import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(ROOT)
R = "experiments/results/"
ogb = json.load(open(R + "e2e_ogb_stream_results.json"))
scale = json.load(open(R + "e2e_scale_timing_results.json"))
diag = json.load(open(R + "e3_label_safe_diagnostics.json"))
acc = json.load(open(R + "e3_pyg_aligned_results.json"))
tim = {r["Dataset"]: r for r in csv.DictReader(open("experiments/timing_summary.csv"))}
wil = list(csv.DictReader(open("experiments/wilcoxon_summary.csv")))
res = {r["Method"]: r for r in csv.DictReader(open("experiments/results_summary.csv"))} \
      if False else None

import subprocess, tempfile
_pdf = os.path.join(ROOT, "docs", "manuscript", "main.pdf")
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as _t:
    _txt = _t.name
subprocess.run(["pdftotext", _pdf, _txt], check=True)
pdf = open(_txt).read()
pi = ogb["per_insertion_ms"]
by = {g["graph"].lower(): g for g in scale["results"]}
dg = {d["dataset"]: d for d in diag["datasets"]}
n_rest = ogb["n_total"] - ogb["n_init"]

def fmt(x, nd=1):
    s = f"{x:,.{nd}f}"
    return s

CHECKS = [
    # (label, string that must appear in the PDF, artifact expression it came from)
    ("OGB mean ms",        f"{pi['total_mean']:.1f}",        "per_insertion_ms.total_mean"),
    ("OGB median ms",      f"{pi['total_median']:.1f}",      "per_insertion_ms.total_median"),
    ("OGB p90 ms",         f"{pi['total_p90']:.1f}",         "per_insertion_ms.total_p90"),
    ("OGB max ms",         f"{pi['total_max']:.1f}",         "per_insertion_ms.total_max"),
    ("OGB delta ms",       f"{pi['delta_mean']:.1f}",        "per_insertion_ms.delta_mean"),
    ("OGB W ms",           f"{pi['w_mean']:.1f}",            "per_insertion_ms.w_mean"),
    ("OGB bookkeep ms",    f"{pi['bookkeeping_mean']:.1f}",  "per_insertion_ms.bookkeeping_mean"),
    ("OGB speedup",        fmt(ogb["speedup_end_to_end_stateful"], 0),
                                                             "speedup_end_to_end_stateful"),
    ("OGB peak RSS",       f"{ogb['peak_rss_gb']:.1f}",      "peak_rss_gb"),
    ("OGB t_adj",          f"{ogb['touch']['t_adj_mean']:.1f}", "touch.t_adj_mean"),
    ("OGB s_delta max",    str(ogb["touch"]["s_delta_max"]),  "touch.s_delta_max"),
    ("OGB n_init",         fmt(ogb["n_init"], 0),            "n_init"),
    ("OGB n_final",        fmt(ogb["n_init"] + ogb["k_inserts"], 0), "n_init + k_inserts"),
    ("OGB n_total",        fmt(ogb["n_total"], 0),           "n_total"),
    ("OGB stream rest",    fmt(n_rest, 0),                   "n_total - n_init"),
    ("OGB anchor last",    f"{ogb['batch_anchors'][-1]['batch_total_s']:.1f}",
                                                             "batch_anchors[-1].batch_total_s"),
    ("proj minutes",       f"{n_rest * pi['total_mean'] / 60000:.1f}",
                                                             "n_rest * mean / 60000"),
    ("proj days",          f"{n_rest * ogb['batch_anchors'][-1]['batch_total_s'] / 86400:.1f}",
                                                             "n_rest * anchor / 86400"),
    ("Cora speedup",       tim["Cora"]["Speedup (fair)"],     "timing_summary Speedup (fair)"),
    ("CiteSeer speedup",   tim["CiteSeer"]["Speedup (fair)"], "timing_summary Speedup (fair)"),
    ("PubMed scale spd",   f"{by['pubmed']['speedup_end_to_end_fair']:.0f}",
                                                             "scale pubmed speedup"),
    ("SBM-25k spd",        f"{by['sbm-25000']['speedup_end_to_end_fair']:.0f}",
                                                             "scale SBM-25000 speedup"),
    ("SBM-6k spd",         f"{by['sbm-6000']['speedup_end_to_end_fair']:.0f}",
                                                             "scale SBM-6000 speedup"),
    ("Cora t_adj",         f"{by['cora']['touch_mean']:.1f}", "scale cora touch_mean"),
    ("CiteSeer t_adj",     f"{by['citeseer']['touch_mean']:.1f}", "scale citeseer touch_mean"),
    ("Cora delta=inf %",   f"{dg['cora']['frac_delta_infinite']*100:.1f}", "diag cora frac_delta_infinite"),
    ("Cite delta=inf %",   f"{dg['citeseer']['frac_delta_infinite']*100:.1f}", "diag citeseer frac_delta_infinite"),
    ("Cora zero-edge %",   f"{dg['cora']['frac_rough_edges_zero']*100:.1f}", "diag cora frac_rough_edges_zero"),
    ("Cite zero-edge %",   f"{dg['citeseer']['frac_rough_edges_zero']*100:.1f}", "diag citeseer frac_rough_edges_zero"),
    ("Cora nonzero edges", str(dg["cora"]["n_rough_edges_nonzero"]), "diag cora n_rough_edges_nonzero"),
    ("Cite nonzero edges", str(dg["citeseer"]["n_rough_edges_nonzero"]), "diag citeseer n_rough_edges_nonzero"),
    ("Cora selfloop psi",  f"{dg['cora']['mean_self_loop_coeff_psi_only']:.2f}", "diag cora selfloop psi"),
    ("Cite selfloop psi",  f"{dg['citeseer']['mean_self_loop_coeff_psi_only']:.2f}", "diag citeseer selfloop psi"),
    ("Cora selfloop plain",f"{dg['cora']['mean_self_loop_coeff_plain']:.2f}", "diag cora selfloop plain"),
    ("Cite selfloop plain",f"{dg['citeseer']['mean_self_loop_coeff_plain']:.2f}", "diag citeseer selfloop plain"),
]
for w in wil:
    CHECKS.append((f"Wilcoxon {w['Comparison'][:12]} {w['Dataset']}",
                   w["Wilcoxon $p$"], "wilcoxon_summary.csv"))

bad = 0
for label, val, src in CHECKS:
    hit = val in pdf
    print(("  OK   " if hit else "  MISS ") + f"{label:24s} {val:>12s}   <- {src}")
    if not hit:
        bad += 1
print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} numbers found verbatim in the PDF text")
sys.exit(1 if bad else 0)
