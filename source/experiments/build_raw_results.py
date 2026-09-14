#!/usr/bin/env python3
"""
Build raw_results.csv (single source of truth, per-seed long format) and
summary CSVs for table/figure generation, from e3_pyg_aligned_results.json.

Per table_figure_skill.md: no number in a manuscript table/figure should be
typed by hand — everything traces back to raw_results.csv here.

Usage: python3 experiments/build_raw_results.py
Outputs:
  experiments/raw_results.csv          — per-seed, long format (source of truth)
  experiments/results_summary.csv      — mean±std accuracy table (wide)
  experiments/timing_summary.csv       — per-insertion timing table
  experiments/ablation_summary.csv     — psi-only vs hybrid vs plain vs RG
  experiments/accuracy_parity_plot.csv — long format for the parity figure
"""
import csv
import json
import os

import numpy as np
from scipy import stats

RESULTS = os.path.join(os.path.dirname(__file__), "results", "e3_pyg_aligned_results.json")
TIMING_RESULTS = os.path.join(os.path.dirname(__file__), "results", "e2e_fair_timing_results.json")
OUT_DIR = os.path.dirname(__file__)

METHOD_LABELS = {
    "GCN_plain": "GCN-plain",
    "RG_GCN_static": "RG-GCN-static (simplified)",
    "INRS_psi_only": "INRS-psi-only",
    "INRS_hybrid_grid": "INRS-hybrid-grid",
    "INRS_hybrid_learned": "INRS-hybrid-learned",
}
METHODS_ORDER = ["GCN_plain", "RG_GCN_static", "INRS_psi_only", "INRS_hybrid_grid", "INRS_hybrid_learned"]
DATASET_LABELS = {"cora": "Cora", "citeseer": "CiteSeer"}


def fmt_p(p):
    """Report p-values without printing a rounded zero (review 2026-09-10, m5)."""
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def main():
    with open(RESULTS) as f:
        d = json.load(f)

    # ---- raw_results.csv: per-seed, long format, source of truth ----
    raw_rows = []
    for ds in d["datasets"]:
        name = ds["dataset"]
        for run in ds["seed_runs"]:
            seed = run["seed"]
            for method in METHODS_ORDER:
                if method not in run:
                    continue
                m = run[method]
                raw_rows.append({
                    "dataset": DATASET_LABELS.get(name, name),
                    "method": METHOD_LABELS[method],
                    "seed": seed,
                    "test_acc": f"{m['test_acc']:.4f}",
                    "val_acc": f"{m['val_acc']:.4f}",
                    "alpha": m.get("alpha", ""),
                })

    with open(os.path.join(OUT_DIR, "raw_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "seed", "test_acc", "val_acc", "alpha"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"[PASS] Wrote raw_results.csv ({len(raw_rows)} rows)")

    # ---- results_summary.csv: wide, mean±std, for tab_accuracy.tex ----
    summary_rows = []
    for method in METHODS_ORDER:
        row = {"Method": METHOD_LABELS[method]}
        for ds in d["datasets"]:
            name = DATASET_LABELS.get(ds["dataset"], ds["dataset"])
            agg = ds["aggregate"][method]
            row[name] = f"{agg['test_mean']*100:.2f} $\\pm$ {agg['test_std']*100:.2f}"
        summary_rows.append(row)

    ds_names = [DATASET_LABELS.get(ds["dataset"], ds["dataset"]) for ds in d["datasets"]]
    with open(os.path.join(OUT_DIR, "results_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method"] + ds_names)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[PASS] Wrote results_summary.csv ({len(summary_rows)} rows)")

    # ---- ablation_summary.csv: same data, ablation framing (psi-only emphasized) ----
    with open(os.path.join(OUT_DIR, "ablation_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Variant"] + ds_names)
        w.writeheader()
        for method in ["GCN_plain", "INRS_psi_only", "INRS_hybrid_grid"]:
            row = {"Variant": METHOD_LABELS[method]}
            for ds in d["datasets"]:
                name = DATASET_LABELS.get(ds["dataset"], ds["dataset"])
                agg = ds["aggregate"][method]
                row[name] = f"{agg['test_mean']*100:.2f} $\\pm$ {agg['test_std']*100:.2f}"
            w.writerow(row)
    print("[PASS] Wrote ablation_summary.csv")

    # ---- accuracy_parity_plot.csv: long format for the parity figure ----
    with open(os.path.join(OUT_DIR, "accuracy_parity_plot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "test_mean", "test_std"])
        w.writeheader()
        for ds in d["datasets"]:
            name = DATASET_LABELS.get(ds["dataset"], ds["dataset"])
            for method in ["GCN_plain", "INRS_hybrid_grid", "INRS_hybrid_learned"]:
                agg = ds["aggregate"][method]
                w.writerow({
                    "dataset": name,
                    "method": METHOD_LABELS[method],
                    "test_mean": f"{agg['test_mean']*100:.3f}",
                    "test_std": f"{agg['test_std']*100:.3f}",
                })
    print("[PASS] Wrote accuracy_parity_plot.csv")

    # ---- timing_summary.csv: end-to-end per-insertion incremental vs batch
    # rebuild (delta + W together), from e2e_fair_timing_results.json.
    # review_opus_round2.md MAJOR-E2-R2: the headline is now the ratio against
    # a *matched vectorized* batch (BLAS-GEMM delta + edge-list NumPy W); the
    # naive-loop ratio is retained only as an explicitly-labelled
    # constant-factor reference. review_opus_round2.md MINOR-E13: the Speedup
    # columns are means of per-seed ratios, not the quotient of the printed
    # mean-time columns (stated in the table note). ----
    with open(TIMING_RESULTS) as f:
        dt = json.load(f)["results"]
    fieldnames = [
        "Dataset", "Inc. (ms)", "Batch, fair (ms)", "Speedup (fair)",
        "Batch, naive-loop (ms)", "Speedup (naive-loop)",
    ]
    with open(os.path.join(OUT_DIR, "timing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in dt:
            name = DATASET_LABELS.get(row["dataset"], row["dataset"])
            inc_ms = row["inc_w_ms"][0] + row["inc_delta_ms"][0]
            batch_fair_ms = row["batch_w_vectorized_ms"][0] + row["batch_delta_vectorized_ms"][0]
            batch_naive_ms = row["batch_w_naive_ms"][0] + row["batch_delta_naive_ms"][0]
            w.writerow({
                "Dataset": name,
                "Inc. (ms)": f"{inc_ms:.2f}",
                "Batch, fair (ms)": f"{batch_fair_ms:.1f}",
                "Speedup (fair)": f"{row['speedup_end_to_end_fair']:.0f}",
                "Batch, naive-loop (ms)": f"{batch_naive_ms:.1f}",
                "Speedup (naive-loop)": f"{row['speedup_end_to_end_naive_reference']:.0f}",
            })
    print("[PASS] Wrote timing_summary.csv")

    # ---- timing_plot.csv: long format for the speedup figure (fair batch) ----
    with open(os.path.join(OUT_DIR, "timing_plot.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_x", "time_ms", "method"])
        w.writeheader()
        for i, row in enumerate(dt):
            inc_ms = row["inc_w_ms"][0] + row["inc_delta_ms"][0]
            batch_fair_ms = row["batch_w_vectorized_ms"][0] + row["batch_delta_vectorized_ms"][0]
            w.writerow({"dataset_x": i, "time_ms": f"{inc_ms:.4f}", "method": "Incremental delta+W"})
            w.writerow({"dataset_x": i, "time_ms": f"{batch_fair_ms:.4f}", "method": "Batch delta+W rebuild"})
    print("[PASS] Wrote timing_plot.csv")

    # ---- wilcoxon_summary.csv: paired signed-rank test, hybrid vs plain,
    # plus mean difference, 95% CI, effect size, and a TOST equivalence test
    # against a pre-stated +/-1pp margin (review_opus_round1.md MAJOR-E4:
    # at the old n=5, the minimum attainable two-sided Wilcoxon p is 0.0625,
    # so p>=0.05 alone cannot support a parity claim -- it is now reported
    # alongside CI/effect-size/TOST, not in place of them). ----
    # +/-1 percentage point equivalence margin. NOT pre-registered: the frozen
    # protocol (experiments/protocol.yaml) records a 5-seed Wilcoxon test; the
    # margin and the 10-seed re-run were both adopted at revision, after the
    # first-round results had been seen. Reported as exploratory.
    TOST_MARGIN = 0.01
    with open(os.path.join(OUT_DIR, "wilcoxon_summary.csv"), "w", newline="") as f:
        fieldnames = [
            "Comparison", "Dataset", "N", "Mean $\\Delta$ (pp)", "95% CI (pp)",
            "Cohen's $d_z$", "Wilcoxon $p$", "TOST $p$ ($\\pm$1pp)",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for hybrid_method, row_label in [
            ("INRS_hybrid_grid", "Grid $\\alpha$ vs.\\ plain"),
            ("INRS_hybrid_learned", "Learned $\\alpha$ vs.\\ plain"),
        ]:
            for ds in d["datasets"]:
                name = DATASET_LABELS.get(ds["dataset"], ds["dataset"])
                plain = np.array([r["GCN_plain"]["test_acc"] for r in ds["seed_runs"]])
                hyb = np.array([r[hybrid_method]["test_acc"] for r in ds["seed_runs"]])
                diff = hyb - plain
                n = len(diff)
                mean_diff = float(np.mean(diff))
                sd = float(np.std(diff, ddof=1))
                se = sd / np.sqrt(n) if sd > 0 else 0.0
                df = n - 1
                t_crit = stats.t.ppf(0.975, df)
                ci_lo = mean_diff - t_crit * se
                ci_hi = mean_diff + t_crit * se
                cohens_d = mean_diff / sd if sd > 0 else float("nan")
                # Test accuracy on these splits lives on a 1/1000 grid (1000
                # test nodes), but the JSON stores float32-derived values, so
                # exact ties are lost to representation noise and the signed-rank
                # test sees spurious non-zero differences. Snap to the grid
                # first -- and subtract on the INTEGER grid (counts of correct
                # test nodes), not on two rounded floats: np.round(x,3) is not
                # exactly representable, so differences that are equal in
                # correct-count can differ in the last bit and get ranked apart.
                # (review 2026-09-10 round 2, finding r2: this moved Cora
                # learned-alpha from p=0.28590 to p=0.28547.)
                n_test = int(ds["mask_info"]["test"])  # Planetoid public split
                diff_grid = (np.rint(hyb * n_test).astype(np.int64)
                             - np.rint(plain * n_test).astype(np.int64))
                if np.allclose(diff_grid, 0.0):
                    wil_p = 1.0
                else:
                    wil_p = stats.wilcoxon(diff_grid).pvalue

                # TOST: two one-sided t-tests against +/-TOST_MARGIN. Reject
                # both nulls (H0: diff <= -margin; H0: diff >= +margin) at
                # alpha=0.05 => conclude equivalence within the margin.
                if se > 0:
                    t_lo = (mean_diff - (-TOST_MARGIN)) / se
                    t_hi = (mean_diff - TOST_MARGIN) / se
                    p_lo = 1 - stats.t.cdf(t_lo, df)  # H1: diff > -margin
                    p_hi = stats.t.cdf(t_hi, df)      # H1: diff < +margin
                    tost_p = max(p_lo, p_hi)
                else:
                    tost_p = 0.0 if abs(mean_diff) < TOST_MARGIN else 1.0

                w.writerow({
                    "Comparison": row_label,
                    "Dataset": name,
                    "N": n,
                    "Mean $\\Delta$ (pp)": f"{mean_diff * 100:.2f}",
                    "95% CI (pp)": f"[{ci_lo * 100:.2f}, {ci_hi * 100:.2f}]",
                    "Cohen's $d_z$": f"{cohens_d:.2f}",
                    "Wilcoxon $p$": fmt_p(wil_p),
                    "TOST $p$ ($\\pm$1pp)": fmt_p(tost_p),
                })
    print("[PASS] Wrote wilcoxon_summary.csv (mean diff, 95% CI, Cohen's d, Wilcoxon p, TOST p vs +/-1pp)")


if __name__ == "__main__":
    main()
