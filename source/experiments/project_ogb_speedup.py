"""
Tier-1 projection: extrapolate the |V|-scaling laws fitted on the measured
scale study (|V| = 2e3 .. 2e4, both arms measured) to ogbn-arxiv
(|V| = 169,343, |B| = 128), where a per-insertion batch recomputation cannot
be run.

Fits, in log space, on experiments/results/e2e_scale_timing_results.json:
  batch delta (ms)  ~  a * (|V|^2 * |B|)      [O(|V|^2 |B|) all-pairs enemy search]
  incremental delta ~  b * (|V|   * |B|)      [O(|V| |B|)   single pass]
  batch W (ms)      ~  c * (|E|   * |B|)      [O(|E| |B|)   edge-list]
  incremental W     ~  d * (|Tadj| * |V|)     [dense-adjacency variant, Fig 6]

Reports the projected per-insertion times and speedup at arxiv scale, with a
prediction interval from the fit residuals, plus the (fit-free) memory-
infeasibility argument.

Output: experiments/results/ogb_projection.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
SCALE = os.path.join(HERE, "results", "e2e_scale_timing_results.json")

# |V|_active, |B|, mean degree (for |E|) per scale-study graph
META = {
    "cora":      dict(B=1433, deg=3.9),
    "citeseer":  dict(B=3703, deg=2.7),
    "SBM-6000":  dict(B=100,  deg=10.0),
    "SBM-12000": dict(B=100,  deg=10.0),
    "pubmed":    dict(B=500,  deg=4.5),
    "SBM-25000": dict(B=100,  deg=10.0),
}
ARXIV = dict(V=169343, B=128, E=1166243 // 2)  # undirected edge count


def fit_loglog(x, ymeas):
    """ymeas ~ k * x ; fit log k and check the exponent is ~1."""
    lx, ly = np.log(x), np.log(ymeas)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = np.exp(intercept) * x ** slope
    resid = np.log(ymeas) - np.log(pred)
    return slope, float(np.exp(intercept)), float(np.std(resid, ddof=1))


def main():
    d = json.load(open(SCALE))["results"]
    rows = {r["graph"]: r for r in d}

    V, B, E, Tadj = [], [], [], []
    bd, idl, bw, iw = [], [], [], []
    for name, m in META.items():
        r = rows[name]
        v = r["n_active_start"]
        V.append(v); B.append(m["B"]); E.append(m["deg"] * v / 2)
        Tadj.append(r.get("touch_mean", 5.0) if "touch_mean" in r else 5.0)
        bd.append(r["batch_delta_vec_ms"]); idl.append(r["inc_delta_ms"])
        bw.append(r["batch_w_vec_ms"]);     iw.append(r["inc_w_ms"])
    V = np.array(V, float); B = np.array(B, float); E = np.array(E, float)
    Tadj = np.array(Tadj, float)
    bd, idl, bw, iw = map(lambda a: np.array(a, float), (bd, idl, bw, iw))

    fits = {}
    for label, xfeat, ymeas, xproj in [
        ("batch_delta", V**2 * B, bd, ARXIV["V"]**2 * ARXIV["B"]),
        ("inc_delta",   V * B,    idl, ARXIV["V"] * ARXIV["B"]),
        ("batch_W",     E * B,    bw, ARXIV["E"] * ARXIV["B"]),
        ("inc_W_dense", Tadj * V, iw, 5.0 * ARXIV["V"]),
    ]:
        slope, k, sd = fit_loglog(xfeat, ymeas)
        proj = k * xproj ** slope
        lo, hi = proj * np.exp(-1.96 * sd), proj * np.exp(1.96 * sd)
        fits[label] = dict(exponent=round(slope, 3), const=k, resid_sd=round(sd, 3),
                           proj_ms=proj, proj_lo_ms=lo, proj_hi_ms=hi)
        print(f"{label:13} ~ x^{slope:.2f}  (want 1.0)  resid_sd={sd:.2f}  "
              f"-> arxiv {proj:,.1f} ms  [{lo:,.1f}, {hi:,.1f}]")

    inc_tot = fits["inc_delta"]["proj_ms"] + fits["inc_W_dense"]["proj_ms"]
    bat_tot = fits["batch_delta"]["proj_ms"] + fits["batch_W"]["proj_ms"]
    speedup = bat_tot / inc_tot
    # delta-only ratio (the dominant, implementation-clean term)
    d_ratio = fits["batch_delta"]["proj_ms"] / fits["inc_delta"]["proj_ms"]

    mem_dense_delta_gb = ARXIV["V"] ** 2 * 8 / 1e9
    mem_dense_adj_gb = ARXIV["V"] ** 2 / 1e9

    out = dict(
        target=ARXIV, fits=fits,
        projected_inc_total_ms=inc_tot,
        projected_batch_total_ms=bat_tot,
        projected_speedup_end_to_end=speedup,
        projected_delta_only_ratio=d_ratio,
        dense_delta_matrix_gb=mem_dense_delta_gb,
        dense_adj_matrix_gb=mem_dense_adj_gb,
        note=("Batch per-insertion recomputation at this scale also requires a "
              f"{ARXIV['V']}x{ARXIV['V']} distance working set "
              f"({mem_dense_delta_gb:.0f} GB fp64) / dense adjacency "
              f"({mem_dense_adj_gb:.0f} GB), infeasible on commodity hardware "
              "independent of the timing fit."),
    )
    with open(os.path.join(HERE, "results", "ogb_projection.json"), "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"PROJECTED at ogbn-arxiv (|V|={ARXIV['V']:,}, |B|={ARXIV['B']}):")
    print(f"  incremental delta+W : {inc_tot:,.1f} ms / insertion")
    print(f"  batch delta+W       : {bat_tot/1000:,.1f} s / insertion "
          f"({bat_tot:,.0f} ms)")
    print(f"  end-to-end speedup  : {speedup:,.0f}x   (delta-only ratio {d_ratio:,.0f}x)")
    print(f"  + dense delta matrix would need {mem_dense_delta_gb:,.0f} GB (fp64) "
          f"-> batch not runnable per-insertion regardless")
    print(f"\nSaved: {os.path.join(HERE, 'results', 'ogb_projection.json')}")


if __name__ == "__main__":
    main()
