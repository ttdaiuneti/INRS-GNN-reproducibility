"""
Scale study: does the fair end-to-end incremental-vs-batch speedup grow with
|V|, as Proposition 4 predicts (incremental delta O(|V|.|B|) vs batch delta
O(|V|^2.|B|))?  review_opus_round2.md standing concern: "the regime where
|V|^2 recomputation actually hurts is exactly the regime not evaluated."

For each graph we take an ~|V|-node active state and perform K single-node
insertions, timing per insertion:
  - incremental:  incremental_insert (delta) + incremental_rough_adjacency_update
                  (W, touch set only)
  - fair batch:   calc_deltas_vectorized (one BLAS GEMM) + batch_rough_adjacency_
                  vectorized (edge-list NumPy) -- both bit-compatible with the
                  loop implementations, matched to the incremental side's kernels

The naive Python-loop batch comparator is NOT run here (O(|V|^2) per call is
prohibitive at |V|=25k); e2e_fair_timing.py covers it on Cora/CiteSeer.

Graphs: Cora, CiteSeer, PubMed (real Planetoid), plus synthetic SBM graphs at
|V| in {6k, 12k, 25k} (5 classes, 100 Gaussian features, mean degree ~10).

Output: experiments/results/e2e_scale_timing_results.json
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid_pyg
from theory.incremental_nrs import (
    calc_deltas,
    calc_deltas_vectorized,
    incremental_insert,
)
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency,
    batch_rough_adjacency_vectorized,
    incremental_rough_adjacency_update,
    structural_touch_set,
)

REAL = ["cora", "citeseer", "pubmed"]
SBM_SIZES = [6000, 12000, 25000]
SBM_CLASSES = 5
SBM_FEATURES = 100
SBM_MEAN_DEGREE = 10
SEEDS = [0, 1, 2]
K_INSERTS = 8
INIT_FRAC = 0.8


def make_sbm(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, SBM_CLASSES, size=n)
    centers = rng.normal(0, 2.0, size=(SBM_CLASSES, SBM_FEATURES))
    X = centers[y] + rng.normal(0, 1.0, size=(n, SBM_FEATURES))
    # sparse SBM-ish adjacency: each node draws ~mean_degree neighbours,
    # 80% from its own class, 20% global-random.
    adj = np.zeros((n, n), dtype=bool)
    idx_by_class = [np.where(y == c)[0] for c in range(SBM_CLASSES)]
    half = SBM_MEAN_DEGREE // 2
    for i in range(n):
        same = idx_by_class[y[i]]
        n_same = max(1, int(round(half * 0.8)))
        n_glob = max(1, half - n_same)
        js = np.concatenate([
            rng.choice(same, size=n_same, replace=True),
            rng.integers(0, n, size=n_glob),
        ])
        js = js[js != i]
        adj[i, js] = True
        adj[js, i] = True
    return X.astype(np.float64), y.astype(np.int64), adj


def load_real(name):
    X, y, adj, *_ = load_planetoid_pyg(name)
    return np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int64), adj


def spot_check_correctness(X, y, adj, seed, k=6):
    """One incremental-vs-batch W comparison at scale-study size.

    e1_full_scale_correctness.py verifies max|W_inc - W_batch| = 0 over the
    entire matrix on Cora/CiteSeer; Theorem 3 makes this size-independent, but
    a reviewer will still ask whether the update holds at |V| ~ 10^4 with
    real-valued features. This streams k insertions on an SBM graph (Gaussian
    features) while maintaining W incrementally and asserts exact agreement
    with a full rebuild from the same final state. The batch reference is
    batch_rough_adjacency (the same per-pair np.linalg.norm distance kernel
    the incremental update uses), matching e1's methodology.
    """
    n = X.shape[0]
    B_idx = np.arange(X.shape[1])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init = np.sort(perm[:n_init])
    stream = perm[n_init:n_init + k]

    Xa, ya = X[init].copy(), y[init].copy()
    adj_a = np.ascontiguousarray(adj[np.ix_(init, init)])
    deltas = calc_deltas(Xa, ya)
    W = batch_rough_adjacency(Xa, ya, deltas, adj_a, B_idx)
    order = list(init)

    for gidx in stream:
        n_old = len(ya)
        deltas_old = deltas
        Xa, ya, deltas, _, _ = incremental_insert(Xa, ya, deltas, X[gidx], y[gidx])
        S_delta = set(np.flatnonzero(deltas[:n_old] != deltas_old).tolist())
        neighbors = [j for j, o in enumerate(order) if adj[gidx, o]]
        touch = structural_touch_set(n_old, neighbors)

        adj_big = np.zeros((n_old + 1, n_old + 1), dtype=bool)
        adj_big[:n_old, :n_old] = adj_a
        for nb in neighbors:
            adj_big[nb, n_old] = adj_big[n_old, nb] = True
        adj_a = adj_big
        W_big = np.zeros((n_old + 1, n_old + 1))
        W_big[:n_old, :n_old] = W
        W = W_big
        order.append(gidx)

        updates, _ = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_a, B_idx, n_old, touch, S_delta
        )
        for (v, u), w in updates.items():
            W[v, u] = w

    W_batch = batch_rough_adjacency(Xa, ya, calc_deltas(Xa, ya), adj_a, B_idx)
    max_w = float(np.max(np.abs(W - W_batch)))
    assert max_w == 0.0, f"spot check FAILED: max|W_inc - W_batch| = {max_w:.3e}"
    print(f"  [spot check] SBM |V|={len(ya)} |B|={X.shape[1]}: "
          f"max|W_inc - W_batch| = 0 over {W.shape[0]}x{W.shape[1]} matrix", flush=True)


def time_one(X, y, adj, seed):
    n = X.shape[0]
    B_idx = np.arange(X.shape[1])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = np.sort(perm[:n_init])
    stream_idx = perm[n_init:n_init + K_INSERTS]

    Xa = X[init_idx].copy()
    ya = y[init_idx].copy()
    adj_a = np.ascontiguousarray(adj[np.ix_(init_idx, init_idx)])
    deltas = calc_deltas(Xa, ya)

    inc_d, inc_w, bat_d, bat_w, touch_n, sdelta_n = [], [], [], [], [], []
    for gidx in stream_idx:
        n_old = len(ya)
        deltas_old = deltas
        # --- incremental delta ---
        t0 = time.perf_counter()
        Xn, yn, deltas_n, _, _ = incremental_insert(Xa, ya, deltas, X[gidx], y[gidx])
        inc_d.append(time.perf_counter() - t0)

        S_delta = set(np.flatnonzero(deltas_n[:n_old] != deltas_old).tolist())
        neighbors = [j for j in range(n_old) if adj[init_idx[j], gidx]] if n_old else []
        touch = structural_touch_set(n_old, neighbors)
        touch_n.append(len(touch | S_delta))
        sdelta_n.append(len(S_delta))

        adj_big = np.zeros((n_old + 1, n_old + 1), dtype=bool)
        adj_big[:n_old, :n_old] = adj_a
        for nb in neighbors:
            adj_big[nb, n_old] = adj_big[n_old, nb] = True

        # --- incremental W update (touch set only) ---
        t0 = time.perf_counter()
        incremental_rough_adjacency_update(
            Xn, yn, deltas_n, adj_big, B_idx, n_old, touch, S_delta
        )
        inc_w.append(time.perf_counter() - t0)

        # --- fair batch: delta (gram GEMM) + W (edge-list NumPy) at the
        #     post-insertion state ---
        t0 = time.perf_counter()
        db = calc_deltas_vectorized(Xn, yn)
        bat_d.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        batch_rough_adjacency_vectorized(Xn, yn, db, adj_big, B_idx)
        bat_w.append(time.perf_counter() - t0)

        # advance the active state by the insertion so |V| tracks the stream
        Xa, ya, deltas = Xn, yn, deltas_n
        adj_a = adj_big
        init_idx = np.append(init_idx, gidx)

    def ms(a):
        return float(np.mean(a) * 1000)

    inc_total = np.array(inc_d) + np.array(inc_w)
    bat_total = np.array(bat_d) + np.array(bat_w)
    return {
        "seed": seed,
        "inc_delta_ms": ms(inc_d),
        "inc_w_ms": ms(inc_w),
        "batch_delta_vec_ms": ms(bat_d),
        "batch_w_vec_ms": ms(bat_w),
        "speedup_end_to_end_fair": float(np.mean(bat_total / inc_total)),
        "touch_mean": float(np.mean(touch_n)),
        "s_delta_mean": float(np.mean(sdelta_n)),
    }


def summarize(label, n, per_seed):
    def col(k):
        return np.array([r[k] for r in per_seed])
    return {
        "graph": label,
        "n_active_start": n,
        "n_seeds": len(per_seed),
        "inc_delta_ms": float(np.mean(col("inc_delta_ms"))),
        "inc_w_ms": float(np.mean(col("inc_w_ms"))),
        "batch_delta_vec_ms": float(np.mean(col("batch_delta_vec_ms"))),
        "batch_w_vec_ms": float(np.mean(col("batch_w_vec_ms"))),
        "speedup_end_to_end_fair": float(np.mean(col("speedup_end_to_end_fair"))),
        "speedup_std": float(np.std(col("speedup_end_to_end_fair"))),
        "touch_mean": float(np.mean(col("touch_mean"))),
        "s_delta_mean": float(np.mean(col("s_delta_mean"))),
        "per_seed": per_seed,
    }


def main():
    print(f"numpy={np.__version__} scipy={scipy.__version__} sklearn={sklearn.__version__}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    results = []

    # correctness spot check at scale-study size before any timing
    Xc, yc, adjc = make_sbm(SBM_SIZES[0], 0)
    spot_check_correctness(Xc, yc, adjc, 0)
    del Xc, adjc

    for name in REAL:
        X, y, adj = load_real(name)
        n0 = int(X.shape[0] * INIT_FRAC)
        per = [time_one(X, y, adj, s) for s in SEEDS]
        agg = summarize(name, n0, per)
        results.append(agg)
        print(f"  {name:10s} n~{n0:6d}: fair {agg['speedup_end_to_end_fair']:7.1f}x  "
              f"(inc d={agg['inc_delta_ms']:.2f} w={agg['inc_w_ms']:.2f} | "
              f"batch d={agg['batch_delta_vec_ms']:.1f} w={agg['batch_w_vec_ms']:.1f} ms)", flush=True)

    for n in SBM_SIZES:
        per = []
        for s in SEEDS:
            X, y, adj = make_sbm(n, s)
            per.append(time_one(X, y, adj, s))
            del X, adj
        n0 = int(n * INIT_FRAC)
        agg = summarize(f"SBM-{n}", n0, per)
        results.append(agg)
        print(f"  SBM-{n:<6d} n~{n0:6d}: fair {agg['speedup_end_to_end_fair']:7.1f}x  "
              f"(inc d={agg['inc_delta_ms']:.2f} w={agg['inc_w_ms']:.2f} | "
              f"batch d={agg['batch_delta_vec_ms']:.1f} w={agg['batch_w_vec_ms']:.1f} ms)", flush=True)

    out = os.path.join(os.path.dirname(__file__), "results", "e2e_scale_timing_results.json")
    with open(out, "w") as f:
        json.dump({
            "env": {"numpy": np.__version__, "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__, "platform": platform.platform()},
            "config": {"seeds": SEEDS, "k_inserts": K_INSERTS, "init_frac": INIT_FRAC,
                       "sbm_classes": SBM_CLASSES, "sbm_features": SBM_FEATURES,
                       "sbm_mean_degree": SBM_MEAN_DEGREE},
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
