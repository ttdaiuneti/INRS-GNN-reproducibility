"""
Stateful streaming benchmark on ogbn-arxiv (review 2026-09-10, finding M3).

The earlier script (e2e_ogb_scale.py) called incremental_insert K times from the
SAME initial state and timed two kernels in isolation, so its 28 ms figure was a
kernel microbenchmark around a fixed snapshot, not the cost of maintaining state
along a stream. This script maintains real state:

  * delta_B is carried forward (each insertion updates the array the next one reads),
  * W^rough is materialised as a value array parallel to the CSR edge list and
    every incremental update is written into it,
  * the active set grows, and the arriving node's edges become visible through an
    active mask over a graph laid out ONCE (active nodes occupy a contiguous
    prefix, so no per-insertion CSR is rebuilt and no feature block is copied),
  * per-insertion wall-clock covers the delta pass, the W update AND the
    bookkeeping (mask flip, update application), not only the two kernels.

Correctness is re-established against from-scratch recomputation at checkpoints:
delta_B against calc_deltas_blocked, and the full maintained W against
batch_rough_adjacency_edgelist over the whole active subgraph (not only the
entries the incremental step happened to touch).

Batch anchors are measured at the same checkpoints, so the reported ratio
compares like with like.

Output: experiments/results/e2e_ogb_stream_results.json
"""
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_ROOT, "shared"))
sys.path.insert(0, _ROOT)
from core.incremental_nrs import calc_deltas_blocked  # noqa: E402
from core.rough_adjacency import (  # noqa: E402
    batch_rough_adjacency_edgelist,
    incremental_rough_adjacency_update_sparse,
)

OGB_DIR = os.path.join(_ROOT, "data", "ogb")
OUT = os.path.join(_ROOT, "results", "e2e_ogb_stream_results.json")
INIT_FRAC = 0.8
K_INSERTS = 200
CHECKPOINTS = (0, 100, 200)   # insertions completed when a batch anchor is taken
SEED = 0
# Declared tolerance for the checkpoint verification. The update and the
# rebuild evaluate the same real-valued expressions in different orders, so
# agreement is expected to fp64 rounding, not bit-for-bit (Corollary 1).
CHECK_TOL = 1e-9
BLOCK = 2048


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def load_arxiv():
    X = np.load(os.path.join(OGB_DIR, "arxiv_X.npy"))
    y = np.load(os.path.join(OGB_DIR, "arxiv_y.npy"))
    ei = np.load(os.path.join(OGB_DIR, "arxiv_edge_index.npy"))
    n = X.shape[0]
    r = np.concatenate([ei[0], ei[1]])
    c = np.concatenate([ei[1], ei[0]])
    adj = csr_matrix((np.ones(r.size, bool), (r, c)), shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = True
    return np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int64), adj


def edge_pos(indptr, indices, v, u):
    """Index of column u within CSR row v (indices are sorted)."""
    lo, hi = indptr[v], indptr[v + 1]
    k = lo + int(np.searchsorted(indices[lo:hi], u))
    return k if k < hi and indices[k] == u else -1


def edge_pos_strict(indptr, indices, v, u):
    """edge_pos, but a missing entry is a bug in the layout, not a skip.

    Every edge handed to us comes from a submatrix of the laid-out CSR, so it
    must be locatable. Silently ignoring a miss would let a mapping error pass
    as a clean correctness check (review 2026-09-10 round 2, finding r6).
    """
    k = edge_pos(indptr, indices, v, u)
    if k < 0:
        raise AssertionError(f"edge ({v},{u}) not present in the laid-out CSR")
    return k


def main():
    print(f"numpy={np.__version__} scipy={scipy.__version__} "
          f"platform={platform.platform()}", flush=True)
    X, y, adj = load_arxiv()
    n = X.shape[0]
    print(f"ogbn-arxiv: |V|={n} |B|={X.shape[1]} |E|={adj.nnz // 2}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = np.sort(perm[:n_init])
    stream_idx = perm[n_init:n_init + K_INSERTS]

    # One permutation, applied once: active nodes are the contiguous prefix
    # [0, n_active) of Xp/yp, so the delta pass reads a view, never a copy.
    order = np.concatenate([init_idx, stream_idx])
    Xp = np.ascontiguousarray(X[order])
    yp = y[order]
    adj_p = adj[order][:, order].tocsr()
    adj_p.sort_indices()
    n_nodes_p = order.size
    del X, adj

    active = np.zeros(n_nodes_p, dtype=bool)
    active[:n_init] = True
    B_idx = np.arange(Xp.shape[1])

    # ---- initial state: one batch build, as any deployment would pay once ----
    t0 = time.perf_counter()
    deltas = np.full(n_nodes_p, np.inf)
    deltas[:n_init] = calc_deltas_blocked(Xp[:n_init], yp[:n_init], block=BLOCK)
    init_delta_s = time.perf_counter() - t0
    print(f"  initial batch delta (|V|={n_init}): {init_delta_s:.1f} s", flush=True)

    adj_act = adj_p[:n_init][:, :n_init].tocsr()
    t0 = time.perf_counter()
    ev, eu, wv = batch_rough_adjacency_edgelist(Xp[:n_init], yp[:n_init],
                                                deltas[:n_init], adj_act, B_idx)
    init_w_s = time.perf_counter() - t0
    print(f"  initial batch W (|E|={ev.size}): {init_w_s:.2f} s", flush=True)

    # maintained W: one value per CSR entry of the laid-out graph
    w_arr = np.zeros(adj_p.nnz)
    indptr, indices = adj_p.indptr, adj_p.indices
    for a, b, ww in zip(ev.tolist(), eu.tolist(), wv.tolist()):
        for (r_, c_) in ((a, b), (b, a)):
            w_arr[edge_pos_strict(indptr, indices, r_, c_)] = ww

    records, anchors, checks = [], [], []

    def batch_anchor(m, tag):
        """From-scratch rebuild of both quantities at the current active size."""
        t = time.perf_counter()
        d_ref = calc_deltas_blocked(Xp[:m], yp[:m], block=BLOCK)
        b_delta = time.perf_counter() - t
        sub = adj_p[:m][:, :m].tocsr()
        t = time.perf_counter()
        ev2, eu2, w2 = batch_rough_adjacency_edgelist(Xp[:m], yp[:m], d_ref, sub, B_idx)
        b_w = time.perf_counter() - t
        # independent correctness check of the maintained state at this point
        finite = np.isfinite(d_ref) & np.isfinite(deltas[:m])
        d_diff = float(np.max(np.abs(d_ref[finite] - deltas[:m][finite]))) if finite.any() else 0.0
        inf_mismatch = int(np.sum(np.isinf(d_ref) != np.isinf(deltas[:m])))
        w_diff, n_checked = 0.0, 0
        for a, b, ww in zip(ev2.tolist(), eu2.tolist(), w2.tolist()):
            k = edge_pos_strict(indptr, indices, a, b)
            w_diff = max(w_diff, abs(w_arr[k] - ww))
            n_checked += 1

        # --- the checkpoint DECIDES pass/fail; it does not merely record ---
        # (review 2026-09-10 round 2, finding r6)
        assert not np.isnan(d_ref).any(), f"{tag}: NaN in rebuilt radii"
        assert not np.isnan(deltas[:m]).any(), f"{tag}: NaN in maintained radii"
        assert not np.isnan(w_arr).any(), f"{tag}: NaN in maintained W"
        assert inf_mismatch == 0, (
            f"{tag}: {inf_mismatch} nodes disagree on finite/infinite radius")
        if d_diff > CHECK_TOL or w_diff > CHECK_TOL:
            bad = np.flatnonzero(finite & (np.abs(d_ref - deltas[:m]) > CHECK_TOL))
            diagnostic = []
            for idx in bad:
                dist = np.sqrt(np.sum((Xp[:m] - Xp[idx]) ** 2, axis=1))
                dist[yp[:m] == yp[idx]] = np.inf
                enemy_idx = int(np.argmin(dist))
                diagnostic.append(dict(node=int(idx), nearest_enemy=enemy_idx,
                    direct_radius=float(dist[enemy_idx]), gram_radius=float(d_ref[idx]),
                    maintained_radius=float(deltas[idx]),
                    features_equal=bool(np.array_equal(Xp[idx], Xp[enemy_idx]))))
            with open(OUT + '.failure.json', 'w') as failure:
                json.dump(dict(tag=tag, delta_diff=d_diff, weight_diff=w_diff,
                    diagnostic=diagnostic, records=records, anchors=anchors, checks=checks), failure, indent=2)
            print('FAILURE DIAGNOSTICS', diagnostic, flush=True)
        assert d_diff <= CHECK_TOL, f"{tag}: max|dd|={d_diff:.3e} > {CHECK_TOL:.0e}"
        assert w_diff <= CHECK_TOL, f"{tag}: max|dW|={w_diff:.3e} > {CHECK_TOL:.0e}"
        assert n_checked == ev2.size, (
            f"{tag}: verified {n_checked} of {ev2.size} rebuilt edges")
        # "entire maintained W" means the entire undirected support, so check
        # the symmetry invariant too rather than assuming it.
        Wm = csr_matrix((w_arr, indices, indptr), shape=adj_p.shape)
        asym = float(abs(Wm - Wm.T).max()) if Wm.nnz else 0.0
        assert asym == 0.0, f"{tag}: maintained W is asymmetric by {asym:.3e}"
        # Nothing outside the active block may carry weight.
        assert float(abs(Wm[m:]).max() if m < n_nodes_p else 0.0) == 0.0, (
            f"{tag}: nonzero weight on inactive rows")

        anchors.append({"tag": tag, "n_active": int(m),
                        "batch_delta_s": b_delta, "batch_w_s": b_w,
                        "batch_total_s": b_delta + b_w})
        checks.append({"tag": tag, "n_active": int(m),
                       "max_abs_delta_diff": d_diff,
                       "n_inf_mismatch": inf_mismatch,
                       "max_abs_w_diff_full_support": w_diff,
                       "n_edges_checked": n_checked,
                       "n_edges_rebuilt": int(ev2.size),
                       "max_abs_asymmetry": asym,
                       "tolerance": CHECK_TOL,
                       "passed": True})
        print(f"  [anchor {tag}] |V|={m} batch delta {b_delta:.1f}s + W {b_w:.2f}s "
              f"| maintained state: max|dd|={d_diff:.3e}, inf mismatch={inf_mismatch}, "
              f"max|dW| over {n_checked} edges = {w_diff:.3e}", flush=True)

    batch_anchor(n_init, "t=0")

    for t in range(K_INSERTS):
        v_new = n_init + t
        m = v_new                      # active count before this insertion
        t_all = time.perf_counter()

        # --- delta pass: Theorem 1, over the active prefix (a view, no copy) ---
        t0 = time.perf_counter()
        x_new = Xp[v_new]
        diff = Xp[:m] - x_new
        dist_to = np.sqrt(np.sum(diff * diff, axis=1))
        enemy = yp[:m] != yp[v_new]
        closer = dist_to < deltas[:m]
        shrink_mask = enemy & closer
        deltas[:m][shrink_mask] = dist_to[shrink_mask]
        d_new = float(dist_to[enemy].min()) if enemy.any() else np.inf
        deltas[v_new] = d_new
        S_delta = np.flatnonzero(shrink_mask)
        t_delta = time.perf_counter() - t0

        # --- bookkeeping: the node joins the active set ---
        t0 = time.perf_counter()
        active[v_new] = True
        nbrs = indices[indptr[v_new]:indptr[v_new + 1]]
        touch = {v_new} | {int(u) for u in nbrs if active[u]}
        t_book1 = time.perf_counter() - t0

        # --- W update over T_adj, then write it into the maintained array ---
        t0 = time.perf_counter()
        updates, T_adj = incremental_rough_adjacency_update_sparse(
            Xp, yp, deltas, adj_p, B_idx, touch, set(S_delta.tolist()),
            active_mask=active,
        )
        t_w = time.perf_counter() - t0

        t0 = time.perf_counter()
        for (a, b), ww in updates.items():
            k = edge_pos(indptr, indices, a, b)
            if k >= 0:
                w_arr[k] = ww
        t_book2 = time.perf_counter() - t0

        total = time.perf_counter() - t_all
        records.append({
            "step": t, "n_active": int(m + 1),
            "delta_ms": t_delta * 1000, "w_ms": t_w * 1000,
            "bookkeeping_ms": (t_book1 + t_book2) * 1000,
            "total_ms": total * 1000,
            "s_delta": int(S_delta.size), "t_adj": int(len(T_adj)),
            "n_updates": len(updates),
        })
        if (t + 1) in CHECKPOINTS:
            batch_anchor(m + 1, f"t={t + 1}")

    tot = np.array([r["total_ms"] for r in records])
    dl = np.array([r["delta_ms"] for r in records])
    wl = np.array([r["w_ms"] for r in records])
    bk = np.array([r["bookkeeping_ms"] for r in records])
    last = anchors[-1]
    speedup = last["batch_total_s"] * 1000.0 / float(tot.mean())

    out = {
        "dataset": "ogbn-arxiv",
        "design": "stateful stream: state carried forward, W materialised and "
                  "verified against a full from-scratch rebuild at checkpoints",
        "n_total": int(n), "n_init": int(n_init), "k_inserts": K_INSERTS,
        "seed": SEED, "block": BLOCK,
        "initial_build_delta_s": init_delta_s, "initial_build_w_s": init_w_s,
        "per_insertion_ms": {
            "total_mean": float(tot.mean()), "total_std": float(tot.std()),
            "total_median": float(np.median(tot)),
            "total_p90": float(np.percentile(tot, 90)),
            "total_max": float(tot.max()),
            "delta_mean": float(dl.mean()), "w_mean": float(wl.mean()),
            "bookkeeping_mean": float(bk.mean()),
        },
        "touch": {
            "t_adj_mean": float(np.mean([r["t_adj"] for r in records])),
            "s_delta_mean": float(np.mean([r["s_delta"] for r in records])),
            "s_delta_max": int(np.max([r["s_delta"] for r in records])),
        },
        "batch_anchors": anchors,
        "correctness_checks": checks,
        "speedup_end_to_end_stateful": speedup,
        "peak_rss_gb": peak_rss_gb(),
        "per_insertion_records": records,
        "env": {"numpy": np.__version__, "scipy": scipy.__version__,
                "platform": platform.platform()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nper-insertion total: mean {tot.mean():.1f} ms (median {np.median(tot):.1f}, "
          f"p90 {np.percentile(tot, 90):.1f}, max {tot.max():.1f})")
    print(f"  delta {dl.mean():.1f} | W {wl.mean():.1f} | bookkeeping {bk.mean():.1f} ms")
    print(f"batch anchor at end: {last['batch_total_s']:.1f} s  ->  speedup {speedup:.0f}x")
    print(f"peak RSS {peak_rss_gb():.1f} GB")
    print(f"[PASS] {len(checks)} checkpoints verified within tol {CHECK_TOL:.0e} "
          f"(max|dd| {max(c['max_abs_delta_diff'] for c in checks):.2e}, "
          f"max|dW| {max(c['max_abs_w_diff_full_support'] for c in checks):.2e}, "
          f"0 inf mismatches, W symmetric); wrote {OUT}")


if __name__ == "__main__":
    main()
