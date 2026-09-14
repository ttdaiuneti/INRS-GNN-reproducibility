"""
OGB-scale point for the |V|-scaling study: incremental NRS maintenance on
ogbn-arxiv (|V| = 169,343, |B| = 128, ~1.17M edges), where per-insertion batch
recomputation is infeasible.

review_opus_round2.md standing concern (and the ASOC significance case): the
regime where O(|V|^2 |B|) batch recomputation actually hurts was never
evaluated. This script:

  - MEASURES the incremental per-insertion cost (delta via one O(|V|.|B|) pass;
    rough-weight via the sparse-adjacency O(|T_adj|.dbar) update) over K
    held-out insertions -- fully feasible.
  - MEASURES one full batch delta recomputation on the 80% active set using the
    blocked Gram kernel (calc_deltas_blocked), and one edge-list batch W, to
    anchor the batch per-insertion cost. A dense |V|x|V| distance matrix
    (2.3e11 fp64 = 1.9 TB) or dense adjacency (2.9e10 bool = 29 GB) is not
    built anywhere.
  - reports the measured speedup and the stream-level projection.

Output: experiments/results/e2e_ogb_scale_results.json
SUPERSEDED by e2e_ogb_stream.py (review 2026-09-10, finding M3): this script
calls incremental_insert K times from the SAME initial state and times two
kernels in isolation, so it measures kernel cost around a fixed snapshot, not
stateful streaming maintenance. Kept for provenance; the manuscript reports
e2e_ogb_stream.py.

Requires the cached arrays produced by `python3 experiments/fetch_data.py`
(shared/data/ogb/arxiv_{X,y,edge_index}.npy).
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theory.incremental_nrs import (
    calc_deltas_blocked,
    incremental_insert,
)
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency_edgelist,
    incremental_rough_adjacency_update_sparse,
    structural_touch_set,
)

OGB_DIR = os.path.join(os.path.dirname(__file__), "..", "shared", "data", "ogb")
INIT_FRAC = 0.8
K_INSERTS = 20
SEED = 0
BLOCK = 2048


def load_arxiv():
    X = np.load(os.path.join(OGB_DIR, "arxiv_X.npy"))
    y = np.load(os.path.join(OGB_DIR, "arxiv_y.npy"))
    ei = np.load(os.path.join(OGB_DIR, "arxiv_edge_index.npy"))
    n = X.shape[0]
    # symmetrise the directed edge list
    r = np.concatenate([ei[0], ei[1]])
    c = np.concatenate([ei[1], ei[0]])
    data = np.ones(r.size, dtype=bool)
    adj = csr_matrix((data, (r, c)), shape=(n, n))
    adj.sum_duplicates()
    adj.data[:] = True
    return np.ascontiguousarray(X, dtype=np.float64), y.astype(np.int64), adj


def main():
    print(f"numpy={np.__version__} scipy={scipy.__version__} "
          f"platform={platform.platform()}", flush=True)
    X, y, adj = load_arxiv()
    n = X.shape[0]
    n_edges = adj.nnz // 2
    print(f"ogbn-arxiv: |V|={n} |B|={X.shape[1]} |E|={n_edges} "
          f"classes={len(np.unique(y))}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = np.sort(perm[:n_init])
    stream_idx = perm[n_init:n_init + K_INSERTS]
    active_mask = np.zeros(n, dtype=bool)
    active_mask[init_idx] = True

    Xa = X[init_idx].copy()
    ya = y[init_idx].copy()

    # ---- batch delta on the active set: ONE blocked call (anchor) ----
    t0 = time.perf_counter()
    deltas_a = calc_deltas_blocked(Xa, ya, block=BLOCK)
    batch_delta_s = time.perf_counter() - t0
    print(f"  batch delta (blocked, |V|={n_init}): {batch_delta_s:.1f} s "
          f"[+inf nodes: {int(np.isinf(deltas_a).sum())}]", flush=True)

    # ---- batch W on the active subgraph: ONE edge-list call (anchor) ----
    adj_init = adj[init_idx][:, init_idx].tocsr()
    t0 = time.perf_counter()
    _ev, _eu, _w = batch_rough_adjacency_edgelist(Xa, ya, deltas_a, adj_init, np.arange(X.shape[1]))
    batch_w_s = time.perf_counter() - t0
    print(f"  batch W (edge-list, |E|={adj_init.nnz // 2}): {batch_w_s*1000:.1f} ms",
          flush=True)

    # ---- incremental per-insertion: delta pass + sparse W update ----
    # The rough-weight update touches only T_adj = {v_new} u nbrs(v_new) u S_delta
    # and, per node, only its true neighbours. We pass the FULL graph CSR with an
    # active_mask so no per-insertion CSR growth is needed; the arriving node's
    # own edges are added as a tiny local CSR block.
    B_idx = np.arange(X.shape[1])
    global_to_local = {int(g): i for i, g in enumerate(init_idx)}
    base_coo = adj_init.tocoo()
    v_new_local = n_init  # arriving node sits at the end of the active block
    inc_delta_ms, inc_w_ms, touch_sizes, sdelta_sizes = [], [], [], []
    spot_max_w_diff = None

    for gidx in stream_idx:
        gidx = int(gidx)
        # delta: O(|V|.|B|) single pass over the active set (adjacency-free)
        t0 = time.perf_counter()
        _, _, deltas_new, _, _ = incremental_insert(Xa, ya, deltas_a, X[gidx], y[gidx])
        inc_delta_ms.append((time.perf_counter() - t0) * 1000)

        S_delta_local = np.flatnonzero(deltas_new[:n_init] != deltas_a)
        sdelta_sizes.append(int(S_delta_local.size))

        g_nbrs = adj.indices[adj.indptr[gidx]:adj.indptr[gidx + 1]]
        nbrs_local = [global_to_local[int(u)] for u in g_nbrs if active_mask[u]]
        touch = structural_touch_set(n_init, nbrs_local)

        Xp = np.vstack([Xa, X[gidx][None, :]])
        yp = np.append(ya, y[gidx])
        add_r = np.asarray(nbrs_local, dtype=np.int64)
        add_c = np.full(add_r.size, v_new_local, dtype=np.int64)
        allr = np.concatenate([base_coo.row, add_r, add_c])
        allc = np.concatenate([base_coo.col, add_c, add_r])
        adj_p = csr_matrix((np.ones(allr.size, bool), (allr, allc)),
                           shape=(n_init + 1, n_init + 1))
        adj_p.data[:] = True

        t0 = time.perf_counter()
        updates, T_adj = incremental_rough_adjacency_update_sparse(
            Xp, yp, deltas_new, adj_p, B_idx, touch, set(S_delta_local.tolist())
        )
        inc_w_ms.append((time.perf_counter() - t0) * 1000)
        touch_sizes.append(len(T_adj))

        # correctness spot check on the first insertion: every incremental W
        # update must equal a from-scratch edge-list rebuild at this state
        if spot_max_w_diff is None:
            ev2, eu2, w2 = batch_rough_adjacency_edgelist(
                Xp, yp, deltas_new, adj_p, B_idx
            )
            wref = {}
            for a, b, ww in zip(ev2.tolist(), eu2.tolist(), w2.tolist()):
                wref[(a, b)] = ww
                wref[(b, a)] = ww
            spot_max_w_diff = 0.0
            for (a, b), ww in updates.items():
                if (a, b) in wref:
                    spot_max_w_diff = max(spot_max_w_diff, abs(wref[(a, b)] - ww))
                elif ww != 0.0:
                    spot_max_w_diff = max(spot_max_w_diff, abs(ww))
            print(f"  [spot check] first insertion: {len(updates)} incremental "
                  f"W updates, max|inc - batch| = {spot_max_w_diff:.3e}", flush=True)

    inc_d = float(np.mean(inc_delta_ms))
    inc_w = float(np.mean(inc_w_ms))
    inc_tot = inc_d + inc_w
    batch_tot_s = batch_delta_s + batch_w_s
    speedup = batch_tot_s * 1000 / inc_tot

    result = {
        "dataset": "ogbn-arxiv",
        "n_total": n, "n_active": n_init, "n_features": int(X.shape[1]),
        "n_edges": int(n_edges),
        "k_inserts": K_INSERTS, "seed": SEED, "block": BLOCK,
        "inc_delta_ms_mean": inc_d, "inc_delta_ms_std": float(np.std(inc_delta_ms)),
        "inc_w_ms_mean": inc_w, "inc_w_ms_std": float(np.std(inc_w_ms)),
        "inc_total_ms": inc_tot,
        "batch_delta_s": batch_delta_s, "batch_w_ms": batch_w_s * 1000,
        "batch_total_s": batch_tot_s,
        "speedup_end_to_end": speedup,
        "touch_mean": float(np.mean(touch_sizes)),
        "s_delta_mean": float(np.mean(sdelta_sizes)),
        "spot_check_max_w_diff": spot_max_w_diff,
        "dense_delta_matrix_gb": n_init * n_init * 8 / 1e9,
        "dense_adj_matrix_gb": n * n / 1e9,
        "env": {"numpy": np.__version__, "scipy": scipy.__version__,
                "platform": platform.platform()},
    }
    out = os.path.join(os.path.dirname(__file__), "results", "e2e_ogb_scale_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in
                      ["inc_delta_ms_mean", "inc_w_ms_mean", "inc_total_ms",
                       "batch_delta_s", "batch_w_ms", "speedup_end_to_end",
                       "touch_mean", "s_delta_mean", "dense_delta_matrix_gb"]},
                     indent=2))
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
