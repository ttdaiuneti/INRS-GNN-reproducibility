"""
E1 -- Full-scale incremental-vs-batch correctness check.

Fixes MAJOR-E1 / MAJOR-T3 from review_opus_round1.md: the only artifact
previously claiming full-graph W/logit parity (e0_gnn_results.json) compared
an oracle-delta incremental W against a label-safe-delta batch W -- two
different delta definitions -- which is why it read max_diff=1.0 rather than
testing anything. This script uses ONE delta definition throughout (oracle,
matching Theorem 1-3's actual hypotheses -- Assumption 4a in
docs/proofs/formal_framework.tex), the FULL node count and FULL feature
dimension B (not the 400-node / |B|=50 subgraph in theory/e0_check_results.json),
and compares the ENTIRE final n x n W matrix -- not just entries in the
touched set T_adj -- against an independent full batch rebuild. This is what
directly tests Theorem 3 / Corollary 1 at the scale the Abstract claims it
("bit-exact ... maximum weight error < 1e-7") for.

See docs/manuscript/sections/experiments.tex Table 3 (tab:correctness) and
Abstract. Label-safe delta (Assumption 4b, used for downstream GCN
classification in e0_gnn_prototype.py / e3_pyg_aligned_benchmark.py) is a
separate, distinct claim (Remark 1) not tested here.
"""
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.data import load_planetoid
from core.incremental_nrs import calc_deltas, incremental_insert
from core.rough_adjacency import (
    batch_rough_adjacency,
    incremental_rough_adjacency_update,
    structural_touch_set,
)

SEED = 42
INIT_FRAC = 0.8
DATASETS = ["cora", "citeseer"]


def run_dataset(name):
    X, y, adj = load_planetoid(name)
    n = X.shape[0]
    B_idx = np.arange(X.shape[1])  # full feature dimension, no top-k subsampling
    XB = X[:, B_idx]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    Xa = XB[init_idx].copy()
    ya = y[init_idx].copy()
    deltas = calc_deltas(Xa, ya)
    order = list(init_idx)
    adj_sub = adj[np.ix_(init_idx, init_idx)].copy()
    W = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_idx)

    t0 = time.time()
    for gidx in stream_idx:
        n_old = len(ya)
        deltas_old = deltas.copy()
        Xa, ya, deltas, _, _ = incremental_insert(Xa, ya, deltas, XB[gidx], y[gidx])
        S_delta = {i for i in range(n_old) if deltas[i] != deltas_old[i]}
        neighbors = [j for j, o in enumerate(order) if adj[gidx, o]]
        touch = structural_touch_set(n_old, neighbors)

        adj_big = np.zeros((n_old + 1, n_old + 1), dtype=bool)
        adj_big[:n_old, :n_old] = adj_sub
        for nb in neighbors:
            adj_big[nb, n_old] = adj_big[n_old, nb] = True
        adj_sub = adj_big

        W_big = np.zeros((n_old + 1, n_old + 1))
        W_big[:n_old, :n_old] = W
        W = W_big
        order.append(gidx)

        updates, T_adj = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_sub, B_idx, n_old, touch, S_delta
        )
        for (v, u), w in updates.items():
            W[v, u] = w
    stream_time = time.time() - t0

    # Independent full rebuild from the identical final state, same oracle
    # delta definition -- this is a fresh calc_deltas() call, not a re-use of
    # `deltas` from the streaming loop, so it genuinely re-derives from scratch.
    t0 = time.time()
    deltas_batch = calc_deltas(Xa, ya)
    W_batch = batch_rough_adjacency(Xa, ya, deltas_batch, adj_sub, B_idx)
    batch_time = time.time() - t0

    max_delta_diff = float(np.max(np.abs(deltas - deltas_batch)))
    # Entire matrix, including entries NEVER touched during streaming -- this
    # is precisely what Theorem 3 claims and what a touched-only diff cannot
    # catch (an untouched-entry bug would not show up there).
    max_w_diff_full = float(np.max(np.abs(W - W_batch)))

    return {
        "dataset": name,
        "n_nodes_final": len(ya),
        "n_features_B": len(B_idx),
        "n_stream_steps": len(stream_idx),
        "delta_definition": "oracle (Theorem 1-3 hypotheses, Assumption 4a)",
        "max_delta_diff_inc_vs_batch": max_delta_diff,
        "max_W_diff_inc_vs_batch_full_matrix": max_w_diff_full,
        "stream_time_s": stream_time,
        "final_batch_rebuild_time_s": batch_time,
        "note": (
            "Compares the entire n x n W matrix (all pairs), not only "
            "entries in the touched set T_adj, directly testing Theorem 3 / "
            "Corollary 1 at full graph scale with a single consistent "
            "(oracle) delta definition throughout."
        ),
    }


def main():
    results = []
    for ds in DATASETS:
        print(f"=== {ds} ===", flush=True)
        r = run_dataset(ds)
        print(
            f"  n={r['n_nodes_final']} |B|={r['n_features_B']} "
            f"max_delta_diff={r['max_delta_diff_inc_vs_batch']:.3e} "
            f"max_W_diff_full={r['max_W_diff_inc_vs_batch_full_matrix']:.3e}",
            flush=True,
        )
        results.append(r)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e1_full_scale_correctness.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
