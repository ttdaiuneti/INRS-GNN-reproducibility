"""
E0 — verify incremental δ_B + rough adjacency vs batch (Cora subset).
Granule-size column is a δ-derived consistency check, not a separate test.
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.sparse import vstack
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theory.incremental_nrs import calc_deltas, calc_granule_sizes, incremental_insert
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency,
    incremental_rough_adjacency_update,
    structural_touch_set,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "data", "planetoid")
SEED = 42
N_SUB = 400
TOP_B = 50


def load_cora_subset(n_sub, seed):
    names = ["x", "y", "tx", "ty", "allx", "ally"]
    objs = []
    for name in names:
        with open(os.path.join(DATA_ROOT, f"ind.cora.{name}"), "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally = objs
    test_idx = [
        int(line.strip())
        for line in open(os.path.join(DATA_ROOT, "ind.cora.test.index"))
    ]
    test_idx_range = np.sort(test_idx)
    features = vstack([allx, tx]).tolil()
    labels = np.vstack([ally, ty])
    labels[test_idx, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=-1)
    X = np.array(features.todense())
    with open(os.path.join(DATA_ROOT, "ind.cora.graph"), "rb") as f:
        graph = pickle.load(f, encoding="latin1")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(labels), size=min(n_sub, len(labels)), replace=False)
    X, labels = X[pick], labels[pick]
    old_to_new = {old: i for i, old in enumerate(pick)}
    n = len(pick)
    adj = np.zeros((n, n), dtype=bool)
    for i, old_i in enumerate(pick):
        for old_j in graph.get(old_i, []):
            if old_j in old_to_new:
                j = old_to_new[old_j]
                adj[i, j] = adj[j, i] = True
    return X, labels, adj, pick


def select_top_variance(X, k):
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    return np.argsort(X.var(axis=0))[-k:]


def run_check():
    X, y, adj_full, pick = load_cora_subset(N_SUB, SEED)
    B = select_top_variance(X, TOP_B)
    XB = MinMaxScaler().fit_transform(X[:, B])
    B_idx = np.arange(TOP_B)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_init = int(len(y) * 0.8)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    Xa = XB[init_idx].copy()
    ya = y[init_idx].copy()
    deltas = calc_deltas(Xa, ya)
    order = list(init_idx)
    adj_sub = adj_full[np.ix_(init_idx, init_idx)].copy()

    max_delta_diff = 0.0
    max_granule_diff = 0
    max_w_diff = 0.0

    for idx in stream_idx:
        n_old = len(ya)
        deltas_old = deltas.copy()

        Xa, ya, deltas, _gain, _shrink = incremental_insert(
            Xa, ya, deltas, XB[idx], y[idx]
        )
        S_delta = {i for i in range(n_old) if deltas[i] != deltas_old[i]}

        deltas_batch = calc_deltas(Xa, ya)
        max_delta_diff = max(
            max_delta_diff, float(np.max(np.abs(deltas - deltas_batch)))
        )

        # Granules are a DERIVED quantity: G_B(v) = {u : dist_B(v,u) < delta_B(v)}
        # is recomputed from delta_B, not maintained incrementally (see manuscript
        # Remark "Granules are a derived quantity in our pipeline"). This compares
        # granule sizes derived from the incremental delta vs. from the batch
        # delta -- given max_delta_abs_diff == 0 it is a consistency check, a
        # direct consequence of delta-exactness, not a test of a separate
        # incremental granule-maintenance routine (there is none).
        sizes_inc = calc_granule_sizes(Xa, ya, deltas)
        sizes_batch = calc_granule_sizes(Xa, ya, deltas_batch)
        max_granule_diff = max(
            max_granule_diff, int(np.max(np.abs(sizes_inc - sizes_batch)))
        )

        # structural neighbors of new node in current active set
        neighbors = []
        for j, gidx in enumerate(order):
            if adj_full[idx, gidx]:
                neighbors.append(j)
        touch = structural_touch_set(n_old, neighbors)

        adj_big = np.zeros((n_old + 1, n_old + 1), dtype=bool)
        adj_big[:n_old, :n_old] = adj_sub
        for nb in neighbors:
            adj_big[nb, n_old] = adj_big[n_old, nb] = True
        adj_sub = adj_big
        order.append(idx)

        W_batch = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_idx)
        updates, _T_adj = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_sub, B_idx, n_old, touch, S_delta
        )
        for (v, u), w in updates.items():
            max_w_diff = max(max_w_diff, abs(W_batch[v, u] - w))

    result = {
        "n_sub": N_SUB,
        "n_stream": len(stream_idx),
        "max_delta_abs_diff": max_delta_diff,
        "max_granule_size_diff": max_granule_diff,
        "max_rough_weight_diff": max_w_diff,
        "delta_pass": max_delta_diff < 1e-9,
        "granule_pass": max_granule_diff == 0,
        "adj_pass": max_w_diff < 1e-9,
        "all_pass": (
            max_delta_diff < 1e-9
            and max_granule_diff == 0
            and max_w_diff < 1e-9
        ),
    }
    out = os.path.join(os.path.dirname(__file__), "e0_check_results.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_check()
