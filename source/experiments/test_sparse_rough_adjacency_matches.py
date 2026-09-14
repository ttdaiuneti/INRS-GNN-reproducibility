"""
Verify the sparse-adjacency rough-adjacency routines match the dense ones
bit-for-bit on Cora/CiteSeer (binary features).

  - incremental_rough_adjacency_update_sparse (CSR neighbour iteration,
    O(|T_adj|.dbar)) vs incremental_rough_adjacency_update (dense scan,
    O(|T_adj|.|V|)): identical `updates` dict.
  - batch_rough_adjacency_edgelist (COO over CSR upper triangle) vs
    batch_rough_adjacency (dense loop): identical weight on every shared edge.

These are the routines e2e_ogb_scale.py runs at |V| ~ 1.7e5, where a dense
|V| x |V| adjacency / W is infeasible.

Run: python3 experiments/test_sparse_rough_adjacency_matches.py
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid
from theory.incremental_nrs import calc_deltas, incremental_insert
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency,
    batch_rough_adjacency_edgelist,
    incremental_rough_adjacency_update,
    incremental_rough_adjacency_update_sparse,
    structural_touch_set,
)


def _check(name, n_steps=25, seed=0):
    X, y, adj = load_planetoid(name)
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    init, stream = np.sort(perm[:-n_steps]), perm[-n_steps:]

    Xa, ya = X[init].copy(), y[init].copy()
    adj_a = adj[np.ix_(init, init)].copy()
    deltas = calc_deltas(Xa, ya)
    B_idx = np.arange(X.shape[1])
    order = list(init)

    # ---- batch edgelist vs dense, at the init state ----
    Wd = batch_rough_adjacency(Xa, ya, deltas, adj_a, B_idx)
    ev, eu, w = batch_rough_adjacency_edgelist(
        Xa, ya, deltas, csr_matrix(adj_a), B_idx
    )
    assert np.max(np.abs(Wd[ev, eu] - w)) == 0.0, f"{name}: edgelist batch W != dense"
    assert (Wd > 0).sum() == 2 * (w > 0).sum(), f"{name}: edgelist edge count off"

    # ---- incremental: sparse update dict == dense update dict, per step ----
    max_step_diff = 0.0
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
        order.append(gidx)

        up_dense, _ = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_a, B_idx, n_old, touch, S_delta
        )
        up_sparse, _ = incremental_rough_adjacency_update_sparse(
            Xa, ya, deltas, csr_matrix(adj_a), B_idx, touch, S_delta
        )
        assert set(up_dense) == set(up_sparse), (
            f"{name}: sparse/dense update keys differ at node {gidx} "
            f"({len(up_dense)} vs {len(up_sparse)})"
        )
        for k in up_dense:
            max_step_diff = max(max_step_diff, abs(up_dense[k] - up_sparse[k]))

    assert max_step_diff == 0.0, f"{name}: sparse update value diff {max_step_diff:.3e}"
    print(f"OK {name}: n={len(ya)} |B|={X.shape[1]}  edgelist batch W exact, "
          f"sparse inc update exact over {n_steps} inserts")


def main():
    for ds in ("cora", "citeseer"):
        _check(ds)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
