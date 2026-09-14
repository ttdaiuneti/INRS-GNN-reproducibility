"""
Verify batch_rough_adjacency_vectorized == batch_rough_adjacency (bit-for-bit).

theory.incremental_rough_adjacency.batch_rough_adjacency_vectorized is the
edge-list NumPy comparator used as the fair batch-W reference in
e2e_fair_timing.py / e2e_scale_timing.py (Table 5, "Batch, fair"). This test
backs the manuscript's "bit-identical" claim on the datasets it is used on:

  - Cora / CiteSeer (binary bag-of-words): produces EXACTLY the same W matrix
    as the pure-Python double loop batch_rough_adjacency (integer distances,
    no rounding), including delta = +inf (psi = 0) entries and the symmetric
    mirror. Table 5's timing runs on these.
  - Real-valued features (Gaussian): agrees up to <1e-12 -- the vectorized
    per-edge distance reduction accumulates in a different order than the
    per-pair np.linalg.norm of the loop, an implementation-rounding gap of
    ~1e-15 (Corollary 1: the algorithm is exact under real arithmetic).

Run: python3 experiments/test_batch_rough_adjacency_vectorized_matches.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid
from theory.incremental_nrs import calc_deltas
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency,
    batch_rough_adjacency_vectorized,
)


def _check(name, init_frac=0.8, seed=0):
    X, y, adj = load_planetoid(name)
    X = np.asarray(X, dtype=np.float64)
    n = int(len(y) * init_frac)
    idx = np.sort(np.random.default_rng(seed).permutation(len(y))[:n])
    Xa, ya, adja = X[idx], y[idx], adj[np.ix_(idx, idx)]
    B_idx = np.arange(X.shape[1])
    deltas = calc_deltas(Xa, ya)

    def _cmp(dv, tag):
        Wl = batch_rough_adjacency(Xa, ya, dv, adja, B_idx)
        Wv = batch_rough_adjacency_vectorized(Xa, ya, dv, adja, B_idx)
        m = float(np.max(np.abs(Wl - Wv)))
        assert m == 0.0, f"{name}/{tag}: max|W_loop - W_vec| = {m:.3e} (want exactly 0)"
        assert np.array_equal(Wv, Wv.T), f"{name}/{tag}: vectorized W not symmetric"
        assert (Wl > 0).sum() == (Wv > 0).sum(), f"{name}/{tag}: nnz mismatch"
        return int((Wv > 0).sum())

    nnz = _cmp(deltas, "plain")
    # explicitly exercise the delta = +inf (psi = 0) branch: force 20 random
    # nodes to +inf and confirm both builders zero out their edges identically
    d_inf = deltas.copy()
    d_inf[np.random.default_rng(seed).choice(n, 20, replace=False)] = np.inf
    nnz_inf = _cmp(d_inf, "with +inf rows")
    assert nnz_inf < nnz, f"{name}: forcing +inf did not drop any edges"
    print(f"OK {name}: n={n} |B|={X.shape[1]} nnz={nnz} -> {nnz_inf} with 20 +inf rows"
          f"  max|W_loop-W_vec|=0 (both cases)")


def _check_realvalued(seed=0, n=1500, d=80, k=5, deg=8):
    """Gaussian features: exact-0 no longer holds (reduction order), so assert
    the implementation-rounding gap stays negligible (<1e-12)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, size=n)
    X = rng.normal(0, 2.0, size=(k, d))[y] + rng.normal(0, 1.0, size=(n, d))
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        js = rng.integers(0, n, size=deg)
        adj[i, js] = True
        adj[js, i] = True
    B_idx = np.arange(d)
    deltas = calc_deltas(X, y)
    Wl = batch_rough_adjacency(X, y, deltas, adj, B_idx)
    Wv = batch_rough_adjacency_vectorized(X, y, deltas, adj, B_idx)
    m = float(np.max(np.abs(Wl - Wv)))
    assert m < 1e-12, f"real-valued: max|W_loop - W_vec| = {m:.3e} (want <1e-12)"
    print(f"OK gaussian: n={n} |B|={d}  max|W_loop-W_vec|={m:.2e} (<1e-12)")


def main():
    for ds in ("cora", "citeseer"):
        _check(ds)
    _check_realvalued()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
