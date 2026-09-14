"""
Verification for the fair batch-δ comparator (review_opus_round2.md MAJOR-E2-R2).

Round 2 found that the end-to-end speedup was measured against a batch path
whose δ recomputation was an unvectorized Python row loop
(theory.incremental_nrs.calc_deltas_rowwise) accounting for 94–99 % of the
batch cost. theory.incremental_nrs.calc_deltas_vectorized is the fair,
BLAS-backed replacement used as the batch-δ comparator in
e2e_fair_timing.py. This test checks two things:

  1. calc_deltas_vectorized (BLAS GEMM, timing-only comparator) reproduces
     calc_deltas_rowwise up to Gram-identity rounding (max abs diff < 1e-6)
     with an identical +∞ pattern, on the full Cora and CiteSeer matrices.
  2. The vectorized incremental_insert gain/shrink update is element-for-element
     identical to the original per-node loop, using the same direct-difference
     distance kernel (so incremental vs. batch δ stay bit-exact, Corollary 1).

Run: python3 experiments/test_batch_delta_vectorized_matches.py
"""
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid
from theory.incremental_nrs import (
    calc_deltas_rowwise,
    calc_deltas_vectorized,
    incremental_insert,
)

TOL = 1e-6


def _check_delta_comparator(name):
    X, y, _ = load_planetoid(name)
    X = np.asarray(X, dtype=np.float64)
    d_ref = calc_deltas_rowwise(X, y)           # direct-difference (naive ref + correctness path)
    d_vec = calc_deltas_vectorized(X, y)        # one BLAS GEMM (timing-only comparator)

    inf_ref, inf_d = np.isinf(d_ref), np.isinf(d_vec)
    assert np.array_equal(inf_ref, inf_d), (
        f"{name}: +inf pattern differs ({inf_ref.sum()} vs {inf_d.sum()})"
    )
    finite = ~inf_ref
    max_abs = float(np.max(np.abs(d_ref[finite] - d_vec[finite])))
    assert max_abs < TOL, f"{name}: max|Δδ| {max_abs:.2e} >= {TOL:.0e}"
    print(f"OK {name}: n={len(y)} |B|={X.shape[1]} calc_deltas_vectorized (GEMM) "
          f"vs direct-diff max|Δδ|={max_abs:.2e} inf_nodes={int(inf_ref.sum())}")


def _loop_reference(X, y, deltas, x_new, y_new):
    """Pre-vectorization gain/shrink/new_deltas computation with the same
    direct-difference distance kernel incremental_insert uses, so this
    isolates the gain/shrink masking logic."""
    n = len(y)
    diff = X - x_new
    dist_to = np.sqrt(np.sum(diff * diff, axis=1))
    gain, shrink = set(), set()
    new_deltas = deltas.copy()
    for i in range(n):
        if y[i] == y_new:
            if dist_to[i] < deltas[i]:
                gain.add(i)
        else:
            if dist_to[i] < deltas[i]:
                shrink.add(i)
                new_deltas[i] = dist_to[i]
    return gain, shrink, new_deltas


def _check_incremental_insert(name, n_steps=40, seed=0):
    X, y, _ = load_planetoid(name)
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    init, stream = perm[:-n_steps], perm[-n_steps:]

    Xa, ya = X[init].copy(), y[init].copy()
    deltas = calc_deltas_rowwise(Xa, ya)
    for gidx in stream:
        g_ref, s_ref, nd_ref = _loop_reference(Xa, ya, deltas, X[gidx], y[gidx])
        Xa, ya, deltas, g_vec, s_vec = incremental_insert(
            Xa, ya, deltas, X[gidx], y[gidx]
        )
        assert g_vec == g_ref, f"{name}: gain set mismatch at node {gidx}"
        assert s_vec == s_ref, f"{name}: shrink set mismatch at node {gidx}"
        # deltas returned includes the new node; compare the |V_old| prefix
        assert np.array_equal(deltas[:-1], nd_ref), (
            f"{name}: new_deltas mismatch at node {gidx}"
        )
    print(f"OK {name}: incremental_insert vectorized == loop reference over {n_steps} inserts")


def main():
    for ds in ("cora", "citeseer"):
        _check_delta_comparator(ds)
        _check_incremental_insert(ds)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
