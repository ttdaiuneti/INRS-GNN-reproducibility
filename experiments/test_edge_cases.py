"""Edge-case tests of exact maintenance (Theorems 1-3, Algorithm 1 and 2, minimal touch set).

Every case is a stream of arrivals. After each arrival the maintained state is compared, with exact
equality and not a tolerance, with a rebuild by the batch definition:
  radii    shared/theory/incremental_nrs.incremental_insert   vs calc_deltas
  weights  incremental_rough_adjacency_update on T = {v_new} u S_delta (minimal touch set)
           vs batch_rough_adjacency
  granules experiments/minimal_touch/granule_check.insert (member sets) vs batch_state
It also checks Theorem 3(ii): every re-evaluated weight on a shrinking node changes, except on edges
with distance 0 or with an endpoint of infinite radius.
Cases cover: one class only, infinite to finite radii, first node of a new class, distance 0 between
classes, ties at the minimal enemy distance, an arrival at exactly the current radius, an edge with
both ends in T, a hub arrival, a ring arrival that shrinks half of all radii, duplicates within a class,
and 300 random small streams on an integer grid, where ties are frequent.
Run: python3 experiments/test_edge_cases.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT / "experiments" / "minimal_touch"))
from theory.incremental_nrs import calc_deltas, incremental_insert  # noqa: E402
from theory.incremental_rough_adjacency import (  # noqa: E402
    batch_rough_adjacency, incremental_rough_adjacency_update, nrs_psi)
import granule_check as gc  # noqa: E402


def run_stream(X, y, edges, n0, name):
    """edges: set of (i, j), i < j, over all nodes. Arrivals are nodes n0, n0+1, ..."""
    N = len(y)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    B = np.arange(X.shape[1])
    A = np.zeros((N, N), dtype=bool)
    for i, j in edges:
        A[i, j] = A[j, i] = True
    D = gc.pairwise_direct(X)
    Xa, ya = X[:n0].copy(), y[:n0].copy()
    deltas = calc_deltas(Xa, ya)
    W = batch_rough_adjacency(Xa, ya, deltas, A[:n0, :n0], B)
    act = np.arange(n0)
    rad, gran = gc.batch_state(D, y, act)
    stats = dict(shrink=0, both_in_T=0, inf_to_finite=0, zero_dist=0, tie=0, steps=0)
    for z in range(n0, N):
        n_old = z
        old = deltas.copy()
        Xa, ya, deltas, _, S = incremental_insert(Xa, ya, deltas, X[z], y[z])
        # strictness at the boundary: an enemy at exactly the old radius must not shrink it
        for v in range(n_old):
            if y[v] != y[z] and D[v, z] == old[v] and np.isfinite(old[v]):
                assert v not in S, f"{name}: node {v} shrank on a tie"
                stats["tie"] += 1
        stats["inf_to_finite"] += int(sum(np.isinf(old[v]) and np.isfinite(deltas[v]) for v in range(n_old)))
        stats["zero_dist"] += int(sum(y[v] != y[z] and D[v, z] == 0 for v in range(n_old)))
        stats["shrink"] += len(S)
        Wn = np.zeros((n_old + 1, n_old + 1))
        Wn[:n_old, :n_old] = W
        adj_now = A[:n_old + 1, :n_old + 1]
        updates, T = incremental_rough_adjacency_update(Xa, ya, deltas, adj_now, B, n_old, {n_old}, S)
        assert T == {n_old} | set(S)
        T_list = sorted(T)
        stats["both_in_T"] += sum(1 for i in T_list for j in T_list if i < j and adj_now[i, j])
        # Theorem 3(ii): rows of S_delta change, except degenerate edges
        for (v, u), w in updates.items():
            if v in S and u != n_old and u < n_old:
                rho = D[v, u]
                if rho > 0 and np.isfinite(deltas[u]):
                    w_old = nrs_psi(rho, old[v], old[u])
                    assert w < w_old or (w_old == 0 < w), f"{name}: weight on ({v},{u}) did not change"
        for (v, u), w in updates.items():
            Wn[v, u] = w
        W = Wn
        # exactness against a rebuild
        d_ref = calc_deltas(Xa, ya)
        assert np.array_equal(deltas, d_ref), f"{name}: radii differ at arrival {z}"
        W_ref = batch_rough_adjacency(Xa, ya, d_ref, adj_now, B)
        assert np.array_equal(W, W_ref), f"{name}: weights differ at arrival {z}"
        act, _ = gc.insert(D, y, act, rad, gran, z)
        br, bg = gc.batch_state(D, y, act)
        assert br == rad and bg == gran, f"{name}: granules differ at arrival {z}"
        stats["steps"] += 1
    return stats


def case_single_class():
    # five nodes of class 0 (radii infinite), then the first enemy, then more of both classes
    X = [[0], [1], [2], [3], [4], [2.5], [10], [0.5]]
    y = [0, 0, 0, 0, 0, 1, 1, 0]
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (0, 7)}
    return X, y, edges, 5


def case_zero_distance():
    # an enemy with identical features: radius 0, empty granule, weight with distance 0
    X = [[0, 0], [1, 1], [3, 3], [1, 1], [1, 1], [0, 0]]
    y = [0, 1, 0, 0, 1, 1]
    edges = {(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (0, 5)}
    return X, y, edges, 3


def case_ties():
    # two enemies at the same minimal distance, then an arrival at exactly the current radius
    X = [[0], [2], [-2], [2], [-2], [1], [0.5]]
    y = [0, 1, 1, 1, 1, 1, 1]
    edges = {(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 6)}
    return X, y, edges, 3


def case_edge_in_T():
    # the arrival is adjacent to a node whose radius it shrinks: edge with both ends in T
    X = [[0], [3], [6], [1]]
    y = [0, 1, 0, 1]
    edges = {(0, 1), (1, 2), (1, 3), (0, 3)}
    return X, y, edges, 3


def case_hub():
    n = 12
    X = [[float(i)] for i in range(n)] + [[5.5]]
    y = [i % 2 for i in range(n)] + [0]
    edges = {(i, i + 1) for i in range(n - 1)} | {(i, n) for i in range(n)}
    return X, y, edges, n


def case_ring_enemy():
    n = 40
    rng = np.random.default_rng(0)
    y = np.r_[np.repeat([0, 1], n // 2), 1]
    X = rng.normal(0, .02, (n, 4))
    X[y[:n] == 1, 0] += 10
    X = np.vstack([X, np.zeros(4)])
    edges = {(i, (i + 1) % n if (i + 1) % n > i else 0) for i in range(n)}
    edges = {(min(a, b), max(a, b)) for a, b in edges if a != b} | {(0, n), (n // 2, n)}
    return X, y, edges, n


def case_duplicates():
    # identical features inside one class (friend at distance 0) and across classes
    X = [[1], [1], [1], [4], [4], [1], [4]]
    y = [0, 0, 1, 1, 0, 1, 0]
    edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)}
    return X, y, edges, 2


def random_case(seed):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(6, 16))
    k = int(rng.integers(1, 4))
    X = rng.integers(0, 4, size=(N, int(rng.integers(1, 3))))
    y = rng.integers(0, k, size=N)
    p = rng.uniform(0.1, 0.6)
    edges = {(i, j) for i in range(N) for j in range(i + 1, N) if rng.random() < p}
    return X, y, edges, int(rng.integers(1, N - 1))


def main():
    cases = {"single class, infinite to finite": case_single_class(), "distance zero between classes": case_zero_distance(),
             "ties at the minimal distance": case_ties(), "edge with both ends in T": case_edge_in_T(),
             "hub arrival": case_hub(), "ring, half of all radii shrink": case_ring_enemy(),
             "duplicates within and across classes": case_duplicates()}
    total = dict(shrink=0, both_in_T=0, inf_to_finite=0, zero_dist=0, tie=0, steps=0)
    named = {}
    for name, (X, y, E, n0) in cases.items():
        s = run_stream(X, y, E, n0, name)
        named[name] = s
        print(f"[PASS] {name}: {s}")
        for k in total:
            total[k] += s[k]
    for seed in range(300):
        X, y, E, n0 = random_case(seed)
        s = run_stream(X, y, E, n0, f"random {seed}")
        for k in total:
            total[k] += s[k]
    (ROOT / "experiments" / "minimal_touch" / "results" / "edge_cases.json").write_text(
        json.dumps(dict(named_cases=len(cases), random_streams=300, totals=total, named=named), indent=2))
    print(f"[PASS] 300 random streams. Totals over all streams: {total}")
    # the suite must actually exercise the cases it names
    for k in ("shrink", "both_in_T", "inf_to_finite", "zero_dist", "tie"):
        assert total[k] > 0, f"no case exercised '{k}'"
    print("[PASS] all edge cases")


if __name__ == "__main__":
    main()
