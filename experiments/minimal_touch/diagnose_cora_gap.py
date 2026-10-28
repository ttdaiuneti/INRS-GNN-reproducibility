"""Why do Table 7 (full stream) and Table 8 (fixed state) give different Cora/CiteSeer speedups?

Two candidate causes, measured separately with the same kernels and the minimal touch set:
 (a) graph size at which the rebuild is timed: Table 7 samples rebuilds along the
     whole stream (|V| from 80% to 100% of the nodes); Table 8 times them near the
     80% state. A vectorized rebuild is quadratic in |V|.
 (b) cache state of the incremental step: Table 8 runs a rebuild between
     consecutive incremental steps; Table 7 runs incremental steps back to back,
     with a rebuild only at six sampled steps.
For each dataset and seed 0-2 we stream the first K arrivals after the 80% state
twice from the same initial state: once back to back (A) and once with a
vectorized rebuild before every incremental step (B). We also time the
vectorized rebuild at the 80% state and on the full graph.
Usage: python3 experiments/minimal_touch/diagnose_cora_gap.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT))
from graph_data import load_planetoid  # noqa: E402
from theory.incremental_nrs import calc_deltas, calc_deltas_vectorized, incremental_insert  # noqa: E402
from theory.incremental_rough_adjacency import (  # noqa: E402
    batch_rough_adjacency_vectorized, incremental_rough_adjacency_update)

K = 60


def rebuild(Xa, ya, adj, B):
    d = calc_deltas_vectorized(Xa, ya)
    batch_rough_adjacency_vectorized(Xa, ya, d, adj, B)


def stream(X, y, adj, init, arrivals, with_rebuild):
    B = np.arange(X.shape[1])
    Xa, ya = X[init].copy(), y[init].copy()
    order = list(init)
    adj_a = np.ascontiguousarray(adj[np.ix_(init, init)])
    deltas = calc_deltas(Xa, ya)
    times = []
    for g in arrivals:
        if with_rebuild:
            rebuild(Xa, ya, adj_a, B)
        n_old = len(ya)
        t0 = time.perf_counter()
        Xa, ya, deltas, _, S = incremental_insert(Xa, ya, deltas, X[g], y[g])
        nb = [j for j, o in enumerate(order) if adj[g, o]]
        big = np.zeros((n_old + 1, n_old + 1), dtype=bool)
        big[:n_old, :n_old] = adj_a
        for j in nb:
            big[j, n_old] = big[n_old, j] = True
        adj_a = big
        incremental_rough_adjacency_update(Xa, ya, deltas, adj_a, B, n_old, {n_old}, S)
        times.append(time.perf_counter() - t0)
        order.append(g)
    return np.array(times) * 1000


def main():
    out = []
    for name in ["cora", "citeseer"]:
        X, y, adj = load_planetoid(name)
        X = np.asarray(X, dtype=np.float64)
        B = np.arange(X.shape[1])
        for seed in [0, 1, 2]:
            perm = np.random.default_rng(seed).permutation(len(y))
            n0 = int(0.8 * len(y))
            init, arr = np.sort(perm[:n0]), perm[n0:n0 + K]
            a = stream(X, y, adj, init, arr, False)
            b = stream(X, y, adj, init, arr, True)
            t80, tall = [], []
            for _ in range(3):
                t = time.perf_counter(); rebuild(X[init], y[init], np.ascontiguousarray(adj[np.ix_(init, init)]), B); t80.append(time.perf_counter() - t)
                t = time.perf_counter(); rebuild(X, y, np.ascontiguousarray(adj), B); tall.append(time.perf_counter() - t)
            r = dict(dataset=name, seed=seed, n80=int(n0), n_all=int(len(y)),
                     inc_back_to_back_ms=float(np.median(a)), inc_after_rebuild_ms=float(np.median(b)),
                     rebuild_80_ms=float(np.median(t80)) * 1000, rebuild_all_ms=float(np.median(tall)) * 1000)
            print(r, flush=True)
            out.append(r)
    (HERE / "results" / "diagnose_cora_gap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
