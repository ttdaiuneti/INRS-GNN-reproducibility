"""Effect of the feature dtype (float32 in e2e_fair_timing.py, float64 in e2e_scale_timing.py)
on incremental and rebuild time, at the 80% state and on the full graph. Seeds 0-2, 60 arrivals.
Usage: python3 experiments/minimal_touch/diagnose_dtype.py"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (ROOT / "experiments", ROOT / "shared", ROOT): sys.path.insert(0, str(p))
from graph_data import load_planetoid
from theory.incremental_nrs import calc_deltas, calc_deltas_vectorized, incremental_insert
from theory.incremental_rough_adjacency import batch_rough_adjacency_vectorized

def run(X, y, adj, seed, K=60):
    perm = np.random.default_rng(seed).permutation(len(y)); n0 = int(0.8 * len(y))
    init, arr = np.sort(perm[:n0]), perm[n0:n0 + K]
    Xa, ya = X[init].copy(), y[init].copy(); d = calc_deltas(Xa, ya); t = []
    for g in arr:
        t0 = time.perf_counter(); Xa, ya, d, _, _ = incremental_insert(Xa, ya, d, X[g], y[g]); t.append(time.perf_counter() - t0)
    B = np.arange(X.shape[1]); rb = {}
    for tag, idx in (("80", init), ("all", np.arange(len(y)))):
        A = np.ascontiguousarray(adj[np.ix_(idx, idx)]); ts = []
        for _ in range(3):
            t0 = time.perf_counter(); dd = calc_deltas_vectorized(X[idx], y[idx]); batch_rough_adjacency_vectorized(X[idx], y[idx], dd, A, B); ts.append(time.perf_counter() - t0)
        rb[tag] = float(np.median(ts)) * 1000
    return float(np.median(t)) * 1000, rb

out = []
for name in ["cora", "citeseer"]:
    X, y, adj = load_planetoid(name)
    for dt in ["float32", "float64"]:
        Xd = np.ascontiguousarray(X, dtype=dt)
        for seed in [0, 1, 2]:
            inc, rb = run(Xd, y, adj, seed)
            r = dict(dataset=name, dtype=dt, seed=seed, inc_delta_ms=inc, rebuild_80_ms=rb["80"], rebuild_all_ms=rb["all"])
            print(r, flush=True); out.append(r)
(HERE / "results" / "diagnose_dtype.json").write_text(json.dumps(out, indent=2))
