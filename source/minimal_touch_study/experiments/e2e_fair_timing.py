"""
E2e -- Fair, end-to-end, multi-seed per-insertion timing.

Fixes MAJOR-E2 / E3 / E6 / E7 from review_opus_round1.md and MAJOR-E2-R2 from
review_opus_round2.md:
  - E2 (R1): the original "164-292x" comparator (theory.incremental_rough_
    adjacency.batch_rough_adjacency) is an unoptimized O(|V|^2) Python double
    loop. This script adds batch_rough_adjacency_vectorized (bit-identical,
    edge-list NumPy) as a matched comparator for the W step.
  - E2-R2 (R2): fixing only batch W left the end-to-end headline resting on
    batch DELTA, which was still theory.incremental_nrs.calc_deltas ->
    calc_deltas_rowwise, a Python row loop that is 94-99% of batch cost. This
    script now also times calc_deltas_vectorized (one BLAS GEMM, bit-compatible
    -- see test_batch_delta_vectorized_matches.py) as the fair batch-delta
    comparator, and reports the fair end-to-end speedup (vectorized delta +
    vectorized W vs. incremental) as the headline, keeping the naive-reference
    ratio only as an explicitly-labelled matched-naive-loop number. The
    incremental delta step (incremental_nrs.incremental_insert) is itself
    vectorized so both sides are measured on the same implementation footing.
  - E3: times batch delta recomputation alongside batch W, so an end-to-end
    incremental-vs-batch speedup (delta + W together) is reported, not only W.
  - E6: repeats over 5 seeds (42, 0, 1, 2, 3), reports mean +/- std.
  - E7: hardware/software versions collected into Table 2 (tab:config).

Output: experiments/results/e2e_fair_timing_results.json
"""
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid
from theory.incremental_nrs import (
    calc_deltas,
    calc_deltas_vectorized,
    incremental_insert,
)
from theory.incremental_rough_adjacency import (
    batch_rough_adjacency,
    batch_rough_adjacency_vectorized,
    incremental_rough_adjacency_update,
    structural_touch_set,
)

DATASETS = ["cora", "citeseer"]
SEEDS = [42, 0, 1, 2, 3]
INIT_FRAC = 0.8
# batch_delta (calc_deltas from scratch, O(n^2*d)) is a near-deterministic
# function of the current active-set size and dominates wall time (Cora
# ~2.6s/call, CiteSeer ~23-26s/call): a handful of samples per seed is
# enough to estimate its mean/std precisely without ~25 near-duplicate
# calls. inc_w/inc_delta are still measured at EVERY stream step (cheap,
# millisecond scale) -- only the expensive batch-comparator sampling is
# reduced.
MAX_BATCH_SAMPLES = 5


def run_one(name, seed):
    X, y, adj = load_planetoid(name)
    B_idx = np.arange(X.shape[1])
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]
    n_stream = len(stream_idx)

    sample_rng = np.random.default_rng(seed)
    if n_stream <= MAX_BATCH_SAMPLES:
        batch_sample_steps = set(range(n_stream))
    else:
        batch_sample_steps = set(
            sample_rng.choice(n_stream, MAX_BATCH_SAMPLES, replace=False)
        )
        batch_sample_steps.add(n_stream - 1)

    Xa = X[init_idx].copy()
    ya = y[init_idx].copy()
    deltas = calc_deltas(Xa, ya)
    order = list(init_idx)
    adj_sub = adj[np.ix_(init_idx, init_idx)].copy()
    W = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_idx)

    inc_w_times, inc_delta_times = [], []
    batch_w_naive_times, batch_w_vec_times = [], []
    batch_delta_naive_times, batch_delta_vec_times = [], []
    touch_sizes = []

    for step, gidx in enumerate(stream_idx):
        n_old = len(ya)
        deltas_old = deltas.copy()

        t0 = time.perf_counter()
        Xa, ya, deltas, _, _ = incremental_insert(Xa, ya, deltas, X[gidx], y[gidx])
        inc_delta_times.append(time.perf_counter() - t0)

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

        t0 = time.perf_counter()
        updates, T_adj = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_sub, B_idx, n_old, touch, S_delta
        )
        for (v, u), w in updates.items():
            W[v, u] = w
        inc_w_times.append(time.perf_counter() - t0)
        touch_sizes.append(len(T_adj))

        if step in batch_sample_steps:
            t0 = time.perf_counter()
            deltas_batch = calc_deltas(Xa, ya)
            batch_delta_naive_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            deltas_batch_vec = calc_deltas_vectorized(Xa, ya)
            batch_delta_vec_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            batch_rough_adjacency(Xa, ya, deltas_batch, adj_sub, B_idx)
            batch_w_naive_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            batch_rough_adjacency_vectorized(Xa, ya, deltas_batch, adj_sub, B_idx)
            batch_w_vec_times.append(time.perf_counter() - t0)

    return {
        "seed": seed,
        "inc_w_mean_s": float(np.mean(inc_w_times)),
        "inc_delta_mean_s": float(np.mean(inc_delta_times)),
        "batch_w_naive_mean_s": float(np.mean(batch_w_naive_times)),
        "batch_w_vectorized_mean_s": float(np.mean(batch_w_vec_times)),
        "batch_delta_naive_mean_s": float(np.mean(batch_delta_naive_times)),
        "batch_delta_vectorized_mean_s": float(np.mean(batch_delta_vec_times)),
        "touch_mean": float(np.mean(touch_sizes)),
        "n_stream_steps": n_stream,
        "n_batch_samples": len(batch_w_naive_times),
    }


def aggregate(name, per_seed):
    def col(key):
        return np.array([r[key] for r in per_seed])

    inc_w = col("inc_w_mean_s")
    inc_delta = col("inc_delta_mean_s")
    batch_w_naive = col("batch_w_naive_mean_s")
    batch_w_vec = col("batch_w_vectorized_mean_s")
    batch_delta_naive = col("batch_delta_naive_mean_s")
    batch_delta_vec = col("batch_delta_vectorized_mean_s")

    inc_total = inc_w + inc_delta
    # Matched-naive-loop reference: both batch terms are the unvectorized
    # Python-loop implementations. Kept only as an explicitly-labelled
    # constant-factor artefact, NOT the headline (review_opus_round2.md
    # MAJOR-E2-R2).
    batch_total_naive = batch_w_naive + batch_delta_naive
    # Fair reference: both batch terms vectorized (edge-list NumPy W, BLAS
    # GEMM delta), bit-compatible output. This is the honest end-to-end
    # speedup and the number that goes in the abstract/table/figure.
    batch_total_fair = batch_w_vec + batch_delta_vec

    def ms(a):
        return float(np.mean(a) * 1000), float(np.std(a) * 1000)

    return {
        "dataset": name,
        "n_seeds": len(per_seed),
        "seeds": [r["seed"] for r in per_seed],
        "inc_w_ms": ms(inc_w),
        "inc_delta_ms": ms(inc_delta),
        "batch_w_naive_ms": ms(batch_w_naive),
        "batch_w_vectorized_ms": ms(batch_w_vec),
        "batch_delta_naive_ms": ms(batch_delta_naive),
        "batch_delta_vectorized_ms": ms(batch_delta_vec),
        "touch_mean": float(np.mean(col("touch_mean"))),
        "n_stream_steps": int(per_seed[0]["n_stream_steps"]),
        # W-only ratios (what the original "164-292x" measured)
        "speedup_W_only_naive": float(np.mean(batch_w_naive / inc_w)),
        "speedup_W_only_vectorized": float(np.mean(batch_w_vec / inc_w)),
        # end-to-end ratios (delta + W together)
        "speedup_end_to_end_fair": float(np.mean(batch_total_fair / inc_total)),
        "speedup_end_to_end_naive_reference": float(
            np.mean(batch_total_naive / inc_total)
        ),
        "per_seed": per_seed,
    }


def main():
    print(f"torch={torch.__version__} numpy={np.__version__} scipy={scipy.__version__} "
          f"sklearn={sklearn.__version__}", flush=True)
    print(f"platform={platform.platform()} processor={platform.processor()}", flush=True)

    results = []
    for ds in DATASETS:
        print(f"\n=== {ds} ===", flush=True)
        per_seed = []
        for seed in SEEDS:
            r = run_one(ds, seed)
            per_seed.append(r)
            print(
                f"  seed={seed}: inc_w={r['inc_w_mean_s']*1000:.2f}ms "
                f"inc_delta={r['inc_delta_mean_s']*1000:.2f}ms "
                f"batch_w_naive={r['batch_w_naive_mean_s']*1000:.2f}ms "
                f"batch_w_vec={r['batch_w_vectorized_mean_s']*1000:.2f}ms "
                f"batch_delta_naive={r['batch_delta_naive_mean_s']*1000:.2f}ms "
                f"batch_delta_vec={r['batch_delta_vectorized_mean_s']*1000:.2f}ms",
                flush=True,
            )
        agg = aggregate(ds, per_seed)
        print(
            f"  AGG: speedup_W_naive={agg['speedup_W_only_naive']:.1f}x "
            f"speedup_W_vec={agg['speedup_W_only_vectorized']:.1f}x "
            f"speedup_e2e_FAIR={agg['speedup_end_to_end_fair']:.1f}x "
            f"speedup_e2e_naive_ref={agg['speedup_end_to_end_naive_reference']:.1f}x",
            flush=True,
        )
        results.append(agg)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e2e_fair_timing_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "env": {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
