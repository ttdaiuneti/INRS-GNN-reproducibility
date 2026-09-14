"""Incremental rough-adjacency stream builders (Paper 1 only)."""
import time

import numpy as np

from core.incremental_nrs import calc_deltas, incremental_insert
from core.rough_adjacency import (
    batch_rough_adjacency,
    incremental_rough_adjacency_update,
    structural_touch_set,
)


def build_rough_adjacency_stream(
    XB, y, adj, B_cols, init_idx, stream_idx, seed=42, use_true_incremental=True
):
    """
    Build W via node stream. Returns W_final on active subgraph, timings, parity stats.
    use_true_incremental: local W updates on T_adj (fast) vs full batch rebuild at end.
    """
    Xa = XB[init_idx].copy()
    ya = y[init_idx].copy()
    deltas = calc_deltas(Xa, ya)
    order = list(init_idx)
    adj_sub = adj[np.ix_(init_idx, init_idx)].copy()
    W = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_cols)

    touch_sizes = []
    t0 = time.time()
    for gidx in stream_idx:
        n_old = len(ya)
        deltas_old = deltas.copy()
        Xa, ya, deltas, _, _ = incremental_insert(
            Xa, ya, deltas, XB[gidx], y[gidx]
        )
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

        if use_true_incremental:
            updates, T_adj = incremental_rough_adjacency_update(
                Xa, ya, deltas, adj_sub, B_cols, n_old, touch, S_delta
            )
            touch_sizes.append(len(T_adj))
            for (v, u), w in updates.items():
                W[v, u] = w

    stream_time = time.time() - t0

    if not use_true_incremental:
        W = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_cols)

    t0 = time.time()
    W_batch = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_cols)
    batch_rebuild_time = time.time() - t0

    max_w_diff = float(np.max(np.abs(W - W_batch)))
    touch_mean = float(np.mean(touch_sizes)) if touch_sizes else 0.0
    touch_max = int(max(touch_sizes)) if touch_sizes else 0

    return {
        "W": W,
        "stream_time_s": stream_time,
        "batch_rebuild_time_s": batch_rebuild_time,
        "max_W_stream_vs_batch_diff": max_w_diff,
        "order": order,
        "touch_mean": touch_mean,
        "touch_max": touch_max,
        "n_stream_steps": len(stream_idx),
        "use_true_incremental": use_true_incremental,
    }


def build_rough_adjacency_stream_tuple(XB, y, adj, B_cols, init_idx, stream_idx, seed=42):
    """Tuple return for e0 compatibility (legacy path: batch rebuild at end)."""
    out = build_rough_adjacency_stream(
        XB, y, adj, B_cols, init_idx, stream_idx, seed, use_true_incremental=False
    )
    return (
        out["W"],
        out["stream_time_s"],
        out["batch_rebuild_time_s"],
        out["max_W_stream_vs_batch_diff"],
        out["order"],
    )


def remap_adjacency_to_original(W_sub, order, n_total):
    """Map W built on permuted active order back to global node indices."""
    inv = np.zeros(n_total, dtype=int)
    for pos, gidx in enumerate(order):
        inv[gidx] = pos
    W_orig = np.zeros((n_total, n_total))
    for g1 in range(n_total):
        for g2 in range(n_total):
            W_orig[g1, g2] = W_sub[inv[g1], inv[g2]]
    return W_orig


def profile_stream_insert_timings(
    XB, y, adj, B_cols, init_idx, stream_idx, max_batch_samples=30, seed=42
):
    """
    Per stream insertion: time incremental W touch update vs full batch W rebuild
    on the same post-insert graph state (deltas already updated incrementally).
    """
    rng = np.random.default_rng(seed)
    n_stream = len(stream_idx)
    if n_stream <= max_batch_samples:
        batch_sample_steps = set(range(n_stream))
    else:
        batch_sample_steps = set(rng.choice(n_stream, max_batch_samples, replace=False))
        batch_sample_steps.add(n_stream - 1)

    Xa = XB[init_idx].copy()
    ya = y[init_idx].copy()
    deltas = calc_deltas(Xa, ya)
    order = list(init_idx)
    adj_sub = adj[np.ix_(init_idx, init_idx)].copy()
    W = batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_cols)

    inc_w_times = []
    batch_w_times = []
    touch_sizes = []
    delta_inc_times = []

    for step, gidx in enumerate(stream_idx):
        n_old = len(ya)
        deltas_old = deltas.copy()

        t_delta = time.time()
        Xa, ya, deltas, _, _ = incremental_insert(
            Xa, ya, deltas, XB[gidx], y[gidx]
        )
        delta_inc_times.append(time.time() - t_delta)

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

        t_w = time.time()
        updates, T_adj = incremental_rough_adjacency_update(
            Xa, ya, deltas, adj_sub, B_cols, n_old, touch, S_delta
        )
        for (v, u), w in updates.items():
            W[v, u] = w
        inc_w_times.append(time.time() - t_w)
        touch_sizes.append(len(T_adj))

        if step in batch_sample_steps:
            t_b = time.time()
            batch_rough_adjacency(Xa, ya, deltas, adj_sub, B_cols)
            batch_w_times.append(time.time() - t_b)

    inc_mean = float(np.mean(inc_w_times))
    batch_mean = float(np.mean(batch_w_times))
    return {
        "inc_w_mean_s": inc_mean,
        "inc_w_median_s": float(np.median(inc_w_times)),
        "inc_w_total_s": float(np.sum(inc_w_times)),
        "delta_inc_mean_s": float(np.mean(delta_inc_times)),
        "delta_inc_total_s": float(np.sum(delta_inc_times)),
        "batch_w_mean_s": batch_mean,
        "batch_w_median_s": float(np.median(batch_w_times)),
        "batch_w_total_est_s": batch_mean * n_stream,
        "speedup_batch_over_inc_per_step": batch_mean / inc_mean if inc_mean > 0 else None,
        "touch_mean": float(np.mean(touch_sizes)),
        "touch_max": int(max(touch_sizes)),
        "n_stream_steps": n_stream,
        "n_batch_samples": len(batch_w_times),
    }
