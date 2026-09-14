"""
E1 — Hybrid adjacency GCN: plain vs RG-GCN static vs INRS-GNN hybrid.
Full MinMax features, true incremental W stream, alpha grid on val.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.e0_gnn_prototype import (
    GCN,
    EPOCHS,
    HIDDEN,
    INIT_FRAC,
    SEED,
    build_label_safe_deltas,
    eval_acc,
    normalize_adj,
    set_seed,
    train_gcn,
)
from experiments.graph_data import (
    load_planetoid,
    planetoid_train_test_masks,
)
from experiments.graph_rough import (
    build_rough_adjacency_stream,
    remap_adjacency_to_original,
)
from theory.incremental_rough_adjacency import batch_rg_adjacency
from theory.torch_nrs_mps import batch_rough_adjacency_torch, get_device

ALPHAS = [0.5, 1.0, 2.0, 5.0]
DATASETS = ["cora", "citeseer"]


def run_gcn_once(x, a_np, y, train_m, val_m, test_m, device, label):
    set_seed(SEED)
    a = torch.tensor(a_np, dtype=torch.float32, device=device)
    a_norm = normalize_adj(a)
    n_classes = int(torch.unique(y).numel())
    model = GCN(x.shape[1], HIDDEN, n_classes, dropout=0.5).to(device)
    t0 = time.time()
    val_acc = train_gcn(model, x, a_norm, y, train_m, val_m, device)
    train_time = time.time() - t0
    test_acc = eval_acc(model, x, a_norm, y, test_m, device)
    return {
        "name": label,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_time_s": train_time,
    }


def run_dataset(name, device, mps_dev):
    print(f"\n=== {name} ===", flush=True)
    X, y, adj = load_planetoid(name)
    n = X.shape[0]
    train_mask, val_mask, test_mask = planetoid_train_test_masks(name, n)

    X_full = MinMaxScaler().fit_transform(X)
    B_cols = np.arange(X_full.shape[1])
    A_bin = adj.astype(np.float32)

    n_classes = int(len(np.unique(y)))
    t0 = time.time()
    deltas_t, trust_frac = build_label_safe_deltas(
        X_full, y, adj, train_mask, val_mask, n_classes, device, mps_dev
    )
    W_psi_batch = batch_rough_adjacency_torch(X_full, y, deltas_t, adj, mps_dev)
    if mps_dev.type == "mps":
        torch.mps.synchronize()
    batch_delta_w_time = time.time() - t0
    print(f"  trusted_fraction={trust_frac:.3f} (label-safe, W_psi_batch)", flush=True)

    t0 = time.time()
    W_rg = batch_rg_adjacency(X_full, adj, lam=0.5, normalize=True)
    rg_build_time = time.time() - t0

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    # KNOWN GAP (documented, not silently skipped — docs/negative_results.md):
    # the streaming path still uses oracle δ_B (full labels, Assumption 4a in
    # formal_framework.tex) — label-safe streaming was not implemented (would
    # require Tr/y_hat to evolve under the insertion rule, unproven per
    # Remark "Locality of the trusted variant"). W_inc below is therefore NOT
    # label-safe. It is used only for the archived INRS_hybrid_inc accuracy
    # number (superseded — see e3_pyg_aligned_benchmark.py for the valid
    # accuracy source) and for the timing/parity numbers, which do not depend
    # on any label being "safe" (pure structural comparison).
    stream_out = build_rough_adjacency_stream(
        X_full, y, adj, B_cols, init_idx, stream_idx, SEED, use_true_incremental=True
    )
    W_inc = remap_adjacency_to_original(stream_out["W"], stream_out["order"], n)
    max_W_inc_vs_batch = float(np.max(np.abs(W_inc - W_psi_batch)))

    print(
        f"  parity: max_W_inc_vs_batch={max_W_inc_vs_batch:.2e} "
        f"stream_vs_batch_sub={stream_out['max_W_stream_vs_batch_diff']:.2e}",
        flush=True,
    )
    print(
        f"  stream_inc={stream_out['stream_time_s']:.3f}s "
        f"batch_rebuild={stream_out['batch_rebuild_time_s']:.3f}s "
        f"touch_mean={stream_out['touch_mean']:.1f}",
        flush=True,
    )

    x = torch.tensor(X_full, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_m = torch.tensor(train_mask, device=device)
    val_m = torch.tensor(val_mask, device=device)
    test_m = torch.tensor(test_mask, device=device)

    plain = run_gcn_once(x, A_bin, y_t, train_m, val_m, test_m, device, "GCN_plain")
    print(f"  {plain['name']}: test={plain['test_acc']:.4f}", flush=True)

    rg = run_gcn_once(x, W_rg, y_t, train_m, val_m, test_m, device, "RG_GCN_static")
    print(f"  {rg['name']}: test={rg['test_acc']:.4f}", flush=True)

    alpha_results = []
    for alpha in ALPHAS:
        r = run_gcn_once(
            x, A_bin + alpha * W_psi_batch, y_t, train_m, val_m, test_m, device,
            f"INRS_hybrid_a{alpha}",
        )
        r["alpha"] = alpha
        alpha_results.append(r)
        print(f"  {r['name']}: test={r['test_acc']:.4f} val={r['val_acc']:.4f}", flush=True)

    best = max(alpha_results, key=lambda r: r["val_acc"])
    best_alpha = best["alpha"]

    inc = run_gcn_once(
        x, A_bin + best_alpha * W_inc, y_t, train_m, val_m, test_m, device,
        f"INRS_hybrid_inc_a{best_alpha}",
    )
    print(f"  {inc['name']}: test={inc['test_acc']:.4f}", flush=True)

    psi_only = run_gcn_once(x, W_psi_batch, y_t, train_m, val_m, test_m, device, "INRS_psi_only")
    print(f"  {psi_only['name']}: test={psi_only['test_acc']:.4f}", flush=True)

    stream_s = stream_out["stream_time_s"]
    batch_s = stream_out["batch_rebuild_time_s"]

    return {
        "dataset": name,
        "n_nodes": n,
        "n_features": X_full.shape[1],
        "batch_delta_w_build_s": batch_delta_w_time,
        "trusted_fraction_batch": trust_frac,
        "rg_build_s": rg_build_time,
        "stream_incremental_s": stream_s,
        "single_batch_rebuild_s": batch_s,
        "speedup_stream_vs_single_batch": batch_s / stream_s if stream_s > 0 else None,
        "touch_mean": stream_out["touch_mean"],
        "touch_max": stream_out["touch_max"],
        "touch_over_n_mean": stream_out["touch_mean"] / n,
        "n_stream_steps": stream_out["n_stream_steps"],
        "max_W_inc_vs_batch_full_graph": max_W_inc_vs_batch,
        "max_W_stream_vs_batch_subgraph": stream_out["max_W_stream_vs_batch_diff"],
        "best_alpha": best_alpha,
        "alpha_grid": alpha_results,
        "models": {
            "GCN_plain": plain,
            "RG_GCN_static": rg,
            "INRS_hybrid_best": best,
            "INRS_hybrid_inc": inc,
            "INRS_psi_only": psi_only,
        },
        "summary": {
            "GCN_plain_test": plain["test_acc"],
            "RG_GCN_static_test": rg["test_acc"],
            "INRS_hybrid_best_test": best["test_acc"],
            "INRS_hybrid_inc_test": inc["test_acc"],
            "INRS_psi_only_test": psi_only["test_acc"],
            "hybrid_vs_plain_delta": best["test_acc"] - plain["test_acc"],
            "hybrid_vs_rg_delta": best["test_acc"] - rg["test_acc"],
        },
    }


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mps_dev = get_device()
    print(f"Device GCN: {device} | δ/W build: {mps_dev}", flush=True)

    all_results = [run_dataset(ds, device, mps_dev) for ds in DATASETS]

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e1_hybrid_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)

    for r in all_results:
        s = r["summary"]
        print(
            f"{r['dataset']}: plain={s['GCN_plain_test']:.3f} "
            f"RG={s['RG_GCN_static_test']:.3f} "
            f"hybrid(a={r['best_alpha']})={s['INRS_hybrid_best_test']:.3f} "
            f"Δplain={s['hybrid_vs_plain_delta']:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
