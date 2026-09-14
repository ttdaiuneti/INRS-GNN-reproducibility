"""
Diagnostic — why GCN_rough_NRS test acc drops vs GCN_plain.
Checks: weight stats, effective degree, hybrids, full features, RG-style weights.
"""
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.e0_gnn_prototype import (
    GCN,
    EPOCHS,
    HIDDEN,
    LR,
    SEED,
    TOP_B,
    build_label_safe_deltas,
    normalize_adj,
    set_seed,
    train_gcn,
    eval_acc,
)
from experiments.graph_data import (
    build_feature_matrix,
    load_planetoid,
    planetoid_train_test_masks,
    select_top_variance_features,
)
from theory.incremental_rough_adjacency import batch_rough_adjacency, paired_rg_weights
from theory.torch_nrs_mps import batch_rough_adjacency_torch, get_device

# fix typo if WEIGHT_DECAY
WEIGHT_DECAY = 5e-4

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def adj_stats(A_np, name):
    n = A_np.shape[0]
    on_edges = A_np[adj_mask] if adj_mask is not None else A_np[A_np > 0]
    deg = A_np.sum(axis=1)
    deg_n = deg + 1  # with self-loop convention
    return {
        "name": name,
        "edge_weight_mean": float(np.mean(on_edges)) if on_edges.size else 0,
        "edge_weight_std": float(np.std(on_edges)) if on_edges.size else 0,
        "edge_weight_min": float(np.min(on_edges)) if on_edges.size else 0,
        "edge_weight_max": float(np.max(on_edges)) if on_edges.size else 0,
        "row_sum_mean": float(np.mean(deg)),
        "row_sum_min": float(np.min(deg)),
        "row_sum_max": float(np.max(deg)),
        "frac_edges_below_0.1": float(np.mean(on_edges < 0.1)) if on_edges.size else 0,
        "frac_edges_below_0.5": float(np.mean(on_edges < 0.5)) if on_edges.size else 0,
    }


def build_rg_adjacency(XB, adj, lam=0.5):
    """RG-GCN style: paired min/max on half/half features (heuristic split)."""
    d = XB.shape[1]
    mid = d // 2
    B_strict = np.arange(0, mid)
    B_loose = np.arange(mid, d)
    n = XB.shape[0]
    W = np.zeros((n, n))
    for v in range(n):
        for u in range(v + 1, n):
            if adj[v, u]:
                w = paired_rg_weights(XB[v], XB[u], B_strict, B_loose, lam)
                W[v, u] = W[u, v] = w
    return W


def run_gcn(name, x, a_np, y, train_m, val_m, test_m):
    set_seed(SEED)
    a = torch.tensor(a_np, dtype=torch.float32, device=DEVICE)
    a_norm = normalize_adj(a)
    model = GCN(x.shape[1], HIDDEN, len(np.unique(y.cpu())), dropout=0.5).to(DEVICE)
    val = train_gcn(model, x, a_norm, y, train_m, val_m, DEVICE)
    test = eval_acc(model, x, a_norm, y, test_m, DEVICE)
    return {"name": name, "val_acc": val, "test_acc": test}


def diagnose_dataset(name):
    global adj_mask
    X, y, adj = load_planetoid(name)
    train_mask, val_mask, test_mask = planetoid_train_test_masks(name, len(y))
    n_classes = int(len(np.unique(y)))
    mps_dev = get_device()

    B_idx = select_top_variance_features(X, TOP_B)
    XB = build_feature_matrix(X, B_idx)
    B_cols = np.arange(XB.shape[1])

    # Label-safe (2026-08-31) — see build_label_safe_deltas / negative_results.md.
    deltas_t, trust_frac = build_label_safe_deltas(
        XB, y, adj, train_mask, val_mask, n_classes, DEVICE, mps_dev
    )
    deltas = deltas_t.cpu().numpy()
    W_psi = batch_rough_adjacency(XB, y, deltas, adj, B_cols)

    W_rg = build_rg_adjacency(XB, adj)
    w_max = W_rg.max()
    W_rg_norm = W_rg / w_max if w_max > 0 else W_rg

    adj_mask = adj.astype(bool)
    stats = [
        adj_stats(adj.astype(float), "plain_binary"),
        adj_stats(W_psi, "psi_rough"),
        adj_stats(W_rg, "rg_paired_raw"),
        adj_stats(W_rg_norm, "rg_paired_norm01"),
    ]

    alphas = [0.5, 1.0, 2.0, 5.0]

    x_50 = torch.tensor(XB, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    train_m = torch.tensor(train_mask, device=DEVICE)
    val_m = torch.tensor(val_mask, device=DEVICE)
    test_m = torch.tensor(test_mask, device=DEVICE)

    acc_results = []
    acc_results.append(run_gcn("GCN_plain_A", x_50, adj.astype(float), y_t, train_m, val_m, test_m))
    acc_results.append(run_gcn("GCN_psi_W_only", x_50, W_psi, y_t, train_m, val_m, test_m))
    acc_results.append(run_gcn("GCN_rg_raw", x_50, W_rg, y_t, train_m, val_m, test_m))
    acc_results.append(run_gcn("GCN_rg_norm01", x_50, W_rg_norm, y_t, train_m, val_m, test_m))
    for a in alphas:
        acc_results.append(
            run_gcn(f"GCN_A+{a}W_psi", x_50, adj.astype(float) + a * W_psi, y_t, train_m, val_m, test_m)
        )

    from sklearn.preprocessing import MinMaxScaler
    X_full = MinMaxScaler().fit_transform(X)
    x_full = torch.tensor(X_full, dtype=torch.float32, device=DEVICE)
    deltas_full_t, trust_frac_full = build_label_safe_deltas(
        X_full, y, adj, train_mask, val_mask, n_classes, DEVICE, mps_dev
    )
    deltas_full = deltas_full_t.cpu().numpy()
    W_psi_full = batch_rough_adjacency(X_full, y, deltas_full, adj, np.arange(X_full.shape[1]))
    acc_results.append(run_gcn("GCN_plain_A_full1433", x_full, adj.astype(float), y_t, train_m, val_m, test_m))
    acc_results.append(run_gcn("GCN_psi_full1433", x_full, W_psi_full, y_t, train_m, val_m, test_m))

    rho_on_edges = []
    psi_on_edges = []
    delta_stats = {
        "mean": float(np.mean(deltas)),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
        "frac_inf": float(np.mean(np.isinf(deltas))),
        "frac_zero": float(np.mean(deltas == 0)),
    }
    for v in range(len(y)):
        for u in range(v + 1, len(y)):
            if adj[v, u]:
                rho = np.linalg.norm(XB[v] - XB[u])
                rho_on_edges.append(rho)
                psi_on_edges.append(W_psi[v, u])

    ratios = []
    for v in range(len(y)):
        for u in range(v + 1, len(y)):
            if adj[v, u]:
                rho = np.linalg.norm(XB[v] - XB[u])
                denom = deltas[v] + deltas[u] + 1e-9
                if denom > 1e-6:
                    ratios.append(rho / denom)

    return {
        "dataset": name,
        "trusted_fraction": trust_frac,
        "trusted_fraction_full": trust_frac_full,
        "adjacency_stats": stats,
        "delta_B_stats": delta_stats,
        "edge_rho_mean": float(np.mean(rho_on_edges)),
        "edge_psi_mean": float(np.mean(psi_on_edges)),
        "edge_rho_over_delta_sum_mean": float(np.mean(ratios)) if ratios else None,
        "accuracy_sweep": acc_results,
    }


def main():
    results = []
    for ds in ["cora", "citeseer"]:
        results.append(diagnose_dataset(ds))

    out_path = os.path.join(os.path.dirname(__file__), "results", "e0_acc_diagnosis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
