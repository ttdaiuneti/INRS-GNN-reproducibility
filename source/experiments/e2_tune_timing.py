"""
E2 — Learnable hybrid α + per-insert timing ablation + PCA-256 speed path.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.e0_gnn_prototype import (
    EPOCHS,
    HIDDEN,
    INIT_FRAC,
    LR,
    SEED,
    WEIGHT_DECAY,
    build_label_safe_deltas,
    normalize_adj,
    set_seed,
)
from experiments.graph_data import (
    load_planetoid,
    planetoid_train_test_masks,
)
from experiments.graph_rough import profile_stream_insert_timings
from theory.torch_nrs_mps import batch_rough_adjacency_torch, get_device

DATASETS = ["cora", "citeseer"]
PCA_DIM = 256
MAX_BATCH_TIMING_SAMPLES = 25


class HybridGCN(nn.Module):
    """GCN with learnable α: Ã = normalize(A + softplus(α_raw) · W_ψ)."""

    def __init__(self, in_dim, hidden, out_dim, a_bin, w_psi, dropout=0.5, alpha_init=1.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.register_buffer("a_bin", a_bin)
        self.register_buffer("w_psi", w_psi)
        self.alpha_raw = nn.Parameter(
            torch.tensor(float(np.log(max(alpha_init, 1e-3))), dtype=torch.float32)
        )

    def alpha(self):
        return F.softplus(self.alpha_raw)

    def adj_norm(self):
        return normalize_adj(self.a_bin + self.alpha() * self.w_psi)

    def forward(self, x):
        a_norm = self.adj_norm()
        h = F.relu(self.lin1(a_norm @ x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return a_norm @ self.lin2(h)


def train_hybrid_gcn(model, x, y, train_mask, val_mask, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = 0.0
    best_state = None
    y_t = y.to(device)

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[train_mask], y_t[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x).argmax(dim=1)
            val_acc = (pred[val_mask] == y_t[val_mask]).float().mean().item()
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "alpha": float(model.alpha().cpu()),
            }

    if best_state:
        model.load_state_dict(best_state["model"])
        alpha_out = float(best_state["alpha"])
    else:
        alpha_out = float(model.alpha().cpu())
    return best_val, alpha_out


def build_features(X, mode):
    if mode == "full":
        return MinMaxScaler().fit_transform(X), "full"
    if mode == "pca256":
        n_comp = min(PCA_DIM, X.shape[0], X.shape[1])
        Xp = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)
        return MinMaxScaler().fit_transform(Xp), f"pca{n_comp}"
    raise ValueError(mode)


def prepare_w_psi(X_feat, y, adj, train_mask, val_mask, n_classes, device, mps_dev):
    """Label-safe (2026-08-31) — see build_label_safe_deltas / negative_results.md."""
    t0 = time.time()
    deltas_t, trust_frac = build_label_safe_deltas(
        X_feat, y, adj, train_mask, val_mask, n_classes, device, mps_dev
    )
    W = batch_rough_adjacency_torch(X_feat, y, deltas_t, adj, mps_dev)
    if mps_dev.type == "mps":
        torch.mps.synchronize()
    return W, time.time() - t0, trust_frac


def run_learnable_alpha(name, device, mps_dev, feat_mode):
    X, y, adj = load_planetoid(name)
    n = X.shape[0]
    train_mask, val_mask, test_mask = planetoid_train_test_masks(name, n)
    X_feat, feat_label = build_features(X, feat_mode)
    n_classes = int(len(np.unique(y)))

    W_psi, w_build_s, trust_frac = prepare_w_psi(
        X_feat, y, adj, train_mask, val_mask, n_classes, device, mps_dev
    )
    A_bin = torch.tensor(adj.astype(np.float32), device=device)
    W_t = torch.tensor(W_psi, dtype=torch.float32, device=device)

    x = torch.tensor(X_feat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_m = torch.tensor(train_mask, device=device)
    val_m = torch.tensor(val_mask, device=device)
    test_m = torch.tensor(test_mask, device=device)

    set_seed(SEED)
    model = HybridGCN(x.shape[1], HIDDEN, n_classes, A_bin, W_t).to(device)
    t0 = time.time()
    val_acc, alpha = train_hybrid_gcn(model, x, y_t, train_m, val_m, device)
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        test_acc = (pred[test_m] == y_t[test_m]).float().mean().item()

    return {
        "feat_mode": feat_label,
        "w_build_s": w_build_s,
        "trusted_fraction": trust_frac,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "learned_alpha": alpha,
        "train_time_s": train_time,
    }


def run_timing(name, feat_mode, max_batch_samples=MAX_BATCH_TIMING_SAMPLES):
    X, y, adj = load_planetoid(name)
    X_feat, feat_label = build_features(X, feat_mode)
    B_cols = np.arange(X_feat.shape[1])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_init = int(len(y) * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    profile = profile_stream_insert_timings(
        X_feat, y, adj, B_cols, init_idx, stream_idx,
        max_batch_samples=max_batch_samples, seed=SEED,
    )
    profile["feat_mode"] = feat_label
    return profile


def run_dataset(name, device, mps_dev):
    print(f"\n=== {name} ===", flush=True)

    timing_full = run_timing(name, "full")
    print(
        f"  timing full: inc_w={timing_full['inc_w_mean_s']*1000:.2f}ms "
        f"batch_w={timing_full['batch_w_mean_s']*1000:.2f}ms "
        f"speedup={timing_full['speedup_batch_over_inc_per_step']:.1f}x",
        flush=True,
    )

    timing_pca = run_timing(name, "pca256")
    print(
        f"  timing pca: inc_w={timing_pca['inc_w_mean_s']*1000:.2f}ms "
        f"batch_w={timing_pca['batch_w_mean_s']*1000:.2f}ms "
        f"speedup={timing_pca['speedup_batch_over_inc_per_step']:.1f}x",
        flush=True,
    )

    hybrid_full = run_learnable_alpha(name, device, mps_dev, "full")
    print(
        f"  hybrid full: α={hybrid_full['learned_alpha']:.3f} "
        f"test={hybrid_full['test_acc']:.4f}",
        flush=True,
    )

    hybrid_pca = run_learnable_alpha(name, device, mps_dev, "pca256")
    print(
        f"  hybrid pca: α={hybrid_pca['learned_alpha']:.3f} "
        f"test={hybrid_pca['test_acc']:.4f}",
        flush=True,
    )

    return {
        "dataset": name,
        "timing_full_features": timing_full,
        "timing_pca256": timing_pca,
        "hybrid_learnable_full": hybrid_full,
        "hybrid_learnable_pca256": hybrid_pca,
    }


def main():
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mps_dev = get_device()
    print(f"Device GCN: {device} | δ/W: {mps_dev}", flush=True)

    results = [run_dataset(ds, device, mps_dev) for ds in DATASETS]

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e2_tune_timing_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
