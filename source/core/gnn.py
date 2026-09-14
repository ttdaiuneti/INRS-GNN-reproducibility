"""
E0 GNN prototype — GCN on plain adjacency vs NRS rough adjacency.
Verifies incremental W matches batch; compares test accuracy (Cora / Citeseer).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.data import (
    build_feature_matrix,
    load_planetoid,
    planetoid_train_test_masks,
    select_top_variance_features,
)
from core.graph_rough import (
    build_rough_adjacency_stream_tuple,
    remap_adjacency_to_original,
)
from core.rough_partition import build_trusted_labels
from core.torch_nrs import (
    batch_rough_adjacency_torch,
    calc_deltas_torch,
    calc_deltas_torch_trusted,
    get_device,
)

SEED = 42
TOP_B = 50
HIDDEN = 64
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
INIT_FRAC = 0.8
DATASETS = ["cora", "citeseer"]
TRUSTED_TAU = 0.8  # pseudo-label confidence threshold for label-safe δ_B


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_adj(A):
    """Symmetric normalization with self-loops."""
    A = A + torch.eye(A.shape[0], device=A.device)
    deg = A.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, a_norm):
        h = F.relu(self.lin1(a_norm @ x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = a_norm @ self.lin2(h)
        return h


def train_gcn(model, x, a_norm, y, train_mask, val_mask, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = 0.0
    best_state = None
    y_t = y.to(device)

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x, a_norm)
        loss = F.cross_entropy(logits[train_mask], y_t[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(x, a_norm).argmax(dim=1)
            val_acc = (pred[val_mask] == y_t[val_mask]).float().mean().item()
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_val


def eval_acc(model, x, a_norm, y, mask, device):
    model.eval()
    with torch.no_grad():
        pred = model(x, a_norm).argmax(dim=1)
        return (pred[mask] == y.to(device)[mask]).float().mean().item()


def train_warmup_pseudolabels(x, a_norm, y_t, train_m, val_m, n_classes, device, seed=SEED,
                               hidden=HIDDEN):
    """
    Plain-adjacency GCN trained on train_mask only, model-selected on val_mask
    (never test). Used only to produce pseudo-labels + confidence for nodes
    outside train_mask, feeding build_trusted_labels() below — ground-truth
    val/test labels are never read. Shared by all e0-e3 scripts to keep the
    label-safe δ_B computation consistent (see docs/negative_results.md).
    `hidden` defaults to this module's HIDDEN=64 (e0/e1/e2 prototype size);
    e3_pyg_aligned_benchmark.py passes its own HIDDEN_DIM=16 (PyG citation
    benchmark default) to match its main GCN.
    """
    set_seed(seed)
    model = GCN(x.shape[1], hidden, n_classes, dropout=0.5).to(device)
    train_gcn(model, x, a_norm, y_t, train_m, val_m, device)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, a_norm), dim=1)
        conf, pred = probs.max(dim=1)
    return pred.cpu().numpy(), conf.cpu().numpy()


def build_label_safe_deltas(X, y, adj, train_mask, val_mask, n_classes, device, mps_dev,
                             seed=SEED, tau=TRUSTED_TAU, hidden=HIDDEN):
    """
    Label-safe δ_B for any downstream ML pipeline (E0-E3): enemy candidates for
    δ_B are restricted to train ground truth + high-confidence pseudo-labels
    from a plain-adjacency warm-up GCN — never ground-truth val/test labels.
    See theory/torch_nrs_mps.py::calc_deltas_torch_trusted,
    docs/proofs/formal_framework.tex Definition (Label-safe / trusted safety
    radius), docs/negative_results.md. Returns (deltas_tensor, trusted_fraction).
    """
    a_bin = torch.tensor(adj.astype(np.float32), device=device)
    a_norm_plain = normalize_adj(a_bin)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_m = torch.tensor(train_mask, device=device)
    val_m = torch.tensor(val_mask, device=device)

    pred, conf = train_warmup_pseudolabels(
        x_t, a_norm_plain, y_t, train_m, val_m, n_classes, device, seed, hidden=hidden
    )
    y_hat, trusted = build_trusted_labels(y, train_mask, pred, conf, tau)
    deltas_t = calc_deltas_torch_trusted(X, y_hat, trusted, mps_dev)
    return deltas_t, float(np.mean(trusted))


def run_dataset(name, device):
    print(f"\n=== {name} ===", flush=True)
    X, y, adj = load_planetoid(name)
    n, n_classes = X.shape[0], len(np.unique(y))
    B_idx = select_top_variance_features(X, TOP_B)
    XB = build_feature_matrix(X, B_idx)

    train_mask, val_mask, test_mask = planetoid_train_test_masks(name, n)

    # δ + W^rough on Metal (MPS); GCN train cũng MPS. Label-safe (2026-08-31):
    # enemy candidates for δ_B restricted to train + confident pseudo-labels —
    # see build_label_safe_deltas / docs/negative_results.md.
    mps_dev = get_device()
    t0 = time.time()
    deltas_t, trust_frac = build_label_safe_deltas(
        XB, y, adj, train_mask, val_mask, n_classes, device, mps_dev
    )
    W_rough = batch_rough_adjacency_torch(XB, y, deltas_t, adj, mps_dev)
    if mps_dev.type == "mps":
        torch.mps.synchronize()
    w_build_full = time.time() - t0
    compute_device = str(mps_dev)
    print(f"  trusted_fraction={trust_frac:.3f} (label-safe)", flush=True)

    B_cols = np.arange(XB.shape[1])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_init = int(n * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]
    # NOTE: this stream check still uses the oracle δ_B (Definition, full labels
    # — valid here, it's a structural/complexity check, never fed to a model that
    # predicts those labels; Assumption 4a in formal_framework.tex). W_rough
    # above is now label-safe (Assumption 4b), so W_inc_orig and W_rough use
    # DIFFERENT δ definitions — max_w_vs_full below reflects that formula gap,
    # not an incremental-vs-batch bug. See docs/negative_results.md.
    W_inc, stream_t, batch_t, max_w_diff, order = build_rough_adjacency_stream_tuple(
        XB, y, adj, B_cols, init_idx, stream_idx, SEED
    )
    W_inc_orig = remap_adjacency_to_original(W_inc, order, n)
    max_w_vs_full = float(np.max(np.abs(W_inc_orig - W_rough)))

    x = torch.tensor(XB, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    a_plain = torch.tensor(adj.astype(np.float32), device=device)
    a_rough = torch.tensor(W_rough, dtype=torch.float32, device=device)
    a_rough_inc = torch.tensor(W_inc_orig, dtype=torch.float32, device=device)

    a_plain_norm = normalize_adj(a_plain)
    a_rough_norm = normalize_adj(a_rough)
    a_rough_inc_norm = normalize_adj(a_rough_inc)

    train_m = torch.tensor(train_mask, device=device)
    val_m = torch.tensor(val_mask, device=device)
    test_m = torch.tensor(test_mask, device=device)

    # Forward parity: batch W vs incremental W
    model_probe = GCN(XB.shape[1], HIDDEN, n_classes).to(device)
    model_probe.eval()
    with torch.no_grad():
        out_batch = model_probe(x, a_rough_norm)
        out_inc = model_probe(x, a_rough_inc_norm)
    max_logit_diff = float(torch.max(torch.abs(out_batch - out_inc)).cpu())
    max_a_diff = float(torch.max(torch.abs(a_rough - a_rough_inc)).cpu())
    print(
        f"  parity: max_W_diff={max_w_vs_full:.2e} max_A_tensor={max_a_diff:.2e} "
        f"max_logit={max_logit_diff:.2e}",
        flush=True,
    )

    results = {}
    for label, a_norm in [
        ("GCN_plain", a_plain_norm),
        ("GCN_rough_NRS", a_rough_norm),
    ]:
        set_seed(SEED)
        model = GCN(XB.shape[1], HIDDEN, n_classes).to(device)
        t0 = time.time()
        val_acc = train_gcn(model, x, a_norm, y_t, train_m, val_m, device)
        train_time = time.time() - t0
        train_acc = eval_acc(model, x, a_norm, y_t, train_m, device)
        test_acc = eval_acc(model, x, a_norm, y_t, test_m, device)
        results[label] = {
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_time_s": train_time,
        }
        print(
            f"  {label}: test={test_acc:.4f} val={val_acc:.4f} train={train_acc:.4f}",
            flush=True,
        )

    out = {
        "dataset": name,
        "n_nodes": n,
        "n_classes": n_classes,
        "n_features_B": len(B_idx),
        "train_nodes": int(train_mask.sum()),
        "test_nodes": int(test_mask.sum()),
        "W_rough_full_build_s": w_build_full,
        "W_build_device": str(compute_device),
        "trusted_fraction": trust_frac,
        "W_stream_incremental_s": stream_t,
        "W_stream_batch_rebuild_s": batch_t,
        "max_W_inc_vs_batch_diff_subgraph": max_w_diff,
        "max_W_inc_vs_full_batch_diff": max_w_vs_full,
        "max_gcn_logit_diff_batch_vs_inc_W": max_logit_diff,
        "models": results,
        "rough_vs_plain_test_delta": (
            results["GCN_rough_NRS"]["test_acc"] - results["GCN_plain"]["test_acc"]
        ),
    }
    return out


def main():
    set_seed(SEED)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device (GCN + δ/W build): {device}", flush=True)

    all_results = [run_dataset(ds, device) for ds in DATASETS]
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e0_gnn_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
