"""
E3 — PyG-aligned Planetoid benchmark (public split, NormalizeFeatures).

Careful protocol:
  - Data/masks via torch_geometric Planetoid (public split)
  - Cora: 140 train / 500 val / 1000 test
  - Citeseer: 120 train / 500 val / 1000 test
  - Features: row L1 normalize (PyG NormalizeFeatures)
  - GCN: 2-layer, hidden=16, dropout=0.5, Adam lr=0.01, wd=5e-4, 200 epochs
  - Sanity: PyG GCNConv vs our GCN_plain must agree within tolerance
  - Multi-seed mean ± std for all methods

LEAKAGE FIX (2026-08-31): W_psi/δ_B previously used calc_deltas_torch(X, y, ...)
with y = full label vector (train+val+test), so a test node's rough-adjacency
weight was computed from its own ground-truth label — classic transductive label
leakage feeding straight into the GCN's input structure. Now uses
calc_deltas_torch_trusted with trusted labels = train ground truth + high-confidence
(tau=0.8) pseudo-labels from a plain-adjacency warm-up GCN (selected on val_mask,
never test). See theory/torch_nrs_mps.py and test_label_safe_leakage_paper1.py.
Numbers from before this fix (docs/e3_pyg_aligned_report.md, go_nogo.md E2 gate)
are superseded — treat as archived/invalid.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.gnn import (
    GCN,
    EPOCHS,
    HIDDEN,
    LR,
    SEED,
    WEIGHT_DECAY,
    eval_acc,
    normalize_adj,
    set_seed,
    train_gcn,
    train_warmup_pseudolabels,
)
from core.hybrid import HybridGCN, train_hybrid_gcn
from core.data import load_planetoid_pyg, verify_pyg_masks, PYG_NAME_MAP
from core.rough_adjacency import batch_rg_adjacency
from core.rough_partition import build_trusted_labels
from core.torch_nrs import (
    batch_rough_adjacency_torch,
    calc_deltas_torch_trusted,
    get_device,
)

TRUSTED_TAU = 0.8  # pseudo-label confidence threshold (matches Paper 2 default)

DATASETS = ["cora", "citeseer"]
# 10 seeds (was 5): at n=5 the minimum attainable two-sided Wilcoxon p-value
# is 0.0625 > 0.05, so no comparison could ever reach significance regardless
# of effect size -- review_opus_round1.md MAJOR-E4. n=10 gives a minimum
# attainable p of ~0.002 and materially tighter CIs.
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
ALPHAS = [0.5, 1.0, 2.0, 5.0]
SANITY_TOLERANCE = 0.015  # max |test_acc diff| vs PyG GCNConv reference
HIDDEN_DIM = 16  # PyG citation benchmark default


class PyGGCNReference(torch.nn.Module):
    """Reference GCNConv stack matching PyG citation benchmark."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_pyg_reference(model, data, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_val = 0.0
    best_test = 0.0
    best_state = None

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (
                (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            )
            test_acc = (
                (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            )
        if val_acc >= best_val:
            best_val = val_acc
            best_test = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_val, best_test


def load_pyg_data_object(name):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    root = os.path.join(os.path.dirname(__file__), "..", "shared", "data", "pyg")
    dataset = Planetoid(
        root, PYG_NAME_MAP[name.lower()], transform=T.NormalizeFeatures()
    )
    return dataset[0], dataset.num_classes


def run_gcn_plain(x, a_norm, y, train_m, val_m, test_m, n_classes, device, seed):
    set_seed(seed)
    model = GCN(x.shape[1], HIDDEN_DIM, n_classes, dropout=0.5).to(device)
    t0 = time.time()
    val_acc = train_gcn(model, x, a_norm, y, train_m, val_m, device)
    train_time = time.time() - t0
    test_acc = eval_acc(model, x, a_norm, y, test_m, device)
    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_time_s": train_time,
    }


def run_sanity_check(name, device):
    """Our dense GCN vs PyG GCNConv on same PyG data."""
    data, n_classes = load_pyg_data_object(name)
    data = data.to(device)

    X, y, adj, train_np, val_np, test_np, _ = load_planetoid_pyg(name)
    n = X.shape[0]
    a_dense = torch.zeros(n, n, device=device)
    a_dense[data.edge_index[0], data.edge_index[1]] = 1.0
    a_norm = normalize_adj(a_dense)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_m = torch.tensor(train_np, device=device)
    val_m = torch.tensor(val_np, device=device)
    test_m = torch.tensor(test_np, device=device)

    set_seed(SEED)
    plain = run_gcn_plain(
        x, a_norm, y_t, train_m, val_m, test_m, n_classes, device, SEED
    )

    set_seed(SEED)
    ref = PyGGCNReference(data.num_features, HIDDEN_DIM, n_classes).to(device)
    ref_val, ref_test = train_pyg_reference(ref, data, device)

    diff = abs(plain["test_acc"] - ref_test)
    ok = diff <= SANITY_TOLERANCE
    return {
        "dataset": name,
        "our_gcn_test": plain["test_acc"],
        "pyg_gcnconv_test": ref_test,
        "test_diff": diff,
        "pass": ok,
    }


def build_adjacency_tensors(adj, device):
    a_bin = torch.tensor(adj.astype(np.float32), device=device)
    a_norm = normalize_adj(a_bin)
    return a_bin, a_norm


def prepare_rough_matrices(
    X, y, adj, mps_dev, gcn_device, a_norm_plain, train_mask, val_mask, n_classes, seed
):
    """
    Label-safe rough matrices. δ_B's enemy candidates are restricted to
    `trusted` nodes (train ground truth + confident pseudo-labels) — see
    calc_deltas_torch_trusted. Ground-truth val/test labels never enter W_psi.
    """
    x_t = torch.tensor(X, dtype=torch.float32, device=gcn_device)
    y_t = torch.tensor(y, dtype=torch.long, device=gcn_device)
    train_m = torch.tensor(train_mask, device=gcn_device)
    val_m = torch.tensor(val_mask, device=gcn_device)
    pred, conf = train_warmup_pseudolabels(
        x_t, a_norm_plain, y_t, train_m, val_m, n_classes, gcn_device, seed,
        hidden=HIDDEN_DIM,
    )
    y_hat, trusted = build_trusted_labels(y, train_mask, pred, conf, TRUSTED_TAU)

    t0 = time.time()
    deltas_t = calc_deltas_torch_trusted(X, y_hat, trusted, mps_dev)
    W_psi = batch_rough_adjacency_torch(X, y, deltas_t, adj, mps_dev)
    if mps_dev.type == "mps":
        torch.mps.synchronize()
    w_time = time.time() - t0

    t0 = time.time()
    W_rg = batch_rg_adjacency(X, adj, lam=0.5, normalize=True)
    rg_time = time.time() - t0

    trust_frac = float(np.mean(trusted))
    return W_psi, W_rg, w_time, rg_time, trust_frac


def run_seed(name, seed, device, mps_dev, W_psi, W_rg, pack):
    X, y, adj, train_np, val_np, test_np, n_classes = pack
    x = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    train_m = torch.tensor(train_np, device=device)
    val_m = torch.tensor(val_np, device=device)
    test_m = torch.tensor(test_np, device=device)
    a_bin, a_norm = build_adjacency_tensors(adj, device)

    out = {"seed": seed}

    out["GCN_plain"] = run_gcn_plain(
        x, a_norm, y_t, train_m, val_m, test_m, n_classes, device, seed
    )

    set_seed(seed)
    a_rg = torch.tensor(W_rg, dtype=torch.float32, device=device)
    a_rg_norm = normalize_adj(a_rg)
    rg_model = GCN(x.shape[1], HIDDEN_DIM, n_classes, dropout=0.5).to(device)
    rg_val = train_gcn(rg_model, x, a_rg_norm, y_t, train_m, val_m, device)
    rg_test = eval_acc(rg_model, x, a_rg_norm, y_t, test_m, device)
    out["RG_GCN_static"] = {"val_acc": rg_val, "test_acc": rg_test}

    # Ablation: W_psi alone (no binary A) — canonical PyG-aligned, label-safe
    # protocol, so it is directly comparable to GCN_plain/hybrid above (earlier
    # psi-only numbers used a different legacy loader — see negative_results.md).
    set_seed(seed)
    a_psi = torch.tensor(W_psi, dtype=torch.float32, device=device)
    a_psi_norm = normalize_adj(a_psi)
    psi_model = GCN(x.shape[1], HIDDEN_DIM, n_classes, dropout=0.5).to(device)
    psi_val = train_gcn(psi_model, x, a_psi_norm, y_t, train_m, val_m, device)
    psi_test = eval_acc(psi_model, x, a_psi_norm, y_t, test_m, device)
    out["INRS_psi_only"] = {"val_acc": psi_val, "test_acc": psi_test}

    alpha_results = []
    for alpha in ALPHAS:
        a_hybrid = a_bin + alpha * torch.tensor(W_psi, dtype=torch.float32, device=device)
        a_hybrid_norm = normalize_adj(a_hybrid)
        set_seed(seed)
        model = GCN(x.shape[1], HIDDEN_DIM, n_classes, dropout=0.5).to(device)
        val_acc = train_gcn(model, x, a_hybrid_norm, y_t, train_m, val_m, device)
        test_acc = eval_acc(model, x, a_hybrid_norm, y_t, test_m, device)
        alpha_results.append(
            {"alpha": alpha, "val_acc": val_acc, "test_acc": test_acc}
        )
    best = max(alpha_results, key=lambda r: r["val_acc"])
    out["INRS_hybrid_grid"] = best
    out["INRS_hybrid_alpha_grid"] = alpha_results

    set_seed(seed)
    w_t = torch.tensor(W_psi, dtype=torch.float32, device=device)
    hybrid = HybridGCN(x.shape[1], HIDDEN_DIM, n_classes, a_bin, w_t).to(device)
    t0 = time.time()
    h_val, h_alpha = train_hybrid_gcn(hybrid, x, y_t, train_m, val_m, device)
    h_time = time.time() - t0
    hybrid.eval()
    with torch.no_grad():
        pred = hybrid(x).argmax(dim=1)
        h_test = (pred[test_m] == y_t[test_m]).float().mean().item()
    out["INRS_hybrid_learned"] = {
        "val_acc": h_val,
        "test_acc": h_test,
        "learned_alpha": h_alpha,
        "train_time_s": h_time,
    }

    return out


def aggregate_seed_runs(seed_runs, key):
    tests = [r[key]["test_acc"] for r in seed_runs]
    vals = [r[key]["val_acc"] for r in seed_runs]
    return {
        "test_mean": float(np.mean(tests)),
        "test_std": float(np.std(tests)),
        "val_mean": float(np.mean(vals)),
        "val_std": float(np.std(vals)),
        "test_per_seed": tests,
    }


def run_dataset(name, device, mps_dev):
    print(f"\n=== {name} ===", flush=True)
    mask_info = verify_pyg_masks(name)
    print(f"  masks: {mask_info}", flush=True)

    pack = load_planetoid_pyg(name)
    X, y, adj, train_mask, val_mask, test_mask, n_classes = pack
    a_bin_plain = torch.tensor(adj.astype(np.float32), device=device)
    a_norm_plain = normalize_adj(a_bin_plain)

    W_psi, W_rg, w_time, rg_time, trust_frac = prepare_rough_matrices(
        X, y, adj, mps_dev, device, a_norm_plain, train_mask, val_mask, n_classes, SEED
    )
    print(
        f"  W_psi build {w_time:.2f}s, RG build {rg_time:.2f}s, "
        f"trusted_fraction={trust_frac:.3f} (label-safe)",
        flush=True,
    )

    seed_runs = []
    for seed in SEEDS:
        print(f"  seed {seed}...", flush=True)
        seed_runs.append(run_seed(name, seed, device, mps_dev, W_psi, W_rg, pack))

    agg = {
        "GCN_plain": aggregate_seed_runs(seed_runs, "GCN_plain"),
        "RG_GCN_static": aggregate_seed_runs(seed_runs, "RG_GCN_static"),
        "INRS_psi_only": aggregate_seed_runs(seed_runs, "INRS_psi_only"),
        "INRS_hybrid_grid": aggregate_seed_runs(seed_runs, "INRS_hybrid_grid"),
        "INRS_hybrid_learned": aggregate_seed_runs(seed_runs, "INRS_hybrid_learned"),
    }

    # best test alpha per seed from grid
    grid_tests = [r["INRS_hybrid_grid"]["test_acc"] for r in seed_runs]
    best_alpha_per_seed = [r["INRS_hybrid_grid"]["alpha"] for r in seed_runs]

    print(
        f"  plain {agg['GCN_plain']['test_mean']:.3f}±{agg['GCN_plain']['test_std']:.3f} | "
        f"RG {agg['RG_GCN_static']['test_mean']:.3f} | "
        f"psi_only {agg['INRS_psi_only']['test_mean']:.3f} | "
        f"hybrid_grid {agg['INRS_hybrid_grid']['test_mean']:.3f} | "
        f"hybrid_learned {agg['INRS_hybrid_learned']['test_mean']:.3f}",
        flush=True,
    )

    return {
        "dataset": name,
        "mask_info": mask_info,
        "w_psi_build_s": w_time,
        "w_rg_build_s": rg_time,
        "trusted_fraction": trust_frac,
        "trusted_tau": TRUSTED_TAU,
        "seeds": SEEDS,
        "seed_runs": seed_runs,
        "aggregate": agg,
        "grid_best_alpha_per_seed": best_alpha_per_seed,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mps_dev = get_device()
    print(f"Device GCN: {device} | δ/W: {mps_dev}", flush=True)
    print(f"Config: hidden={HIDDEN_DIM}, epochs={EPOCHS}, seeds={SEEDS}", flush=True)

    sanity = []
    for ds in DATASETS:
        s = run_sanity_check(ds, device)
        sanity.append(s)
        status = "PASS" if s["pass"] else "FAIL"
        print(
            f"Sanity {ds}: our={s['our_gcn_test']:.4f} pyg={s['pyg_gcnconv_test']:.4f} "
            f"diff={s['test_diff']:.4f} [{status}]",
            flush=True,
        )

    if not all(s["pass"] for s in sanity):
        print("ABORT: sanity check failed — fix GCN implementation before benchmark.", flush=True)
        sys.exit(1)

    results = [run_dataset(ds, device, mps_dev) for ds in DATASETS]

    out = {
        "config": {
            "hidden": HIDDEN_DIM,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds": SEEDS,
            "alphas": ALPHAS,
            "data": "PyG Planetoid public + NormalizeFeatures",
            "sanity_tolerance": SANITY_TOLERANCE,
        },
        "sanity_checks": sanity,
        "datasets": results,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e3_pyg_aligned_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
