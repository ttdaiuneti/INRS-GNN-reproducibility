#!/usr/bin/env python3
"""
Label-safe diagnostics measured on the SAME rough matrices the accuracy
benchmark uses (e3_pyg_aligned_benchmark.prepare_rough_matrices, seed=SEED).

Motivated by the 2026-09-10 review, finding M6: the manuscript quoted
delta = +inf fractions (63.4% / 86.0%) taken from the legacy E0 diagnostic,
whose trusted fraction is a different number (36.6% / 14.0%), while the
accuracy and ablation results come from this E3 protocol (15.7% / 4.4%).
Mixing the two cannot be right; every quantity used to explain the E3
ablation is recomputed here from the E3 operator itself.

Usage: python3 experiments/e3_label_safe_diagnostics.py
Output: experiments/results/e3_label_safe_diagnostics.json
"""
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "shared"))
sys.path.insert(0, _ROOT)

from experiments.e0_gnn_prototype import (  # noqa: E402
    SEED,
    normalize_adj,
    set_seed,
    train_warmup_pseudolabels,
)
from experiments.e3_pyg_aligned_benchmark import DATASETS, HIDDEN_DIM, TRUSTED_TAU  # noqa: E402
from experiments.graph_data import load_planetoid_pyg  # noqa: E402
from theory.rough_neighborhood_partition import build_trusted_labels  # noqa: E402
from theory.torch_nrs_mps import (  # noqa: E402
    batch_rough_adjacency_torch,
    calc_deltas_torch_trusted,
    get_device,
)

OUT = os.path.join(os.path.dirname(__file__), "results",
                   "e3_label_safe_diagnostics.json")


def mean_self_loop_coeff(A_dense):
    """Mean diagonal coefficient of the symmetrically normalized (A + I).

    NOT a fraction of row mass: symmetric normalisation does not make rows sum
    to one, so this is the mean of the diagonal entries themselves (review
    2026-09-10 round 2, section 7).
    """
    n = normalize_adj(A_dense)
    return float(torch.diagonal(n).mean().item())


def diagnose(name, device, mps_dev):
    X, y, adj, train_mask, val_mask, test_mask, n_classes = load_planetoid_pyg(name)
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    a_bin = torch.tensor(adj.astype(np.float32), device=device)
    a_norm_plain = normalize_adj(a_bin)
    train_m = torch.tensor(train_mask, device=device)
    val_m = torch.tensor(val_mask, device=device)

    set_seed(SEED)
    pred, conf = train_warmup_pseudolabels(
        x_t, a_norm_plain, y_t, train_m, val_m, n_classes, device, SEED,
        hidden=HIDDEN_DIM,
    )
    y_hat, trusted = build_trusted_labels(y, train_mask, pred, conf, TRUSTED_TAU)
    deltas = np.asarray(calc_deltas_torch_trusted(X, y_hat, trusted, mps_dev))
    W = np.asarray(batch_rough_adjacency_torch(X, y, deltas, adj, mps_dev))

    n = X.shape[0]
    finite = np.isfinite(deltas)
    trusted = np.asarray(trusted, dtype=bool)
    iu = np.triu_indices(n, k=1)
    edge_mask = adj[iu].astype(bool)
    w_edges = W[iu][edge_mask]
    n_edges = int(edge_mask.sum())
    nz = w_edges > 0.0

    rough_deg = W.sum(axis=1)
    struct_deg = adj.sum(axis=1)

    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    return {
        "dataset": name,
        "n_nodes": n,
        "n_trusted": int(trusted.sum()),
        "trusted_fraction": float(trusted.mean()),
        "trusted_classes_present": int(len(np.unique(y_hat[trusted]))),
        "n_delta_finite": int(finite.sum()),
        "n_delta_infinite": int((~finite).sum()),
        "frac_delta_infinite": float((~finite).mean()),
        "frac_delta_infinite_among_trusted": float((~finite[trusted]).mean()),
        "n_edges_undirected": n_edges,
        "n_rough_edges_nonzero": int(nz.sum()),
        "frac_rough_edges_zero": float(1.0 - nz.mean()),
        "frac_edges_weight_below_half": float((w_edges < 0.5).mean()),
        "frac_nonzero_edges_weight_below_half": float((w_edges[nz] < 0.5).mean()) if nz.any() else 0.0,
        "mean_weight_over_all_edges": float(w_edges.mean()),
        "mean_weight_over_nonzero_edges": float(w_edges[nz].mean()) if nz.any() else 0.0,
        "median_weight_over_nonzero_edges": float(np.median(w_edges[nz])) if nz.any() else 0.0,
        "mean_rough_weighted_degree": float(rough_deg.mean()),
        "mean_structural_degree": float(struct_deg.mean()),
        "frac_nodes_zero_rough_degree": float((rough_deg == 0.0).mean()),
        "frac_nodes_isolated_in_A": float((struct_deg == 0.0).mean()),
        "mean_self_loop_coeff_psi_only": mean_self_loop_coeff(W_t),
        "mean_self_loop_coeff_plain": mean_self_loop_coeff(a_bin),
        "mean_self_loop_coeff_hybrid_alpha1": mean_self_loop_coeff(a_bin + W_t),
        "trusted_tau": TRUSTED_TAU,
        "warmup_seed": SEED,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mps_dev = get_device()
    out = {"protocol": "e3_pyg_aligned_benchmark.prepare_rough_matrices",
           "note": "single warm-up per dataset at seed=SEED, as in the accuracy run",
           "datasets": []}
    for name in DATASETS:
        d = diagnose(name, device, mps_dev)
        out["datasets"].append(d)
        print(f"{name}: trusted {d['trusted_fraction']:.4f} | "
              f"delta=inf {d['frac_delta_infinite']:.4f} | "
              f"zero rough edges {d['frac_rough_edges_zero']:.4f} | "
              f"mean self-loop coeff psi-only {d['mean_self_loop_coeff_psi_only']:.4f} "
              f"vs plain {d['mean_self_loop_coeff_plain']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[PASS] wrote {OUT}")


if __name__ == "__main__":
    main()
