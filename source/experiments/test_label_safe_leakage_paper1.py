"""
Leakage test for Paper 1's label-safe δ_B / W_psi (E3 benchmark).
Usage: python3 experiments/test_label_safe_leakage_paper1.py

Mirrors experiments/test_label_safe_leakage.py (Paper 2): scramble held-out
(val+test) ground-truth labels and confirm the label-safe δ/W_psi computation
is bit-exact, proving it never reads those labels. Also checks the OLD
(unsafe) calc_deltas_torch DOES change under the same scramble, to document
the bug this fix addresses.
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ws = _ROOT
sys.path.insert(0, os.path.join(_ws, "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.graph_data import load_planetoid_pyg
from theory.rough_neighborhood_partition import build_trusted_labels
from theory.torch_nrs_mps import calc_deltas_torch, calc_deltas_torch_trusted


def assert_bit_exact(a, b, name):
    a_np = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    b_np = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    if not np.array_equal(a_np, b_np):
        raise AssertionError(f"{name} changed after scrambling held-out labels")


def main():
    X, y, adj, train_mask, val_mask, test_mask, _ = load_planetoid_pyg("cora")
    rng = np.random.default_rng(0)
    n_classes = int(y.max()) + 1

    # Fixed dummy pseudo-labels/confidence — irrelevant to this test's claim,
    # only need trusted == train_mask ∪ {high-conf extras} to stay fixed across
    # the scramble, since pred/conf are model outputs (not ground-truth) and
    # thus untouched by scrambling y itself.
    dummy_pred = rng.integers(0, n_classes, size=len(y))
    dummy_conf = rng.random(len(y))

    y_hat, trusted = build_trusted_labels(y, train_mask, dummy_pred, dummy_conf, 0.8)
    deltas_safe = calc_deltas_torch_trusted(X, y_hat, trusted)

    y_scrambled = y.copy()
    hold = val_mask | test_mask
    y_scrambled[hold] = rng.integers(0, n_classes, size=int(hold.sum()))

    y_hat2, trusted2 = build_trusted_labels(
        y_scrambled, train_mask, dummy_pred, dummy_conf, 0.8
    )
    deltas_safe2 = calc_deltas_torch_trusted(X, y_hat2, trusted2)

    assert_bit_exact(trusted, trusted2, "trusted mask")
    assert_bit_exact(y_hat, y_hat2, "trusted labels")
    assert_bit_exact(deltas_safe, deltas_safe2, "label-safe deltas (calc_deltas_torch_trusted)")
    print("OK: calc_deltas_torch_trusted is bit-exact under held-out label scramble.")

    # Document the bug: the OLD unsafe path DOES change under the same scramble.
    deltas_unsafe = calc_deltas_torch(X, y)
    deltas_unsafe2 = calc_deltas_torch(X, y_scrambled)
    changed = not np.array_equal(
        deltas_unsafe.cpu().numpy(), deltas_unsafe2.cpu().numpy()
    )
    if not changed:
        raise AssertionError(
            "Expected calc_deltas_torch (unsafe) to change under label scramble — "
            "if it didn't, this regression test no longer demonstrates the leak."
        )
    print("CONFIRMED: calc_deltas_torch (old/unsafe) changes under the same scramble — "
          "i.e. it does read held-out labels. Do not use it for GNN experiments.")


if __name__ == "__main__":
    main()
