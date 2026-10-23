"""Planetoid graph loaders shared by INRS-GNN and RNG-GCN."""
import os
import pickle

import numpy as np
from scipy.sparse import lil_matrix, vstack
from sklearn.preprocessing import MinMaxScaler

_SHARED_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(_SHARED_ROOT, "data", "planetoid")
PYG_DATA_ROOT = os.path.join(_SHARED_ROOT, "data", "pyg")

PYG_NAME_MAP = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}

# PyG public split: 20 labels/class train, 500 val, 1000 test (transductive).
PYG_PUBLIC_TRAIN = {"cora": 140, "citeseer": 120, "pubmed": 60}


def parse_index_file(path):
    return [int(line.strip()) for line in open(path)]


def load_planetoid(name):
    names = ["x", "y", "tx", "ty", "allx", "ally"]
    objs = []
    for nm in names:
        with open(os.path.join(DATA_ROOT, f"ind.{name}.{nm}"), "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally = objs
    test_idx_reorder = parse_index_file(
        os.path.join(DATA_ROOT, f"ind.{name}.test.index")
    )
    test_idx_range = np.sort(test_idx_reorder)
    if name == "citeseer":
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_reorder), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_reorder), :] = ty
        ty = ty_extended
    features = vstack([allx, tx]).tolil()
    labels = np.vstack([ally, ty])
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=-1)
    X = np.array(features.todense())
    with open(os.path.join(DATA_ROOT, f"ind.{name}.graph"), "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    n = X.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    for i, neighbors in graph.items():
        for j in neighbors:
            adj[i, j] = adj[j, i] = True
    return X, labels, adj


def planetoid_train_test_masks(name, n):
    """
    Masks for legacy pickle loader only — node order differs from PyG.
    For benchmark accuracy use load_planetoid_pyg() instead.
    """
    test_idx = parse_index_file(
        os.path.join(DATA_ROOT, f"ind.{name}.test.index")
    )
    name_l = name.lower()
    n_train = PYG_PUBLIC_TRAIN.get(name_l, 140)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[0:n_train] = True
    val_mask = np.zeros(n, dtype=bool)
    val_mask[n_train:n_train + 500] = True
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_planetoid_pyg(name, root=None, normalize_features=True):
    """
    Canonical Planetoid benchmark via PyG public split.
    Returns X (L1-normalized if normalize_features), y, adj, train/val/test masks, n_classes.
    """
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    name_l = name.lower()
    if name_l not in PYG_NAME_MAP:
        raise ValueError(f"Unknown dataset {name}; use cora|citeseer|pubmed")

    root = root or PYG_DATA_ROOT
    transform = T.NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root, PYG_NAME_MAP[name_l], transform=transform)
    data = dataset[0]

    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    n = data.num_nodes
    adj = np.zeros((n, n), dtype=bool)
    edge_index = data.edge_index.cpu().numpy()
    adj[edge_index[0], edge_index[1]] = True

    train_mask = data.train_mask.cpu().numpy().astype(bool)
    val_mask = data.val_mask.cpu().numpy().astype(bool)
    test_mask = data.test_mask.cpu().numpy().astype(bool)

    expected_train = PYG_PUBLIC_TRAIN[name_l]
    assert int(train_mask.sum()) == expected_train
    assert int(val_mask.sum()) == 500
    assert int(test_mask.sum()) == 1000

    return X, y, adj, train_mask, val_mask, test_mask, dataset.num_classes


def verify_pyg_masks(name):
    """Print mask stats for sanity checks."""
    X, y, adj, tr, va, te, nc = load_planetoid_pyg(name)
    return {
        "dataset": name,
        "n_nodes": X.shape[0],
        "n_features": X.shape[1],
        "n_classes": nc,
        "train": int(tr.sum()),
        "val": int(va.sum()),
        "test": int(te.sum()),
        "feature_row_sum_mean": float(X.sum(axis=1).mean()),
    }


def select_top_variance_features(X, k):
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    return np.argsort(X.var(axis=0))[-k:]


def build_feature_matrix(X, B_idx):
    return MinMaxScaler().fit_transform(X[:, B_idx])
