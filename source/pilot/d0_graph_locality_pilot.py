"""
D0 graph locality pilot — kiem tra tinh cuc bo cua delta_B(x_i) khi chen node tuan tu
tren graph node set (Cora, Citeseer, SBM synthetic).

Cau hoi song con (kill criterion #1 trong docs/idea_frame.md): khi chen mot node moi,
co bao nhieu % node hien co bi doi delta_B (ban kinh an toan = khoang cach toi lang gieng
khac lop gan nhat trong feature space)?

Tai su dung logic tu IJAR-MHMG-Incremental/pilot/d2_locality_pilot.py, ap dung cho
node feature matrix cua citation graphs (planetoid) va SBM synthetic.
"""
import json
import os
import pickle
import time
import urllib.request

import numpy as np
from scipy.sparse import lil_matrix, vstack
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

SEED = 42
INIT_FRAC = 0.8
TOP_FEATURES = 50  # B co dinh: top variance features (giam chi phi pilot high-dim)
LOCALITY_THRESHOLD = 0.30

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "shared", "data", "planetoid")
DATASETS = [
    {"id": "G1", "name": "Cora", "kind": "planetoid", "dataset": "cora"},
    {"id": "G2", "name": "Citeseer", "kind": "planetoid", "dataset": "citeseer"},
    {"id": "G3", "name": "SBM-synthetic", "kind": "sbm", "n": 2000, "communities": 5},
]

PLANETOID_FILES = [
    "ind.{ds}.x",
    "ind.{ds}.y",
    "ind.{ds}.tx",
    "ind.{ds}.ty",
    "ind.{ds}.allx",
    "ind.{ds}.ally",
    "ind.{ds}.test.index",
    "ind.{ds}.graph",
]
PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/"


def ensure_planetoid(ds):
    os.makedirs(DATA_ROOT, exist_ok=True)
    for pattern in PLANETOID_FILES:
        fname = pattern.format(ds=ds)
        path = os.path.join(DATA_ROOT, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            url = PLANETOID_BASE + fname
            print(f"  downloading {fname} ...", flush=True)
            urllib.request.urlretrieve(url, path)


def parse_index_file(filename):
    return [int(line.strip()) for line in open(filename)]


def load_planetoid(dataset):
    ensure_planetoid(dataset)
    names = ["x", "y", "tx", "ty", "allx", "ally"]
    objs = []
    for name in names:
        with open(os.path.join(DATA_ROOT, f"ind.{dataset}.{name}"), "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally = objs
    test_idx_reorder = parse_index_file(
        os.path.join(DATA_ROOT, f"ind.{dataset}.test.index")
    )
    test_idx_range = np.sort(test_idx_reorder)
    if dataset == "citeseer":
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
    features = np.array(features.todense())
    with open(os.path.join(DATA_ROOT, f"ind.{dataset}.graph"), "rb") as f:
        graph = pickle.load(f, encoding="latin1")
    return features, labels, graph


def make_sbm_graph(n, communities, seed):
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n,
        n_features=60,
        n_informative=25,
        n_redundant=10,
        n_classes=communities,
        class_sep=1.2,
        random_state=seed,
    )
    graph = {i: [] for i in range(n)}
    p_in, p_out = 0.08, 0.01
    for i in range(n):
        for j in range(i + 1, n):
            same = y[i] == y[j]
            p = p_in if same else p_out
            if rng.random() < p:
                graph[i].append(j)
                graph[j].append(i)
    return X, y, graph


def select_top_variance_features(X, k):
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    var = X.var(axis=0)
    return np.argsort(var)[-k:]


def calc_initial_deltas(X_sub, y):
    n = len(y)
    if n == 0:
        return np.array([])
    diff = X_sub[:, None, :] - X_sub[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    deltas = np.full(n, np.inf)
    for i in range(n):
        enemy_mask = y != y[i]
        ed = dist_matrix[i][enemy_mask]
        if len(ed) > 0:
            deltas[i] = ed.min()
    return deltas


def run_pilot_for_dataset(ds):
    ds_id, name = ds["id"], ds["name"]
    print(f"\n=== {ds_id} {name} ===", flush=True)

    if ds["kind"] == "planetoid":
        X, y, graph = load_planetoid(ds["dataset"])
    else:
        X, y, graph = make_sbm_graph(ds["n"], ds["communities"], SEED)

    n_total = len(y)
    n_edges = sum(len(v) for v in graph.values()) // 2
    print(f"  |V|={n_total}, |E|~={n_edges}, |C|={X.shape[1]}", flush=True)

    B = select_top_variance_features(X, TOP_FEATURES)
    XB = MinMaxScaler().fit_transform(X[:, B])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_init = int(n_total * INIT_FRAC)
    init_idx = perm[:n_init]
    stream_idx = perm[n_init:]

    order = list(init_idx)
    active_X = XB[init_idx].copy()
    active_y = y[init_idx].copy()
    active_deltas = calc_initial_deltas(active_X, active_y)

    affected_fracs = []
    n_active_trace = []

    t0 = time.time()
    for k, idx in enumerate(stream_idx):
        x_new = XB[idx]
        y_new = y[idx]
        n_active = len(active_y)

        diff = active_X - x_new
        dist_to_active = np.sqrt(np.sum(diff * diff, axis=1))
        enemy_mask = active_y != y_new

        affected_mask = enemy_mask & (dist_to_active < active_deltas)
        n_affected = int(affected_mask.sum())
        affected_fracs.append(n_affected / n_active if n_active > 0 else 0.0)
        n_active_trace.append(n_active)

        active_deltas[affected_mask] = dist_to_active[affected_mask]
        enemy_dist = dist_to_active[enemy_mask]
        delta_new = float(enemy_dist.min()) if enemy_dist.size > 0 else np.inf

        active_X = np.vstack([active_X, x_new[None, :]])
        active_y = np.append(active_y, y_new)
        active_deltas = np.append(active_deltas, delta_new)
        order.append(idx)

    incremental_stream_time = time.time() - t0
    print(
        f"  streaming {len(stream_idx)} insertions: {incremental_stream_time:.2f}s",
        flush=True,
    )

    naive_checkpoints = sorted(set([0, len(stream_idx) // 2, len(stream_idx) - 1]))
    naive_vs_smart = []
    replay_X = XB[init_idx].copy()
    replay_y = y[init_idx].copy()
    for k, idx in enumerate(stream_idx):
        if k in naive_checkpoints:
            t0 = time.time()
            _ = calc_initial_deltas(replay_X, replay_y)
            naive_time = time.time() - t0
            smart_time_est = incremental_stream_time / max(len(stream_idx), 1)
            naive_vs_smart.append(
                {
                    "step": k,
                    "n_active": len(replay_y),
                    "naive_full_recompute_s": naive_time,
                    "smart_update_avg_s": smart_time_est,
                    "speedup_x": (
                        (naive_time / smart_time_est) if smart_time_est > 0 else None
                    ),
                }
            )
        replay_X = np.vstack([replay_X, XB[idx][None, :]])
        replay_y = np.append(replay_y, y[idx])

    mean_affected = float(np.mean(affected_fracs)) if affected_fracs else None
    locality_pass = mean_affected is not None and mean_affected < LOCALITY_THRESHOLD

    quality_check = None
    if locality_pass:
        t0 = time.time()
        full_deltas = calc_initial_deltas(XB, y)
        batch_time = time.time() - t0
        mapped_full = full_deltas[np.array(order)]
        max_delta_diff = float(np.max(np.abs(active_deltas - mapped_full)))
        quality_check = {
            "top_features_B": int(len(B)),
            "batch_recompute_all_nodes_s": batch_time,
            "max_delta_abs_diff_after_stream_vs_batch": max_delta_diff,
            "incremental_matches_batch": max_delta_diff < 1e-9,
        }
        print(
            f"  [locality PASS] max_delta_diff={max_delta_diff:.2e}, "
            f"batch_time={batch_time:.2f}s",
            flush=True,
        )
    else:
        print(
            f"  [locality FAIL threshold {LOCALITY_THRESHOLD}] "
            f"mean_affected={mean_affected:.4f}",
            flush=True,
        )

    return {
        "id": ds_id,
        "dataset": name,
        "n_total": n_total,
        "n_init": int(n_init),
        "n_stream": int(len(stream_idx)),
        "n_features_raw": int(X.shape[1]),
        "n_features_B": int(len(B)),
        "mean_affected_fraction": mean_affected,
        "median_affected_fraction": (
            float(np.median(affected_fracs)) if affected_fracs else None
        ),
        "max_affected_fraction": (
            float(np.max(affected_fracs)) if affected_fracs else None
        ),
        "affected_fraction_trend_corr_with_n_active": (
            float(np.corrcoef(n_active_trace, affected_fracs)[0, 1])
            if len(affected_fracs) > 2
            else None
        ),
        "incremental_stream_time_s": incremental_stream_time,
        "naive_vs_smart_checkpoints": naive_vs_smart,
        "locality_pass_lt_30pct": locality_pass,
        "quality_check": quality_check,
    }


def main():
    results = [run_pilot_for_dataset(ds) for ds in DATASETS]
    out_path = os.path.join(os.path.dirname(__file__), "pilot_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDa luu ket qua: {out_path}", flush=True)
    passed = sum(1 for r in results if r.get("locality_pass_lt_30pct"))
    print(f"Locality pass: {passed}/{len(results)} datasets", flush=True)


if __name__ == "__main__":
    main()
