"""Indexed versus Gram rebuild of nearest-enemy radii (protocol.json, frozen first).

    python3 experiments/indexed_rebuild/run.py            # all runs, each in a subprocess
    python3 experiments/indexed_rebuild/run.py one <dataset> <seed> <variant>
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = HERE / "results"
sys.path.insert(0, str(ROOT / "shared"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ogb_robustness"))
P = json.loads((HERE / "protocol.json").read_text())
VARIANTS = ["gram", "kd_enemy", "ball_enemy", "kd_class", "ball_class"]
CAP_S = 1800
CIT_REPS = 5
# The cap applies to the whole subprocess (load, all repetitions, reference check);
# loading and the reference take < 600 s, so CAP_S + 600 bounds the timed part by CAP_S.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(dataset, seed):
    if dataset in ("cora", "citeseer"):
        from experiments.graph_data import load_planetoid_pyg
        X, y, *_ = load_planetoid_pyg(dataset)
        return np.ascontiguousarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)
    d = ROOT / "shared" / "data" / "ogb"
    X = np.load(d / "arxiv_X.npy")
    y = np.load(d / "arxiv_y.npy").astype(np.int64).ravel()
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_init = int(n * 0.8)
    order = np.concatenate([np.sort(perm[:n_init]), perm[n_init:n_init + 1000]])
    return np.ascontiguousarray(X[order], dtype=np.float64), y[order]


def gram(X, y, dataset):
    if dataset == "ogbn-arxiv":
        from stable_batch import calc_deltas_blocked_stable
        return calc_deltas_blocked_stable(X, y, block=2048)
    from theory.incremental_nrs import calc_deltas_vectorized
    return calc_deltas_vectorized(X, y)


def indexed(X, y, algo, mode):
    from sklearn.neighbors import NearestNeighbors
    out = np.full(len(y), np.inf)
    for c in np.unique(y):
        inc = y == c
        if inc.all():
            continue
        if mode == "enemy":
            nn = NearestNeighbors(n_neighbors=1, algorithm=algo, n_jobs=-1).fit(X[~inc])
            out[inc] = nn.kneighbors(X[inc])[0][:, 0]
        else:
            nn = NearestNeighbors(n_neighbors=1, algorithm=algo, n_jobs=-1).fit(X[inc])
            idx = np.flatnonzero(~inc)
            out[idx] = np.minimum(out[idx], nn.kneighbors(X[idx])[0][:, 0])
    return out


def compute(X, y, dataset, variant):
    if variant == "gram":
        return gram(X, y, dataset)
    algo, mode = variant.split("_")
    return indexed(X, y, {"kd": "kd_tree", "ball": "ball_tree"}[algo], mode)


def reference(X, y, dataset):
    if dataset == "ogbn-arxiv":
        return gram(X, y, dataset)
    from theory.incremental_nrs import calc_deltas
    return calc_deltas(X, y)


def one(dataset, seed, variant):
    X, y = load(dataset, seed)
    reps = CIT_REPS if dataset != "ogbn-arxiv" else 1
    times, d = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        d = compute(X, y, dataset, variant)
        times.append(time.perf_counter() - t0)
    ref = reference(X, y, dataset)
    fin = np.isfinite(ref) & np.isfinite(d)
    rec = {"dataset": dataset, "seed": seed, "variant": variant, "n": int(len(y)), "d": int(X.shape[1]),
           "n_classes": int(len(np.unique(y))), "times_s": times, "median_s": float(np.median(times)),
           "max_abs_diff": float(np.max(np.abs(ref[fin] - d[fin]))) if fin.any() else 0.0,
           "inf_mismatch": int(np.sum(np.isinf(ref) != np.isinf(d)))}
    rec["exact"] = rec["max_abs_diff"] <= 1e-9 and rec["inf_mismatch"] == 0
    return rec


def main():
    RES.mkdir(exist_ok=True)
    runs = [(ds, 0) for ds in ("cora", "citeseer")] + [("ogbn-arxiv", s) for s in (0, 1, 2)]
    for ds, seed in runs:
        for v in VARIANTS:
            out = RES / f"{ds}_seed{seed}_{v}.json"
            if out.exists():
                continue
            first = RES / f"{ds}_seed0_{v}.json"
            if seed > 0 and json.loads(first.read_text()).get("status") == "exceeded cap":
                out.write_text(json.dumps({"dataset": ds, "seed": seed, "variant": v,
                                           "status": "skipped after cap on seed 0"}, indent=2))
                continue
            t0 = time.time()
            try:
                r = subprocess.run([sys.executable, str(Path(__file__)), "one", ds, str(seed), v],
                                   capture_output=True, text=True, timeout=CAP_S + 600)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-2000:])
                rec = json.loads(r.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                rec = {"dataset": ds, "seed": seed, "variant": v, "status": "exceeded cap", "cap_s": CAP_S}
            rec.update(protocol_sha256=sha(HERE / "protocol.json"), runner_sha256=sha(Path(__file__)),
                       wall_s=time.time() - t0)
            out.write_text(json.dumps(rec, indent=2))
            print(ds, seed, v, rec.get("median_s"), rec.get("exact"), rec.get("max_abs_diff"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        print(json.dumps(one(sys.argv[2], int(sys.argv[3]), sys.argv[4])))
    else:
        main()
