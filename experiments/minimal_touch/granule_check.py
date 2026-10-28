"""Exactness check of the granule part of Algorithm 1 (stored granules).

The timed experiments never stored granules; this script implements lines 8,
11 and 15 of Algorithm 1 with granules kept as unordered member sets and
compares every granule's MEMBER SET (not only its size) with the batch
definition after every arrival.

Distances come from one precomputed direct-difference matrix D, shared by the
incremental and the batch side, so the check isolates the set logic of
Theorem 2 / Algorithm 1 from floating-point distance evaluation.

Streams: Cora and CiteSeer (all features, row-L1 features as in the paper),
seeds 0-4, random 80% initial set, remaining nodes in random order; plus the
ring 'enemy' arrival of the stress study (n = 256, 1024), where one arrival
shrinks n/2 radii and evicts most granule members.
Usage: python3 experiments/minimal_touch/granule_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))


def pairwise_direct(X, block=256):
    n = X.shape[0]
    D = np.empty((n, n))
    for s in range(0, n, block):
        e = min(s + block, n)
        diff = X[s:e, None, :] - X[None, :, :]
        D[s:e] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return D


def batch_state(D, y, act):
    """Radii and granules of the active node list `act` by Definition 2."""
    sub = D[np.ix_(act, act)]
    ya = y[act]
    enemy = ya[:, None] != ya[None, :]
    rad = np.where(enemy, sub, np.inf).min(axis=1)
    gran = [set(act[np.flatnonzero(sub[i] < rad[i])].tolist()) for i in range(len(act))]
    return dict(zip(act.tolist(), rad.tolist())), dict(zip(act.tolist(), gran))


def insert(D, y, act, rad, gran, z):
    """Algorithm 1 with stored granules; returns shrink set."""
    S, d_new = [], np.inf
    for u in act.tolist():
        d = D[u, z]
        if y[u] != y[z]:
            d_new = min(d_new, d)
            if d < rad[u]:
                rad[u] = d
                S.append(u)
                gran[u] = {w for w in gran[u] if D[u, w] < rad[u]}      # line 8
        elif d < rad[u]:
            gran[u].add(z)                                               # line 11
    rad[z] = d_new
    new_act = np.append(act, z)
    gran[z] = set(new_act[D[z, new_act] < d_new].tolist())               # line 15
    return new_act, S


def run_stream(D, y, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n0 = int(frac * len(y))
    act = np.sort(perm[:n0])
    rad, gran = batch_state(D, y, act)
    mism_r = mism_g = steps = evicted = gained = 0
    for z in perm[n0:].tolist():
        before = {u: len(gran[u]) for u in act.tolist()}
        act, S = insert(D, y, act, rad, gran, z)
        evicted += sum(before[u] - len(gran[u]) for u in S)
        gained += sum(1 for u in before if z in gran[u])
        br, bg = batch_state(D, y, act)
        mism_r += sum(1 for u in act.tolist() if br[u] != rad[u])
        mism_g += sum(1 for u in act.tolist() if bg[u] != gran[u])
        steps += 1
    return dict(steps=steps, radius_mismatches=mism_r, granule_mismatches=mism_g,
                members_evicted=evicted, granule_gains=gained)


def ring_enemy(n, seed, features=16):
    rng = np.random.default_rng(seed)
    y = np.r_[np.repeat([0, 1], n // 2), 1]
    X = rng.normal(0, .02, (n, features))
    X[y[:n] == 1, 0] += 10
    X = np.vstack([X, np.zeros(features)])          # enemy arrival at the origin
    D = pairwise_direct(X)
    act = np.arange(n)
    rad, gran = batch_state(D, y, act)
    before = {u: len(gran[u]) for u in act.tolist()}
    act, S = insert(D, y, act, rad, gran, n)
    br, bg = batch_state(D, y, act)
    return dict(n=n, seed=seed, shrink=len(S),
                members_evicted=sum(before[u] - len(gran[u]) for u in S),
                radius_mismatches=sum(1 for u in act.tolist() if br[u] != rad[u]),
                granule_mismatches=sum(1 for u in act.tolist() if bg[u] != gran[u]))


def main():
    from graph_data import load_planetoid_pyg  # noqa: E402
    out = {"citation": [], "ring_enemy": []}
    for name in ["cora", "citeseer"]:
        X, y = load_planetoid_pyg(name)[:2]  # L1-normalized features (amendment 3)
        D = pairwise_direct(np.asarray(X, dtype=np.float64))
        y = np.asarray(y)
        for seed in range(5):
            r = run_stream(D, y, seed)
            r.update(dataset=name, seed=seed)
            out["citation"].append(r)
            print(r, flush=True)
    for n in (256, 1024):
        for seed in range(5):
            r = ring_enemy(n, seed)
            out["ring_enemy"].append(r)
            print(r, flush=True)
    assert all(r["radius_mismatches"] == 0 and r["granule_mismatches"] == 0
               for v in out.values() for r in v)
    (HERE / "results" / "granule_check_pyg.json").write_text(json.dumps(out, indent=2))
    print("[PASS] all granule member sets equal the batch definition")


if __name__ == "__main__":
    main()
