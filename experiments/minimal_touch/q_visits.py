"""Edge visits Q_j of Eq. (qtouch) on the ogbn-arxiv streams (post-hoc, no timing).

The timed runs record the number of distinct re-evaluated edges, but Theorem 4
bounds E[Q_j], the number of edge VISITS (an edge with both ends in the touch
set is visited twice), with the mean degree d_j of the graph G_j induced by the
first j nodes. This script replays the same three streams (same permutation,
same stable batch radii), recomputes per arrival
    Q_j = deg_j(v_new) + sum_{v in S_j} deg_j(v),   d_j = 2|E(G_j)| / j,
and checks |S_j| and the distinct-edge count against the recorded runs.
Usage: python3 experiments/minimal_touch/q_visits.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "ogb_robustness"))
import e2e_ogb_stream as b  # noqa: E402
from stable_batch import calc_deltas_blocked_stable  # noqa: E402

P = json.loads((EXP / "ogb_robustness" / "protocol.json").read_text())


def replay(X, y, adj, seed):
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    n_init = int(n * P["initial_fraction"])
    order = np.concatenate([np.sort(perm[:n_init]), perm[n_init:n_init + P["insertions"]]])
    Xp, yp = np.ascontiguousarray(X[order]), y[order]
    A = adj[order][:, order].tocsr()
    A.setdiag(False)
    A.eliminate_zeros()
    indptr, indices = A.indptr, A.indices
    deltas = np.full(order.size, np.inf)
    deltas[:n_init] = calc_deltas_blocked_stable(Xp[:n_init], yp[:n_init], block=P["block"])
    deg = np.zeros(order.size, dtype=np.int64)          # degree in the active graph
    sub = A[:n_init][:, :n_init]
    deg[:n_init] = np.diff(sub.indptr)
    n_edges = int(sub.nnz // 2)
    out = []
    for t in range(P["insertions"]):
        v = n_init + t
        m = v
        dist = np.sqrt(np.sum((Xp[:m] - Xp[v]) ** 2, axis=1))
        enemy = yp[:m] != yp[v]
        mask = enemy & (dist < deltas[:m])
        deltas[:m][mask] = dist[mask]
        deltas[v] = dist[enemy].min() if enemy.any() else np.inf
        S = np.flatnonzero(mask)
        nb = indices[indptr[v]:indptr[v + 1]]
        nb = nb[nb < m]                                  # edges to earlier nodes
        deg[nb] += 1
        deg[v] = nb.size
        n_edges += int(nb.size)
        j = m + 1
        T = set(S.tolist()) | {v}
        Q = int(deg[v] + deg[S].sum())
        inner = sum(1 for a in T for u in indices[indptr[a]:indptr[a + 1]] if u <= v and u in T and u != a) // 2
        out.append(dict(s_delta=int(S.size), Q=Q, distinct=Q - inner, dbar=2 * n_edges / j))
    return out


def main():
    X, y, adj = b.load_arxiv()
    res = {}
    for seed in P["seeds"]:
        rec = replay(X, y, adj, seed)
        ref = json.load(open(HERE / "results" / "ogb_minimal" / f"seed_{seed}.json"))["per_insertion_records"]
        assert [r["s_delta"] for r in rec] == [r["s_delta"] for r in ref], "shrink sets differ from timed run"
        mism = sum(r["distinct"] != x["n_updates"] // 2 for r, x in zip(rec, ref))
        Q = np.array([r["Q"] for r in rec]); D = np.array([2 * r["dbar"] for r in rec])
        print(seed, "mean Q", Q.mean(), "mean 2dbar", D.mean(), "distinct mismatches", mism, flush=True)
        res[seed] = dict(records=rec, distinct_mismatches=int(mism))
    (HERE / "results" / "q_visits.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
