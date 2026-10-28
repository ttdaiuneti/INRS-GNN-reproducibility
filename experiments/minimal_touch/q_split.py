"""Split Q_j of q_visits.json into the arrival's own degree and the shrink-set part.

E[deg_j(z) | V_j] = dbar_j exactly and E[sum_{v in S_j} deg_j(v) | V_j] <= dbar_j (proof of
Theorem 4). The own degree needs only the permutation and the graph, so no radius is recomputed.
Statistics are per stream: arrivals of one stream share their node set and are not independent.
Usage: python3 experiments/minimal_touch/q_split.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(EXP))
import e2e_ogb_stream as b  # noqa: E402

P = json.loads((EXP / "ogb_robustness" / "protocol.json").read_text())
qv = json.loads((HERE / "results" / "q_visits.json").read_text())


def stats(a):
    return dict(mean=float(a.mean()), se=float(a.std(ddof=1) / np.sqrt(len(a))),
                median=float(np.median(a)), max=float(a.max()))


X, y, adj = b.load_arxiv()
n = X.shape[0]
out = {}
for seed in P["seeds"]:
    perm = np.random.default_rng(seed).permutation(n)
    n_init = int(n * P["initial_fraction"])
    order = np.concatenate([np.sort(perm[:n_init]), perm[n_init:n_init + P["insertions"]]])
    A = adj[order][:, order].tocsr()
    A.setdiag(False)
    A.eliminate_zeros()
    own = np.array([int(np.sum(A.indices[A.indptr[v]:A.indptr[v + 1]] < v))
                    for v in range(n_init, n_init + P["insertions"])])
    rec = qv[str(seed)]["records"]
    Q = np.array([r["Q"] for r in rec])
    S = np.array([r["s_delta"] for r in rec])
    dbar = np.array([r["dbar"] for r in rec])
    assert np.all(own <= Q)
    out[seed] = dict(shrink=stats(S), own=stats(own), shrink_part=stats(Q - own), Q=stats(Q),
                     dbar_mean=float(dbar.mean()))
    print(seed, {k: (round(v["mean"], 2), round(v["se"], 2)) for k, v in out[seed].items() if k != "dbar_mean"},
          round(out[seed]["dbar_mean"], 2), flush=True)
(HERE / "results" / "q_split.json").write_text(json.dumps(out, indent=2))
