"""Nodes with safety radius 0 on the citation graphs (row-L1-normalized features, no self-loops).
Quoted in the boundary-case remark of Section 2. Usage: python3 experiments/minimal_touch/zero_radius.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "experiments", ROOT / "shared", ROOT):
    sys.path.insert(0, str(p))
from graph_data import load_planetoid_pyg  # noqa: E402
from theory.incremental_nrs import calc_deltas_blocked  # noqa: E402

out = {}
for name in ("cora", "citeseer"):
    X, y = load_planetoid_pyg(name)[:2]
    d = calc_deltas_blocked(np.asarray(X, dtype=np.float64), np.asarray(y), block=512)
    out[name] = dict(n=int(len(y)), zero=int(np.sum(d == 0)), infinite=int(np.sum(np.isinf(d))))
    print(name, out[name])
(HERE / "results" / "zero_radius.json").write_text(json.dumps(out, indent=2))
