"""Minimal-touch-set rerun (protocol.json, frozen before execution).

Each study executes an ORIGINAL experiment script after asserted text
replacements (touch-set construction, output path, expected counts). Kernels in
theory/ and shared/theory/ are untouched. Usage:

    python3 experiments/minimal_touch/run.py stress <conservative|minimal>
    python3 experiments/minimal_touch/run.py ogb <conservative|minimal> <seed>
    python3 experiments/minimal_touch/run.py ogb200
    python3 experiments/minimal_touch/run.py citation
    python3 experiments/minimal_touch/run.py scale
"""
import hashlib
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
RES = HERE / "results"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def patched_module(path, name, replacements):
    """Load `path` as module `name` after applying (old, new, count) edits."""
    src = Path(path).read_text()
    for old, new, count in replacements:
        found = src.count(old)
        assert found == count, f"{path}: expected {count} x {old!r}, found {found}"
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def record(study, policy, extra=None):
    meta = {"study": study, "policy": policy,
            "protocol_sha256": sha(HERE / "protocol.json"),
            "runner_sha256": sha(Path(__file__))}
    meta.update(extra or {})
    return meta


def stress(policy):
    assert policy in ("conservative", "minimal")
    out = RES / f"stress_{policy}"
    reps = [("out=HERE/'results';out.mkdir(exist_ok=False)",
             f"out=Path({str(out)!r});out.mkdir(parents=True,exist_ok=False)", 1)]
    if policy == "minimal":
        reps += [
            ("touch={len(y),nb}", "touch={len(y)}", 1),
            ("assert len(T)==(n//2+1 if scenario=='ring_enemy' else 2)",
             "assert len(T)==(n//2+1 if scenario=='ring_enemy' else 1)", 1),
            ("expected_q={'ring_control':4,'ring_enemy':n+2,'star_leaf':3,'star_hub':n+1}[scenario]",
             "expected_q={'ring_control':1,'ring_enemy':n+2,'star_leaf':1,'star_hub':1}[scenario]", 1),
        ]
    mod = patched_module(EXP / "locality_stress" / "run.py", "stress_run", reps)
    mod.main()
    (out / "minimal_touch_meta.json").write_text(json.dumps(record("stress", policy), indent=2))


def ogb(policy, seed):
    assert policy in ("conservative", "minimal")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(EXP / "ogb_robustness"))
    P = json.loads((EXP / "ogb_robustness" / "protocol.json").read_text())
    assert seed in P["seeds"]
    reps = []
    if policy == "minimal":
        reps.append(("touch = {v_new} | {int(u) for u in nbrs if active[u]}",
                     "touch = {v_new}", 1))
    b = patched_module(EXP / "e2e_ogb_stream.py", "e2e_ogb_stream_mt", reps)
    from stable_batch import calc_deltas_blocked_stable
    b.calc_deltas_blocked = calc_deltas_blocked_stable
    b.SEED = seed
    b.K_INSERTS = P["insertions"]
    b.CHECKPOINTS = tuple(P["checkpoints"])
    out = RES / f"ogb_{policy}"
    out.mkdir(parents=True, exist_ok=True)
    b.OUT = str(out / f"seed_{seed}.json")
    assert b.INIT_FRAC == P["initial_fraction"] and b.BLOCK == P["block"] and b.CHECK_TOL == P["tolerance"]
    assert not Path(b.OUT).exists()
    b.main()
    (out / f"seed_{seed}_meta.json").write_text(json.dumps(record("ogb_robustness", policy, {"seed": seed}), indent=2))


def ogb200():
    out = RES / "ogb_stream200_minimal.json"
    assert not out.exists()
    b = patched_module(EXP / "e2e_ogb_stream.py", "e2e_ogb_stream_200",
                       [("touch = {v_new} | {int(u) for u in nbrs if active[u]}", "touch = {v_new}", 1)])
    b.OUT = str(out)
    b.main()


def citation():
    out = RES / "citation_fair_minimal.json"
    assert not out.exists()
    m = patched_module(EXP / "e2e_fair_timing.py", "e2e_fair_timing_mt", [
        ("touch = structural_touch_set(n_old, neighbors)", "touch = {n_old}", 1),
        ('out_path = os.path.join(out_dir, "e2e_fair_timing_results.json")', f"out_path = {str(out)!r}", 1),
    ])
    m.main()


def scale():
    out = RES / "scale_minimal.json"
    assert not out.exists()
    m = patched_module(EXP / "e2e_scale_timing.py", "e2e_scale_timing_mt", [
        ("touch = structural_touch_set(n_old, neighbors)", "touch = {n_old}", 2),
        ('out = os.path.join(os.path.dirname(__file__), "results", "e2e_scale_timing_results.json")',
         f"out = {str(out)!r}", 1),
    ])
    m.main()


def fullcheck():
    """Amendment 1: full-matrix Cora/CiteSeer check with the minimal touch set."""
    out = RES / "fullcheck_minimal.json"
    assert not out.exists()
    m = patched_module(EXP / "e1_full_scale_correctness.py", "e1_mt", [
        ("touch = structural_touch_set(n_old, neighbors)", "touch = {n_old}", 1),
        ('out_path = os.path.join(out_dir, "e1_full_scale_correctness.json")', f"out_path = {str(out)!r}", 1),
    ])
    results = []
    for seed in [42, 0, 1, 2, 3, 4]:
        m.SEED = seed
        for ds in m.DATASETS:
            r = m.run_dataset(ds)
            r["seed"] = seed
            print(ds, seed, r["max_delta_diff_inc_vs_batch"], r["max_W_diff_inc_vs_batch_full_matrix"], flush=True)
            results.append(r)
    out.write_text(json.dumps(results, indent=2))


LOADER_OLD = 'from experiments.graph_data import load_planetoid\n'
LOADER_NEW = 'from experiments.graph_data import load_planetoid_pyg as _lpp\ndef load_planetoid(name):\n    X, y, adj, *_ = _lpp(name)\n    return np.ascontiguousarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(adj, dtype=bool)\n'


def citation_pyg():
    """Amendment 3: Table 7 with the L1-normalized float64 features of every other study."""
    out = RES / "citation_fair_minimal_pyg.json"
    assert not out.exists()
    m = patched_module(EXP / "e2e_fair_timing.py", "e2e_fair_timing_pyg", [
        (LOADER_OLD, LOADER_NEW, 1),
        ("touch = structural_touch_set(n_old, neighbors)", "touch = {n_old}", 1),
        ('out_path = os.path.join(out_dir, "e2e_fair_timing_results.json")', f"out_path = {str(out)!r}", 1),
    ])
    m.main()


def fullcheck_pyg():
    """Amendment 3: full-matrix check with the L1-normalized float64 features."""
    out = RES / "fullcheck_minimal_pyg.json"
    assert not out.exists()
    m = patched_module(EXP / "e1_full_scale_correctness.py", "e1_pyg", [
        (LOADER_OLD, LOADER_NEW, 1),
        ("touch = structural_touch_set(n_old, neighbors)", "touch = {n_old}", 1),
    ])
    results = []
    for seed in [42, 0, 1, 2, 3, 4]:
        m.SEED = seed
        for ds in m.DATASETS:
            r = m.run_dataset(ds)
            r["seed"] = seed
            print(ds, seed, r["max_delta_diff_inc_vs_batch"], r["max_W_diff_inc_vs_batch_full_matrix"], flush=True)
            results.append(r)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "stress":
        stress(sys.argv[2])
    elif cmd == "ogb":
        ogb(sys.argv[2], int(sys.argv[3]))
    elif cmd == "ogb200":
        ogb200()
    elif cmd == "citation":
        citation()
    elif cmd == "scale":
        scale()
    elif cmd == "fullcheck":
        fullcheck()
    elif cmd == "citation_pyg":
        citation_pyg()
    elif cmd == "fullcheck_pyg":
        fullcheck_pyg()
    else:
        raise SystemExit(__doc__)
