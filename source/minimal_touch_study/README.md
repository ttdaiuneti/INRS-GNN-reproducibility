# Minimal touch set study (revised manuscript, October 2026)

This folder holds the code and results behind the timing, exactness and
boundary-case numbers of the revised manuscript
("Exact Incremental Maintenance of Neighborhood Rough Set Radii and Edge
Weights on Growing Graphs"). It keeps the directory layout of the working
project, so the scripts run without path changes. The older results in
`../../results/` are kept for reference; where a number appears in both, the
revised manuscript uses the one in this folder.

## What is here

| Path | Content |
|---|---|
| `experiments/minimal_touch/protocol.json` | Protocol, frozen before the first run |
| `experiments/minimal_touch/AMENDMENTS.md` | Every change made after the protocol was frozen, with reasons |
| `experiments/minimal_touch/run.py` | Timing and exactness runs with the minimal touch set {v_new} ∪ S_δ and the old touch set |
| `experiments/minimal_touch/granule_check.py` | Stored-granule check (Algorithm 1, lines 8, 11, 15) |
| `experiments/minimal_touch/q_visits.py`, `q_split.py` | Edge visits Q_j and its split, for Theorem 4 (Table 10) |
| `experiments/minimal_touch/zero_radius.py` | Number of nodes with radius 0 (Section 2) |
| `experiments/test_edge_cases.py` | Boundary-case streams with exact-equality checks (Appendix B) |
| `experiments/minimal_touch/summarize.py` | Builds all manuscript tables and macros from the results |
| `experiments/minimal_touch/results/` | Raw results, per-arrival records and logs of every run |
| `theory/`, `shared/theory/` | The incremental kernels used by all runs (unchanged by this study) |

The other files under `experiments/` are the original runners that `run.py`
patches (touch-set construction and output path only, each replacement
asserted to occur once) and the inputs that `summarize.py` reads.
`experiments/e0_gnn_prototype.py` is included only because the trusted-label
summary verifies its hash.

## Regenerating the tables

No data download is needed. From this folder:

```
mkdir -p docs/manuscript_v2/tables docs/manuscript_v2/figures
python3 experiments/minimal_touch/summarize.py
```

This writes `docs/manuscript_v2/tables/*.tex`, including
`experiment_numbers.tex`, which holds every number quoted in the text. Before
release, the output was compared with the manuscript sources: all tables,
macros and the figure wrapper were identical.

## Rerunning the experiments

Fetch the data with `python3 ../experiments/fetch_data.py` and make it
visible here:

```
ln -s ../../core/data shared/data
```

Then, for example, `python3 experiments/minimal_touch/run.py citation_pyg`,
`python3 experiments/minimal_touch/granule_check.py` or
`python3 experiments/test_edge_cases.py`. See `run.py` for the list of
studies. Timings depend on the machine; the manuscript reports one Apple M4 Pro
with 48 GB of memory (see the `env` field of each result file).
