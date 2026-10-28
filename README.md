# INRS-GNN

Code, protocols and results for the paper *Exact Incremental Maintenance of
Neighborhood Rough Set Radii and Edge Weights on Growing Graphs*.

The repository holds every result reported in the paper and the scripts that
turn these results into the tables, the numbers quoted in the text and
Figure 3. Regenerating them needs no data download.

## Layout

| Path | Content |
|---|---|
| `theory/`, `shared/theory/` | Incremental kernels: radii (`incremental_nrs.py`) and rough edge weights (`incremental_rough_adjacency.py`) |
| `experiments/minimal_touch/` | Timing and exactness runs (`run.py`), stored-granule check, edge-visit counts, zero radii; `results/` holds raw results, per-arrival records and run logs |
| `experiments/indexed_rebuild/` | Rebuild with exact KD-tree and ball-tree indexes, compared with the all-pairs rebuild |
| `experiments/locality_stress/` | Ring and star graphs with adverse arrivals |
| `experiments/label_safe_stream/` | Streams with trusted labels fixed at arrival |
| `experiments/full_pipeline_controls/` | Classification controls: plain GCN, NRS weights and three reweighting controls |
| `experiments/ogb_robustness/` | Numerically stable rebuild used as the reference on ogbn-arxiv |
| `experiments/test_edge_cases.py` | Boundary cases (ties, zero distances, infinite radii) with exact-equality checks |
| `experiments/*.py` (other) | Runners that `minimal_touch/run.py` executes, data loading and table helpers |
| `experiments/results_summary.csv`, `wilcoxon_summary.csv`, `results/` | Stored classification results read by the table scripts |

Each study folder has a `protocol.json` written before its runs. Result
metadata record the SHA-256 of the protocol and of the runner.
`experiments/minimal_touch/AMENDMENTS.md` lists what was added after that
protocol was fixed, and why.

After the runs, comments were shortened in `theory/incremental_rough_adjacency.py`
and `shared/theory/incremental_nrs.py`, and two description strings in
`experiments/minimal_touch/protocol.json`. The code and all protocol settings
are unchanged. The SHA-256 values recorded in the results refer to the files
as they were when the studies ran. For the two Python files,
`experiments/comment_edits.json` stores the recorded SHA-256 together with a
hash of the code without comments and docstrings, and
`experiments/code_hash.py` checks that the current files have the same code
hash. The summary scripts use this check.

## Installation

Python 3.9 was used. Install the pinned dependencies:

```
pip install -r requirements.txt
```

## Regenerating the tables, numbers and Figure 3

From the repository root:

```
mkdir -p paper/tables paper/figures docs
python3 experiments/full_pipeline_controls/summarize.py > /dev/null
cp experiments/full_pipeline_controls/tab_full_pipeline.tex paper/tables/
python3 experiments/minimal_touch/summarize.py
python3 experiments/indexed_rebuild/summarize.py
python3 experiments/build_classification_tables.py
python3 experiments/minimal_touch/plot_scale_speedup.py
```

The scripts write `paper/tables/*.tex` and `paper/figures/`, and two short
run reports in `docs/`. Before writing, the summary scripts check that the
source files match the SHA-256 values recorded when the runs finished. The file
`paper/tables/experiment_numbers.tex` defines every number quoted in the text
as a LaTeX macro; `indexed_numbers.tex` does the same for the indexed rebuilds.
The figure uses the Charis SIL font when it is installed with TeX; otherwise
it falls back to the default font.

## Rerunning the experiments

Download the data (Planetoid through PyTorch Geometric, ogbn-arxiv through
`ogb`; the script checks the SHA-256 of the cached arrays):

```
python3 experiments/fetch_data.py
```

Then, for example:

```
python3 experiments/minimal_touch/run.py citation_pyg
python3 experiments/minimal_touch/run.py scale
python3 experiments/minimal_touch/run.py ogb minimal 0
python3 experiments/minimal_touch/run.py stress minimal
python3 experiments/minimal_touch/run.py fullcheck_pyg
python3 experiments/minimal_touch/granule_check.py
python3 experiments/minimal_touch/q_visits.py
python3 experiments/indexed_rebuild/run.py
python3 experiments/test_edge_cases.py
```

`run.py` lists all studies. Timings depend on the machine. The paper reports
an Apple M4 Pro with 48 GB of memory; the `env` field of each result file
records the software versions.

## Names used in the code

| In the code | In the paper |
|---|---|
| `minimal` policy | touch set $\{v_{new}\} \cup S_\delta$ |
| `conservative` policy | extended touch set $\{v_{new}\} \cup N(v_{new}) \cup S_\delta$ |
| `delta`, `deltas` | radii $\delta$ |
| `W`, rough adjacency | rough edge weights |
| batch | full rebuild |

Result files ending in `_pyg.json` load Cora and CiteSeer through PyTorch
Geometric (row-normalized features, no self-loops), as stated in the paper.
The files without this suffix used raw binary features and are kept for
completeness (see `AMENDMENTS.md`, item 3).
