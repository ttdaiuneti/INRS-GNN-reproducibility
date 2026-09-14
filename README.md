# INRS-GNN reproducibility package

This directory contains the code, frozen protocols and experimental records
supporting the manuscript **“Incremental Maintenance of Neighborhood Rough Set
Structure on Evolving Graphs: Locality Guarantees and Efficient Updates.”**

Permanent repository location:

<https://github.com/ttdaiuneti/INRS-GNN/tree/main/reproducibility>

## Contents

- `source/experiments/`: experiment runners, checks and table/figure generators.
- `source/theory/`: exact incremental NRS and rough-adjacency implementation.
- `source/shared/`: vendored shared loaders and NRS routines; no submodule is
  required inside this package.
- `source/pilot/`: synthetic locality pilot.
- `results/canonical/`: JSON and CSV artifacts behind the main accuracy,
  correctness, timing and diagnostic results.
- `results/full_pipeline_controls/`: ten-seed downstream control study.
- `results/label_safe_stream/`: immutable-assignment label-safe stream checks.
- `results/locality_stress/`: all 300 controlled stress timing records and
  aggregates.
- `results/ogb_robustness/`: three 1,000-insertion ogbn-arxiv streams, logs,
  checkpoint checks and aggregates.
- `ARTIFACT_INDEX.csv`: mapping from manuscript evidence to source and results.
- `MANIFEST.sha256`: checksums for every distributed file except the manifest.

Downloaded datasets and local caches are deliberately excluded. Cora,
CiteSeer and PubMed are downloaded by PyTorch Geometric. The ogbn-arxiv arrays
are obtained through OGB and checked against the SHA-256 digests embedded in
`source/experiments/fetch_data.py`.

## Environment

The reported artifacts were produced with Python 3.9.6 on macOS arm64. Install
the pinned packages in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$(pwd)/source:$(pwd)/source/shared:${PYTHONPATH}"
```

Some pinned binary packages may require a compatible Python/platform build.
The result JSON and metadata files retain the recorded runtime information and
source/protocol hashes where the corresponding experiment supplied them.

## Verify the released package

From this directory, verify file integrity and the presence of every required
artifact:

```bash
python3 verify_package.py
```

The check uses only the Python standard library and does not rerun training.

## Reproduce the main artifacts

Fetch or verify dataset caches under `source/shared/data`:

```bash
python source/experiments/fetch_data.py
```

Run the primary experiments from this directory:

```bash
python source/experiments/e1_full_scale_correctness.py
python source/experiments/e3_pyg_aligned_benchmark.py
python source/experiments/e3_label_safe_diagnostics.py
python source/experiments/e2e_fair_timing.py
python source/experiments/e2e_scale_timing.py
python source/experiments/e2e_ogb_stream.py
```

These scripts write new outputs below `source/experiments/results/`. Preserve
the released `results/` tree as the immutable record used by the paper.
Training and large-graph timing runs can be long and hardware dependent.

The additive experiment directories contain their own frozen protocols and
instructions:

```bash
python source/experiments/locality_stress/run.py
python source/experiments/locality_stress/summarize.py

python source/experiments/label_safe_stream/run.py
python source/experiments/label_safe_stream/summarize.py

python source/experiments/full_pipeline_controls/run.py --check-only
python source/experiments/full_pipeline_controls/run.py
python source/experiments/full_pipeline_controls/summarize.py

python source/experiments/ogb_robustness/run.py
python source/experiments/ogb_robustness/summarize.py
```

Each runner's overwrite and caching rules are documented in its local README.
Accuracy values should agree under the pinned protocol and seeds; wall-clock
latencies can vary across machines.

## Result provenance

`ARTIFACT_INDEX.csv` identifies the code and released result directory for each
evidence block. Generated tables and figures are derived from JSON/CSV records;
the paper's claims do not depend on downloaded result files or private data.

The repository currently carries no open-source licence. Copyright remains
with the author; the package is public for peer review and reproducibility.

