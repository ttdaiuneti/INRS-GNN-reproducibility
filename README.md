# INRS-GNN

This repository contains only the implementation (`source/`) and experimental
results (`results/`) supporting the INRS-GNN manuscript. The reusable operators
are in `source/core/`; the paper-facing runners are in `source/experiments/`.

Install dependencies with `pip install -r requirements.txt` and set
`PYTHONPATH="$(pwd)/source:$(pwd)/source/shared"` before running scripts.

## Revised manuscript (October 2026)

The timing, exactness and boundary-case results of the revised manuscript are in
`source/minimal_touch_study/`, together with the code that produced them and a
script that regenerates every table and number of the paper from the stored
results. See `source/minimal_touch_study/README.md`. The results in `results/`
belong to the earlier version and are kept for reference.
