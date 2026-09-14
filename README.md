# INRS-GNN

This repository contains only the implementation (`source/`) and experimental
results (`results/`) supporting the INRS-GNN manuscript. The reusable operators
are in `source/core/`; the paper-facing runners are in `source/experiments/`.

Install dependencies with `pip install -r requirements.txt` and set
`PYTHONPATH="$(pwd)/source:$(pwd)/source/shared"` before running scripts.
