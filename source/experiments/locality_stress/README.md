# Controlled locality stress test

This is an additive oracle-label maintenance experiment. It does not change the existing GCN protocol or raw results. The exact protocol was written before the first benchmark run and its SHA256 is recorded with execution timestamps in `results/metadata.json`.

## Reproduce

From the project root, using Python with NumPy and SciPy:

```sh
python3 experiments/locality_stress/run.py
python3 experiments/locality_stress/summarize.py
```

`run.py` deliberately refuses to overwrite an existing `results/` directory. Before a new run, archive the current directory under a unique name, retaining its metadata and raw CSV. Then execute both commands; do not combine records from different runs. The summarizer invokes the repository's table generator and updates the manuscript table and numerical prose macros.

## Artifacts

- `protocol.json`: frozen construction, sizes, seed set, timing boundary, checks and aggregation.
- `run.py`: executes existing production incremental and batch kernels; validates against direct-difference radii and scalar weight computation over every graph edge.
- `results/raw_results.csv`: all 300 timing records, 60 graph/seed cases, five repeats each. Repeats do not count as independent graphs.
- `results/metadata.json`: protocol and source hashes, environment, timestamps, peak process RSS and raw hash.
- `results/summary.csv`: unrounded means and population SD across five seed medians; timing components retained.
- `results/table.csv`: formatted values consumed by `build_tables_from_csv.py`.
- `../../docs/locality_stress_results_2026-09-11.md`: interpretation and limitations.
- `../../docs/manuscript/tables/tab_locality_stress.tex`: generated table.
- `../../docs/manuscript/tables/locality_stress_numbers.tex`: generated numerical macros for prose.

No datasets are downloaded. Initial graphs have 256, 1024 or 2048 nodes and 16 features. Each seed pairs the same feature realization across ring/star and control/adverse scenarios. Thread limits are requested before NumPy import. Measurement order alternates. Initialization, reference checks and replica cloning are excluded; state extension and materialized weight writes are included for both methods.

Suite peak RSS includes initialization and verification, so it must not be reported as insertion memory. The sparse implementation still processes incident edges in Python; observed crossovers are properties of the measured kernels and these controlled geometries, not universal algorithm thresholds. All cases, including incremental slowdowns, are retained.
