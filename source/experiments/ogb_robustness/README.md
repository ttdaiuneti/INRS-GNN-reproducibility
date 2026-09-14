# OGB robustness extension

Frozen additive experiment: three independent seeded streams, 1,000 insertions each and full-state verification at steps 0, 500 and 1,000. Each seed changes the initial set and arrival order together. Original 200-step results are preserved.

Run from the repository root with the existing NumPy/SciPy environment and locally cached arxiv arrays:

```sh
python3 experiments/ogb_robustness/run.py
python3 experiments/ogb_robustness/summarize.py
```

The runner executes serial fresh processes, reusing the original incremental kernels. Version 2 refines near-zero Gram distances in the batch reference; see INCIDENT.md for the retained failed attempt and direct-distance diagnosis. The full guard cost is included in batch timing. It refuses to overwrite `results/`; archive that directory with a unique name before an intentional rerun. Do not mix different runs. Source/protocol and dataset hashes are saved before execution; result hashes are saved after completion. Each worker retains a log.

The summarizer validates all steps/checkpoints and generates raw_results.csv, summary.csv, checks.csv, blocks.csv and table.csv. Manuscript table and numerical macros are generated from those validated aggregates. Timing statistics within a stream and variation between the three stream means are distinct; the latter uses population SD, not a confidence interval.

Initialization and checkpoint validation are excluded from insertion timing. Peak RSS includes both. Topology is laid out in advance and replayed through an active mask. The benchmark does not measure unknown-edge ingestion, classification, or the full remaining 20% stream. CPU thread settings are inherited and recorded, not silently changed to match the separately single-thread-limited synthetic stress test.
