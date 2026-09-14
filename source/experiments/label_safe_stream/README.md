# Immutable-assignment label-safe insertion check

From repository root:

```sh
python3 experiments/label_safe_stream/run.py
python3 experiments/label_safe_stream/summarize.py
```

The runner refuses to overwrite results/. Preserve any existing results under a unique archive name before intentionally rerunning. The summarizer revalidates source hashes and all records before creating the manuscript table. Old E3 accuracy and OGB timing artifacts are not modified.

Protocol was written before running. See protocol.json, results/metadata.json, per-dataset/seed JSON and verified_hashes.json. The initial caller-signature failure is retained separately (INCIDENT.md).

All train/validation nodes are initially active. Test labels are removed from training tensors; only train labels supply trusted ground truth. Validation labels select the initial checkpoint. Each non-training assignment is made once by the frozen model on the active induced graph. This is a restricted immutable-trust experiment, not a complete adaptive streaming classifier. There are no reported accuracy or latency comparisons.

The final GCN probe compares pre-softmax outputs of the same fixed parameters with alpha=1; a zero error is not evidence of prediction quality. Warm-up and operator construction are repeated independently for each of three seeds per dataset, but these are not replacements for the ten-seed snapshot accuracy study.
