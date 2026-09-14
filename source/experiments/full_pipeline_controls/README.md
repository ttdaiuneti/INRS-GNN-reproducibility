# Full-pipeline reweighting controls

Additive experiment; does not overwrite historical E3, OGB, or stress results.

Run from any working directory:

```sh
python3 INRS-GNN/experiments/full_pipeline_controls/run.py --check-only
python3 INRS-GNN/experiments/full_pipeline_controls/run.py
python3 INRS-GNN/experiments/full_pipeline_controls/summarize.py
```

Each completed dataset/seed is atomic and resumable only with identical source hashes. `protocol.json` was written before results. `results/metadata.json` records source hashes, environment and timestamps. Ten seeds repeat warm-up and downstream initialization; graph controls vary with the fresh pseudo-label model. Data and realized assignments/support have hashes in each seed record. Test labels are replaced before model fitting and are accessed for accuracy only after validation selection.

GCN forward caches the fixed normalized adjacency-feature product AX. The implementation is the same GCN as historical code, with regression checks for outputs and parameter gradients. CPU one thread; four lr/wd candidates of 200 epochs per method, fixed alpha=1, fixed tau=.8, width 16. Equal downstream trial/epoch budget does not mean equal total compute. Warm-up is a shared extra cost for NRS-dependent controls, recorded separately. Ties select earliest epoch/first config.

Do not append seeds/configs after looking at test outcomes. A temporal utility experiment is conditional on the locked gate; no-go is an allowed scientific outcome and does not prove no possible NRS benefit.
