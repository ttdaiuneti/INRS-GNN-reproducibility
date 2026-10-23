# Amendments to protocol minimal-touch-v1 (protocol.json is unchanged)

1. 2026-10-23, after the timing runs finished: added `run.py fullcheck`, which repeats
   experiments/e1_full_scale_correctness.py (full-matrix W and radii, Cora/CiteSeer, all features)
   with the minimal touch set, for seeds 42 and 0-4. Reason: the full-matrix check in the submitted
   paper used the old touch set; the minimal set had been checked only on ogbn-arxiv checkpoints and
   the stress study.
2. 2026-10-23: added `granule_check.py` (stored granules, Algorithm 1 lines 8/11/15), because the
   code base had no incremental granule routine and the earlier "granule sizes" row only compared
   granules derived from radii.
3. 2026-10-23: the citation timing (Table 7), the full-matrix check and the granule check had loaded
   Cora/CiteSeer through shared/graph_data.load_planetoid, which returns raw binary bag-of-words
   features (float32 for Cora, float64 for CiteSeer) and 124 self-loops on CiteSeer. All other studies
   (scaling, classification, trusted-label streams) use load_planetoid_pyg: row-L1-normalized float64
   features, no self-loops, as the manuscript states. The three studies are rerun with load_planetoid_pyg
   (`run.py citation_pyg`, `run.py fullcheck_pyg`, granule_check.py now writing granule_check_pyg.json).
   The earlier outputs are kept. Diagnostics: diagnose_cora_gap.py, diagnose_dtype.py.
4. 2026-10-23: post-hoc analysis, no timing. The OGB runs record distinct re-evaluated edges, but
   Theorem 4 bounds edge VISITS Q_j, and the manuscript compared the distinct count with 2*dbar taken
   at the final checkpoint. `q_visits.py` replays the three minimal-touch streams (same permutation,
   same stable radii), recomputes Q_j and 2*dbar_j per arrival, and asserts that shrink-set sizes and
   distinct-edge counts equal the timed runs (3000/3000). Output results/q_visits.json; macros OgbQ*,
   OgbTwoDbar* in summarize.py.
5. 2026-10-23: post-hoc, no timing. `q_split.py` splits Q_j into deg_j(z) (expectation exactly
   dbar_j) and the shrink-set part (expectation <= dbar_j), per stream with standard errors, because
   arrivals within a stream are not independent. Table tab_random_order and macros OgbSplitMaxZ,
   OgbShrinkStream* in summarize.py. Finding: in streams 0 and 1 the excess of Q over 2*dbar comes from
   the shrink-set part (high-degree shrinking nodes), not from a high-degree arrival; all deviations
   are within 1.5 standard errors, in both directions.
6. 2026-10-23: reporting only, no new run. summarize.py gained three macros for the Discussion and
   Introduction: StressRingRebuildMs and StressRingRebuildEvals (median rebuild time of the smallest
   ring with an enemy arrival, and that time divided by the per-weight cost EdgeEvalUs), and
   OgbShortActiveEnd (active nodes at the final rebuild of the 200-arrival stream, which is the state
   whose rebuild time OgbShortBatchS reports; OgbShortActive is the state at the start of the stream).
   Note on item 5: the statement that the excess "comes from high-degree shrinking nodes" is an
   interpretation; the data show only that the shrink-set part lies above dbar_j with median zero.
   The manuscript does not make the causal claim.
