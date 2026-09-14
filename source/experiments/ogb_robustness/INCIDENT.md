# OGB robustness incident — 2026-09-11

Attempt 1 stopped at seed 0, checkpoint t=1000. Initial and t=500 checks passed; maximum radius discrepancy at t=1000 was 1.192e-7, above the frozen 1e-9 tolerance. No complete result was emitted and seeds 1/2 were not started. Original attempt protocol, runner, metadata and log are retained in protocol_attempt1.json, run_attempt1.py and results_attempt1/.

The tolerance is unchanged. A diagnostic replay records each discrepant row, its maintained radius, Gram-rebuild radius, independently computed direct-difference nearest-enemy radius and exact feature equality. Diagnostic timings are not benchmark results. Interpretation and any correction will be appended after the evidence is available.

## Confirmed cause and correction

The diagnostic replay identified exactly two nodes (active-layout indices 58640 and 136122) with different labels and exactly identical feature vectors. The independent direct-distance minimum and maintained radius are both 0. The Gram reference reports 1.1920928955078125e-7; the corresponding max weight discrepancy is 2.3915809499452934e-8. This is cancellation error in the batch reference, not evidence of an incorrect incremental update. Diagnostics are retained in diagnostic_seed0.json.failure.json.

Version 2 adds a conservative near-zero guard to the blocked Gram rebuild: directly recompute all enemy candidates in a near-minimum band for flagged rows. Ordinary distances retain the Gram path. The guard and refinement are included in batch timing. Neither incremental kernels nor tolerance nor seeds/stream length/checkpoints change. Tests against direct full recomputation cover exact opposite-label duplicates, near duplicates, competing enemies, three feature scales, three block sizes, no enemies and empty inputs. All pass. Every seed is rerun from scratch under v2; failed/diagnostic timings are excluded from the new result table and retained separately. This amended protocol is transparent, not preregistered or blinded.
