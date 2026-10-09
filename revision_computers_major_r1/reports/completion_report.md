# Completion report

Generated: 2026-10-09T15:54:44.319372Z

## Counts and status

- Legacy Qwen main retained: 33,600.
- External GPT-6-Luna requests attempted: 6720/6,720 — completed; usable text: 6719; recorded failures: 1.
- Robustness condition records: 96,000; valid text: 95,994; recorded failures: 6.
- Evaluation index: 112,313 usable; 7 explicit generation failures excluded; all 112,320 planned identities accounted for: True.
- Qwen judge valid/cache records: 112,266; terminal evaluation failures: 47.
- GPT judge valid records: 112,313; terminal evaluation failures: 0.
- Fixed references: 640 generated; Qwen screening valid: 639.
- Judge stability retest: all 240 selected identities accounted for by both judges; valid retest scores were Qwen 239/240 and GPT 240/240, with terminal failures retained separately.
- Three anchor protocols × three encoders: six-model analysis complete on all valid outputs.

## API usage and cost

- GPT estimated cost from recorded usage and pilot: USD 183.6565.
- Conservative total with interrupted in-flight-call and unrecorded comparison-pilot allowances: USD 183.9378.
- Operational cap: USD 240.00; estimated headroom: USD 56.34.
- Actual account balance: unavailable from relay.
- External GPT-6-Luna generation cost: USD 0.4769.
- GPT hashed response IDs captured: 109881; missing from early records: 2432. The missing early IDs are an audit limitation and are not reconstructed.

## Reproduction entry points

Exact credential-free commands are preserved in `reports/runbook.md`.

- Audit: `python scripts/audit_inputs.py --config config/experiment.yaml --output-root .`
- Robustness preparation/generation: `prepare_robustness.py`, `run_qwen_robustness.py`, then `consolidate_outputs.py`.
- External main generation: `run_gpt_external_main.py` (resume-safe; token budget and response status recorded per request).
- References/judges: `run_gpt_references.py`, `build_*_evaluation_index.py`, `run_qwen_judge.py`, `run_gpt_judge.py`.
- Embeddings/geometry: `embed_dataset.py`, then `compute_anchor_metrics.py` once per encoder.
- Final analyses: `analyze_reference_quality.py`, `analyze_geometry.py`, `analyze_judges.py`, `analyze_robustness_full.py`, `analyze_stability_retest.py`, `select_failure_cases.py`.
- Tests: `pytest -q revision_computers_major_r1/tests` from the repository root.
- Final identity/count/hash audit: `python scripts/audit_final_coverage.py --output-root .`.

## Worktree and authorization boundaries

- Legacy directory was used read-only; new writes are contained under `revision_computers_major_r1/`.
- No human labels were created or inferred.
- No manuscript submission or email was performed.
- On the author's explicit instruction, this completed package was prepared for delivery to the repository's public `main` branch. The exact delivery commit is recorded in Git history.
- The new revision tree contains 281 files totaling 3.38 GiB; JSONL, NPZ, and Parquet payloads use Git LFS. See `manifests/environment_and_hashes.json` for the audited base commit, environment, exact paths, and critical hashes.
