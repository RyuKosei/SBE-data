# Computers major revision R1

This directory contains only the 2026-10-09 major-revision work for
StyleBlend-Bench.  The legacy `revision_computers_final/` tree is an immutable
input.  No human ratings are created or inferred here.

External API credentials and the employee-routing header must be supplied
through `SBE_IAI_API_KEY` and `SBE_IAI_EMP_ID`, respectively. They must never
be written to configs, manifests, logs, JSONL, or command-line arguments.

The attempted DeepSeek routes were blocked before inference by an internal
quota/currency configuration.  The author subsequently replaced that planned
condition with the lightweight `gpt-6-luna` alias as the sixth, non-Qwen
generator.  It is distinct from the `gpt-6-sol` judge but shares the GPT
family/provider, a limitation that must be disclosed.  Neither alias is
treated as a frozen snapshot.

The locked GPT model for this run is the internal `gpt-6-sol` alias.  It was
selected before formal calls after the recorded 80-output pilot; the alias is
not treated as a frozen snapshot.  Formal GPT spending is guarded by the
USD 240 operational cap (inside the author's USD 300 ceiling), with token-based
estimates in `manifests/api_usage_and_budget.json`.  The relay does not expose
the actual wallet balance.

Main reproducibility entry points, in dependency order:

1. `audit_inputs.py`, `prepare_robustness.py`, `run_gpt_external_main.py`,
   and the local robustness generation runners.
2. `consolidate_outputs.py` and the three `embed_dataset.py` runs.
3. `build_evaluation_index.py`, the Qwen/GPT judge runners, and the reference
   screening pipeline.
4. `compute_anchor_metrics.py` once per encoder.
5. `analyze_geometry.py`, `analyze_judges.py`,
   `analyze_robustness_full.py`, and `analyze_stability_retest.py`.
6. `write_revision_reports.py`, `summarize_api_usage.py`, and
   `write_environment_manifest.py` for the final handoff.

All scripts take an explicit output root or operate only within this directory;
none writes into `revision_computers_final/`.
