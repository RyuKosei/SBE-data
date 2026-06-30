# Implementation TODO

## Ready

- Repository skeleton.
- Config files for models, generation, evaluation, and style axes.
- Deterministic seed scenario generator.
- OpenAI-compatible async generation script with resume support.
- Embedding projection script with projection ratio, interpolation distance, off-axis drift, nearest-neighbor rank, local `transformers` backend, and an offline hash backend for pipeline smoke tests.
- Scalar judge script with JSON parse retry.
- Pairwise local-comparison judge script.
- Analysis script for merged multi-model tables and figures.
- Pilot generation and scalar judging for 4B, 9B, and 35B-A3B.
- Offline pilot flow check under `results/pilot_hash_flow_with_pairwise_dry/`.
- Downloaded `BAAI/bge-m3` to `/home/data_cpfs/zicheng/models/bge-m3`.
- Ran bge-m3 pilot projection for all three pilot generation files.
- Produced `results/pilot_bge_m3_flow/`.
- Downloaded `shibing624/text2vec-base-chinese` to `/home/data_cpfs/zicheng/models/text2vec-base-chinese`.
- Ran text2vec pilot projection for all three pilot generation files.
- Added bootstrap CI by scenario and length-error diagnostics to `05_analyze.py`.
- Added scenario candidate generation and rule-based curation scripts:
  - `scripts/07_generate_scenario_candidates.py`
  - `scripts/06_curate_scenarios.py`

## Needs Cluster Input

- Decide whether an unrelated judge model is available.
- Confirm whether the previous SGLang endpoints are still alive before running real pairwise or new generations.

## Next Engineering Steps

1. Generate and curate scenario candidates for the 3-axis curated pilot.
2. Manually inspect 10 examples per axis.
3. Rerun the 3-axis pilot with 15-20 curated scenarios per axis.
4. Prepare launch commands for full main generation only after curated pilot metrics look sane.

## Research Checks

- Do not overclaim novelty.
- Keep tested model and judge model separation in final reporting.
- Report failures and parse rates, not only successful generations.
- Treat source docs as guidance; keep implementation simple and reproducible first.
