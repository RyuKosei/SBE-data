# StyleBlend-Bench

StyleBlend-Bench is a Chinese benchmark and reproducibility package for evaluating continuous style-ratio control in black-box language models. The accompanying study uses model-specific endpoint anchors, excludes the two mechanically perfect endpoints from primary errors, and reports calibration, off-axis drift, trajectory geometry, content diagnostics, robustness, and serving efficiency for five Qwen3.5 and Qwen3.6 checkpoints.

## Paper experiment package

The final package for the *Computers* submission is under [`revision_computers_final/`](revision_computers_final/). It contains:

- 320 frozen scenarios across eight style axes;
- 33,600 main generations and 9,600 prompt/temperature robustness records;
- embeddings from BGE-M3, text2vec-base-chinese, and multilingual-E5-large-instruct;
- sample-, scenario-, and model-level metrics, statistical analyses, tables, figures, and run manifests;
- source scripts, frozen configurations, tests, and a minimal reproducibility package.

The manuscript does not report human-perception results or shared-anchor rankings. Prepared but uncollected human and shared-anchor branches remain explicitly marked `external_pending` and are not used in the paper's conclusions.

## Benchmark construction

The released construction trail includes:

- candidate scenarios: [`data/scenarios/candidate_scenarios_8axis_v1.jsonl`](data/scenarios/candidate_scenarios_8axis_v1.jsonl);
- final scenarios: [`data/scenarios/styleblend_8axis_scenarios_v1.jsonl`](data/scenarios/styleblend_8axis_scenarios_v1.jsonl);
- candidate drafting: [`scripts/07_generate_scenario_candidates.py`](scripts/07_generate_scenario_candidates.py);
- deterministic filtering: [`scripts/06_curate_scenarios.py`](scripts/06_curate_scenarios.py);
- domain-stratified selection and validation: [`scripts/09_review_scenario_pool.py`](scripts/09_review_scenario_pool.py);
- validation summary: [`results/scenario_pool_8axis_v1/tables/scenario_validation_summary.csv`](results/scenario_pool_8axis_v1/tables/scenario_validation_summary.csv).

## Minimal reproduction

Create an environment with Python 3.10 or later, then run:

```bash
python3 -m pip install -r revision_computers_final/reproducibility_package/requirements.txt
bash revision_computers_final/reproducibility_package/run_all.sh
```

A successful run reports 11 passing tests and evaluates the included synthetic seven-point trajectory.

To regenerate the paper's derived tables and figures from the completed outputs, run from the repository root:

```bash
python3 revision_computers_final/scripts/aggregate_metrics.py
python3 revision_computers_final/scripts/run_ablations.py
python3 revision_computers_final/scripts/run_statistical_analysis.py
python3 revision_computers_final/scripts/make_tables_and_figures.py
```

Model and encoder checkpoints are not redistributed. Their public identifiers and frozen revisions are recorded in [`revision_computers_final/config/`](revision_computers_final/config/). Absolute cluster paths retained in frozen configurations and manifests document the original run environment; local users should replace them with their own checkpoint and output locations.

## Repository map

- [`revision_computers_final/PACKAGE_CONTENTS.md`](revision_computers_final/PACKAGE_CONTENTS.md): delivery inventory and validation status.
- [`revision_computers_final/completion_report.md`](revision_computers_final/completion_report.md): experiment completion report and claim boundaries.
- [`revision_computers_final/data/`](revision_computers_final/data/): consolidated generations, embeddings, and frozen data artifacts.
- [`revision_computers_final/metrics/`](revision_computers_final/metrics/): derived metrics.
- [`revision_computers_final/statistics/`](revision_computers_final/statistics/): uncertainty estimates and hypothesis tests.
- [`revision_computers_final/tables/`](revision_computers_final/tables/) and [`revision_computers_final/figures/`](revision_computers_final/figures/): manuscript assets.
- [`revision_computers_final/scripts/`](revision_computers_final/scripts/): analysis and generation code.

## License and citation

Source code is released under the MIT License. Benchmark records, generated outputs, derived data, figures, and documentation are released under CC BY 4.0 unless a file states otherwise. See [`LICENSE`](LICENSE), [`DATA_LICENSE.md`](DATA_LICENSE.md), and [`CITATION.cff`](CITATION.cff).
