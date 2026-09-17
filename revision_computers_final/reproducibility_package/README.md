# Minimal reproducibility package

This package reproduces the self-anchored, interior-only metric sanity checks without copying restricted checkpoints or the full generated corpus. The full cluster run is controlled by the configurations in `config/`; frozen copies are included here.

Run:

```bash
bash run_all.sh
```

The command runs the metric unit tests and evaluates the included synthetic seven-point example. A successful run reports five interior points, a projection decomposition residual below `1e-6`, and the constant-0.5 baseline values for both the legacy and final ratio grids.

To regenerate the full paper assets from completed cluster outputs, run from the project root:

```bash
python revision_computers_final/scripts/aggregate_metrics.py && \
python revision_computers_final/scripts/run_ablations.py && \
python revision_computers_final/scripts/run_statistical_analysis.py && \
python revision_computers_final/scripts/make_tables_and_figures.py
```

Shared-anchor and human-agreement analyses deliberately stop with `external_pending` until approved shared endpoints and genuine ratings are supplied.
