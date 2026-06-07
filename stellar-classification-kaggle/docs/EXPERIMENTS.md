# Experiment Workflow

The project uses numbered scripts to keep the experiment history readable.

## Script Groups

- `01_baseline_autogluon.py`: AutoGluon baseline and smoke test.
- `02_feature_exploration.py`: automatic and grouped feature exploration.
- `03_layer1_oof_experiments.py`: layer-1 OOF model training.
- `04_stack_oof.py`: merge OOF runs and train stackers.
- `05_two_stage_threshold_experiment.py`: threshold search and class-specific adjustment.
- `06_autogluon_oof_extensions.py`: AutoGluon extension experiments.
- `07_small_hyperparam_sweep.py`: small hyperparameter sweeps.
- `08_register_submissions.py`: scan and register submission metadata.
- `09_disagreement_arbitration.py`: compare and arbitrate disagreeing predictions.
- `10+`: later scripts for materialization, guard thresholds, local evaluation, and stack feature experiments.

## Run Directory Convention

Each experiment writes artifacts under:

```text
outputs/<workflow>/<run_name>/
```

This keeps runs separate and makes it easier to compare submissions.

## Submission Registry

Generated submissions are not committed, but metadata is tracked through:

```text
submissions/submission_registry.csv
submissions/submission_registry.jsonl
```

The registry records:

- submission path
- SHA256
- row count
- duplicate or missing IDs
- class distribution
- workflow
- run name
- script
- model name
- metrics
- key parameters

This is useful for avoiding repeated submissions and for explaining why a particular file was created.

## Public Portfolio Cleanup

The GitHub-ready version intentionally removes:

- raw data
- generated submission CSVs
- model binaries
- OOF arrays
- processed caches

The goal is to show the workflow and engineering quality without uploading competition data or large artifacts.
