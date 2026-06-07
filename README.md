# Stellar Classification Kaggle Project

This repository contains a cleaned portfolio version of my Kaggle workflow for the **Stellar Classification** competition.

The task is a tabular multi-class classification problem for astronomical objects:

- `GALAXY`
- `QSO`
- `STAR`

The local experiment pipeline was built around feature engineering, cross-validation, OOF prediction management, stacking, threshold experiments, AutoGluon baselines, and submission tracking.

> Data files and generated submissions are intentionally excluded from this public repository. See `data/raw/.gitkeep` and the setup section below.

## Highlights

- Worked with roughly **400K rows** of tabular astronomy data.
- Built reusable feature engineering utilities for magnitudes, colors, redshift, SDSS-inspired features, and hard-slice diagnostics.
- Ran model families including AutoGluon, XGBoost, LightGBM, CatBoost, Random Forest, MLP, and RealMLP-style neural tabular experiments.
- Used 5-Fold cross-validation and OOF predictions for model comparison and stacking.
- Explored Logistic Regression stacking, AutoGluon stacking, threshold tuning, disagreement arbitration, and public-LB-safe evaluation.
- Maintained submission metadata and a registry for reproducible experiment tracking.
- Current best public leaderboard result during the ongoing competition reached approximately **Top 20%**.

## Repository Layout

```text
stellar-classification-kaggle/
  configs/                 Example run configs
  data/
    raw/                   Put train.csv, test.csv, sample_submission.csv here
    processed/             Optional local feature caches
  docs/
    APPROACH.md            Modeling strategy and feature engineering notes
    EXPERIMENTS.md         Experiment workflow and tracking notes
  notebooks/               Lightweight exploratory notebooks
  scripts/                 Numbered experiment scripts
  src/stellar/             Shared reusable package
  submissions/             Registry metadata, no generated CSV submissions
  requirements.txt
  requirements-optional.txt
```

## Tech Stack

- Python
- pandas / numpy / scipy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- AutoGluon
- joblib
- Jupyter

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

For the full experiment suite:

```powershell
python -m pip install -r requirements.txt -r requirements-optional.txt
```

Place competition data under:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

## Typical Workflow

### 1. Baseline

```powershell
python scripts/01_baseline_autogluon.py --dry-run --run-name dry_run
python scripts/01_baseline_autogluon.py --run-name baseline_medium_15m
```

### 2. Feature Exploration

```powershell
python scripts/02_feature_exploration.py --run-name feature_exploration
```

### 3. Layer-1 OOF Models

```powershell
python scripts/03_layer1_oof_experiments.py --suite smoke --dry-run --run-name smoke_check
python scripts/03_layer1_oof_experiments.py --run-name balanced_5fold
```

### 4. Stacking

```powershell
python scripts/04_stack_oof.py --run-name stack_lr_all_models
```

### 5. Submission Registry

```powershell
python scripts/08_register_submissions.py --write
```

## What Is Tracked

The public repo keeps:

- reusable source code
- experiment scripts
- example configs
- notebooks
- submission registry metadata
- documentation

The public repo excludes:

- raw Kaggle data
- processed feature caches
- trained model files
- OOF matrices
- generated submission CSV files
- local package caches

## Modeling Approach

See [docs/APPROACH.md](docs/APPROACH.md).

## Experiment Notes

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Portfolio Notes

This project demonstrates practical machine learning workflow skills:

- tabular feature engineering
- model comparison
- cross-validation design
- OOF stacking
- metric-driven iteration
- experiment registry
- reproducible project organization
