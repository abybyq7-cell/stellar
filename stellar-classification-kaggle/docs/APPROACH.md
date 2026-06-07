# Modeling Approach

## Problem

The competition is a multi-class tabular classification task for astronomical object labels:

- `GALAXY`
- `QSO`
- `STAR`

The main modeling challenge is not only optimizing global accuracy, but also improving recall for harder classes and reducing confusion between visually similar object categories.

## Validation

The project uses stratified 5-Fold cross-validation for stable local estimates.

OOF predictions are saved and reused for:

- model comparison
- stacking
- threshold experiments
- disagreement analysis
- public leaderboard sanity checks

## Feature Engineering

Reusable feature engineering lives in `src/stellar/features.py`.

Feature groups include:

- magnitude color differences such as `u_minus_g`, `g_minus_r`, `r_minus_i`, `i_minus_z`
- magnitude summary statistics such as mean, std, range
- redshift interactions
- angle cycle features using sine and cosine transforms
- SDSS-inspired color and slope features
- hard-slice diagnostic features for STAR/GALAXY and low-redshift cases

## Models

The experiment suite covers:

- AutoGluon tabular models
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Histogram-based models
- MLP / RealMLP-style neural tabular experiments
- Logistic Regression stackers

## Ensembling

The project explores:

- simple averaging
- weighted ensembling
- OOF-based Logistic Regression stacking
- AutoGluon stacking
- threshold tuning
- disagreement arbitration between strong submissions

## Metric Strategy

The public metric was treated as a recall-sensitive balanced objective. The local workflow tracks:

- accuracy
- balanced accuracy
- weighted accuracy
- log loss
- MCC where useful

The main lesson from this competition was that local CV and public leaderboard behavior can diverge, so each submission is registered with metadata and compared against both local and public feedback.
