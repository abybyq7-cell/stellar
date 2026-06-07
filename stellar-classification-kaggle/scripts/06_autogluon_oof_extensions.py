from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


CLASSES = ["GALAXY", "QSO", "STAR"]
WEIGHT_COL = "__sample_weight__"
BINARY_LABEL_COL = "__binary_label__"


@dataclass(frozen=True)
class AGExperiment:
    name: str
    kind: str
    feature_set: str
    seed: int
    weighting: str
    model_family: str = ""
    qso_model_family: str = ""
    star_model_family: str = ""


def load_layer1_helpers():
    path = PROJECT_ROOT / "scripts" / "03_layer1_oof_experiments.py"
    spec = importlib.util.spec_from_file_location("layer1_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["layer1_helpers"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    def int_or_auto(value: str) -> int | str:
        if str(value).lower() == "auto":
            return "auto"
        return int(value)

    parser = argparse.ArgumentParser(
        description=(
            "Train additional 5-fold AutoGluon layer-1 OOF models, including "
            "weighted hard-case variants and a learned two-stage binary model."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "layer1_oof")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=["focused", "wide", "two_stage_only", "fresh_diverse", "smoke"],
        default="focused",
    )
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument(
        "--autofe-dir",
        type=Path,
        default=OUTPUTS_DIR / "feature_exploration" / "medium_autofe_groupby",
    )
    parser.add_argument("--groupby-top-n", type=int, default=16)
    parser.add_argument(
        "--base-proba-train",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full_15_lr_all" / "lr_logits_oof_train.csv",
        help="OOF probabilities used to find previous STAR/QSO -> GALAXY hard cases.",
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        default=RAW_DATA_DIR / "train.csv",
        help="CSV with id and class, used if --base-proba-train does not include class.",
    )
    parser.add_argument("--time-limit-per-fit", type=int, default=90)
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--num-cpus", type=int_or_auto, default=8)
    parser.add_argument("--num-gpus", type=int_or_auto, default=0)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--star-weight", type=float, default=1.75)
    parser.add_argument("--qso-weight", type=float, default=1.50)
    parser.add_argument("--hard-star-weight", type=float, default=2.60)
    parser.add_argument("--hard-qso-weight", type=float, default=2.20)
    parser.add_argument("--non-hard-weight", type=float, default=1.0)
    parser.add_argument("--skip-test-pred", action="store_true")
    parser.add_argument("--save-single-submissions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def build_experiments(args: argparse.Namespace) -> list[AGExperiment]:
    if args.suite == "smoke":
        experiments = [
            AGExperiment(
                "ag_gbm_smoke_s1301_targeted_starw",
                "multiclass",
                "targeted",
                1301,
                "star",
                model_family="gbm",
            ),
            AGExperiment(
                "ag2_smoke_s1401_targeted_hardsgq",
                "two_stage",
                "targeted",
                1401,
                "hard_sgq_to_galaxy",
                qso_model_family="gbm",
                star_model_family="gbm_xt",
            ),
        ]
    elif args.suite == "two_stage_only":
        experiments = [
            AGExperiment(
                "ag2_gbmxt_s1401_targeted_hardsgq",
                "two_stage",
                "targeted",
                1401,
                "hard_sgq_to_galaxy",
                qso_model_family="gbm",
                star_model_family="gbm_xt",
            ),
            AGExperiment(
                "ag2_catgbm_s1402_targeted_wide_starqso",
                "two_stage",
                "targeted_wide",
                1402,
                "star_qso",
                qso_model_family="gbm_xt",
                star_model_family="cat",
            ),
        ]
    elif args.suite == "wide":
        experiments = [
            AGExperiment(
                "ag_gbm_s1301_targeted_starw",
                "multiclass",
                "targeted",
                1301,
                "star",
                model_family="gbm",
            ),
            AGExperiment(
                "ag_gbmxt_s1302_targeted_wide_qsow",
                "multiclass",
                "targeted_wide",
                1302,
                "qso",
                model_family="gbm_xt",
            ),
            AGExperiment(
                "ag_cat_s1303_autofe_hardsgq",
                "multiclass",
                "autofe",
                1303,
                "hard_sgq_to_galaxy",
                model_family="cat",
            ),
            AGExperiment(
                "ag_xgb_s1304_wide_starqso",
                "multiclass",
                "wide",
                1304,
                "star_qso",
                model_family="xgb",
            ),
            AGExperiment(
                "ag2_gbmxt_s1401_targeted_hardsgq",
                "two_stage",
                "targeted",
                1401,
                "hard_sgq_to_galaxy",
                qso_model_family="gbm",
                star_model_family="gbm_xt",
            ),
            AGExperiment(
                "ag2_catgbm_s1402_targeted_wide_starqso",
                "two_stage",
                "targeted_wide",
                1402,
                "star_qso",
                qso_model_family="gbm_xt",
                star_model_family="cat",
            ),
        ]
    elif args.suite == "fresh_diverse":
        experiments = [
            AGExperiment(
                "ag_gbm_s2305_targeted_wide_starw",
                "multiclass",
                "targeted_wide",
                2305,
                "star",
                model_family="gbm",
            ),
            AGExperiment(
                "ag_rf_s2306_targeted_hardsgq",
                "multiclass",
                "targeted",
                2306,
                "hard_sgq_to_galaxy",
                model_family="rf",
            ),
            AGExperiment(
                "ag_nn_s2307_targeted_wide_starqso",
                "multiclass",
                "targeted_wide",
                2307,
                "star_qso",
                model_family="nn_torch",
            ),
            AGExperiment(
                "ag2_xgbgbm_s2308_targeted_wide_hardsgq",
                "two_stage",
                "targeted_wide",
                2308,
                "hard_sgq_to_galaxy",
                qso_model_family="xgb",
                star_model_family="gbm",
            ),
        ]
    else:
        experiments = [
            AGExperiment(
                "ag_gbm_s1301_targeted_starw",
                "multiclass",
                "targeted",
                1301,
                "star",
                model_family="gbm",
            ),
            AGExperiment(
                "ag_gbmxt_s1302_targeted_wide_qsow",
                "multiclass",
                "targeted_wide",
                1302,
                "qso",
                model_family="gbm_xt",
            ),
            AGExperiment(
                "ag_cat_s1303_autofe_hardsgq",
                "multiclass",
                "autofe",
                1303,
                "hard_sgq_to_galaxy",
                model_family="cat",
            ),
            AGExperiment(
                "ag2_gbmxt_s1401_targeted_hardsgq",
                "two_stage",
                "targeted",
                1401,
                "hard_sgq_to_galaxy",
                qso_model_family="gbm",
                star_model_family="gbm_xt",
            ),
        ]

    if args.experiments:
        wanted = set(args.experiments)
        experiments = [exp for exp in experiments if exp.name in wanted]
    return experiments


def hyperparameters_for_family(family: str, seed: int) -> dict:
    if family == "gbm":
        return {
            "GBM": [
                {
                    "random_state": seed,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                }
            ]
        }
    if family == "gbm_xt":
        return {
            "GBM": [
                {
                    "extra_trees": True,
                    "random_state": seed,
                    "ag_args": {"name_suffix": f"_XT_s{seed}"},
                }
            ]
        }
    if family == "cat":
        return {
            "CAT": [
                {
                    "random_seed": seed,
                    "depth": 6,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                }
            ]
        }
    if family == "xgb":
        return {
            "XGB": [
                {
                    "random_state": seed,
                    "max_depth": 5,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                }
            ]
        }
    if family == "rf":
        return {
            "RF": [
                {
                    "n_estimators": 260,
                    "max_leaf_nodes": 40000,
                    "max_features": "sqrt",
                    "random_state": seed,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                }
            ]
        }
    if family == "nn_torch":
        return {
            "NN_TORCH": [
                {
                    "seed_value": seed,
                    "num_epochs": 24,
                    "learning_rate": 3e-4,
                    "weight_decay": 1e-5,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                }
            ]
        }
    if family == "mix":
        return {
            "GBM": [
                {
                    "random_state": seed,
                    "ag_args": {"name_suffix": f"_s{seed}"},
                },
                {
                    "extra_trees": True,
                    "random_state": seed + 17,
                    "ag_args": {"name_suffix": f"_XT_s{seed}"},
                },
            ],
            "CAT": [{"random_seed": seed, "depth": 6}],
        }
    raise ValueError(f"Unknown AutoGluon model family: {family}")


def detect_probability_prefix(df: pd.DataFrame) -> str:
    candidates = []
    for col in df.columns:
        for cls in CLASSES:
            suffix = f"__{cls}"
            if col.endswith(suffix):
                candidates.append(col[: -len(suffix)])
    valid = sorted(
        {
            prefix
            for prefix in candidates
            if all(f"{prefix}__{cls}" in df.columns for cls in CLASSES)
        }
    )
    if len(valid) != 1:
        raise ValueError(f"Could not detect one probability prefix in base OOF: {valid}")
    return valid[0]


def load_previous_hard_ids(base_proba_path: Path, label_source_path: Path) -> dict[str, set[int]]:
    if not base_proba_path.exists():
        return {"STAR": set(), "QSO": set()}

    proba_df = pd.read_csv(base_proba_path)
    prefix = detect_probability_prefix(proba_df)
    if LABEL_COL not in proba_df.columns:
        labels = pd.read_csv(label_source_path, usecols=lambda col: col in {ID_COL, LABEL_COL})
        proba_df = proba_df.merge(labels, on=ID_COL, how="left")
    if LABEL_COL not in proba_df.columns:
        return {"STAR": set(), "QSO": set()}

    cols = [f"{prefix}__{cls}" for cls in CLASSES]
    pred = np.asarray(CLASSES)[proba_df[cols].to_numpy(dtype="float64").argmax(axis=1)]
    out = {}
    for cls in ["STAR", "QSO"]:
        mask = proba_df[LABEL_COL].astype(str).eq(cls) & pd.Series(pred).eq("GALAXY")
        out[cls] = set(proba_df.loc[mask.to_numpy(), ID_COL].astype(int).tolist())
    return out


def sample_weights_for_scheme(
    train_sample: pd.DataFrame,
    scheme: str,
    hard_ids: dict[str, set[int]],
    args: argparse.Namespace,
) -> np.ndarray:
    weights = np.full(len(train_sample), args.non_hard_weight, dtype="float32")
    labels = train_sample[LABEL_COL].astype(str)
    ids = train_sample[ID_COL].astype(int)

    if scheme in {"star", "star_qso"}:
        weights[labels.eq("STAR").to_numpy()] *= args.star_weight
    if scheme in {"qso", "star_qso"}:
        weights[labels.eq("QSO").to_numpy()] *= args.qso_weight
    if scheme == "hard_sgq_to_galaxy":
        star_hard = ids.isin(hard_ids.get("STAR", set())).to_numpy()
        qso_hard = ids.isin(hard_ids.get("QSO", set())).to_numpy()
        weights[labels.eq("STAR").to_numpy()] *= 1.20
        weights[labels.eq("QSO").to_numpy()] *= 1.10
        weights[star_hard] *= args.hard_star_weight
        weights[qso_hard] *= args.hard_qso_weight

    weights = np.clip(weights, 0.25, 10.0)
    weights /= float(weights.mean())
    return weights.astype("float32")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_ag_train(
    X: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    weights: np.ndarray | None,
    label_col: str,
) -> pd.DataFrame:
    train_ag = X.copy()
    train_ag[label_col] = np.asarray(labels)
    if weights is not None:
        train_ag[WEIGHT_COL] = weights.astype("float32")
    return train_ag


def fit_autogluon_predictor(
    train_ag: pd.DataFrame,
    label_col: str,
    problem_type: str,
    model_path: Path,
    seed: int,
    family: str,
    args: argparse.Namespace,
    positive_class: str | None = None,
):
    from autogluon.tabular import TabularPredictor

    if model_path.exists():
        shutil.rmtree(model_path)
    set_random_seed(seed)
    predictor = TabularPredictor(
        label=label_col,
        problem_type=problem_type,
        eval_metric="log_loss",
        path=str(model_path),
        verbosity=args.verbosity,
        sample_weight=WEIGHT_COL if WEIGHT_COL in train_ag.columns else None,
        positive_class=positive_class,
    )
    predictor.fit(
        train_data=train_ag,
        presets=args.presets,
        time_limit=args.time_limit_per_fit if args.time_limit_per_fit > 0 else None,
        hyperparameters=hyperparameters_for_family(family, seed),
        fit_weighted_ensemble=False,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
    )
    return predictor


def align_multiclass_proba(proba: pd.DataFrame | np.ndarray, classes: list[str]) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        missing = [cls for cls in classes if cls not in proba.columns]
        if missing:
            raise ValueError(f"Missing probability columns from AutoGluon output: {missing}")
        values = proba[classes].to_numpy(dtype="float64")
    else:
        values = np.asarray(proba, dtype="float64")
        if values.shape[1] != len(classes):
            raise ValueError(f"Unexpected probability shape: {values.shape}")
    values = np.clip(values, 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values.astype("float32")


def positive_probability(
    predictor,
    X: pd.DataFrame,
    positive_label: str,
) -> np.ndarray:
    proba = predictor.predict_proba(X)
    if isinstance(proba, pd.DataFrame):
        if positive_label in proba.columns:
            return proba[positive_label].to_numpy(dtype="float32")
        str_cols = {str(col): col for col in proba.columns}
        if positive_label in str_cols:
            return proba[str_cols[positive_label]].to_numpy(dtype="float32")
        raise ValueError(f"Positive label {positive_label!r} not in columns {proba.columns.tolist()}")

    labels = [str(label) for label in getattr(predictor, "class_labels", [])]
    if positive_label in labels:
        return np.asarray(proba, dtype="float32")[:, labels.index(positive_label)]
    if np.asarray(proba).shape[1] == 2:
        return np.asarray(proba, dtype="float32")[:, 1]
    raise ValueError(f"Cannot align binary probabilities for {positive_label!r}")


def combined_two_stage_proba(p_qso: np.ndarray, p_star_given_other: np.ndarray) -> np.ndarray:
    p_qso = np.clip(p_qso.astype("float64"), 1e-8, 1.0 - 1e-8)
    p_star_given_other = np.clip(p_star_given_other.astype("float64"), 1e-8, 1.0 - 1e-8)
    p_other = 1.0 - p_qso
    proba = np.column_stack(
        [
            p_other * (1.0 - p_star_given_other),
            p_qso,
            p_other * p_star_given_other,
        ]
    )
    proba = np.clip(proba, 1e-12, None)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype("float32")


def score_oof(exp: AGExperiment, y_enc: np.ndarray, proba: np.ndarray, fold_rows: list[dict]) -> dict:
    pred = proba.argmax(axis=1)
    overall = {
        "experiment": exp.name,
        "model_type": f"autogluon_{exp.model_family or 'two_stage'}",
        "feature_set": exp.feature_set,
        "seed": exp.seed,
        "weighting": exp.weighting,
        "fold": "overall",
        "accuracy": accuracy_score(y_enc, pred),
        "balanced_accuracy": balanced_accuracy_score(y_enc, pred),
        "log_loss": log_loss(y_enc, proba, labels=np.arange(len(CLASSES))),
        "fit_predict_seconds": sum(row["fit_predict_seconds"] for row in fold_rows),
        "best_iteration": None,
    }
    return overall


def save_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    proba: np.ndarray,
) -> None:
    sub = sample_submission.copy()
    sub[ID_COL] = test_ids.values
    sub[LABEL_COL] = np.asarray(CLASSES)[proba.argmax(axis=1)]
    sub.to_csv(path, index=False)


def overall_metrics_from_rows(rows: list[dict]) -> dict:
    for row in rows:
        if str(row.get("fold")) == "overall":
            return {
                key: row[key]
                for key in ["accuracy", "balanced_accuracy", "log_loss"]
                if key in row
            }
    return {}


def write_experiment_outputs(
    exp: AGExperiment,
    run_dir: Path,
    oof_train: pd.DataFrame,
    test_ids: pd.Series,
    oof_proba: np.ndarray,
    test_proba: np.ndarray | None,
    fold_rows: list[dict],
    sample_submission: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    prefix = f"l1__{exp.name}"
    exp_dir = run_dir / "experiments" / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COL: oof_train[ID_COL].values,
            "source_index": oof_train["source_index"].values,
            LABEL_COL: oof_train[LABEL_COL].values,
            **{f"{prefix}__{cls}": oof_proba[:, i] for i, cls in enumerate(CLASSES)},
        }
    ).to_csv(exp_dir / "oof_train.csv", index=False)
    if test_proba is not None:
        pd.DataFrame(
            {
                ID_COL: test_ids.values,
                **{f"{prefix}__{cls}": test_proba[:, i] for i, cls in enumerate(CLASSES)},
            }
        ).to_csv(exp_dir / "oof_test.csv", index=False)
        if args.save_single_submissions:
            sub_dir = run_dir / "single_model_submissions"
            sub_dir.mkdir(parents=True, exist_ok=True)
            submission_path = sub_dir / f"{exp.name}.csv"
            save_submission(submission_path, sample_submission, test_ids, test_proba)
            register_submission(
                submission_path,
                run_dir=run_dir,
                script=Path(__file__).name,
                submission_type="autogluon_oof_extension_single_model",
                model_name=exp.name,
                metrics=overall_metrics_from_rows(fold_rows),
                params={
                    "experiment": asdict(exp),
                    "suite": args.suite,
                    "sample_rows": args.sample_rows,
                    "n_splits": args.n_splits,
                    "time_limit_per_fit": args.time_limit_per_fit,
                    "presets": args.presets,
                    "star_weight": args.star_weight,
                    "qso_weight": args.qso_weight,
                    "hard_star_weight": args.hard_star_weight,
                    "hard_qso_weight": args.hard_qso_weight,
                },
                extra={"fold_metrics_path": f"experiments/{exp.name}/fold_metrics.csv"},
            )
    save_json(exp_dir / "config.json", asdict(exp))
    pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)


def run_multiclass_experiment(
    exp: AGExperiment,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    y_enc: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:
    oof = np.zeros((len(X), len(CLASSES)), dtype="float32")
    test_sum = np.zeros((len(X_test), len(CLASSES)), dtype="float32")
    fold_rows = []

    for fold, (tr, va) in enumerate(folds, start=1):
        print(f"[{exp.name}] fold {fold}/{len(folds)} multiclass {exp.model_family}")
        started = time.time()
        predictor = fit_autogluon_predictor(
            prepare_ag_train(X.iloc[tr].reset_index(drop=True), y.iloc[tr], weights[tr], LABEL_COL),
            LABEL_COL,
            "multiclass",
            run_dir / "ag_models" / exp.name / f"fold{fold}",
            exp.seed + fold,
            exp.model_family,
            args,
        )
        valid_proba = align_multiclass_proba(
            predictor.predict_proba(X.iloc[va].reset_index(drop=True)),
            CLASSES,
        )
        oof[va] = valid_proba
        test_proba = None
        if not args.skip_test_pred:
            test_proba = align_multiclass_proba(predictor.predict_proba(X_test), CLASSES)
            test_sum += test_proba / len(folds)

        valid_pred = valid_proba.argmax(axis=1)
        fold_rows.append(
            {
                "experiment": exp.name,
                "model_type": f"autogluon_{exp.model_family}",
                "feature_set": exp.feature_set,
                "seed": exp.seed,
                "weighting": exp.weighting,
                "fold": fold,
                "accuracy": accuracy_score(y_enc[va], valid_pred),
                "balanced_accuracy": balanced_accuracy_score(y_enc[va], valid_pred),
                "log_loss": log_loss(y_enc[va], valid_proba, labels=np.arange(len(CLASSES))),
                "fit_predict_seconds": time.time() - started,
                "best_iteration": None,
            }
        )

    return oof, None if args.skip_test_pred else test_sum, fold_rows


def run_two_stage_experiment(
    exp: AGExperiment,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    y_enc: np.ndarray,
    weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:
    p_qso_oof = np.zeros(len(X), dtype="float32")
    p_star_oof = np.zeros(len(X), dtype="float32")
    p_qso_test_sum = np.zeros(len(X_test), dtype="float32")
    p_star_test_sum = np.zeros(len(X_test), dtype="float32")
    fold_rows = []

    for fold, (tr, va) in enumerate(folds, start=1):
        print(f"[{exp.name}] fold {fold}/{len(folds)} binary QSO vs OTHER")
        started = time.time()
        qso_labels = np.where(y.iloc[tr].astype(str).eq("QSO"), "QSO", "OTHER")
        qso_weights = weights[tr].copy()
        qso_predictor = fit_autogluon_predictor(
            prepare_ag_train(X.iloc[tr].reset_index(drop=True), qso_labels, qso_weights, BINARY_LABEL_COL),
            BINARY_LABEL_COL,
            "binary",
            run_dir / "ag_models" / exp.name / f"fold{fold}_qso",
            exp.seed + fold,
            exp.qso_model_family,
            args,
            positive_class="QSO",
        )
        p_qso_valid = positive_probability(qso_predictor, X.iloc[va].reset_index(drop=True), "QSO")
        p_qso_oof[va] = p_qso_valid
        if not args.skip_test_pred:
            p_qso_test_sum += positive_probability(qso_predictor, X_test, "QSO") / len(folds)

        non_qso_tr = tr[y.iloc[tr].astype(str).ne("QSO").to_numpy()]
        sg_labels = y.iloc[non_qso_tr].astype(str).to_numpy()
        sg_weights = weights[non_qso_tr].copy()
        print(f"[{exp.name}] fold {fold}/{len(folds)} binary STAR vs GALAXY")
        sg_predictor = fit_autogluon_predictor(
            prepare_ag_train(
                X.iloc[non_qso_tr].reset_index(drop=True),
                sg_labels,
                sg_weights,
                BINARY_LABEL_COL,
            ),
            BINARY_LABEL_COL,
            "binary",
            run_dir / "ag_models" / exp.name / f"fold{fold}_star_galaxy",
            exp.seed + 100 + fold,
            exp.star_model_family,
            args,
            positive_class="STAR",
        )
        p_star_valid = positive_probability(sg_predictor, X.iloc[va].reset_index(drop=True), "STAR")
        p_star_oof[va] = p_star_valid
        if not args.skip_test_pred:
            p_star_test_sum += positive_probability(sg_predictor, X_test, "STAR") / len(folds)

        valid_proba = combined_two_stage_proba(p_qso_valid, p_star_valid)
        valid_pred = valid_proba.argmax(axis=1)
        fold_rows.append(
            {
                "experiment": exp.name,
                "model_type": "autogluon_two_stage",
                "feature_set": exp.feature_set,
                "seed": exp.seed,
                "weighting": exp.weighting,
                "fold": fold,
                "qso_model_family": exp.qso_model_family,
                "star_model_family": exp.star_model_family,
                "accuracy": accuracy_score(y_enc[va], valid_pred),
                "balanced_accuracy": balanced_accuracy_score(y_enc[va], valid_pred),
                "log_loss": log_loss(y_enc[va], valid_proba, labels=np.arange(len(CLASSES))),
                "fit_predict_seconds": time.time() - started,
                "best_iteration": None,
            }
        )

    oof = combined_two_stage_proba(p_qso_oof, p_star_oof)
    test = None
    if not args.skip_test_pred:
        test = combined_two_stage_proba(p_qso_test_sum, p_star_test_sum)
    return oof, test, fold_rows


def write_run_summary(run_dir: Path, metadata: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# AutoGluon OOF Extensions",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Data",
        "",
        f"- Train sample shape: {metadata['train_sample_shape']}",
        f"- Test shape: {metadata['test_shape']}",
        f"- Classes: {metadata['classes']}",
        f"- Folds: {metadata['args']['n_splits']}",
        f"- Time limit per AutoGluon fit: {metadata['args']['time_limit_per_fit']} seconds",
        "",
        "## Overall Scores",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['experiment']}` ({row['feature_set']}, {row['model_type']}): "
            f"accuracy={row['accuracy']:.6f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}, "
            f"log_loss={row['log_loss']:.6f}"
        )
    lines.append("")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.suite == "smoke":
        args.sample_rows = min(args.sample_rows if args.sample_rows > 0 else 2500, 2500)
        args.groupby_top_n = min(args.groupby_top_n, 4)
        args.time_limit_per_fit = min(args.time_limit_per_fit, 25)

    run_dir = make_run_dir(args.output_dir, args.run_name)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")

    train, test, sample_submission = read_inputs(args.data_dir)
    layer1 = load_layer1_helpers()
    train_sample = layer1.stratified_sample_train(
        train, LABEL_COL, args.sample_rows, args.sample_seed
    )

    feature_sets, feature_metadata = layer1.build_feature_sets(train_sample, test, args)
    experiments = build_experiments(args)
    experiments = [exp for exp in experiments if exp.feature_set in feature_sets]

    label_encoder = LabelEncoder()
    y = train_sample[LABEL_COL].copy()
    y_enc = label_encoder.fit_transform(y)
    classes = label_encoder.classes_.tolist()
    if classes != CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")

    hard_ids = load_previous_hard_ids(args.base_proba_train, args.label_source)
    weight_by_experiment = {
        exp.name: sample_weights_for_scheme(train_sample, exp.weighting, hard_ids, args)
        for exp in experiments
    }

    metadata = {
        "args": vars(args),
        "train_full_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_sample_shape": list(train_sample.shape),
        "class_counts_sample": y.value_counts().to_dict(),
        "classes": classes,
        "feature_sets": feature_metadata,
        "hard_case_counts": {key: len(value) for key, value in hard_ids.items()},
        "experiments": [asdict(exp) for exp in experiments],
        "sample_weight_summary": {
            name: {
                "min": float(weights.min()),
                "mean": float(weights.mean()),
                "max": float(weights.max()),
                "std": float(weights.std()),
            }
            for name, weights in weight_by_experiment.items()
        },
    }
    save_json(run_dir / "manifest.json", metadata)
    save_json(
        run_dir / "feature_set_columns.json",
        {name: frames[0].columns.tolist() for name, frames in feature_sets.items()},
    )

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str)[:6000])
        print("Dry run complete. No models were trained.")
        return

    folds = list(
        StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.fold_seed,
        ).split(np.zeros(len(y_enc)), y_enc)
    )
    oof_train = pd.DataFrame(
        {
            ID_COL: train_sample[ID_COL].values,
            "source_index": train_sample["source_index"].values,
            LABEL_COL: y.values,
        }
    )
    oof_test = pd.DataFrame({ID_COL: test[ID_COL].values})
    all_rows: list[dict] = []

    for exp in experiments:
        started = time.time()
        X, X_test = feature_sets[exp.feature_set]
        weights = weight_by_experiment[exp.name]
        print(f"Starting experiment: {exp.name} ({exp.kind}, {exp.feature_set}, {exp.weighting})")
        if exp.kind == "multiclass":
            oof_proba, test_proba, fold_rows = run_multiclass_experiment(
                exp, X, X_test, y, y_enc, weights, folds, args, run_dir
            )
        elif exp.kind == "two_stage":
            oof_proba, test_proba, fold_rows = run_two_stage_experiment(
                exp, X, X_test, y, y_enc, weights, folds, args, run_dir
            )
        else:
            raise ValueError(f"Unknown experiment kind: {exp.kind}")

        overall = score_oof(exp, y_enc, oof_proba, fold_rows)
        fold_rows.append(overall)
        for row in fold_rows:
            row["experiment_wall_seconds"] = time.time() - started
        all_rows.extend(fold_rows)

        prefix = f"l1__{exp.name}"
        for i, cls in enumerate(CLASSES):
            oof_train[f"{prefix}__{cls}"] = oof_proba[:, i]
            if test_proba is not None:
                oof_test[f"{prefix}__{cls}"] = test_proba[:, i]
        write_experiment_outputs(
            exp,
            run_dir,
            oof_train,
            test[ID_COL],
            oof_proba,
            test_proba,
            fold_rows,
            sample_submission,
            args,
        )
        oof_train.to_csv(run_dir / "oof_train.csv", index=False)
        if not args.skip_test_pred:
            oof_test.to_csv(run_dir / "oof_test.csv", index=False)
        pd.DataFrame(all_rows).to_csv(run_dir / "experiment_scores.csv", index=False)
        print(
            f"[{exp.name}] OOF accuracy={overall['accuracy']:.6f}, "
            f"balanced={overall['balanced_accuracy']:.6f}, log_loss={overall['log_loss']:.6f}"
        )

    summary = (
        pd.DataFrame(all_rows)
        .query("fold == 'overall'")
        .sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    )
    summary.to_csv(run_dir / "summary_overall.csv", index=False)
    write_run_summary(run_dir, metadata, summary)
    print("AutoGluon OOF extension complete.")
    print(summary[["experiment", "model_type", "feature_set", "accuracy", "balanced_accuracy", "log_loss"]])
    print(f"OOF train: {(run_dir / 'oof_train.csv').resolve()}")
    if not args.skip_test_pred:
        print(f"OOF test: {(run_dir / 'oof_test.csv').resolve()}")


if __name__ == "__main__":
    main()
