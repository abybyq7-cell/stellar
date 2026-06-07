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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
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
RAW_PHOTOZ_FEATURES = ["u", "g", "r", "i", "z", "redshift"]
LOW_ORDER_FEATURES = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
    "u_minus_g",
    "g_minus_r",
    "r_minus_i",
    "i_minus_z",
    "u_minus_r",
    "g_minus_i",
    "r_minus_z",
    "u_minus_z",
    "mag_mean",
    "mag_std",
    "mag_min",
    "mag_max",
    "mag_range",
]


@dataclass(frozen=True)
class SupplementalExperiment:
    name: str
    model_type: str
    feature_set: str
    seed: int
    weighting: str
    params: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supplemental diverse 5-fold layer-1 OOF models."
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
        choices=["diverse", "local", "disjoint", "nn", "nn_smoke", "smoke"],
        default="diverse",
    )
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument(
        "--autofe-dir",
        type=Path,
        default=OUTPUTS_DIR / "feature_exploration" / "medium_autofe_groupby",
    )
    parser.add_argument("--groupby-top-n", type=int, default=16)
    parser.add_argument(
        "--pred015-train",
        type=Path,
        default=OUTPUTS_DIR / "disagreement_arbitration" / "reconstructed_015_train_pred.csv",
    )
    parser.add_argument(
        "--pred017-train",
        type=Path,
        default=OUTPUTS_DIR
        / "two_stage_threshold"
        / "full21_weighted_lr_threshold_grid"
        / "best_weighted_oof_train_pred.csv",
    )
    parser.add_argument("--star-weight", type=float, default=1.35)
    parser.add_argument("--qso-weight", type=float, default=1.20)
    parser.add_argument("--disagreement-weight", type=float, default=1.70)
    parser.add_argument("--both-wrong-weight", type=float, default=2.40)
    parser.add_argument("--star-gal-hard-weight", type=float, default=2.25)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--nn-time-limit-per-fit", type=int, default=180)
    parser.add_argument("--nn-device", type=str, default="cpu")
    parser.add_argument("--skip-test-pred", action="store_true")
    parser.add_argument("--save-single-submissions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_layer1_helpers():
    path = PROJECT_ROOT / "scripts" / "03_layer1_oof_experiments.py"
    spec = importlib.util.spec_from_file_location("layer1_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["layer1_helpers"] = module
    spec.loader.exec_module(module)
    return module


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def hgb_params(seed: int, aggressive: bool = False) -> dict:
    params = {
        "learning_rate": 0.045,
        "max_iter": 520,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 45,
        "l2_regularization": 0.12,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 40,
        "random_state": seed,
        "verbose": 0,
    }
    if aggressive:
        params.update(
            {
                "learning_rate": 0.035,
                "max_iter": 680,
                "max_leaf_nodes": 95,
                "min_samples_leaf": 28,
                "l2_regularization": 0.05,
            }
        )
    return params


def rf_params(seed: int) -> dict:
    return {
        "n_estimators": 360,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 6,
        "max_features": "sqrt",
        "class_weight": None,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": seed,
    }


def sklearn_mlp_params(seed: int, hidden: tuple[int, ...] = (192, 96), max_iter: int = 80) -> dict:
    return {
        "hidden_layer_sizes": hidden,
        "activation": "relu",
        "solver": "adam",
        "alpha": 2e-4,
        "batch_size": 4096,
        "learning_rate": "adaptive",
        "learning_rate_init": 8e-4,
        "max_iter": max_iter,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 12,
        "tol": 1e-4,
        "random_state": seed,
        "verbose": False,
    }


def realmlp_params(
    seed: int,
    small: bool = False,
    hidden: list[int] | None = None,
    n_epochs: int = 36,
) -> dict:
    params = {
        "device": "cpu",
        "random_state": seed,
        "n_epochs": n_epochs,
        "batch_size": 4096,
        "predict_batch_size": 8192,
        "hidden_sizes": hidden or [192, 192],
        "p_drop": 0.10,
        "wd": 0.015,
        "lr": 0.025,
        "val_metric_name": "class_error",
        "verbosity": 0,
    }
    if small:
        params.update(
            {
                "n_epochs": min(n_epochs, 20),
                "hidden_sizes": hidden or [128, 128],
                "p_drop": 0.05,
                "lr": 0.02,
            }
        )
    return params


def build_experiments(args: argparse.Namespace) -> list[SupplementalExperiment]:
    if args.suite == "smoke":
        experiments = [
            SupplementalExperiment(
                "hgb_smoke_local_guard_s3101",
                "hgb",
                "local_guard",
                3101,
                "star_gal_hard",
                {**hgb_params(3101), "max_iter": 40},
            ),
            SupplementalExperiment(
                "rf_smoke_targeted_s3102",
                "rf",
                "targeted",
                3102,
                "disagreement",
                {**rf_params(3102), "n_estimators": 30},
            ),
        ]
    elif args.suite == "nn_smoke":
        experiments = [
            SupplementalExperiment(
                "skmlp_smoke_targeted_s4101",
                "sklearn_mlp",
                "targeted",
                4101,
                "star_qso",
                sklearn_mlp_params(4101, hidden=(96,), max_iter=25),
            ),
            SupplementalExperiment(
                "realmlp_td_s_smoke_local_guard_s4102",
                "realmlp_td_s",
                "local_guard",
                4102,
                "none",
                realmlp_params(4102, small=True, n_epochs=10),
            ),
            SupplementalExperiment(
                "realmlp_td_smoke_targeted_wide_s4103",
                "realmlp_td",
                "targeted_wide",
                4103,
                "none",
                realmlp_params(4103, small=True, n_epochs=10),
            ),
        ]
    elif args.suite == "local":
        experiments = [
            SupplementalExperiment(
                "hgb_local_guard_starhard_s3101",
                "hgb",
                "local_guard",
                3101,
                "star_gal_hard",
                hgb_params(3101, aggressive=True),
            ),
            SupplementalExperiment(
                "hgb_local_guard_bothwrong_s3103",
                "hgb",
                "local_guard",
                3103,
                "both_wrong",
                hgb_params(3103),
            ),
        ]
    elif args.suite == "disjoint":
        experiments = [
            SupplementalExperiment(
                "hgb_raw_photoz_disagreement_s3201",
                "hgb",
                "raw_photoz",
                3201,
                "disagreement",
                {**hgb_params(3201), "max_leaf_nodes": 31, "min_samples_leaf": 70},
            ),
            SupplementalExperiment(
                "rf_raw_photoz_starhard_s3202",
                "rf",
                "raw_photoz",
                3202,
                "star_gal_hard",
                {**rf_params(3202), "n_estimators": 420, "min_samples_leaf": 3},
            ),
            SupplementalExperiment(
                "hgb_reduced_low_order_bothwrong_s3203",
                "hgb",
                "reduced_low_order",
                3203,
                "both_wrong",
                hgb_params(3203),
            ),
            SupplementalExperiment(
                "rf_reduced_low_order_disagreement_s3204",
                "rf",
                "reduced_low_order",
                3204,
                "disagreement",
                {**rf_params(3204), "n_estimators": 420},
            ),
        ]
    elif args.suite == "nn":
        experiments = [
            SupplementalExperiment(
                "skmlp_targeted_starqso_s4101",
                "sklearn_mlp",
                "targeted",
                4101,
                "star_qso",
                sklearn_mlp_params(4101, hidden=(192, 96), max_iter=90),
            ),
            SupplementalExperiment(
                "skmlp_local_guard_starhard_s4104",
                "sklearn_mlp",
                "local_guard",
                4104,
                "star_gal_hard",
                sklearn_mlp_params(4104, hidden=(256, 128), max_iter=90),
            ),
            SupplementalExperiment(
                "realmlp_td_s_local_guard_s4102",
                "realmlp_td_s",
                "local_guard",
                4102,
                "none",
                realmlp_params(4102, small=True, n_epochs=28),
            ),
            SupplementalExperiment(
                "realmlp_td_targeted_wide_s4103",
                "realmlp_td",
                "targeted_wide",
                4103,
                "none",
                realmlp_params(4103, small=False, hidden=[192, 192], n_epochs=32),
            ),
            SupplementalExperiment(
                "realmlp_td_targeted_s4105",
                "realmlp_td",
                "targeted",
                4105,
                "none",
                realmlp_params(4105, small=True, hidden=[160, 160], n_epochs=28),
            ),
        ]
    else:
        experiments = [
            SupplementalExperiment(
                "hgb_local_guard_starhard_s3101",
                "hgb",
                "local_guard",
                3101,
                "star_gal_hard",
                hgb_params(3101, aggressive=True),
            ),
            SupplementalExperiment(
                "hgb_targeted_bothwrong_s3103",
                "hgb",
                "targeted",
                3103,
                "both_wrong",
                hgb_params(3103),
            ),
            SupplementalExperiment(
                "rf_targeted_disagreement_s3102",
                "rf",
                "targeted",
                3102,
                "disagreement",
                rf_params(3102),
            ),
            SupplementalExperiment(
                "ydf_targeted_wide_s3104",
                "ydf",
                "targeted_wide",
                3104,
                "none",
                {"num_trees": 420, "max_depth": 16},
            ),
            SupplementalExperiment(
                "realmlp_targeted_wide_s3105",
                "realmlp",
                "targeted_wide",
                3105,
                "star_qso",
                {"epochs": 80},
            ),
        ]

    if args.experiments:
        wanted = set(args.experiments)
        experiments = [exp for exp in experiments if exp.name in wanted]
    return experiments


def optional_available(model_type: str) -> bool:
    if model_type in {"hgb", "rf", "sklearn_mlp"}:
        return True
    if model_type == "ydf":
        return importlib.util.find_spec("ydf") is not None
    if model_type in {"realmlp", "realmlp_td", "realmlp_td_s"}:
        return importlib.util.find_spec("pytabkit") is not None
    return False


def build_018_from_015_017(pred015: pd.Series, pred017: pd.Series) -> pd.Series:
    out = pred017.copy()
    use_015 = pred017.eq("GALAXY") & pred015.isin(["QSO", "STAR"])
    out.loc[use_015] = pred015.loc[use_015]
    return out


def read_prediction(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain {LABEL_COL!r}")
    return df[[ID_COL, LABEL_COL]].rename(columns={LABEL_COL: col})


def diagnostic_flags(train_sample: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = train_sample[[ID_COL, LABEL_COL]].copy()
    out["pred015"] = ""
    out["pred017"] = ""
    out["pred018"] = ""
    out["disagreement_015_018"] = False
    out["both_wrong_015_018"] = False
    out["star_gal_hard"] = False

    if not args.pred015_train.exists() or not args.pred017_train.exists():
        return out

    pred015 = read_prediction(args.pred015_train, "pred015")
    pred017 = read_prediction(args.pred017_train, "pred017")
    merged = out[[ID_COL]].merge(pred015, on=ID_COL, how="left").merge(
        pred017, on=ID_COL, how="left"
    )
    merged["pred015"] = merged["pred015"].fillna("")
    merged["pred017"] = merged["pred017"].fillna("")
    merged["pred018"] = build_018_from_015_017(merged["pred015"], merged["pred017"])

    truth = out[LABEL_COL].astype(str).reset_index(drop=True)
    out["pred015"] = merged["pred015"].values
    out["pred017"] = merged["pred017"].values
    out["pred018"] = merged["pred018"].values
    out["disagreement_015_018"] = out["pred015"].ne(out["pred018"])
    out["both_wrong_015_018"] = out["pred015"].ne(truth) & out["pred018"].ne(truth)
    out["star_gal_hard"] = (
        truth.isin(["STAR", "GALAXY"])
        & out["pred015"].isin(["STAR", "GALAXY"])
        & out["pred018"].isin(["STAR", "GALAXY"])
        & (out["pred015"].ne(truth) | out["pred018"].ne(truth))
    )
    return out


def sample_weights_for_experiment(
    train_sample: pd.DataFrame,
    flags: pd.DataFrame,
    exp: SupplementalExperiment,
    args: argparse.Namespace,
) -> np.ndarray:
    labels = train_sample[LABEL_COL].astype(str)
    weights = np.ones(len(train_sample), dtype="float32")
    if exp.weighting in {"star", "star_qso", "star_gal_hard"}:
        weights[labels.eq("STAR").to_numpy()] *= args.star_weight
    if exp.weighting == "star_qso":
        weights[labels.eq("QSO").to_numpy()] *= args.qso_weight
    if exp.weighting in {"disagreement", "star_gal_hard"}:
        weights[flags["disagreement_015_018"].to_numpy()] *= args.disagreement_weight
    if exp.weighting == "both_wrong":
        weights[flags["both_wrong_015_018"].to_numpy()] *= args.both_wrong_weight
    if exp.weighting == "star_gal_hard":
        weights[flags["star_gal_hard"].to_numpy()] *= args.star_gal_hard_weight
    weights = np.clip(weights, 0.25, 10.0)
    weights /= float(weights.mean())
    return weights.astype("float32")


def add_disjoint_feature_sets(
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_metadata: dict,
) -> None:
    raw_train, raw_test = feature_sets["raw"]
    raw_photoz_cols = [col for col in RAW_PHOTOZ_FEATURES if col in raw_train.columns]
    if raw_photoz_cols:
        feature_sets["raw_photoz"] = (
            raw_train[raw_photoz_cols].copy(),
            raw_test[raw_photoz_cols].copy(),
        )

    baseline_train, baseline_test = feature_sets["baseline"]
    low_order_cols = [col for col in LOW_ORDER_FEATURES if col in baseline_train.columns]
    if low_order_cols:
        feature_sets["reduced_low_order"] = (
            baseline_train[low_order_cols].copy(),
            baseline_test[low_order_cols].copy(),
        )

    for name in ["raw_photoz", "reduced_low_order"]:
        if name not in feature_sets:
            continue
        train_frame, test_frame = feature_sets[name]
        feature_metadata[name] = {
            "train_shape": list(train_frame.shape),
            "test_shape": list(test_frame.shape),
            "columns": train_frame.columns.tolist(),
        }


def numeric_frame(df: pd.DataFrame, fit_categories: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    columns = {}
    mappings = fit_categories or {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = series.replace([np.inf, -np.inf], np.nan).astype("float32")
            columns[col] = values.fillna(
                float(values.median()) if values.notna().any() else 0.0
            )
        else:
            key = col
            if key not in mappings:
                values = series.astype("string").fillna("__NA__")
                cats = sorted(values.unique().tolist())
                mappings[key] = {cat: idx for idx, cat in enumerate(cats)}
            columns[col] = (
                series.astype("string")
                .fillna("__NA__")
                .map(mappings[key])
                .fillna(-1)
                .astype("int16")
            )
    out = pd.DataFrame(columns, index=df.index)
    return out, mappings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def align_predict_proba(model: object, proba: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        values = proba.to_numpy(dtype="float64")
        columns = list(proba.columns)
        if all(cls in columns for cls in range(len(CLASSES))):
            values = proba[list(range(len(CLASSES)))].to_numpy(dtype="float64")
        elif all(str(cls) in {str(col) for col in columns} for cls in range(len(CLASSES))):
            lookup = {str(col): col for col in columns}
            values = proba[[lookup[str(cls)] for cls in range(len(CLASSES))]].to_numpy(dtype="float64")
    else:
        values = np.asarray(proba, dtype="float64")

    classes = getattr(model, "classes_", None)
    if classes is not None:
        classes = list(classes)
        if len(classes) == values.shape[1] and classes != list(range(len(CLASSES))):
            aligned = np.zeros((values.shape[0], len(CLASSES)), dtype="float64")
            for idx, cls in enumerate(classes):
                aligned[:, int(cls)] = values[:, idx]
            values = aligned

    if values.shape[1] != len(CLASSES):
        raise ValueError(f"Unexpected probability shape: {values.shape}")
    values = np.clip(values, 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def categorical_column_names(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if str(df[col].dtype) == "category"
        or pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
    ]


def fit_predict_fold(
    exp: SupplementalExperiment,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, object, dict]:
    started = time.time()
    set_seed(exp.seed)
    train_num, mappings = numeric_frame(X_train)
    valid_num, _ = numeric_frame(X_valid, mappings)
    test_num = None
    if X_test is not None:
        test_num, _ = numeric_frame(X_test, mappings)

    if exp.model_type == "hgb":
        model = HistGradientBoostingClassifier(**exp.params)
        model.fit(train_num, y_train, sample_weight=w_train)
    elif exp.model_type == "rf":
        params = exp.params.copy()
        params["n_jobs"] = args.n_jobs
        model = RandomForestClassifier(**params)
        model.fit(train_num, y_train, sample_weight=w_train)
    elif exp.model_type == "sklearn_mlp":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_num)
        valid_scaled = scaler.transform(valid_num)
        test_scaled = scaler.transform(test_num) if test_num is not None else None
        model = MLPClassifier(**exp.params)
        try:
            model.fit(train_scaled, y_train, sample_weight=w_train)
        except TypeError:
            model.fit(train_scaled, y_train)
        valid_proba = align_predict_proba(model, model.predict_proba(valid_scaled))
        test_proba = (
            align_predict_proba(model, model.predict_proba(test_scaled))
            if test_scaled is not None
            else None
        )
        metrics = {
            "accuracy": accuracy_score(y_valid, valid_proba.argmax(axis=1)),
            "balanced_accuracy": balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1)),
            "log_loss": log_loss(y_valid, valid_proba, labels=np.arange(len(CLASSES))),
            "fit_predict_seconds": time.time() - started,
            "best_iteration": getattr(model, "n_iter_", None),
        }
        return valid_proba, test_proba, model, metrics
    elif exp.model_type in {"realmlp", "realmlp_td", "realmlp_td_s"}:
        from pytabkit.models.sklearn.sklearn_interfaces import (
            RealMLP_TD_Classifier,
            RealMLP_TD_S_Classifier,
        )

        params = exp.params.copy()
        params["device"] = args.nn_device
        params["random_state"] = exp.seed
        model_cls = RealMLP_TD_S_Classifier if exp.model_type == "realmlp_td_s" else RealMLP_TD_Classifier
        model = model_cls(**params)
        cat_cols = categorical_column_names(X_train)
        fit_kwargs = {
            "X_val": X_valid.reset_index(drop=True),
            "y_val": y_valid,
            "cat_col_names": cat_cols,
        }
        if args.nn_time_limit_per_fit > 0:
            fit_kwargs["time_to_fit_in_seconds"] = args.nn_time_limit_per_fit
        model.fit(X_train.reset_index(drop=True), y_train, **fit_kwargs)
        valid_proba = align_predict_proba(
            model,
            model.predict_proba(X_valid.reset_index(drop=True)),
        )
        test_proba = (
            align_predict_proba(model, model.predict_proba(X_test.reset_index(drop=True)))
            if X_test is not None
            else None
        )
        metrics = {
            "accuracy": accuracy_score(y_valid, valid_proba.argmax(axis=1)),
            "balanced_accuracy": balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1)),
            "log_loss": log_loss(y_valid, valid_proba, labels=np.arange(len(CLASSES))),
            "fit_predict_seconds": time.time() - started,
            "best_iteration": None,
        }
        return valid_proba, test_proba, model, metrics
    else:
        raise RuntimeError(
            f"{exp.model_type} is not installed or not implemented in this environment."
        )

    valid_proba = align_predict_proba(model, model.predict_proba(valid_num))
    test_proba = (
        align_predict_proba(model, model.predict_proba(test_num))
        if test_num is not None
        else None
    )
    metrics = {
        "accuracy": accuracy_score(y_valid, valid_proba.argmax(axis=1)),
        "balanced_accuracy": balanced_accuracy_score(y_valid, valid_proba.argmax(axis=1)),
        "log_loss": log_loss(y_valid, valid_proba, labels=np.arange(len(CLASSES))),
        "fit_predict_seconds": time.time() - started,
        "best_iteration": getattr(model, "n_iter_", None),
    }
    return valid_proba, test_proba, model, metrics


def save_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    test_proba: np.ndarray,
) -> None:
    sub = sample_submission.copy()
    sub[ID_COL] = test_ids.values
    sub[LABEL_COL] = np.asarray(CLASSES)[test_proba.argmax(axis=1)]
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
    exp: SupplementalExperiment,
    run_dir: Path,
    base_oof_train: pd.DataFrame,
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
            ID_COL: base_oof_train[ID_COL].values,
            "source_index": base_oof_train["source_index"].values,
            LABEL_COL: base_oof_train[LABEL_COL].values,
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
                submission_type="supplemental_oof_single_model",
                model_name=exp.name,
                metrics=overall_metrics_from_rows(fold_rows),
                params={
                    "experiment": asdict(exp),
                    "suite": args.suite,
                    "sample_rows": args.sample_rows,
                    "n_splits": args.n_splits,
                },
                extra={"fold_metrics_path": f"experiments/{exp.name}/fold_metrics.csv"},
            )
    save_json(exp_dir / "config.json", asdict(exp))
    pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)


def run_experiment(
    exp: SupplementalExperiment,
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    y_enc: np.ndarray,
    sample_weights: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    oof_train: pd.DataFrame,
    oof_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict]:
    X, X_test = feature_sets[exp.feature_set]
    oof = np.zeros((len(X), len(CLASSES)), dtype="float32")
    test_sum = np.zeros((len(X_test), len(CLASSES)), dtype="float32")
    rows: list[dict] = []

    for fold, (tr, va) in enumerate(folds, start=1):
        print(f"[{exp.name}] fold {fold}/{len(folds)}")
        valid_proba, test_proba, _, metrics = fit_predict_fold(
            exp,
            X.iloc[tr].reset_index(drop=True),
            y_enc[tr],
            sample_weights[tr],
            X.iloc[va].reset_index(drop=True),
            y_enc[va],
            X_test if not args.skip_test_pred else None,
            args,
        )
        oof[va] = valid_proba.astype("float32")
        if test_proba is not None:
            test_sum += test_proba.astype("float32") / len(folds)
        rows.append(
            {
                "experiment": exp.name,
                "model_type": exp.model_type,
                "feature_set": exp.feature_set,
                "seed": exp.seed,
                "weighting": exp.weighting,
                "fold": fold,
                **metrics,
            }
        )

    pred = oof.argmax(axis=1)
    overall = {
        "experiment": exp.name,
        "model_type": exp.model_type,
        "feature_set": exp.feature_set,
        "seed": exp.seed,
        "weighting": exp.weighting,
        "fold": "overall",
        "accuracy": accuracy_score(y_enc, pred),
        "balanced_accuracy": balanced_accuracy_score(y_enc, pred),
        "log_loss": log_loss(y_enc, oof, labels=np.arange(len(CLASSES))),
        "fit_predict_seconds": sum(row["fit_predict_seconds"] for row in rows),
        "best_iteration": None,
    }
    rows.append(overall)
    for i, cls in enumerate(CLASSES):
        oof_train[f"l1__{exp.name}__{cls}"] = oof[:, i]
        if not args.skip_test_pred:
            oof_test[f"l1__{exp.name}__{cls}"] = test_sum[:, i]
    write_experiment_outputs(
        exp,
        run_dir,
        oof_train,
        test_ids,
        oof,
        None if args.skip_test_pred else test_sum,
        rows,
        sample_submission,
        args,
    )
    print(
        f"[{exp.name}] OOF accuracy={overall['accuracy']:.6f}, "
        f"balanced={overall['balanced_accuracy']:.6f}, log_loss={overall['log_loss']:.6f}"
    )
    return rows


def write_run_summary(run_dir: Path, metadata: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# Supplemental Layer-1 OOF Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        f"- Train sample shape: {metadata['train_sample_shape']}",
        f"- Test shape: {metadata['test_shape']}",
        f"- Classes: {metadata['classes']}",
        f"- Folds: {metadata['args']['n_splits']}",
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
    if args.suite in {"smoke", "nn_smoke"}:
        args.sample_rows = min(args.sample_rows if args.sample_rows > 0 else 2500, 2500)
        args.groupby_top_n = min(args.groupby_top_n, 4)

    run_dir = make_run_dir(args.output_dir, args.run_name)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")

    train, test, sample_submission = read_inputs(args.data_dir)
    layer1 = load_layer1_helpers()
    train_sample = layer1.stratified_sample_train(
        train,
        LABEL_COL,
        args.sample_rows,
        args.sample_seed,
    )
    feature_sets, feature_metadata = layer1.build_feature_sets(train_sample, test, args)
    add_disjoint_feature_sets(feature_sets, feature_metadata)
    experiments = build_experiments(args)
    skipped = [
        {"experiment": exp.name, "model_type": exp.model_type, "reason": "missing_optional_dependency"}
        for exp in experiments
        if not optional_available(exp.model_type)
    ]
    experiments = [
        exp
        for exp in experiments
        if optional_available(exp.model_type) and exp.feature_set in feature_sets
    ]

    y = train_sample[LABEL_COL].copy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    classes = label_encoder.classes_.tolist()
    if classes != CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")

    flags = diagnostic_flags(train_sample, args)
    weights_by_exp = {
        exp.name: sample_weights_for_experiment(train_sample, flags, exp, args)
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
        "experiments": [asdict(exp) for exp in experiments],
        "skipped_experiments": skipped,
        "diagnostic_flag_counts": {
            col: int(flags[col].sum())
            for col in ["disagreement_015_018", "both_wrong_015_018", "star_gal_hard"]
        },
        "sample_weight_summary": {
            name: {
                "min": float(weights.min()),
                "mean": float(weights.mean()),
                "max": float(weights.max()),
                "std": float(weights.std()),
            }
            for name, weights in weights_by_exp.items()
        },
    }
    save_json(run_dir / "manifest.json", metadata)
    save_json(
        run_dir / "feature_set_columns.json",
        {name: frames[0].columns.tolist() for name, frames in feature_sets.items()},
    )
    flags.to_csv(run_dir / "diagnostic_flags_train.csv", index=False)

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
        rows = run_experiment(
            exp,
            feature_sets,
            y_enc,
            weights_by_exp[exp.name],
            folds,
            oof_train,
            oof_test,
            sample_submission,
            test[ID_COL],
            args,
            run_dir,
        )
        for row in rows:
            row["experiment_wall_seconds"] = time.time() - started
        all_rows.extend(rows)
        oof_train.to_csv(run_dir / "oof_train.csv", index=False)
        if not args.skip_test_pred:
            oof_test.to_csv(run_dir / "oof_test.csv", index=False)
        pd.DataFrame(all_rows).to_csv(run_dir / "experiment_scores.csv", index=False)

    summary = (
        pd.DataFrame(all_rows)
        .query("fold == 'overall'")
        .sort_values(["balanced_accuracy", "accuracy"], ascending=False)
    )
    summary.to_csv(run_dir / "summary_overall.csv", index=False)
    write_run_summary(run_dir, metadata, summary)
    print("Supplemental OOF complete.")
    print(summary[["experiment", "model_type", "feature_set", "accuracy", "balanced_accuracy", "log_loss"]])
    if skipped:
        print("Skipped optional experiments:")
        print(pd.DataFrame(skipped).to_string(index=False))
    print(f"OOF train: {(run_dir / 'oof_train.csv').resolve()}")
    if not args.skip_test_pred:
        print(f"OOF test: {(run_dir / 'oof_test.csv').resolve()}")


if __name__ == "__main__":
    main()
