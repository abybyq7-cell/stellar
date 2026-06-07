from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL, MAG_COLS
from stellar.features import add_astronomy_features, add_targeted_features
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASSES)}
INT_TO_CLASS = np.asarray(CLASSES)
DEFAULT_CLASS_METRIC_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}


def parse_range(spec: str) -> np.ndarray:
    parts = [float(part) for part in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"Range must be start:stop:step, got {spec!r}")
    start, stop, step = parts
    if step <= 0:
        raise ValueError("Range step must be positive.")
    values = np.arange(start, stop + step / 2.0, step, dtype="float64")
    return np.round(values, 10)


def parse_float_grid(spec: str) -> np.ndarray:
    spec = str(spec).strip()
    if not spec:
        raise ValueError("Grid specification cannot be empty.")
    if ":" in spec and "," not in spec:
        return parse_range(spec)
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError(f"No values found in {spec!r}")
    return np.asarray(values, dtype="float64")


def parse_bool_grid(spec: str) -> list[bool]:
    mapping = {
        "0": False,
        "1": True,
        "false": False,
        "true": True,
        "no": False,
        "yes": True,
    }
    values = []
    for item in spec.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Boolean grid value must be one of {sorted(mapping)}, got {item!r}")
        values.append(mapping[key])
    if not values:
        raise ValueError(f"No boolean values found in {spec!r}")
    return values


def parse_class_metric_weights(spec: str) -> dict[str, float]:
    weights = DEFAULT_CLASS_METRIC_WEIGHTS.copy()
    if not spec:
        return weights
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", 1)
        weights[name.strip()] = float(value)
    missing = [cls for cls in CLASSES if cls not in weights]
    if missing:
        raise ValueError(f"Missing class metric weights for: {missing}")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a local GALAXY-vs-STAR model and search a guard that flips "
            "local base STAR/QSO predictions back to GALAXY."
        )
    )
    parser.add_argument(
        "--pred015-train",
        type=Path,
        default=OUTPUTS_DIR / "disagreement_arbitration" / "reconstructed_015_train_pred.csv",
    )
    parser.add_argument(
        "--pred015-test",
        type=Path,
        default=OUTPUTS_DIR / "disagreement_arbitration" / "reconstructed_015_test_pred.csv",
    )
    parser.add_argument(
        "--pred017-train",
        type=Path,
        default=OUTPUTS_DIR
        / "two_stage_threshold"
        / "full21_weighted_lr_threshold_grid"
        / "best_weighted_oof_train_pred.csv",
    )
    parser.add_argument(
        "--pred017-test",
        type=Path,
        default=OUTPUTS_DIR
        / "two_stage_threshold"
        / "full21_weighted_lr_threshold_grid"
        / "best_weighted_test_pred.csv",
    )
    parser.add_argument(
        "--pred018-train",
        type=Path,
        default=None,
        help="Optional materialized 018 train predictions. Defaults to reconstructing from 017 and 015.",
    )
    parser.add_argument(
        "--pred018-test",
        type=Path,
        default=None,
        help="Optional materialized 018 test predictions. Defaults to reconstructing from 017 and 015.",
    )
    parser.add_argument(
        "--train-proba",
        type=Path,
        default=OUTPUTS_DIR
        / "stacking"
        / "stack_full21_weighted_lr_cat_reduced"
        / "lr_logits_oof_train.csv",
    )
    parser.add_argument(
        "--test-proba",
        type=Path,
        default=OUTPUTS_DIR
        / "stacking"
        / "stack_full21_weighted_lr_cat_reduced"
        / "lr_logits_oof_test.csv",
    )
    parser.add_argument(
        "--proba-prefix",
        type=str,
        default="auto",
        help="Probability column prefix, for example lr_logits. Use auto to detect it.",
    )
    parser.add_argument("--raw-train", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--raw-test", type=Path, default=RAW_DATA_DIR / "test.csv")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=RAW_DATA_DIR / "sample_submission.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "local_guard_threshold")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--base",
        choices=["015", "017", "018"],
        default="018",
        help="Prediction stream guarded by the local GALAXY rule.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-metric-weights",
        type=str,
        default="GALAXY:1.0,QSO:3.2,STAR:4.6",
        help="Class weights used for weighted accuracy and base local sample weights.",
    )
    parser.add_argument("--p-galaxy-thresholds", type=str, default="0.55:0.95:0.05")
    parser.add_argument(
        "--redshift-abs-max-values",
        type=str,
        default="0.0497,0.08,0.127,0.144,0.20,0.35",
    )
    parser.add_argument(
        "--compact-color-score-max-values",
        type=str,
        default="0.70,0.90,1.10,1.30,1.60,2.00,2.50",
    )
    parser.add_argument(
        "--mag-std-max-values",
        type=str,
        default="0.25,0.35,0.50,0.75,1.00,1.35",
    )
    parser.add_argument(
        "--blue-cloud-required-values",
        type=str,
        default="0,1",
        help="Comma-separated booleans. 1 means require galaxy_population == Blue_Cloud.",
    )
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--guard-injury-penalty", type=float, default=2.0)
    parser.add_argument("--guard-recovery-bonus", type=float, default=0.25)
    parser.add_argument("--disagreement-weight", type=float, default=1.25)
    parser.add_argument("--both-wrong-weight", type=float, default=1.50)
    parser.add_argument(
        "--015-only-correct-true-galaxy-weight",
        dest="only015_correct_true_galaxy_weight",
        type=float,
        default=2.00,
    )
    parser.add_argument("--local-region-weight", type=float, default=1.75)
    parser.add_argument("--local-region-redshift-abs-max", type=float, default=0.144)
    parser.add_argument("--local-region-compact-color-score-max", type=float, default=1.30)
    parser.add_argument("--local-region-mag-std-max", type=float, default=0.75)
    parser.add_argument("--hgb-max-iter", type=int, default=250)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.06)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_pred(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [name for name in [ID_COL, LABEL_COL] if name not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[[ID_COL, LABEL_COL]].copy()
    out = out.rename(columns={LABEL_COL: col})
    return out


def reconstruct_018(pred017: pd.Series, pred015: pd.Series) -> pd.Series:
    out = pred017.astype(str).copy()
    use_015 = pred017.astype(str).eq("GALAXY") & pred015.astype(str).isin(["QSO", "STAR"])
    out.loc[use_015] = pred015.loc[use_015].astype(str)
    return out


def detect_probability_prefix(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested

    prefixes = []
    for col in df.columns:
        for cls in CLASSES:
            suffix = f"__{cls}"
            if col.endswith(suffix):
                prefixes.append(col[: -len(suffix)])
    valid = sorted(
        {
            prefix
            for prefix in prefixes
            if all(f"{prefix}__{cls}" in df.columns for cls in CLASSES)
        }
    )
    if len(valid) != 1:
        raise ValueError(f"Could not auto-detect one probability prefix. Found: {valid}")
    return valid[0]


def normalize_proba(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype("float64"), 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def load_proba(path: Path, prefix_arg: str, prefix: str | None = None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    detected = prefix if prefix is not None else detect_probability_prefix(df, prefix_arg)
    cols = [f"{detected}__{cls}" for cls in CLASSES]
    missing = [col for col in [ID_COL] + cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    proba = normalize_proba(df[cols].to_numpy())
    out = pd.DataFrame(
        {
            ID_COL: df[ID_COL].values,
            **{f"stack_p_{cls}": proba[:, idx] for idx, cls in enumerate(CLASSES)},
        }
    )
    return out, detected


def add_local_guard_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-3

    if "redshift" in df.columns:
        redshift_abs = df["redshift"].abs()
        df["redshift_abs"] = redshift_abs
        df["lowz_abs_le_00497"] = (redshift_abs <= 0.0497).astype("int8")
        df["lowz_abs_le_008"] = (redshift_abs <= 0.08).astype("int8")
        df["lowz_abs_le_0127"] = (redshift_abs <= 0.127).astype("int8")
        df["lowz_abs_le_0144"] = (redshift_abs <= 0.144).astype("int8")
        df["lowz_abs_le_020"] = (redshift_abs <= 0.20).astype("int8")
        df["redshift_abs_inverse"] = 1.0 / (redshift_abs + eps)

    color_cols = ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]
    if all(col in df.columns for col in color_cols):
        color_values = df[color_cols].astype("float64")
        df["compact_color_score"] = np.sqrt(np.square(color_values).mean(axis=1))
        df["compact_color_score_non_u"] = np.sqrt(
            np.square(df[["g_minus_r", "r_minus_i", "i_minus_z"]].astype("float64")).mean(axis=1)
        )
        df["blue_color_score"] = (
            (0.80 - df["u_minus_g"]).clip(lower=0.0)
            + (0.45 - df["g_minus_r"]).clip(lower=0.0)
            + (0.25 - df["r_minus_i"]).clip(lower=0.0)
        )
        df["is_compact_color_090"] = (df["compact_color_score"] <= 0.90).astype("int8")
        df["is_compact_color_130"] = (df["compact_color_score"] <= 1.30).astype("int8")
        df["is_blue_color"] = (df["blue_color_score"] > 0.0).astype("int8")

    if "mag_std" in df.columns:
        df["low_mag_std_025"] = (df["mag_std"] <= 0.25).astype("int8")
        df["low_mag_std_035"] = (df["mag_std"] <= 0.35).astype("int8")
        df["low_mag_std_050"] = (df["mag_std"] <= 0.50).astype("int8")
        df["low_mag_std_075"] = (df["mag_std"] <= 0.75).astype("int8")
        df["low_mag_std_score"] = (0.75 - df["mag_std"]).clip(lower=0.0)

    if "galaxy_population" in df.columns:
        population = df["galaxy_population"].astype("string")
        df["is_blue_cloud"] = (population == "Blue_Cloud").astype("int8")
        df["is_red_sequence"] = (population == "Red_Sequence").astype("int8")

    if all(col in df.columns for col in ["redshift_abs", "compact_color_score", "mag_std"]):
        df["compact_lowz_score"] = (
            (0.144 - df["redshift_abs"]).clip(lower=0.0)
            * (1.30 - df["compact_color_score"]).clip(lower=0.0)
            * (0.75 - df["mag_std"]).clip(lower=0.0)
        )
        if "is_blue_cloud" in df.columns:
            df["blue_cloud_compact_lowz_score"] = df["compact_lowz_score"] * df["is_blue_cloud"]
            df["lowz_blue_cloud_compact_low_mag_std"] = (
                (df["redshift_abs"] <= 0.144)
                & (df["compact_color_score"] <= 1.30)
                & (df["mag_std"] <= 0.75)
                & df["is_blue_cloud"].eq(1)
            ).astype("int8")

    return df


def add_extra_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-3

    if all(col in df.columns for col in MAG_COLS) and "mag_mean" in df.columns:
        for col in MAG_COLS:
            df[f"{col}_minus_mag_mean"] = df[col] - df["mag_mean"]
            df[f"{col}_over_mag_mean"] = df[col] / (df["mag_mean"].abs() + eps)

    for left, right in [
        ("u_minus_g", "g_minus_r"),
        ("g_minus_r", "r_minus_i"),
        ("r_minus_i", "i_minus_z"),
        ("u_minus_r", "r_minus_z"),
        ("g_minus_i", "i_minus_z"),
    ]:
        if left in df.columns and right in df.columns:
            df[f"{left}_over_{right}"] = df[left] / (df[right].abs() + eps)
            df[f"{left}_x_{right}"] = df[left] * df[right]

    if "redshift" in df.columns:
        redshift_abs = df["redshift"].abs()
        df["redshift_log1p_abs"] = np.log1p(redshift_abs)
        df["redshift_sq"] = df["redshift"] ** 2

    if "mag_range" in df.columns and "redshift" in df.columns:
        df["mag_range_x_redshift"] = df["mag_range"] * df["redshift"]
        df["mag_range_over_redshift"] = df["mag_range"] / (df["redshift"].abs() + eps)

    return df


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    out = add_astronomy_features(raw)
    out = add_extra_numeric_features(out)
    out = add_targeted_features(out)
    out = add_local_guard_features(out)
    return out


def add_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for pred_col in ["pred015", "pred017", "pred018", "base_pred"]:
        if pred_col in df.columns:
            codes = df[pred_col].astype(str).map(CLASS_TO_INT).fillna(-1).astype("int8")
            df[f"{pred_col}_code"] = codes
            for cls in CLASSES:
                df[f"{pred_col}_is_{cls}"] = df[pred_col].astype(str).eq(cls).astype("int8")

    if all(f"stack_p_{cls}" in df.columns for cls in CLASSES):
        p = df[[f"stack_p_{cls}" for cls in CLASSES]].to_numpy(dtype="float64")
        p = normalize_proba(p)
        df["stack_margin"] = np.partition(p, -1, axis=1)[:, -1] - np.partition(p, -2, axis=1)[:, -2]
        df["stack_entropy"] = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
        df["stack_galaxy_given_galaxy_star"] = p[:, CLASS_TO_INT["GALAXY"]] / np.clip(
            p[:, CLASS_TO_INT["GALAXY"]] + p[:, CLASS_TO_INT["STAR"]],
            1e-12,
            None,
        )
        df["stack_star_given_galaxy_star"] = p[:, CLASS_TO_INT["STAR"]] / np.clip(
            p[:, CLASS_TO_INT["GALAXY"]] + p[:, CLASS_TO_INT["STAR"]],
            1e-12,
            None,
        )
        df["stack_qso_margin_vs_galaxy"] = p[:, CLASS_TO_INT["QSO"]] - p[:, CLASS_TO_INT["GALAXY"]]
        df["stack_star_margin_vs_galaxy"] = p[:, CLASS_TO_INT["STAR"]] - p[:, CLASS_TO_INT["GALAXY"]]
        stack_pred = p.argmax(axis=1).astype("int8")
        df["stack_pred_code"] = stack_pred
        for cls in CLASSES:
            df[f"stack_pred_is_{cls}"] = (stack_pred == CLASS_TO_INT[cls]).astype("int8")

    if all(col in df.columns for col in ["pred015", "pred018"]):
        df["pred015_ne_pred018"] = df["pred015"].astype(str).ne(df["pred018"].astype(str)).astype("int8")
    if all(col in df.columns for col in ["pred015", "pred017"]):
        df["pred015_ne_pred017"] = df["pred015"].astype(str).ne(df["pred017"].astype(str)).astype("int8")
    if all(col in df.columns for col in ["pred017", "pred018"]):
        df["pred017_ne_pred018"] = df["pred017"].astype(str).ne(df["pred018"].astype(str)).astype("int8")

    return df


def make_model_matrices(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    drop_cols = {ID_COL, LABEL_COL, "true_class", "pred015", "pred017", "pred018", "base_pred"}
    train = train_features.drop(columns=[col for col in drop_cols if col in train_features.columns]).copy()
    test = test_features.drop(columns=[col for col in drop_cols if col in test_features.columns]).copy()

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan
    for col in test.columns:
        if col not in train.columns:
            train[col] = np.nan
    test = test[train.columns]

    for col in train.columns:
        if (
            pd.api.types.is_object_dtype(train[col])
            or pd.api.types.is_string_dtype(train[col])
            or isinstance(train[col].dtype, pd.CategoricalDtype)
        ):
            combined = pd.concat(
                [train[col].astype("string"), test[col].astype("string")],
                ignore_index=True,
            )
            categories = sorted(value for value in combined.dropna().unique().tolist())
            dtype = pd.CategoricalDtype(categories=categories)
            train[col] = train[col].astype("string").astype(dtype).cat.codes.astype("int16")
            test[col] = test[col].astype("string").astype(dtype).cat.codes.astype("int16")

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    numeric_cols = [
        col
        for col in train.columns
        if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_bool_dtype(train[col])
    ]
    train = train[numeric_cols].astype("float32")
    test = test[numeric_cols].astype("float32")
    return train, test, numeric_cols


def merge_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_train = pd.read_csv(args.raw_train)
    raw_test = pd.read_csv(args.raw_test)
    if LABEL_COL not in raw_train.columns:
        raise ValueError(f"{args.raw_train} must contain {LABEL_COL!r}.")

    train_features = build_feature_frame(raw_train.rename(columns={LABEL_COL: "true_class"}))
    test_features = build_feature_frame(raw_test)

    pred015_train = read_pred(args.pred015_train, "pred015")
    pred015_test = read_pred(args.pred015_test, "pred015")
    pred017_train = read_pred(args.pred017_train, "pred017")
    pred017_test = read_pred(args.pred017_test, "pred017")

    train = train_features.merge(pred015_train, on=ID_COL, how="left", validate="one_to_one")
    train = train.merge(pred017_train, on=ID_COL, how="left", validate="one_to_one")
    test = test_features.merge(pred015_test, on=ID_COL, how="left", validate="one_to_one")
    test = test.merge(pred017_test, on=ID_COL, how="left", validate="one_to_one")

    if args.pred018_train is not None and args.pred018_test is not None:
        train = train.merge(read_pred(args.pred018_train, "pred018"), on=ID_COL, how="left", validate="one_to_one")
        test = test.merge(read_pred(args.pred018_test, "pred018"), on=ID_COL, how="left", validate="one_to_one")
        pred018_source = "provided"
    elif args.pred018_train is None and args.pred018_test is None:
        train["pred018"] = reconstruct_018(train["pred017"], train["pred015"])
        test["pred018"] = reconstruct_018(test["pred017"], test["pred015"])
        pred018_source = "reconstructed_from_017_galaxy_to_015_qso_star"
    else:
        raise ValueError("--pred018-train and --pred018-test must be provided together.")

    base_col = f"pred{args.base}"
    train["base_pred"] = train[base_col].astype(str)
    test["base_pred"] = test[base_col].astype(str)

    train_proba, prefix = load_proba(args.train_proba, args.proba_prefix)
    test_proba, _ = load_proba(args.test_proba, args.proba_prefix, prefix=prefix)
    train = train.merge(train_proba, on=ID_COL, how="left", validate="one_to_one")
    test = test.merge(test_proba, on=ID_COL, how="left", validate="one_to_one")

    for frame_name, frame in [("train", train), ("test", test)]:
        required = ["pred015", "pred017", "pred018", "base_pred"] + [f"stack_p_{cls}" for cls in CLASSES]
        missing_counts = {col: int(frame[col].isna().sum()) for col in required if col in frame.columns}
        missing_counts = {col: value for col, value in missing_counts.items() if value}
        if missing_counts:
            raise ValueError(f"{frame_name} alignment left missing values: {missing_counts}")

    train = add_prediction_features(train)
    test = add_prediction_features(test)
    train.attrs["pred018_source"] = pred018_source
    test.attrs["pred018_source"] = pred018_source
    return train, test, prefix


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels.astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels.astype(str).map(CLASS_TO_INT).to_numpy(dtype="int8")


def class_sample_weight(y_true: np.ndarray, class_weights: dict[str, float]) -> np.ndarray:
    weights = np.asarray([class_weights[cls] for cls in CLASSES], dtype="float64")
    return weights[y_true.astype("int64")]


def confusion_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(
        y_true.astype("int64") * len(CLASSES) + y_pred.astype("int64"),
        minlength=len(CLASSES) ** 2,
    ).reshape(len(CLASSES), len(CLASSES))


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> float:
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(CLASSES), dtype="float64"),
        where=support > 0,
    )
    return float(recall.mean())


def metrics_from_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_weights: dict[str, float],
) -> dict:
    confusion = confusion_from_pred(y_true, y_pred)
    weights = np.asarray([class_weights[cls] for cls in CLASSES], dtype="float64")
    weighted_correct = float(np.sum(np.diag(confusion) * weights))
    weighted_total = float(np.sum(confusion.sum(axis=1) * weights))
    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    return {
        "accuracy": correct / total,
        "balanced_accuracy": balanced_accuracy_from_confusion(confusion),
        "weighted_accuracy": weighted_correct / weighted_total if weighted_total else 0.0,
        "errors": total - correct,
        "star_to_galaxy": int(confusion[CLASS_TO_INT["STAR"], CLASS_TO_INT["GALAXY"]]),
        "galaxy_to_star": int(confusion[CLASS_TO_INT["GALAXY"], CLASS_TO_INT["STAR"]]),
        "qso_to_galaxy": int(confusion[CLASS_TO_INT["QSO"], CLASS_TO_INT["GALAXY"]]),
        "galaxy_to_qso": int(confusion[CLASS_TO_INT["GALAXY"], CLASS_TO_INT["QSO"]]),
        "confusion": confusion,
    }


def pred_series_to_int(values: pd.Series) -> np.ndarray:
    unknown = sorted(set(values.astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown predictions: {unknown}")
    return values.astype(str).map(CLASS_TO_INT).to_numpy(dtype="int8")


def local_region_mask(
    frame: pd.DataFrame,
    redshift_abs_max: float,
    compact_color_score_max: float,
    mag_std_max: float,
    blue_cloud_required: bool,
) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    if "redshift_abs" not in frame.columns:
        raise ValueError("Missing redshift_abs for local guard mask.")
    if "compact_color_score" not in frame.columns:
        raise ValueError("Missing compact_color_score for local guard mask.")
    if "mag_std" not in frame.columns:
        raise ValueError("Missing mag_std for local guard mask.")

    mask &= frame["redshift_abs"].to_numpy(dtype="float64") <= redshift_abs_max
    mask &= frame["compact_color_score"].to_numpy(dtype="float64") <= compact_color_score_max
    mask &= frame["mag_std"].to_numpy(dtype="float64") <= mag_std_max
    if blue_cloud_required:
        if "is_blue_cloud" not in frame.columns:
            raise ValueError("blue_cloud_required=True but is_blue_cloud is missing.")
        mask &= frame["is_blue_cloud"].to_numpy(dtype="int8") == 1
    return mask


def build_local_sample_weight(
    train: pd.DataFrame,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    class_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    weights = class_sample_weight(y_true, class_weights)
    pred015 = pred_series_to_int(train["pred015"])
    pred018 = pred_series_to_int(train["pred018"])

    pred015_correct = pred015 == y_true
    base_correct = base_pred == y_true
    disagreement_015_018 = pred015 != pred018
    both_wrong = (~pred015_correct) & (~base_correct)
    only015_correct_true_galaxy = pred015_correct & (~base_correct) & (y_true == CLASS_TO_INT["GALAXY"])
    fixed_local_region = local_region_mask(
        train,
        redshift_abs_max=args.local_region_redshift_abs_max,
        compact_color_score_max=args.local_region_compact_color_score_max,
        mag_std_max=args.local_region_mag_std_max,
        blue_cloud_required=True,
    )

    weights[disagreement_015_018] *= args.disagreement_weight
    weights[both_wrong] *= args.both_wrong_weight
    weights[only015_correct_true_galaxy] *= args.only015_correct_true_galaxy_weight
    weights[fixed_local_region] *= args.local_region_weight

    audit = {
        "disagreement_015_018": int(disagreement_015_018.sum()),
        "both_wrong_015_base": int(both_wrong.sum()),
        "015_only_correct_true_galaxy": int(only015_correct_true_galaxy.sum()),
        "fixed_local_region": int(fixed_local_region.sum()),
        "binary_train_fixed_local_region": int(
            (fixed_local_region & np.isin(y_true, [CLASS_TO_INT["GALAXY"], CLASS_TO_INT["STAR"]])).sum()
        ),
        "sample_weight_min": float(np.min(weights)),
        "sample_weight_mean": float(np.mean(weights)),
        "sample_weight_max": float(np.max(weights)),
    }
    return weights, audit


def positive_class_proba(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Local binary model classes do not include GALAXY=1: {classes}")
    return proba[:, classes.index(1)].astype("float64")


def train_local_model(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y_true: np.ndarray,
    sample_weight: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    binary_mask = np.isin(y_true, [CLASS_TO_INT["GALAXY"], CLASS_TO_INT["STAR"]])
    binary_idx = np.flatnonzero(binary_mask)
    y_binary = (y_true[binary_idx] == CLASS_TO_INT["GALAXY"]).astype("int8")

    class_counts = pd.Series(y_binary).value_counts().to_dict()
    min_class_count = min(class_counts.get(0, 0), class_counts.get(1, 0))
    if min_class_count < args.n_splits:
        raise ValueError(
            f"Need at least n_splits={args.n_splits} examples per binary class, got {class_counts}."
        )

    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_binary = np.full(len(binary_idx), np.nan, dtype="float64")
    train_fold_sum = np.zeros(len(X), dtype="float64")
    test_sum = np.zeros(len(X_test), dtype="float64")
    rows = []

    for fold, (tr_rel, va_rel) in enumerate(folds.split(X.iloc[binary_idx], y_binary), start=1):
        tr_idx = binary_idx[tr_rel]
        va_idx = binary_idx[va_rel]
        model = HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=args.hgb_l2_regularization,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=args.seed + fold,
        )
        model.fit(X.iloc[tr_idx], y_binary[tr_rel], sample_weight=sample_weight[tr_idx])
        va_p = positive_class_proba(model, X.iloc[va_idx])
        oof_binary[va_rel] = va_p
        train_fold_sum += positive_class_proba(model, X) / args.n_splits
        test_sum += positive_class_proba(model, X_test) / args.n_splits

        va_pred = (va_p >= 0.5).astype("int8")
        row = {
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "galaxy_rate_valid": float(y_binary[va_rel].mean()),
            "accuracy_at_050": float((va_pred == y_binary[va_rel]).mean()),
            "log_loss": float(log_loss(y_binary[va_rel], np.column_stack([1.0 - va_p, va_p]), labels=[0, 1])),
        }
        try:
            row["roc_auc"] = float(roc_auc_score(y_binary[va_rel], va_p))
        except ValueError:
            row["roc_auc"] = np.nan
        rows.append(row)

    if np.isnan(oof_binary).any():
        missing = int(np.isnan(oof_binary).sum())
        raise RuntimeError(f"Local model left {missing} binary OOF rows without predictions.")

    train_p = train_fold_sum.copy()
    train_p[binary_idx] = oof_binary
    fold_df = pd.DataFrame(rows)
    overall_pred = (train_p[binary_idx] >= 0.5).astype("int8")
    overall = {
        "fold": "overall_binary_oof",
        "n_train": int(len(binary_idx)),
        "n_valid": int(len(binary_idx)),
        "galaxy_rate_valid": float(y_binary.mean()),
        "accuracy_at_050": float((overall_pred == y_binary).mean()),
        "log_loss": float(log_loss(y_binary, np.column_stack([1.0 - train_p[binary_idx], train_p[binary_idx]]), labels=[0, 1])),
    }
    try:
        overall["roc_auc"] = float(roc_auc_score(y_binary, train_p[binary_idx]))
    except ValueError:
        overall["roc_auc"] = np.nan
    fold_df = pd.concat([fold_df, pd.DataFrame([overall])], ignore_index=True)
    return train_p, test_sum, fold_df


def search_guard_grid(
    train: pd.DataFrame,
    p_galaxy: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    class_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    thresholds = parse_float_grid(args.p_galaxy_thresholds)
    redshift_values = parse_float_grid(args.redshift_abs_max_values)
    compact_values = parse_float_grid(args.compact_color_score_max_values)
    mag_std_values = parse_float_grid(args.mag_std_max_values)
    blue_required_values = parse_bool_grid(args.blue_cloud_required_values)

    baseline = metrics_from_pred(y_true, base_pred, class_weights)
    baseline_correct = base_pred == y_true
    sample_weight = class_sample_weight(y_true, class_weights)
    weighted_total = float(sample_weight.sum())
    baseline_weighted_correct = float(sample_weight[baseline_correct].sum())
    baseline_correct_count = int(baseline_correct.sum())

    pred015 = pred_series_to_int(train["pred015"])
    pred015_correct = pred015 == y_true
    only015_correct = pred015_correct & (~baseline_correct)
    base_is_guardable = np.isin(base_pred, [CLASS_TO_INT["STAR"], CLASS_TO_INT["QSO"]])
    true_galaxy = y_true == CLASS_TO_INT["GALAXY"]
    true_qso = y_true == CLASS_TO_INT["QSO"]
    true_star = y_true == CLASS_TO_INT["STAR"]
    base_star = base_pred == CLASS_TO_INT["STAR"]
    base_qso = base_pred == CLASS_TO_INT["QSO"]

    delta_weighted_if_changed = np.zeros(len(train), dtype="float64")
    delta_correct_if_changed = np.zeros(len(train), dtype="int8")
    recover_mask = base_is_guardable & true_galaxy & (~baseline_correct)
    injure_base_correct_mask = base_is_guardable & (~true_galaxy) & baseline_correct
    delta_weighted_if_changed[recover_mask] = sample_weight[recover_mask]
    delta_weighted_if_changed[injure_base_correct_mask] = -sample_weight[injure_base_correct_mask]
    delta_correct_if_changed[recover_mask] = 1
    delta_correct_if_changed[injure_base_correct_mask] = -1

    rows = []
    for p_threshold in thresholds:
        p_mask = base_is_guardable & (p_galaxy >= float(p_threshold))
        if not p_mask.any():
            for redshift_abs_max in redshift_values:
                for compact_color_score_max in compact_values:
                    for mag_std_max in mag_std_values:
                        for blue_required in blue_required_values:
                            rows.append(
                                {
                                    "p_galaxy_threshold": float(p_threshold),
                                    "redshift_abs_max": float(redshift_abs_max),
                                    "compact_color_score_max": float(compact_color_score_max),
                                    "mag_std_max": float(mag_std_max),
                                    "blue_cloud_required": bool(blue_required),
                                    **grid_metrics_row(
                                        baseline,
                                        baseline_correct_count,
                                        baseline_weighted_correct,
                                        weighted_total,
                                        len(train),
                                        np.zeros(len(train), dtype=bool),
                                        delta_correct_if_changed,
                                        delta_weighted_if_changed,
                                        only015_correct,
                                        true_galaxy,
                                        true_qso,
                                        true_star,
                                        base_star,
                                        base_qso,
                                        args,
                                    ),
                                }
                            )
            continue

        for redshift_abs_max in redshift_values:
            redshift_mask = train["redshift_abs"].to_numpy(dtype="float64") <= float(redshift_abs_max)
            for compact_color_score_max in compact_values:
                compact_mask = train["compact_color_score"].to_numpy(dtype="float64") <= float(compact_color_score_max)
                for mag_std_max in mag_std_values:
                    mag_mask = train["mag_std"].to_numpy(dtype="float64") <= float(mag_std_max)
                    base_local_mask = p_mask & redshift_mask & compact_mask & mag_mask
                    for blue_required in blue_required_values:
                        change_mask = base_local_mask
                        if blue_required:
                            change_mask = change_mask & (train["is_blue_cloud"].to_numpy(dtype="int8") == 1)

                        rows.append(
                            {
                                "p_galaxy_threshold": float(p_threshold),
                                "redshift_abs_max": float(redshift_abs_max),
                                "compact_color_score_max": float(compact_color_score_max),
                                "mag_std_max": float(mag_std_max),
                                "blue_cloud_required": bool(blue_required),
                                **grid_metrics_row(
                                    baseline,
                                    baseline_correct_count,
                                    baseline_weighted_correct,
                                    weighted_total,
                                    len(train),
                                    change_mask,
                                    delta_correct_if_changed,
                                    delta_weighted_if_changed,
                                    only015_correct,
                                    true_galaxy,
                                    true_qso,
                                    true_star,
                                    base_star,
                                    base_qso,
                                    args,
                                ),
                            }
                        )

    results = pd.DataFrame(rows)
    return results, baseline


def grid_metrics_row(
    baseline: dict,
    baseline_correct_count: int,
    baseline_weighted_correct: float,
    weighted_total: float,
    total_rows: int,
    change_mask: np.ndarray,
    delta_correct_if_changed: np.ndarray,
    delta_weighted_if_changed: np.ndarray,
    only015_correct: np.ndarray,
    true_galaxy: np.ndarray,
    true_qso: np.ndarray,
    true_star: np.ndarray,
    base_star: np.ndarray,
    base_qso: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    changed = int(change_mask.sum())
    delta_correct = int(delta_correct_if_changed[change_mask].sum()) if changed else 0
    delta_weighted_correct = float(delta_weighted_if_changed[change_mask].sum()) if changed else 0.0
    correct_count = baseline_correct_count + delta_correct
    weighted_correct = baseline_weighted_correct + delta_weighted_correct

    changed_true_galaxy = int((change_mask & true_galaxy).sum())
    changed_true_qso = int((change_mask & true_qso).sum())
    changed_true_star = int((change_mask & true_star).sum())
    recovered = int((change_mask & true_galaxy).sum())
    injured_base_correct = int((change_mask & (~true_galaxy) & (delta_correct_if_changed < 0)).sum())
    injured_015_only_correct = int((change_mask & only015_correct & (~true_galaxy)).sum())
    recovered_015_only_correct_true_galaxy = int((change_mask & only015_correct & true_galaxy).sum())
    changed_galaxy_from_star = int((change_mask & true_galaxy & base_star).sum())
    changed_galaxy_from_qso = int((change_mask & true_galaxy & base_qso).sum())

    weighted_accuracy = weighted_correct / weighted_total if weighted_total else 0.0
    guard_objective = (
        weighted_accuracy
        - args.guard_injury_penalty * injured_015_only_correct / max(total_rows, 1)
        + args.guard_recovery_bonus * recovered_015_only_correct_true_galaxy / max(total_rows, 1)
    )
    errors = total_rows - correct_count
    return {
        "weighted_accuracy": weighted_accuracy,
        "accuracy": correct_count / total_rows,
        "errors": errors,
        "delta_weighted_accuracy": weighted_accuracy - baseline["weighted_accuracy"],
        "delta_accuracy": correct_count / total_rows - baseline["accuracy"],
        "delta_errors": errors - baseline["errors"],
        "guard_objective": guard_objective,
        "changed_to_galaxy": changed,
        "changed_rate": changed / total_rows,
        "recovered_to_galaxy": recovered,
        "injured_base_correct": injured_base_correct,
        "injured_015_only_correct": injured_015_only_correct,
        "recovered_015_only_correct_true_galaxy": recovered_015_only_correct_true_galaxy,
        "changed_true_galaxy": changed_true_galaxy,
        "changed_true_qso": changed_true_qso,
        "changed_true_star": changed_true_star,
        "star_to_galaxy": int(baseline["star_to_galaxy"] + changed_true_star),
        "qso_to_galaxy": int(baseline["qso_to_galaxy"] + changed_true_qso),
        "galaxy_to_star": int(baseline["galaxy_to_star"] - changed_galaxy_from_star),
        "galaxy_to_qso": int(baseline["galaxy_to_qso"] - changed_galaxy_from_qso),
    }


def apply_guard(
    frame: pd.DataFrame,
    base_pred: np.ndarray,
    p_galaxy: np.ndarray,
    row: pd.Series | dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_mask = local_region_mask(
        frame,
        redshift_abs_max=float(row["redshift_abs_max"]),
        compact_color_score_max=float(row["compact_color_score_max"]),
        mag_std_max=float(row["mag_std_max"]),
        blue_cloud_required=bool(row["blue_cloud_required"]),
    )
    guardable = np.isin(base_pred, [CLASS_TO_INT["STAR"], CLASS_TO_INT["QSO"]])
    change_mask = guardable & local_mask & (p_galaxy >= float(row["p_galaxy_threshold"]))
    pred = base_pred.copy()
    pred[change_mask] = CLASS_TO_INT["GALAXY"]
    return pred, change_mask, local_mask


def save_prediction_csv(path: Path, ids: pd.Series, pred: np.ndarray) -> None:
    pd.DataFrame({ID_COL: ids.values, LABEL_COL: INT_TO_CLASS[pred]}).to_csv(path, index=False)


def save_submission(path: Path, sample_submission_path: Path, test_ids: pd.Series, pred: np.ndarray) -> None:
    sample_submission = pd.read_csv(sample_submission_path)
    pred_df = pd.DataFrame({ID_COL: test_ids.values, LABEL_COL: INT_TO_CLASS[pred]})
    submission = sample_submission[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
    if submission[LABEL_COL].isna().any():
        missing = int(submission[LABEL_COL].isna().sum())
        raise ValueError(f"Submission alignment left {missing} missing predictions.")
    submission.to_csv(path, index=False)


def save_confusion(path: Path, confusion: np.ndarray) -> None:
    pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(path)


def print_dry_run_stats(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prefix: str,
    class_weights: dict[str, float],
    args: argparse.Namespace,
) -> None:
    y_true = encode_labels(train["true_class"])
    base_pred = pred_series_to_int(train["base_pred"])
    base_metrics = metrics_from_pred(y_true, base_pred, class_weights)
    pred015 = pred_series_to_int(train["pred015"])
    pred018 = pred_series_to_int(train["pred018"])
    base_correct = base_pred == y_true
    pred015_correct = pred015 == y_true

    fixed_region_train = local_region_mask(
        train,
        args.local_region_redshift_abs_max,
        args.local_region_compact_color_score_max,
        args.local_region_mag_std_max,
        blue_cloud_required=True,
    )
    fixed_region_test = local_region_mask(
        test,
        args.local_region_redshift_abs_max,
        args.local_region_compact_color_score_max,
        args.local_region_mag_std_max,
        blue_cloud_required=True,
    )
    train_guardable = train["base_pred"].astype(str).isin(["STAR", "QSO"]).to_numpy()
    test_guardable = test["base_pred"].astype(str).isin(["STAR", "QSO"]).to_numpy()
    grid_size = (
        len(parse_float_grid(args.p_galaxy_thresholds))
        * len(parse_float_grid(args.redshift_abs_max_values))
        * len(parse_float_grid(args.compact_color_score_max_values))
        * len(parse_float_grid(args.mag_std_max_values))
        * len(parse_bool_grid(args.blue_cloud_required_values))
    )

    print("Dry run: no local model training will be started.")
    print(f"Probability prefix: {prefix}")
    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Base stream: {args.base}; pred018 source: {train.attrs.get('pred018_source')}")
    print("Base metrics:")
    printable_metrics = {key: value for key, value in base_metrics.items() if key != "confusion"}
    print(json.dumps(printable_metrics, indent=2, sort_keys=True))
    print("Train true class distribution:")
    print(train["true_class"].value_counts().to_string())
    print("Train base prediction distribution:")
    print(train["base_pred"].value_counts().to_string())
    print("Test base prediction distribution:")
    print(test["base_pred"].value_counts().to_string())
    print("Region counters:")
    region = {
        "pred015_ne_pred018": int((pred015 != pred018).sum()),
        "both_wrong_015_base": int(((~pred015_correct) & (~base_correct)).sum()),
        "015_only_correct": int((pred015_correct & (~base_correct)).sum()),
        "015_only_correct_true_galaxy": int(
            (pred015_correct & (~base_correct) & (y_true == CLASS_TO_INT["GALAXY"])).sum()
        ),
        "train_guardable_base_star_qso": int(train_guardable.sum()),
        "test_guardable_base_star_qso": int(test_guardable.sum()),
        "train_fixed_lowz_blue_cloud_compact_low_mag_std": int(fixed_region_train.sum()),
        "test_fixed_lowz_blue_cloud_compact_low_mag_std": int(fixed_region_test.sum()),
        "train_guardable_and_fixed_region": int((train_guardable & fixed_region_train).sum()),
        "test_guardable_and_fixed_region": int((test_guardable & fixed_region_test).sum()),
        "grid_rows": int(grid_size),
    }
    print(json.dumps(region, indent=2, sort_keys=True))
    print("Train fixed region by true class:")
    print(train.loc[fixed_region_train, "true_class"].value_counts().to_string())
    print("Train guardable fixed region by true/base:")
    if (train_guardable & fixed_region_train).any():
        table = pd.crosstab(
            train.loc[train_guardable & fixed_region_train, "true_class"],
            train.loc[train_guardable & fixed_region_train, "base_pred"],
        )
        print(table.to_string())
    else:
        print("(empty)")


def main() -> None:
    args = parse_args()
    class_weights = parse_class_metric_weights(args.class_metric_weights)
    train, test, prefix = merge_inputs(args)

    if args.dry_run:
        print_dry_run_stats(train, test, prefix, class_weights, args)
        return

    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    y_true = encode_labels(train["true_class"])
    base_pred = pred_series_to_int(train["base_pred"])
    test_base_pred = pred_series_to_int(test["base_pred"])

    X, X_test, feature_cols = make_model_matrices(train, test)
    sample_weight, sample_weight_audit = build_local_sample_weight(
        train,
        y_true,
        base_pred,
        class_weights,
        args,
    )
    local_oof_p, local_test_p, fold_scores = train_local_model(
        X,
        X_test,
        y_true,
        sample_weight,
        args,
    )
    fold_scores.to_csv(run_dir / "local_model_fold_scores.csv", index=False)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL].values,
            "true_class": train["true_class"].values,
            "base_pred": train["base_pred"].values,
            "local_p_galaxy": local_oof_p,
            "prediction_source": np.where(
                train["true_class"].astype(str).isin(["GALAXY", "STAR"]),
                "binary_oof",
                "fold_average_not_binary_training",
            ),
        }
    ).to_csv(run_dir / "local_model_oof_train_proba.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: test[ID_COL].values,
            "base_pred": test["base_pred"].values,
            "local_p_galaxy": local_test_p,
        }
    ).to_csv(run_dir / "local_model_test_proba.csv", index=False)

    results, baseline = search_guard_grid(train, local_oof_p, y_true, base_pred, class_weights, args)
    results.to_csv(run_dir / "grid_results.csv", index=False)
    ranked = results.sort_values(
        [
            "weighted_accuracy",
            "injured_015_only_correct",
            "recovered_015_only_correct_true_galaxy",
            "guard_objective",
            "changed_to_galaxy",
        ],
        ascending=[False, True, False, False, True],
    )
    top = ranked.head(args.top_n).copy()
    top.to_csv(run_dir / "top_by_guard_objective.csv", index=False)
    best = top.iloc[0].to_dict()

    best_train_pred, best_train_change_mask, best_train_local_mask = apply_guard(
        train,
        base_pred,
        local_oof_p,
        best,
    )
    best_test_pred, best_test_change_mask, best_test_local_mask = apply_guard(
        test,
        test_base_pred,
        local_test_p,
        best,
    )
    best_metrics = metrics_from_pred(y_true, best_train_pred, class_weights)

    save_prediction_csv(run_dir / "best_guard_oof_train_pred.csv", train[ID_COL], best_train_pred)
    save_prediction_csv(run_dir / "best_guard_test_pred.csv", test[ID_COL], best_test_pred)
    save_submission(run_dir / "guard_submission.csv", args.sample_submission, test[ID_COL], best_test_pred)
    save_confusion(run_dir / "baseline_confusion_matrix.csv", baseline["confusion"])
    save_confusion(run_dir / "best_guard_confusion_matrix.csv", best_metrics["confusion"])

    pd.DataFrame(
        {
            ID_COL: train[ID_COL].values,
            "true_class": train["true_class"].values,
            "pred015": train["pred015"].values,
            "pred017": train["pred017"].values,
            "pred018": train["pred018"].values,
            "base_pred": train["base_pred"].values,
            "guard_pred": INT_TO_CLASS[best_train_pred],
            "local_p_galaxy": local_oof_p,
            "local_mask": best_train_local_mask,
            "changed_to_galaxy": best_train_change_mask,
        }
    ).to_csv(run_dir / "best_guard_oof_train_audit.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: test[ID_COL].values,
            "pred015": test["pred015"].values,
            "pred017": test["pred017"].values,
            "pred018": test["pred018"].values,
            "base_pred": test["base_pred"].values,
            "guard_pred": INT_TO_CLASS[best_test_pred],
            "local_p_galaxy": local_test_p,
            "local_mask": best_test_local_mask,
            "changed_to_galaxy": best_test_change_mask,
        }
    ).to_csv(run_dir / "best_guard_test_audit.csv", index=False)

    score_rows = []
    for name, col in [("015", "pred015"), ("017", "pred017"), ("018", "pred018"), (f"base_{args.base}", "base_pred")]:
        pred = pred_series_to_int(train[col])
        metrics = metrics_from_pred(y_true, pred, class_weights)
        score_rows.append({"model": name, **{key: value for key, value in metrics.items() if key != "confusion"}})
    score_rows.append(
        {
            "model": "local_guard",
            **{key: value for key, value in best_metrics.items() if key != "confusion"},
            "changed_to_galaxy": int(best_train_change_mask.sum()),
            "test_changed_to_galaxy": int(best_test_change_mask.sum()),
            "injured_015_only_correct": int(best["injured_015_only_correct"]),
            "recovered_015_only_correct_true_galaxy": int(best["recovered_015_only_correct_true_galaxy"]),
            "guard_objective": float(best["guard_objective"]),
        }
    )
    pd.DataFrame(score_rows).to_csv(run_dir / "scores.csv", index=False)

    register_submission(
        run_dir / "guard_submission.csv",
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="local_guard_threshold",
        model_name=f"base_{args.base}_galaxy_guard",
        metrics={
            key: best.get(key, best_metrics.get(key))
            for key in [
                "accuracy",
                "weighted_accuracy",
                "delta_accuracy",
                "delta_weighted_accuracy",
                "errors",
                "delta_errors",
                "guard_objective",
                "changed_to_galaxy",
                "injured_015_only_correct",
                "recovered_015_only_correct_true_galaxy",
            ]
        },
        params={
            "base": args.base,
            "pred015_train": args.pred015_train,
            "pred017_train": args.pred017_train,
            "pred018_source": train.attrs.get("pred018_source"),
            "train_proba": args.train_proba,
            "test_proba": args.test_proba,
            "proba_prefix": prefix,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "class_metric_weights": class_weights,
            "best_thresholds": {
                key: best[key]
                for key in [
                    "p_galaxy_threshold",
                    "redshift_abs_max",
                    "compact_color_score_max",
                    "mag_std_max",
                    "blue_cloud_required",
                ]
            },
        },
        extra={
            "grid_results": "grid_results.csv",
            "top_results": "top_by_guard_objective.csv",
            "scores": "scores.csv",
            "local_model_oof": "local_model_oof_train_proba.csv",
            "local_model_test": "local_model_test_proba.csv",
        },
    )

    manifest = {
        "args": vars(args),
        "classes": CLASSES,
        "class_metric_weights": class_weights,
        "proba_prefix": prefix,
        "pred018_source": train.attrs.get("pred018_source"),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "model_feature_count": len(feature_cols),
        "model_feature_columns": feature_cols,
        "sample_weight_audit": sample_weight_audit,
        "grid": {
            "p_galaxy_thresholds": parse_float_grid(args.p_galaxy_thresholds).tolist(),
            "redshift_abs_max_values": parse_float_grid(args.redshift_abs_max_values).tolist(),
            "compact_color_score_max_values": parse_float_grid(args.compact_color_score_max_values).tolist(),
            "mag_std_max_values": parse_float_grid(args.mag_std_max_values).tolist(),
            "blue_cloud_required_values": parse_bool_grid(args.blue_cloud_required_values),
            "rows": int(len(results)),
        },
        "baseline": {key: value for key, value in baseline.items() if key != "confusion"},
        "best": {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value)
            for key, value in best.items()
        },
        "best_metrics": {key: value for key, value in best_metrics.items() if key != "confusion"},
        "test_changed_to_galaxy": int(best_test_change_mask.sum()),
        "test_class_distribution": pd.Series(INT_TO_CLASS[best_test_pred]).value_counts().to_dict(),
    }
    save_json(run_dir / "manifest.json", manifest)

    print("Best guard row:")
    print(pd.DataFrame([best]).to_string(index=False))
    print("Scores:")
    print(pd.read_csv(run_dir / "scores.csv").to_string(index=False))


if __name__ == "__main__":
    main()
