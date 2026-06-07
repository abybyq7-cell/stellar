from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL, MAG_COLS
from stellar.features import add_astronomy_features, add_local_guard_features, add_targeted_features
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASSES)}
INT_TO_CLASS = np.asarray(CLASSES)
DEFAULT_CLASS_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}


@dataclass(frozen=True)
class PairTask:
    name: str
    positive: str
    negative: str
    seed_offset: int


PAIR_TASKS = [
    PairTask("galaxy_vs_star", "GALAXY", "STAR", 11),
    PairTask("galaxy_vs_qso", "GALAXY", "QSO", 29),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train local pairwise binary OOF calibrators and materialize layer-1 blocks."
    )
    parser.add_argument("--raw-train", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--raw-test", type=Path, default=RAW_DATA_DIR / "test.csv")
    parser.add_argument("--sample-submission", type=Path, default=RAW_DATA_DIR / "sample_submission.csv")
    parser.add_argument(
        "--base-train-proba",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full26_nn_balanced_lr" / "lr_logits_oof_train.csv",
    )
    parser.add_argument(
        "--base-test-proba",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full26_nn_balanced_lr" / "lr_logits_oof_test.csv",
    )
    parser.add_argument("--proba-prefix", type=str, default="auto")
    parser.add_argument(
        "--base-train-pred",
        type=Path,
        default=OUTPUTS_DIR
        / "disagreement_arbitration"
        / "full26_nn_weighted_vs_015_arbitration"
        / "arbitrated_oof_train_pred.csv",
    )
    parser.add_argument(
        "--base-test-pred",
        type=Path,
        default=OUTPUTS_DIR
        / "disagreement_arbitration"
        / "full26_nn_weighted_vs_015_arbitration"
        / "arbitrated_test_pred.csv",
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "layer1_oof")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--class-weights", type=str, default="GALAXY:1.0,QSO:3.2,STAR:4.6")
    parser.add_argument("--hard-slice-weight", type=float, default=2.25)
    parser.add_argument("--base-error-weight", type=float, default=1.70)
    parser.add_argument("--galaxy-injury-weight", type=float, default=2.40)
    parser.add_argument("--disagreement015-weight", type=float, default=1.25)
    parser.add_argument("--hgb-max-iter", type=int, default=360)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.045)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=35)
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.05)
    parser.add_argument("--blend-alphas", type=str, default="0.25,0.50,0.75")
    parser.add_argument("--save-single-submissions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_class_weights(spec: str) -> dict[str, float]:
    weights = DEFAULT_CLASS_WEIGHTS.copy()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        weights[key.strip()] = float(value)
    missing = [cls for cls in CLASSES if cls not in weights]
    if missing:
        raise ValueError(f"Missing class weights for: {missing}")
    return weights


def parse_alphas(spec: str) -> list[float]:
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("--blend-alphas must contain at least one value.")
    return values


def detect_probability_prefix(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    valid = []
    for col in df.columns:
        if "__" not in col:
            continue
        prefix = col.rsplit("__", 1)[0]
        if all(f"{prefix}__{cls}" in df.columns for cls in CLASSES):
            valid.append(prefix)
    valid = sorted(set(valid))
    if len(valid) != 1:
        raise ValueError(f"Could not auto-detect one proba prefix. Found: {valid}")
    return valid[0]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype("float64"), 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def load_proba(path: Path, requested_prefix: str, prefix: str | None = None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    detected = prefix or detect_probability_prefix(df, requested_prefix)
    cols = [f"{detected}__{cls}" for cls in CLASSES]
    missing = [col for col in [ID_COL] + cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    p = normalize(df[cols].to_numpy(dtype="float64"))
    out = pd.DataFrame({ID_COL: df[ID_COL].values})
    for idx, cls in enumerate(CLASSES):
        out[f"base_p_{cls}"] = p[:, idx]
    return out, detected


def read_prediction(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain {ID_COL!r} and {LABEL_COL!r}")
    out = df[[ID_COL, LABEL_COL]].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(str)
    return out.rename(columns={LABEL_COL: col})


def maybe_read_prediction(path: Path, col: str) -> pd.DataFrame | None:
    return read_prediction(path, col) if path.exists() else None


def sample_train(train: pd.DataFrame, sample_rows: int, seed: int) -> pd.DataFrame:
    if sample_rows <= 0 or sample_rows >= len(train):
        return train.reset_index(drop=True)
    parts = []
    for _, group in train.groupby(LABEL_COL, sort=False):
        frac = len(group) / len(train)
        n = max(1, int(round(sample_rows * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > sample_rows:
        out = out.sample(n=sample_rows, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def add_hard_slice_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "redshift" in out.columns:
        out["hard_redshift_abs"] = out["redshift"].abs()
        out["hard_lowz"] = out["hard_redshift_abs"].le(0.144).astype("int8")
    if all(col in out.columns for col in MAG_COLS):
        mag = out[MAG_COLS]
        if "mag_std" not in out.columns:
            out["mag_std"] = mag.std(axis=1)
        out["hard_low_mag_std"] = out["mag_std"].le(0.75).astype("int8")
        if "mag_range" not in out.columns:
            out["mag_range"] = mag.max(axis=1) - mag.min(axis=1)
    color_cols = ["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]
    if all(col in out.columns for col in color_cols):
        colors = out[color_cols].astype("float64")
        out["hard_compact_color_score"] = np.sqrt(np.square(colors).mean(axis=1))
        out["hard_compact_color"] = out["hard_compact_color_score"].le(1.30).astype("int8")
    if all(col in out.columns for col in ["g_minus_r", "u_minus_z", "mag_range", "mag_std"]):
        wide = (
            (out["g_minus_r"] > 1.037)
            & (out["u_minus_z"] > 3.834)
            & ((out["mag_range"] > 3.949) | (out["mag_std"] > 1.624))
        )
        out["hard_wide_color_false"] = (~wide).astype("int8")
    if "galaxy_population" in out.columns:
        out["hard_blue_cloud"] = out["galaxy_population"].astype("string").eq("Blue_Cloud").astype("int8")

    needed = [
        "hard_lowz",
        "hard_blue_cloud",
        "hard_compact_color",
        "hard_low_mag_std",
        "hard_wide_color_false",
    ]
    if all(col in out.columns for col in needed):
        out["hard_slice_main"] = np.logical_and.reduce([out[col].astype(bool) for col in needed]).astype("int8")
    return out


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = add_astronomy_features(raw)
    frame = add_targeted_features(frame)
    frame = add_local_guard_features(frame)
    frame = add_hard_slice_features(frame)
    return frame


def add_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    p = out[[f"base_p_{cls}" for cls in CLASSES]].to_numpy(dtype="float64")
    p = normalize(p)
    out["base_margin"] = np.partition(p, -1, axis=1)[:, -1] - np.partition(p, -2, axis=1)[:, -2]
    out["base_entropy"] = -np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=1)
    out["base_galaxy_given_galaxy_star"] = p[:, 0] / np.clip(p[:, 0] + p[:, 2], 1e-12, None)
    out["base_galaxy_given_galaxy_qso"] = p[:, 0] / np.clip(p[:, 0] + p[:, 1], 1e-12, None)
    out["base_star_margin_vs_galaxy"] = p[:, 2] - p[:, 0]
    out["base_qso_margin_vs_galaxy"] = p[:, 1] - p[:, 0]
    base_pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    if "base_pred" not in out.columns:
        out["base_pred"] = base_pred
    for col in ["base_pred", "pred015"]:
        if col not in out.columns:
            continue
        values = out[col].astype(str)
        for cls in CLASSES:
            out[f"{col}_is_{cls}"] = values.eq(cls).astype("int8")
    if "pred015" in out.columns and "base_pred" in out.columns:
        out["pred015_ne_base"] = out["pred015"].astype(str).ne(out["base_pred"].astype(str)).astype("int8")
    return out


def make_model_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    drop_cols = {ID_COL, LABEL_COL, "source_index", "base_pred", "pred015"}
    X = train.drop(columns=[col for col in drop_cols if col in train.columns]).copy()
    X_test = test.drop(columns=[col for col in drop_cols if col in test.columns]).copy()
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    for col in X_test.columns:
        if col not in X.columns:
            X[col] = np.nan
    X_test = X_test[X.columns]

    for col in X.columns:
        if (
            pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        ):
            combined = pd.concat([X[col].astype("string"), X_test[col].astype("string")], ignore_index=True)
            categories = sorted(combined.fillna("__NA__").unique().tolist())
            mapping = {value: idx for idx, value in enumerate(categories)}
            X[col] = X[col].astype("string").fillna("__NA__").map(mapping).fillna(-1).astype("int16")
            X_test[col] = X_test[col].astype("string").fillna("__NA__").map(mapping).fillna(-1).astype("int16")

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])]
    return X[numeric_cols].astype("float32"), X_test[numeric_cols].astype("float32"), numeric_cols


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels.astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels.astype(str).map(CLASS_TO_INT).to_numpy(dtype="int8")


def pair_sample_weight(
    train: pd.DataFrame,
    y_enc: np.ndarray,
    pair_idx: np.ndarray,
    task: PairTask,
    class_weights: dict[str, float],
    args: argparse.Namespace,
) -> np.ndarray:
    labels = train[LABEL_COL].astype(str).to_numpy()
    weights = np.asarray([class_weights[label] for label in labels[pair_idx]], dtype="float64")
    hard = train.get("hard_slice_main", pd.Series(0, index=train.index)).to_numpy(dtype="int8")[pair_idx].astype(bool)
    weights[hard] *= args.hard_slice_weight

    if "base_pred" in train.columns:
        base = train["base_pred"].astype(str).to_numpy()[pair_idx]
        truth = labels[pair_idx]
        weights[base != truth] *= args.base_error_weight
        galaxy_injury = (truth == "GALAXY") & np.isin(base, ["STAR", "QSO"])
        weights[galaxy_injury] *= args.galaxy_injury_weight
    if "pred015" in train.columns and "base_pred" in train.columns:
        pred015 = train["pred015"].astype(str).to_numpy()[pair_idx]
        base = train["base_pred"].astype(str).to_numpy()[pair_idx]
        weights[pred015 != base] *= args.disagreement015_weight

    weights = np.clip(weights, 0.25, 50.0)
    weights /= np.mean(weights)
    return weights.astype("float32")


def positive_proba(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Binary model classes do not include positive label 1: {classes}")
    values = proba[:, classes.index(1)].astype("float64")
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def train_pair_task(
    task: PairTask,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    train: pd.DataFrame,
    y_enc: np.ndarray,
    class_weights: dict[str, float],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pos_idx = CLASS_TO_INT[task.positive]
    neg_idx = CLASS_TO_INT[task.negative]
    pair_mask = np.isin(y_enc, [pos_idx, neg_idx])
    pair_indices = np.flatnonzero(pair_mask)
    y_pair = (y_enc[pair_indices] == pos_idx).astype("int8")
    sample_weight = pair_sample_weight(train, y_enc, pair_indices, task, class_weights, args)
    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + task.seed_offset)

    pair_oof = np.full(len(pair_indices), np.nan, dtype="float64")
    train_avg = np.zeros(len(train), dtype="float64")
    test_avg = np.zeros(len(X_test), dtype="float64")
    rows = []
    for fold, (tr_rel, va_rel) in enumerate(folds.split(X.iloc[pair_indices], y_pair), start=1):
        started = time.time()
        tr_idx = pair_indices[tr_rel]
        va_idx = pair_indices[va_rel]
        model = HistGradientBoostingClassifier(
            max_iter=args.hgb_max_iter,
            learning_rate=args.hgb_learning_rate,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_samples_leaf=args.hgb_min_samples_leaf,
            l2_regularization=args.hgb_l2_regularization,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=35,
            random_state=args.seed + task.seed_offset + fold,
        )
        model.fit(X.iloc[tr_idx], y_pair[tr_rel], sample_weight=sample_weight[tr_rel])
        va_p = positive_proba(model, X.iloc[va_idx])
        pair_oof[va_rel] = va_p
        train_avg += positive_proba(model, X) / args.n_splits
        test_avg += positive_proba(model, X_test) / args.n_splits
        row = {
            "task": task.name,
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "positive_rate_valid": float(y_pair[va_rel].mean()),
            "accuracy_at_050": float(((va_p >= 0.5).astype("int8") == y_pair[va_rel]).mean()),
            "log_loss": float(log_loss(y_pair[va_rel], np.column_stack([1.0 - va_p, va_p]), labels=[0, 1])),
            "fit_predict_seconds": time.time() - started,
            "best_iteration": getattr(model, "n_iter_", None),
        }
        try:
            row["roc_auc"] = float(roc_auc_score(y_pair[va_rel], va_p))
        except ValueError:
            row["roc_auc"] = np.nan
        rows.append(row)
        print(f"[{task.name}] fold {fold}/{args.n_splits}: auc={row['roc_auc']:.6f} logloss={row['log_loss']:.6f}")

    if np.isnan(pair_oof).any():
        raise RuntimeError(f"{task.name} left binary OOF rows without predictions.")
    train_p = train_avg
    train_p[pair_indices] = pair_oof
    overall = {
        "task": task.name,
        "fold": "overall",
        "n_train": int(len(pair_indices)),
        "n_valid": int(len(pair_indices)),
        "positive_rate_valid": float(y_pair.mean()),
        "accuracy_at_050": float(((train_p[pair_indices] >= 0.5).astype("int8") == y_pair).mean()),
        "log_loss": float(log_loss(y_pair, np.column_stack([1.0 - train_p[pair_indices], train_p[pair_indices]]), labels=[0, 1])),
        "fit_predict_seconds": float(np.sum([row["fit_predict_seconds"] for row in rows])),
        "best_iteration": np.nan,
    }
    try:
        overall["roc_auc"] = float(roc_auc_score(y_pair, train_p[pair_indices]))
    except ValueError:
        overall["roc_auc"] = np.nan
    rows.append(overall)
    return train_p, test_avg, pd.DataFrame(rows)


def reallocate_pair(base: np.ndarray, p_positive: np.ndarray, positive: str, negative: str) -> np.ndarray:
    out = base.copy()
    pos = CLASS_TO_INT[positive]
    neg = CLASS_TO_INT[negative]
    total = out[:, pos] + out[:, neg]
    out[:, pos] = total * p_positive
    out[:, neg] = total * (1.0 - p_positive)
    return normalize(out)


def pair_margin_scores(p_galaxy_vs_star: np.ndarray, p_galaxy_vs_qso: np.ndarray) -> np.ndarray:
    eps = 1e-6
    gq = np.log(np.clip(p_galaxy_vs_qso, eps, 1 - eps) / np.clip(1 - p_galaxy_vs_qso, eps, 1 - eps))
    gs = np.log(np.clip(p_galaxy_vs_star, eps, 1 - eps) / np.clip(1 - p_galaxy_vs_star, eps, 1 - eps))
    scores = np.column_stack([np.zeros_like(gq), -gq, -gs])
    scores -= scores.max(axis=1, keepdims=True)
    proba = np.exp(scores)
    return normalize(proba)


def blend_proba(base: np.ndarray, pair_scores: np.ndarray, alpha: float) -> np.ndarray:
    base = normalize(base)
    pair_scores = normalize(pair_scores)
    return normalize((base ** (1.0 - alpha)) * (pair_scores ** alpha))


def metrics_from_proba(y_enc: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_enc, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, pred)),
        "log_loss": float(log_loss(y_enc, proba, labels=np.arange(len(CLASSES)))),
    }


def save_submission(path: Path, sample_submission: pd.DataFrame, test_ids: pd.Series, proba: np.ndarray) -> None:
    out = sample_submission.copy()
    out[ID_COL] = test_ids.values
    out[LABEL_COL] = INT_TO_CLASS[proba.argmax(axis=1)]
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    class_weights = parse_class_weights(args.class_weights)
    raw_train = pd.read_csv(args.raw_train)
    raw_train["source_index"] = np.arange(len(raw_train), dtype="int64")
    raw_train = sample_train(raw_train, args.sample_rows, args.sample_seed)
    raw_test = pd.read_csv(args.raw_test)
    sample_submission = pd.read_csv(args.sample_submission)

    train = build_feature_frame(raw_train)
    test = build_feature_frame(raw_test)
    base_train_proba, prefix = load_proba(args.base_train_proba, args.proba_prefix)
    base_test_proba, _ = load_proba(args.base_test_proba, args.proba_prefix, prefix=prefix)
    train = train.merge(base_train_proba, on=ID_COL, how="left", validate="one_to_one")
    test = test.merge(base_test_proba, on=ID_COL, how="left", validate="one_to_one")
    train = train.merge(read_prediction(args.base_train_pred, "base_pred"), on=ID_COL, how="left", validate="one_to_one")
    test = test.merge(read_prediction(args.base_test_pred, "base_pred"), on=ID_COL, how="left", validate="one_to_one")
    pred015_train = maybe_read_prediction(args.pred015_train, "pred015")
    pred015_test = maybe_read_prediction(args.pred015_test, "pred015")
    if pred015_train is not None and pred015_test is not None:
        train = train.merge(pred015_train, on=ID_COL, how="left", validate="one_to_one")
        test = test.merge(pred015_test, on=ID_COL, how="left", validate="one_to_one")

    for frame_name, frame in [("train", train), ("test", test)]:
        required = [f"base_p_{cls}" for cls in CLASSES] + ["base_pred"]
        missing = {col: int(frame[col].isna().sum()) for col in required if col in frame.columns and frame[col].isna().any()}
        if missing:
            raise ValueError(f"{frame_name} has missing aligned values: {missing}")

    train = add_meta_features(train)
    test = add_meta_features(test)
    y_enc = encode_labels(train[LABEL_COL])
    X, X_test, feature_cols = make_model_matrices(train, test)
    base_train = normalize(train[[f"base_p_{cls}" for cls in CLASSES]].to_numpy(dtype="float64"))
    base_test = normalize(test[[f"base_p_{cls}" for cls in CLASSES]].to_numpy(dtype="float64"))

    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")
    if args.dry_run:
        print(f"Train shape: {train.shape}; test shape: {test.shape}; feature count: {len(feature_cols)}")
        print(train[[LABEL_COL, "base_pred", "hard_slice_main"]].value_counts().head(20).to_string())
        return

    pair_train = {}
    pair_test = {}
    fold_frames = []
    for task in PAIR_TASKS:
        p_train, p_test, scores = train_pair_task(task, X, X_test, train, y_enc, class_weights, args)
        pair_train[task.name] = p_train
        pair_test[task.name] = p_test
        fold_frames.append(scores)
    pair_scores = pd.concat(fold_frames, ignore_index=True)
    pair_scores.to_csv(run_dir / "pair_binary_scores.csv", index=False)

    model_probas: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    sg_name = "pair_sg_hard_realloc_s3301"
    gq_name = "pair_gq_hard_realloc_s3302"
    model_probas[sg_name] = (
        reallocate_pair(base_train, pair_train["galaxy_vs_star"], "GALAXY", "STAR"),
        reallocate_pair(base_test, pair_test["galaxy_vs_star"], "GALAXY", "STAR"),
        "galaxy_vs_star_hard_reallocate",
    )
    model_probas[gq_name] = (
        reallocate_pair(base_train, pair_train["galaxy_vs_qso"], "GALAXY", "QSO"),
        reallocate_pair(base_test, pair_test["galaxy_vs_qso"], "GALAXY", "QSO"),
        "galaxy_vs_qso_hard_reallocate",
    )
    pair_score_train = pair_margin_scores(pair_train["galaxy_vs_star"], pair_train["galaxy_vs_qso"])
    pair_score_test = pair_margin_scores(pair_test["galaxy_vs_star"], pair_test["galaxy_vs_qso"])
    for alpha in parse_alphas(args.blend_alphas):
        suffix = str(alpha).replace(".", "")
        name = f"pair_joint_hard_blend_a{suffix}_s3303"
        model_probas[name] = (
            blend_proba(base_train, pair_score_train, alpha),
            blend_proba(base_test, pair_score_test, alpha),
            f"pairwise_margin_blend_alpha_{alpha}",
        )

    train_payload = {
        ID_COL: train[ID_COL].values,
        "source_index": train["source_index"].values,
        LABEL_COL: train[LABEL_COL].values,
    }
    test_payload = {ID_COL: test[ID_COL].values}
    summary_rows = []
    single_dir = run_dir / "single_model_submissions"
    if args.save_single_submissions:
        single_dir.mkdir(parents=True, exist_ok=True)

    for name, (p_train, p_test, feature_set) in model_probas.items():
        metrics = metrics_from_proba(y_enc, p_train)
        pred = p_train.argmax(axis=1)
        hard_mask = train.get("hard_slice_main", pd.Series(0, index=train.index)).to_numpy(dtype="int8").astype(bool)
        hard_metrics = metrics_from_proba(y_enc[hard_mask], p_train[hard_mask]) if hard_mask.any() else {}
        summary_rows.append(
            {
                "experiment": name,
                "model_type": "pairwise_hgb",
                "feature_set": feature_set,
                "seed": args.seed,
                "weighting": "hard_slice_base_error",
                "fold": "overall",
                **metrics,
                "hard_slice_accuracy": hard_metrics.get("accuracy"),
                "hard_slice_balanced_accuracy": hard_metrics.get("balanced_accuracy"),
                "hard_slice_log_loss": hard_metrics.get("log_loss"),
                "pred_count_GALAXY": int(np.sum(pred == CLASS_TO_INT["GALAXY"])),
                "pred_count_QSO": int(np.sum(pred == CLASS_TO_INT["QSO"])),
                "pred_count_STAR": int(np.sum(pred == CLASS_TO_INT["STAR"])),
            }
        )
        prefix_cols = f"l1__{name}"
        for idx, cls in enumerate(CLASSES):
            train_payload[f"{prefix_cols}__{cls}"] = p_train[:, idx]
            test_payload[f"{prefix_cols}__{cls}"] = p_test[:, idx]
        exp_dir = run_dir / "experiments" / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                ID_COL: train[ID_COL].values,
                "source_index": train["source_index"].values,
                LABEL_COL: train[LABEL_COL].values,
                **{f"{prefix_cols}__{cls}": p_train[:, i] for i, cls in enumerate(CLASSES)},
            }
        ).to_csv(exp_dir / "oof_train.csv", index=False)
        pd.DataFrame(
            {
                ID_COL: test[ID_COL].values,
                **{f"{prefix_cols}__{cls}": p_test[:, i] for i, cls in enumerate(CLASSES)},
            }
        ).to_csv(exp_dir / "oof_test.csv", index=False)
        pd.DataFrame({ID_COL: train[ID_COL].values, LABEL_COL: INT_TO_CLASS[p_train.argmax(axis=1)]}).to_csv(
            exp_dir / "oof_train_pred.csv",
            index=False,
        )
        pd.DataFrame({ID_COL: test[ID_COL].values, LABEL_COL: INT_TO_CLASS[p_test.argmax(axis=1)]}).to_csv(
            exp_dir / "test_pred.csv",
            index=False,
        )
        if args.save_single_submissions:
            sub_path = single_dir / f"{name}.csv"
            save_submission(sub_path, sample_submission, test[ID_COL], p_test)
            register_submission(
                sub_path,
                run_dir=run_dir,
                script=Path(__file__).name,
                submission_type="pairwise_local_single_model",
                model_name=name,
                metrics=metrics,
                params={"feature_set": feature_set, "seed": args.seed, "base_proba": args.base_train_proba},
            )

    pd.DataFrame(train_payload).to_csv(run_dir / "oof_train.csv", index=False)
    pd.DataFrame(test_payload).to_csv(run_dir / "oof_test.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=False,
    )
    summary.to_csv(run_dir / "summary_overall.csv", index=False)
    (run_dir / "feature_set_columns.json").write_text(
        json.dumps(feature_cols, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_json(
        run_dir / "manifest.json",
        {
            "args": vars(args),
            "classes": CLASSES,
            "base_proba_prefix": prefix,
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "feature_count": len(feature_cols),
            "class_weights": class_weights,
            "pair_tasks": [task.__dict__ for task in PAIR_TASKS],
            "summary": summary.to_dict(orient="records"),
        },
    )
    print("Pairwise local OOF complete.")
    print(summary.to_string(index=False))
    print(f"OOF train: {run_dir / 'oof_train.csv'}")
    print(f"OOF test: {run_dir / 'oof_test.csv'}")


if __name__ == "__main__":
    main()
