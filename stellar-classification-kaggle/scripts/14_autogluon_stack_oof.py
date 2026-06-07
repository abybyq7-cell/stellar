from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


CLASSES = ["GALAXY", "QSO", "STAR"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASSES)}
DEFAULT_CLASS_METRIC_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 5-fold AutoGluon level-2 stacker from saved stack features."
    )
    parser.add_argument(
        "--stack-train-features",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full26_nn_balanced_lr" / "stack_train_features.csv",
    )
    parser.add_argument(
        "--stack-test-features",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full26_nn_balanced_lr" / "stack_test_features.csv",
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        default=OUTPUTS_DIR / "stacking" / "stack_full26_nn_balanced_lr" / "lr_logits_oof_train.csv",
    )
    parser.add_argument("--sample-submission", type=Path, default=RAW_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "stacking")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-per-fold", type=int, default=900)
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--eval-metric", type=str, default="balanced_accuracy")
    parser.add_argument(
        "--class-metric-weights",
        type=str,
        default="GALAXY:1.0,QSO:3.2,STAR:4.6",
        help="Class weights used only for reporting weighted accuracy.",
    )
    parser.add_argument(
        "--hyperparameters",
        choices=["fast_diverse", "gbm_lr", "all_default"],
        default="fast_diverse",
    )
    parser.add_argument(
        "--keep-fold-models",
        action="store_true",
        help="Keep AutoGluon fold model directories after probabilities are materialized.",
    )
    return parser.parse_args()


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


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels.astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels.astype(str).map(CLASS_TO_INT).to_numpy(dtype="int8")


def sample_weights_from_y(y_enc: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    class_weights = np.asarray([weights[cls] for cls in CLASSES], dtype="float64")
    return class_weights[y_enc.astype("int64")]


def weighted_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, weights: dict[str, float]) -> float:
    sample_weight = sample_weights_from_y(y_true, weights)
    return float(np.average(y_true == y_pred, weights=sample_weight))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    rows: dict[str, float | int] = {}
    for idx, cls in enumerate(CLASSES):
        support = int(np.sum(y_true == idx))
        pred_count = int(np.sum(y_pred == idx))
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        rows[f"recall_{cls}"] = tp / support if support else 0.0
        rows[f"precision_{cls}"] = tp / pred_count if pred_count else 0.0
        rows[f"pred_count_{cls}"] = pred_count
        rows[f"fn_{cls}"] = fn
        rows[f"fp_{cls}"] = fp
    return rows


def metrics_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_weights: dict[str, float],
) -> dict[str, float | int]:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "weighted_accuracy": weighted_accuracy_score(y_true, pred, class_weights),
        "log_loss": float(log_loss(y_true, proba, labels=np.arange(len(CLASSES)))),
        "errors": int(np.sum(y_true != pred)),
        **per_class_metrics(y_true, pred),
    }


def align_autogluon_proba(proba: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        columns = list(proba.columns)
        if all(cls in columns for cls in CLASSES):
            values = proba[CLASSES].to_numpy(dtype="float64")
        else:
            lookup = {str(col): col for col in columns}
            if not all(cls in lookup for cls in CLASSES):
                raise ValueError(f"Could not align AutoGluon probability columns: {columns}")
            values = proba[[lookup[cls] for cls in CLASSES]].to_numpy(dtype="float64")
    else:
        values = np.asarray(proba, dtype="float64")
    if values.shape[1] != len(CLASSES):
        raise ValueError(f"Expected {len(CLASSES)} probability columns, got shape {values.shape}")
    values = np.clip(values, 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def hyperparameters_for(name: str) -> dict:
    if name == "gbm_lr":
        return {
            "GBM": [
                {},
                {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            ],
            "LR": [{}],
        }
    if name == "all_default":
        return {
            "GBM": [
                {},
                {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            ],
            "CAT": [{}],
            "XGB": [{}],
            "LR": [{}],
        }
    return {
        "GBM": [
            {"num_boost_round": 600, "ag_args": {"name_suffix": "Lite"}},
            {"extra_trees": True, "num_boost_round": 600, "ag_args": {"name_suffix": "XTLite"}},
        ],
        "LR": [{}],
        "RF": [
            {
                "n_estimators": 300,
                "max_depth": 18,
                "ag_args": {"name_suffix": "Lite"},
            }
        ],
    }


def save_proba_frame(
    path: Path,
    ids: pd.Series,
    proba: np.ndarray,
    labels: pd.Series | None = None,
    prefix: str = "ag_stack",
) -> None:
    data = {ID_COL: ids.values}
    if labels is not None:
        data[LABEL_COL] = labels.values
    for idx, cls in enumerate(CLASSES):
        data[f"{prefix}__{cls}"] = proba[:, idx]
    pd.DataFrame(data).to_csv(path, index=False)


def save_submission(path: Path, sample_submission: pd.DataFrame, ids: pd.Series, proba: np.ndarray) -> None:
    out = sample_submission.copy()
    out[ID_COL] = ids.values
    out[LABEL_COL] = np.asarray(CLASSES)[proba.argmax(axis=1)]
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    from autogluon.tabular import TabularPredictor

    X = pd.read_csv(args.stack_train_features)
    X_test = pd.read_csv(args.stack_test_features)
    labels = pd.read_csv(args.label_source, usecols=[ID_COL, LABEL_COL])
    sample_submission = pd.read_csv(args.sample_submission)
    if len(X) != len(labels):
        raise ValueError(f"Feature/label row mismatch: {len(X)} vs {len(labels)}")

    y = labels[LABEL_COL].astype(str)
    y_enc = encode_labels(y)
    class_weights = parse_class_metric_weights(args.class_metric_weights)
    oof = np.zeros((len(X), len(CLASSES)), dtype="float64")
    test_sum = np.zeros((len(X_test), len(CLASSES)), dtype="float64")
    fold_rows = []
    leaderboard_paths = []
    hyperparameters = hyperparameters_for(args.hyperparameters)

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(splitter.split(X, y_enc), start=1):
        fold_dir = run_dir / f"ag_fold{fold}"
        print(f"Fold {fold}: train={len(tr)} valid={len(va)} dir={fold_dir}")
        train_fold = X.iloc[tr].reset_index(drop=True).copy()
        train_fold[LABEL_COL] = y.iloc[tr].to_numpy()
        valid_fold = X.iloc[va].reset_index(drop=True)

        predictor = TabularPredictor(
            label=LABEL_COL,
            problem_type="multiclass",
            eval_metric=args.eval_metric,
            path=str(fold_dir),
            verbosity=2,
        )
        predictor.fit(
            train_data=train_fold,
            presets=args.presets,
            time_limit=args.time_limit_per_fold,
            num_gpus=0,
            hyperparameters=hyperparameters,
        )

        valid_proba = align_autogluon_proba(predictor.predict_proba(valid_fold))
        test_proba = align_autogluon_proba(predictor.predict_proba(X_test))
        oof[va] = valid_proba
        test_sum += test_proba

        row = {
            "stacker": "autogluon_oof",
            "fold": fold,
            **metrics_from_proba(y_enc[va], valid_proba, class_weights),
        }
        fold_rows.append(row)
        leaderboard = predictor.leaderboard(train_fold, silent=True)
        leaderboard_path = run_dir / f"autogluon_leaderboard_fold{fold}.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        leaderboard_paths.append(leaderboard_path.name)
        print(json.dumps(row, ensure_ascii=False, default=str))

        if not args.keep_fold_models:
            del predictor
            shutil.rmtree(fold_dir, ignore_errors=True)

    test_proba = test_sum / args.n_splits
    overall = {
        "stacker": "autogluon_oof",
        "fold": "overall",
        **metrics_from_proba(y_enc, oof, class_weights),
    }
    score_df = pd.DataFrame(fold_rows + [overall])
    score_df.to_csv(run_dir / "autogluon_oof_scores.csv", index=False)
    save_proba_frame(
        run_dir / "autogluon_oof_train.csv",
        labels[ID_COL],
        oof,
        labels=y,
        prefix="ag_stack",
    )
    save_proba_frame(
        run_dir / "autogluon_oof_test.csv",
        sample_submission[ID_COL],
        test_proba,
        labels=None,
        prefix="ag_stack",
    )
    submission_path = run_dir / "autogluon_submission.csv"
    save_submission(submission_path, sample_submission, sample_submission[ID_COL], test_proba)

    register_submission(
        submission_path,
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="stacking_autogluon_oof",
        model_name=f"autogluon_oof_{args.hyperparameters}",
        metrics={
            key: overall[key]
            for key in [
                "accuracy",
                "balanced_accuracy",
                "weighted_accuracy",
                "log_loss",
                "errors",
            ]
        },
        params={
            "stack_train_features": args.stack_train_features,
            "stack_test_features": args.stack_test_features,
            "label_source": args.label_source,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "time_limit_per_fold": args.time_limit_per_fold,
            "presets": args.presets,
            "eval_metric": args.eval_metric,
            "hyperparameters": args.hyperparameters,
            "class_metric_weights": class_weights,
        },
        extra={
            "scores_path": "autogluon_oof_scores.csv",
            "train_proba_path": "autogluon_oof_train.csv",
            "test_proba_path": "autogluon_oof_test.csv",
            "leaderboard_paths": leaderboard_paths,
        },
    )

    save_json(
        run_dir / "manifest.json",
        {
            "args": vars(args),
            "classes": CLASSES,
            "class_metric_weights": class_weights,
            "hyperparameters": hyperparameters,
            "overall": overall,
            "leaderboard_paths": leaderboard_paths,
        },
    )
    print("Overall:")
    print(json.dumps(overall, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
