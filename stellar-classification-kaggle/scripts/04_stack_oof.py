from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
DEFAULT_CLASS_METRIC_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}


@dataclass(frozen=True)
class ModelBlock:
    name: str
    cols: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge layer-1 OOF runs, select diverse models, and train stackers."
    )
    parser.add_argument(
        "--oof-runs",
        nargs="+",
        type=Path,
        default=[
            OUTPUTS_DIR / "layer1_oof" / "layer1_50k_balanced",
            OUTPUTS_DIR / "layer1_oof" / "layer1_50k_targeted_new_features",
        ],
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "stacking")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--corr-threshold", type=float, default=0.9995)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-models", type=int, default=0)
    parser.add_argument("--logit-eps", type=float, default=1e-5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-c-values", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument(
        "--class-metric-weights",
        type=str,
        default="GALAXY:1.0,QSO:3.2,STAR:4.6",
        help="Class weights used for weighted accuracy and optional LR sample weights.",
    )
    parser.add_argument(
        "--lr-class-weighted",
        action="store_true",
        help="Train LR stacker with sample weights derived from --class-metric-weights.",
    )
    parser.add_argument(
        "--model-weight",
        action="append",
        default=[],
        help="Manual feature weight as model:weight. Use 0 to remove a model from stack features.",
    )
    parser.add_argument("--autogluon-time-limit", type=int, default=900)
    parser.add_argument("--autogluon-presets", type=str, default="medium_quality")
    parser.add_argument("--skip-autogluon", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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


def parse_model_feature_weights(specs: list[str]) -> dict[str, float]:
    weights = {}
    for spec in specs or []:
        name, value = spec.split(":", 1)
        weights[name.strip()] = float(value)
    return weights


def sample_weights_from_labels(y_labels: pd.Series, class_weights: dict[str, float]) -> np.ndarray:
    return y_labels.astype(str).map(class_weights).to_numpy(dtype="float64")


def weighted_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
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


def read_oof_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = run_dir / "oof_train.csv"
    test_path = run_dir / "oof_test.csv"
    score_path = run_dir / "summary_overall.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing OOF files in {run_dir}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    scores = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    return train, test, scores


def l1_prefix(col: str) -> str:
    parts = col.split("__")
    return "__".join(parts[:-1])


def model_name_from_prefix(prefix: str) -> str:
    return prefix.removeprefix("l1__")


def get_model_blocks(df: pd.DataFrame) -> list[ModelBlock]:
    prefixes = sorted({l1_prefix(c) for c in df.columns if c.startswith("l1__")})
    blocks = []
    for prefix in prefixes:
        cols = [f"{prefix}__{cls}" for cls in CLASSES]
        if all(col in df.columns for col in cols):
            blocks.append(ModelBlock(model_name_from_prefix(prefix), cols))
    return blocks


def merge_runs(oof_runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_train: pd.DataFrame | None = None
    merged_test: pd.DataFrame | None = None
    all_scores = []

    for run_dir in oof_runs:
        train, test, scores = read_oof_run(run_dir)
        blocks = get_model_blocks(train)
        keep_train = [ID_COL, "source_index", LABEL_COL]
        keep_test = [ID_COL]
        for block in blocks:
            keep_train.extend(block.cols)
            keep_test.extend(block.cols)

        train = train[keep_train].copy()
        test = test[keep_test].copy()
        if merged_train is None:
            merged_train = train
            merged_test = test
        else:
            merged_train = merged_train.merge(
                train.drop(columns=[LABEL_COL]),
                on=[ID_COL, "source_index"],
                how="inner",
            )
            label_map = train[[ID_COL, LABEL_COL]]
            merged_train = merged_train.merge(label_map, on=ID_COL, how="left", suffixes=("", "_new"))
            if not merged_train[LABEL_COL].eq(merged_train[f"{LABEL_COL}_new"]).all():
                raise ValueError(f"Label mismatch while merging {run_dir}")
            merged_train = merged_train.drop(columns=[f"{LABEL_COL}_new"])
            merged_test = merged_test.merge(test, on=ID_COL, how="inner")

        if not scores.empty:
            scores = scores.copy()
            scores["run_dir"] = str(run_dir)
            all_scores.append(scores)

    if merged_train is None or merged_test is None:
        raise ValueError("No OOF runs were loaded.")
    score_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    return merged_train, merged_test, score_df


def block_probability_matrix(df: pd.DataFrame, block: ModelBlock) -> np.ndarray:
    proba = df[block.cols].to_numpy(dtype="float64")
    proba = np.clip(proba, 1e-12, None)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


def score_blocks(
    train: pd.DataFrame,
    y_enc: np.ndarray,
    score_df: pd.DataFrame,
    class_sample_weight: np.ndarray,
) -> pd.DataFrame:
    rows = []
    blocks = get_model_blocks(train)
    score_lookup = {}
    if not score_df.empty and "experiment" in score_df.columns:
        overall = score_df[score_df["fold"].astype(str).eq("overall")] if "fold" in score_df.columns else score_df
        for _, row in overall.iterrows():
            score_lookup[str(row["experiment"])] = row.to_dict()

    for block in blocks:
        proba = block_probability_matrix(train, block)
        pred = proba.argmax(axis=1)
        row = {
            "model": block.name,
            "accuracy": accuracy_score(y_enc, pred),
            "balanced_accuracy": balanced_accuracy_score(y_enc, pred),
            "weighted_accuracy": weighted_accuracy_score(y_enc, pred, class_sample_weight),
            "log_loss": log_loss(y_enc, proba, labels=np.arange(len(CLASSES))),
            **per_class_metrics(y_enc, pred),
        }
        if block.name in score_lookup:
            for key in ["feature_set", "model_type", "seed", "run_dir"]:
                if key in score_lookup[block.name]:
                    row[key] = score_lookup[block.name][key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["weighted_accuracy", "balanced_accuracy", "accuracy"],
        ascending=False,
    )


def prediction_agreement_matrix(train: pd.DataFrame, blocks: list[ModelBlock]) -> pd.DataFrame:
    preds = {}
    for block in blocks:
        preds[block.name] = block_probability_matrix(train, block).argmax(axis=1)
    names = [b.name for b in blocks]
    matrix = pd.DataFrame(index=names, columns=names, dtype="float64")
    for a in names:
        for b in names:
            matrix.loc[a, b] = float(np.mean(preds[a] == preds[b]))
    return matrix


def probability_correlation_matrix(train: pd.DataFrame, blocks: list[ModelBlock]) -> pd.DataFrame:
    vectors = {}
    for block in blocks:
        vectors[block.name] = block_probability_matrix(train, block).reshape(-1)
    mat = pd.DataFrame(vectors).corr()
    return mat


def select_models(
    block_scores: pd.DataFrame,
    corr: pd.DataFrame,
    corr_threshold: float,
    min_accuracy: float,
    max_models: int,
    model_feature_weights: dict[str, float],
) -> tuple[list[str], pd.DataFrame]:
    selected: list[str] = []
    rows = []
    for _, row in block_scores.iterrows():
        name = row["model"]
        manual_weight = float(model_feature_weights.get(name, 1.0))
        if manual_weight <= 0:
            rows.append(
                {
                    **row.to_dict(),
                    "selected": False,
                    "feature_weight": manual_weight,
                    "drop_reason": "manual_weight<=0",
                }
            )
            continue
        if row["accuracy"] < min_accuracy:
            rows.append(
                {
                    **row.to_dict(),
                    "selected": False,
                    "feature_weight": manual_weight,
                    "drop_reason": "below_min_accuracy",
                }
            )
            continue
        too_close_to = None
        for kept in selected:
            if float(corr.loc[name, kept]) >= corr_threshold:
                too_close_to = kept
                break
        if too_close_to is not None:
            rows.append(
                {
                    **row.to_dict(),
                    "selected": False,
                    "feature_weight": manual_weight,
                    "drop_reason": f"corr>={corr_threshold} with {too_close_to}",
                }
            )
            continue
        if max_models > 0 and len(selected) >= max_models:
            rows.append(
                {
                    **row.to_dict(),
                    "selected": False,
                    "feature_weight": manual_weight,
                    "drop_reason": "max_models",
                }
            )
            continue
        selected.append(name)
        rows.append(
            {
                **row.to_dict(),
                "selected": True,
                "feature_weight": manual_weight,
                "drop_reason": "",
            }
        )
    return selected, pd.DataFrame(rows)


def transform_logits(
    df: pd.DataFrame,
    blocks: list[ModelBlock],
    eps: float,
    model_feature_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    features = {}
    model_feature_weights = model_feature_weights or {}
    for block in blocks:
        feature_weight = float(model_feature_weights.get(block.name, 1.0))
        proba = np.clip(block_probability_matrix(df, block), eps, 1.0 - eps)
        proba = proba / proba.sum(axis=1, keepdims=True)
        for i, cls in enumerate(CLASSES):
            features[f"logit__{block.name}__{cls}"] = (
                np.log(proba[:, i] / (1.0 - proba[:, i])) * feature_weight
            )
        features[f"entropy__{block.name}"] = (
            -np.sum(proba * np.log(proba), axis=1) * feature_weight
        )
        features[f"margin__{block.name}"] = (
            np.partition(proba, -1, axis=1)[:, -1]
            - np.partition(proba, -2, axis=1)[:, -2]
        ) * feature_weight
    return pd.DataFrame(features)


def save_submission(path: Path, sample_submission: pd.DataFrame, test_ids: pd.Series, proba: np.ndarray, classes: np.ndarray) -> None:
    sub = sample_submission.copy()
    sub[ID_COL] = test_ids.values
    sub[LABEL_COL] = classes[proba.argmax(axis=1)]
    sub.to_csv(path, index=False)


def run_lr_logits(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    X_test: pd.DataFrame,
    classes: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
    sample_submission: pd.DataFrame,
    train_ids: pd.Series,
    train_labels: pd.Series,
    test_ids: pd.Series,
    class_sample_weight: np.ndarray,
) -> pd.DataFrame:
    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []
    best_payload = None

    for c in args.lr_c_values:
        oof = np.zeros((len(X), len(classes)), dtype="float32")
        test_sum = np.zeros((len(X_test), len(classes)), dtype="float32")
        fold_rows = []
        for fold, (tr, va) in enumerate(folds.split(X, y_enc), start=1):
            model = LogisticRegression(
                C=c,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                class_weight=None,
                random_state=args.seed,
            )
            fit_weight = class_sample_weight[tr] if args.lr_class_weighted else None
            model.fit(X.iloc[tr], y_enc[tr], sample_weight=fit_weight)
            va_proba = model.predict_proba(X.iloc[va])
            test_proba = model.predict_proba(X_test)
            oof[va] = va_proba.astype("float32")
            test_sum += test_proba.astype("float32") / args.n_splits
            fold_rows.append(
                {
                    "stacker": "lr_logits",
                    "C": c,
                    "fold": fold,
                    "accuracy": accuracy_score(y_enc[va], va_proba.argmax(axis=1)),
                    "balanced_accuracy": balanced_accuracy_score(y_enc[va], va_proba.argmax(axis=1)),
                    "weighted_accuracy": weighted_accuracy_score(
                        y_enc[va],
                        va_proba.argmax(axis=1),
                        class_sample_weight[va],
                    ),
                    "log_loss": log_loss(y_enc[va], va_proba, labels=np.arange(len(classes))),
                }
            )

        pred = oof.argmax(axis=1)
        row = {
            "stacker": "lr_logits",
            "C": c,
            "fold": "overall",
            "accuracy": accuracy_score(y_enc, pred),
            "balanced_accuracy": balanced_accuracy_score(y_enc, pred),
            "weighted_accuracy": weighted_accuracy_score(y_enc, pred, class_sample_weight),
            "log_loss": log_loss(y_enc, oof, labels=np.arange(len(classes))),
        }
        rows.extend(fold_rows)
        rows.append(row)
        score_key = "weighted_accuracy" if args.lr_class_weighted else "log_loss"
        if best_payload is None:
            best_payload = {"row": row, "oof": oof, "test": test_sum}
        elif args.lr_class_weighted and row[score_key] > best_payload["row"][score_key]:
            best_payload = {"row": row, "oof": oof, "test": test_sum}
        elif not args.lr_class_weighted and row[score_key] < best_payload["row"][score_key]:
            best_payload = {"row": row, "oof": oof, "test": test_sum}

    score_df = pd.DataFrame(rows)
    score_df.to_csv(run_dir / "lr_logits_scores.csv", index=False)
    if best_payload is not None:
        pd.DataFrame(
            {
                ID_COL: test_ids.values,
                **{f"lr_logits__{cls}": best_payload["test"][:, i] for i, cls in enumerate(classes)},
            }
        ).to_csv(run_dir / "lr_logits_oof_test.csv", index=False)
        pd.DataFrame(
            {
                ID_COL: train_ids.values,
                LABEL_COL: train_labels.values,
                **{f"lr_logits__{cls}": best_payload["oof"][:, i] for i, cls in enumerate(classes)},
            }
        ).to_csv(run_dir / "lr_logits_oof_train.csv", index=False)
        save_submission(
            run_dir / "lr_logits_submission.csv",
            sample_submission,
            test_ids,
            best_payload["test"],
            classes,
        )
        register_submission(
            run_dir / "lr_logits_submission.csv",
            run_dir=run_dir,
            script=Path(__file__).name,
            submission_type="stacking_lr_logits",
            model_name=f"lr_logits_C{best_payload['row']['C']}",
            metrics={
                key: best_payload["row"][key]
                for key in ["accuracy", "balanced_accuracy", "weighted_accuracy", "log_loss"]
            },
            params={
                "C": best_payload["row"]["C"],
                "n_splits": args.n_splits,
                "seed": args.seed,
                "logit_eps": args.logit_eps,
                "oof_runs": args.oof_runs,
                "corr_threshold": args.corr_threshold,
                "min_accuracy": args.min_accuracy,
                "max_models": args.max_models,
                "lr_class_weighted": args.lr_class_weighted,
                "class_metric_weights": args.class_metric_weights,
                "model_feature_weights": args.model_weight,
            },
            extra={"scores_path": "lr_logits_scores.csv"},
        )
    return score_df


def run_autogluon_stacker(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
) -> dict:
    from autogluon.tabular import TabularPredictor

    train_ag = X.copy()
    train_ag[LABEL_COL] = y.values
    model_dir = run_dir / "autogluon_stacker"
    predictor = TabularPredictor(
        label=LABEL_COL,
        problem_type="multiclass",
        eval_metric="log_loss",
        path=str(model_dir),
        verbosity=2,
    )
    predictor.fit(
        train_data=train_ag,
        presets=args.autogluon_presets,
        time_limit=args.autogluon_time_limit,
        num_gpus=0,
        hyperparameters={
            "GBM": [
                {},
                {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            ],
            "CAT": [{}],
            "XGB": [{}],
            "LR": [{}],
        },
    )
    leaderboard = predictor.leaderboard(train_ag, silent=True)
    leaderboard.to_csv(run_dir / "autogluon_leaderboard_train.csv", index=False)
    metrics = predictor.evaluate(train_ag, silent=True)
    save_json(run_dir / "autogluon_train_metrics.json", metrics)
    pred = predictor.predict(X_test)
    sub = sample_submission.copy()
    sub[ID_COL] = test_ids.values
    sub[LABEL_COL] = pred.values
    sub.to_csv(run_dir / "autogluon_submission.csv", index=False)
    register_submission(
        run_dir / "autogluon_submission.csv",
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="stacking_autogluon",
        model_name="autogluon_stacker",
        metrics=metrics,
        params={
            "autogluon_presets": args.autogluon_presets,
            "autogluon_time_limit": args.autogluon_time_limit,
        },
        extra={
            "leaderboard_path": "autogluon_leaderboard_train.csv",
            "metrics_path": "autogluon_train_metrics.json",
        },
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    sample_submission = pd.read_csv(args.data_dir / "sample_submission.csv")
    train, test, score_df = merge_runs(args.oof_runs)
    train = train.sort_values("source_index").reset_index(drop=True)
    test = test.sort_values(ID_COL).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(train[LABEL_COL])
    classes = label_encoder.classes_
    if list(classes) != CLASSES:
        print(f"Class order from LabelEncoder: {classes.tolist()}")
    class_metric_weights = parse_class_metric_weights(args.class_metric_weights)
    model_feature_weights = parse_model_feature_weights(args.model_weight)
    class_sample_weight = sample_weights_from_labels(train[LABEL_COL], class_metric_weights)

    blocks = get_model_blocks(train)
    block_scores = score_blocks(train, y_enc, score_df, class_sample_weight)
    corr = probability_correlation_matrix(train, blocks)
    agreement = prediction_agreement_matrix(train, blocks)

    block_scores.to_csv(run_dir / "base_model_scores.csv", index=False)
    corr.to_csv(run_dir / "base_model_probability_corr.csv")
    agreement.to_csv(run_dir / "base_model_prediction_agreement.csv")

    selected_names, selection = select_models(
        block_scores,
        corr,
        args.corr_threshold,
        args.min_accuracy,
        args.max_models,
        model_feature_weights,
    )
    selection.to_csv(run_dir / "model_selection.csv", index=False)
    selected_blocks = [block for block in blocks if block.name in selected_names]

    metadata = {
        "args": vars(args),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "base_model_count": len(blocks),
        "selected_model_count": len(selected_blocks),
        "selected_models": selected_names,
        "class_metric_weights": class_metric_weights,
        "model_feature_weights": model_feature_weights,
        "lr_class_weighted": args.lr_class_weighted,
    }
    save_json(run_dir / "manifest.json", metadata)

    print("Base model scores:")
    print(
        block_scores[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "weighted_accuracy",
                "recall_QSO",
                "recall_STAR",
                "log_loss",
            ]
        ]
    )
    print("Selected models:")
    print(selection[["model", "accuracy", "weighted_accuracy", "selected", "feature_weight", "drop_reason"]])

    if args.dry_run:
        print("Dry run complete.")
        return

    X = transform_logits(train, selected_blocks, args.logit_eps, model_feature_weights)
    X_test = transform_logits(test, selected_blocks, args.logit_eps, model_feature_weights)
    X.to_csv(run_dir / "stack_train_features.csv", index=False)
    X_test.to_csv(run_dir / "stack_test_features.csv", index=False)

    lr_scores = run_lr_logits(
        X,
        y_enc,
        X_test,
        classes,
        args,
        run_dir,
        sample_submission,
        train[ID_COL],
        train[LABEL_COL],
        test[ID_COL],
        class_sample_weight,
    )
    overall = lr_scores[lr_scores["fold"].astype(str).eq("overall")].copy()
    overall.to_csv(run_dir / "stacker_scores.csv", index=False)
    print("LR logits scores:")
    if args.lr_class_weighted:
        print(overall.sort_values(["weighted_accuracy", "balanced_accuracy"], ascending=False))
    else:
        print(overall.sort_values("log_loss"))

    if not args.skip_autogluon:
        metrics = run_autogluon_stacker(
            X,
            train[LABEL_COL],
            X_test,
            args,
            run_dir,
            sample_submission,
            test[ID_COL],
        )
        print("AutoGluon train metrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
