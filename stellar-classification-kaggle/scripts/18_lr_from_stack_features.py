from __future__ import annotations

import argparse
import json
import sys
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
from stellar.io import save_json
from stellar.paths import RAW_DATA_DIR
from stellar.submissions import register_submission


CLASSES = ["GALAXY", "QSO", "STAR"]
DEFAULT_CLASS_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover LR stacker from saved stack feature CSVs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--sample-submission", type=Path, default=RAW_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.0008, 0.001, 0.0012])
    parser.add_argument("--class-weights", type=str, default="GALAXY:1.0,QSO:3.2,STAR:4.6")
    parser.add_argument("--max-iter", type=int, default=800)
    return parser.parse_args()


def parse_class_weights(spec: str) -> dict[str, float]:
    weights = DEFAULT_CLASS_WEIGHTS.copy()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        weights[key.strip()] = float(value)
    return weights


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    return float(np.average(y_true == y_pred, weights=sample_weight))


def class_sample_weight(labels: pd.Series, class_weights: dict[str, float]) -> np.ndarray:
    return labels.astype(str).map(class_weights).to_numpy(dtype="float64")


def save_submission(path: Path, sample_submission: pd.DataFrame, proba: np.ndarray) -> None:
    out = sample_submission.copy()
    out[LABEL_COL] = np.asarray(CLASSES)[proba.argmax(axis=1)]
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    train_features_path = run_dir / "stack_train_features.csv"
    test_features_path = run_dir / "stack_test_features.csv"
    if not train_features_path.exists() or not test_features_path.exists():
        raise FileNotFoundError(f"Missing stack feature CSVs in {run_dir}")

    X = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    raw_train = pd.read_csv(args.raw_train, usecols=[ID_COL, LABEL_COL])
    sample_submission = pd.read_csv(args.sample_submission)
    if len(X) != len(raw_train):
        raise ValueError(f"Feature/label row mismatch: {len(X)} vs {len(raw_train)}")
    if len(X_test) != len(sample_submission):
        raise ValueError(f"Feature/test row mismatch: {len(X_test)} vs {len(sample_submission)}")

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(raw_train[LABEL_COL])
    classes = label_encoder.classes_.tolist()
    if classes != CLASSES:
        raise ValueError(f"Unexpected class order: {classes}")
    class_weights = parse_class_weights(args.class_weights)
    sample_weight = class_sample_weight(raw_train[LABEL_COL], class_weights)

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows = []
    best = None
    for c in args.c_values:
        oof = np.zeros((len(X), len(CLASSES)), dtype="float64")
        test_sum = np.zeros((len(X_test), len(CLASSES)), dtype="float64")
        for fold, (tr, va) in enumerate(splitter.split(X, y_enc), start=1):
            model = LogisticRegression(
                C=float(c),
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=args.max_iter,
                n_jobs=1,
                random_state=args.seed + fold,
            )
            model.fit(X.iloc[tr], y_enc[tr], sample_weight=sample_weight[tr])
            va_proba = model.predict_proba(X.iloc[va])
            test_proba = model.predict_proba(X_test)
            oof[va] = va_proba
            test_sum += test_proba / args.n_splits
            va_pred = va_proba.argmax(axis=1)
            rows.append(
                {
                    "stacker": "lr_logits_recovered",
                    "C": c,
                    "fold": fold,
                    "accuracy": float(accuracy_score(y_enc[va], va_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_enc[va], va_pred)),
                    "weighted_accuracy": weighted_accuracy(y_enc[va], va_pred, sample_weight[va]),
                    "log_loss": float(log_loss(y_enc[va], va_proba, labels=np.arange(len(CLASSES)))),
                }
            )
        pred = oof.argmax(axis=1)
        overall = {
            "stacker": "lr_logits_recovered",
            "C": c,
            "fold": "overall",
            "accuracy": float(accuracy_score(y_enc, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_enc, pred)),
            "weighted_accuracy": weighted_accuracy(y_enc, pred, sample_weight),
            "log_loss": float(log_loss(y_enc, oof, labels=np.arange(len(CLASSES)))),
        }
        rows.append(overall)
        if best is None or overall["weighted_accuracy"] > best["row"]["weighted_accuracy"]:
            best = {"row": overall, "oof": oof, "test": test_sum}

    if best is None:
        raise RuntimeError("No LR models were trained.")
    score_df = pd.DataFrame(rows)
    score_df.to_csv(run_dir / "lr_logits_recovered_scores.csv", index=False)
    overall_df = score_df[score_df["fold"].astype(str).eq("overall")].copy()
    overall_df.to_csv(run_dir / "stacker_scores.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: raw_train[ID_COL].values,
            LABEL_COL: raw_train[LABEL_COL].values,
            **{f"lr_logits__{cls}": best["oof"][:, i] for i, cls in enumerate(CLASSES)},
        }
    ).to_csv(run_dir / "lr_logits_oof_train.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: sample_submission[ID_COL].values,
            **{f"lr_logits__{cls}": best["test"][:, i] for i, cls in enumerate(CLASSES)},
        }
    ).to_csv(run_dir / "lr_logits_oof_test.csv", index=False)
    save_submission(run_dir / "lr_logits_submission.csv", sample_submission, best["test"])
    register_submission(
        run_dir / "lr_logits_submission.csv",
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="stacking_lr_logits_recovered",
        model_name=f"lr_logits_C{best['row']['C']}",
        metrics={key: best["row"][key] for key in ["accuracy", "balanced_accuracy", "weighted_accuracy", "log_loss"]},
        params={
            "run_dir": run_dir,
            "c_values": args.c_values,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "class_weights": class_weights,
            "max_iter": args.max_iter,
        },
        extra={"scores_path": "lr_logits_recovered_scores.csv"},
    )
    save_json(
        run_dir / "lr_recovery_manifest.json",
        {
            "args": vars(args),
            "class_weights": class_weights,
            "best": best["row"],
            "scores_path": "lr_logits_recovered_scores.csv",
        },
    )
    print("Recovered LR scores:")
    print(overall_df.sort_values(["weighted_accuracy", "balanced_accuracy"], ascending=False).to_string(index=False))
    print(f"Best C: {best['row']['C']}")


if __name__ == "__main__":
    main()
