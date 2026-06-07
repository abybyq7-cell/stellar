from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
INT_TO_CLASS = np.asarray(CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOF utility based flow arbitration between two prediction candidates."
    )
    parser.add_argument("--candidate-a-train", type=Path, required=True)
    parser.add_argument("--candidate-a-test", type=Path, required=True)
    parser.add_argument("--candidate-b-train", type=Path, required=True)
    parser.add_argument("--candidate-b-test", type=Path, required=True)
    parser.add_argument("--candidate-a-name", type=str, default="candidate_a")
    parser.add_argument("--candidate-b-name", type=str, default="candidate_b")
    parser.add_argument("--label-source", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--sample-submission", type=Path, default=RAW_DATA_DIR / "sample_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "disagreement_arbitration")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--class-metric-weights",
        type=str,
        default="GALAXY:1.0,QSO:3.2,STAR:4.6",
    )
    parser.add_argument(
        "--base",
        choices=["a", "b"],
        default="a",
        help="Candidate to start from before applying positive-utility flows.",
    )
    parser.add_argument("--min-flow-n", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=0.0)
    return parser.parse_args()


def parse_class_weights(spec: str) -> dict[str, float]:
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = float(value)
    return out


def load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain {LABEL_COL!r}.")
    return df[[ID_COL, LABEL_COL]].copy()


def load_train_prediction(path: Path, label_source: Path) -> pd.DataFrame:
    df = load_prediction(path)
    labels = pd.read_csv(label_source, usecols=lambda col: col in {ID_COL, LABEL_COL})
    if LABEL_COL in labels.columns:
        labels = labels.rename(columns={LABEL_COL: "true_class"})
    out = df.merge(labels, on=ID_COL, how="left")
    if out["true_class"].isna().any():
        raise ValueError(f"Could not align all labels for {path}")
    return out


def class_weight_array(labels: pd.Series, weights: dict[str, float]) -> np.ndarray:
    return labels.astype(str).map(weights).to_numpy(dtype="float64")


def weighted_accuracy(y_true: pd.Series, y_pred: pd.Series, weights: dict[str, float]) -> float:
    sample_weight = class_weight_array(y_true, weights)
    return float(np.average(y_true.astype(str).to_numpy() == y_pred.astype(str).to_numpy(), weights=sample_weight))


def balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = []
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    for cls in CLASSES:
        mask = yt.eq(cls)
        values.append(float(yp[mask].eq(cls).mean()) if mask.any() else 0.0)
    return float(np.mean(values))


def score_frame(df: pd.DataFrame, pred_col: str, weights: dict[str, float]) -> dict:
    correct = df[pred_col].astype(str).eq(df["true_class"].astype(str))
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": balanced_accuracy(df["true_class"], df[pred_col]),
        "weighted_accuracy": weighted_accuracy(df["true_class"], df[pred_col], weights),
        "errors": int((~correct).sum()),
    }


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    class_weights = parse_class_weights(args.class_metric_weights)
    train_a = load_train_prediction(args.candidate_a_train, args.label_source).rename(
        columns={LABEL_COL: "pred_a"}
    )
    train_b = load_train_prediction(args.candidate_b_train, args.label_source).rename(
        columns={LABEL_COL: "pred_b"}
    )
    train = train_a.merge(train_b[[ID_COL, "pred_b"]], on=ID_COL, how="inner")

    test_a = load_prediction(args.candidate_a_test).rename(columns={LABEL_COL: "pred_a"})
    test_b = load_prediction(args.candidate_b_test).rename(columns={LABEL_COL: "pred_b"})
    test = test_a.merge(test_b, on=ID_COL, how="inner")

    base_col = "pred_a" if args.base == "a" else "pred_b"
    other_col = "pred_b" if args.base == "a" else "pred_a"
    train["arb_pred"] = train[base_col]
    test["arb_pred"] = test[base_col]

    rows = []
    for base_pred in CLASSES:
        for other_pred in CLASSES:
            if base_pred == other_pred:
                continue
            mask = train[base_col].eq(base_pred) & train[other_col].eq(other_pred)
            n = int(mask.sum())
            if n == 0:
                continue
            sample_weight = class_weight_array(train.loc[mask, "true_class"], class_weights)
            before = train.loc[mask, base_col].astype(str).eq(train.loc[mask, "true_class"].astype(str))
            after = train.loc[mask, other_col].astype(str).eq(train.loc[mask, "true_class"].astype(str))
            delta = float(np.sum(sample_weight * (after.astype(int).to_numpy() - before.astype(int).to_numpy())))
            rows.append(
                {
                    "base_pred": base_pred,
                    "other_pred": other_pred,
                    "n": n,
                    "delta_weighted_correct": delta,
                    "mean_delta_weighted_correct": delta / n,
                    "selected": n >= args.min_flow_n and delta > args.min_delta,
                }
            )

    flow_df = pd.DataFrame(rows).sort_values(
        ["selected", "delta_weighted_correct", "n"],
        ascending=[False, False, False],
    )
    flow_df.to_csv(run_dir / "flow_utility.csv", index=False)

    selected_flows = flow_df[flow_df["selected"]].copy()
    for _, row in selected_flows.iterrows():
        base_pred = str(row["base_pred"])
        other_pred = str(row["other_pred"])
        train_mask = train[base_col].eq(base_pred) & train[other_col].eq(other_pred)
        test_mask = test[base_col].eq(base_pred) & test[other_col].eq(other_pred)
        train.loc[train_mask, "arb_pred"] = train.loc[train_mask, other_col]
        test.loc[test_mask, "arb_pred"] = test.loc[test_mask, other_col]

    score_a = score_frame(train, "pred_a", class_weights)
    score_b = score_frame(train, "pred_b", class_weights)
    score_arb = score_frame(train, "arb_pred", class_weights)
    pd.DataFrame(
        [
            {"candidate": args.candidate_a_name, **score_a},
            {"candidate": args.candidate_b_name, **score_b},
            {"candidate": "arbitrated", **score_arb},
        ]
    ).to_csv(run_dir / "scores.csv", index=False)

    sample_submission = pd.read_csv(args.sample_submission)
    submission = sample_submission.copy()
    submission[ID_COL] = test[ID_COL].values
    submission[LABEL_COL] = test["arb_pred"].values
    submission_path = run_dir / "arbitrated_submission.csv"
    submission.to_csv(submission_path, index=False)
    register_submission(
        submission_path,
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="disagreement_arbitration",
        model_name=f"{args.candidate_a_name}_vs_{args.candidate_b_name}",
        metrics=score_arb,
        params={
            "candidate_a_train": args.candidate_a_train,
            "candidate_b_train": args.candidate_b_train,
            "candidate_a_test": args.candidate_a_test,
            "candidate_b_test": args.candidate_b_test,
            "base": args.base,
            "class_metric_weights": class_weights,
            "min_flow_n": args.min_flow_n,
            "min_delta": args.min_delta,
        },
        extra={"flow_utility": "flow_utility.csv", "scores": "scores.csv"},
    )

    save_json(
        run_dir / "manifest.json",
        {
            "args": vars(args),
            "class_metric_weights": class_weights,
            "selected_flows": selected_flows.to_dict(orient="records"),
            "scores": {
                args.candidate_a_name: score_a,
                args.candidate_b_name: score_b,
                "arbitrated": score_arb,
            },
            "test_diff_vs_a": int(test["arb_pred"].ne(test["pred_a"]).sum()),
            "test_diff_vs_b": int(test["arb_pred"].ne(test["pred_b"]).sum()),
            "test_class_distribution": test["arb_pred"].value_counts().to_dict(),
        },
    )
    print("Scores:")
    print(pd.read_csv(run_dir / "scores.csv").to_string(index=False))
    print("Selected flows:")
    print(selected_flows.to_string(index=False))


if __name__ == "__main__":
    main()
