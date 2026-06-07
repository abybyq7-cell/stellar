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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a two-stage post-processing rule: first QSO vs OTHER, then "
            "STAR vs GALAXY, with a confidence threshold for replacing argmax."
        )
    )
    parser.add_argument(
        "--train-proba",
        type=Path,
        default=OUTPUTS_DIR
        / "stacking"
        / "stack_full_selected_lr"
        / "lr_logits_oof_train.csv",
    )
    parser.add_argument(
        "--test-proba",
        type=Path,
        default=OUTPUTS_DIR
        / "stacking"
        / "stack_full_selected_lr"
        / "lr_logits_oof_test.csv",
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        default=OUTPUTS_DIR / "layer1_oof" / "layer1_full_selected_gpu" / "oof_train.csv",
        help="CSV containing id/source_index and the true class for the OOF rows.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=RAW_DATA_DIR / "sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "two_stage_threshold",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--proba-prefix",
        type=str,
        default="auto",
        help="Probability column prefix, for example lr_logits. Use auto to detect it.",
    )
    parser.add_argument("--qso-thresholds", type=str, default="0.40:0.60:0.02")
    parser.add_argument("--star-thresholds", type=str, default="0.40:0.60:0.02")
    parser.add_argument("--replace-thresholds", type=str, default="0.50:0.95:0.03")
    parser.add_argument(
        "--class-metric-weights",
        type=str,
        default="GALAXY:1.0,QSO:3.2,STAR:4.6",
        help="Class weights used for weighted accuracy ranking.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of best grid rows to keep in top_results.csv.",
    )
    parser.add_argument(
        "--save-submission",
        action="store_true",
        help="Write a test submission using the best balanced-accuracy setting.",
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


def sample_weight_for_y(y_true: np.ndarray, class_weights: dict[str, float]) -> np.ndarray:
    weights = np.asarray([class_weights[cls] for cls in CLASSES], dtype="float64")
    return weights[y_true.astype("int64")]


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_weights: dict[str, float]) -> float:
    sample_weight = sample_weight_for_y(y_true, class_weights)
    return float(np.average(y_true == y_pred, weights=sample_weight))


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


def probability_columns(prefix: str) -> list[str]:
    return [f"{prefix}__{cls}" for cls in CLASSES]


def normalize_proba(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype("float64"), 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def load_train_frame(train_proba_path: Path, label_source_path: Path, prefix_arg: str) -> tuple[pd.DataFrame, str]:
    proba_df = pd.read_csv(train_proba_path)
    prefix = detect_probability_prefix(proba_df, prefix_arg)
    cols = probability_columns(prefix)
    missing = [col for col in cols + [ID_COL] if col not in proba_df.columns]
    if missing:
        raise ValueError(f"Missing columns in {train_proba_path}: {missing}")

    if LABEL_COL in proba_df.columns:
        return proba_df, prefix

    labels = pd.read_csv(label_source_path, usecols=lambda col: col in {ID_COL, "source_index", LABEL_COL})
    if LABEL_COL not in labels.columns:
        raise ValueError(f"{label_source_path} does not contain {LABEL_COL!r}.")

    merge_key = ID_COL if ID_COL in labels.columns else "source_index"
    if merge_key not in proba_df.columns:
        raise ValueError(
            f"Cannot align labels: {merge_key!r} is not present in {train_proba_path}."
        )
    out = proba_df.merge(labels[[merge_key, LABEL_COL]], on=merge_key, how="left")
    if out[LABEL_COL].isna().any():
        missing_count = int(out[LABEL_COL].isna().sum())
        raise ValueError(f"Label alignment left {missing_count} rows without labels.")
    return out, prefix


def encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels.astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels.map(CLASS_TO_INT).to_numpy(dtype="int8")


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> float:
    support = confusion.sum(axis=1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(CLASSES), dtype="float64"),
        where=support > 0,
    )
    return float(recall.mean())


def confusion_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(
        y_true.astype("int64") * len(CLASSES) + y_pred.astype("int64"),
        minlength=len(CLASSES) ** 2,
    ).reshape(len(CLASSES), len(CLASSES))


def metrics_from_confusion(confusion: np.ndarray, class_weights: dict[str, float] | None = None) -> dict:
    correct = int(np.trace(confusion))
    total = int(confusion.sum())
    weights = np.asarray(
        [class_weights.get(cls, 1.0) for cls in CLASSES]
        if class_weights is not None
        else [1.0 for _ in CLASSES],
        dtype="float64",
    )
    weighted_correct = float(np.sum(np.diag(confusion) * weights))
    weighted_total = float(np.sum(confusion.sum(axis=1) * weights))
    star_galaxy_swaps = int(
        confusion[CLASS_TO_INT["STAR"], CLASS_TO_INT["GALAXY"]]
        + confusion[CLASS_TO_INT["GALAXY"], CLASS_TO_INT["STAR"]]
    )
    return {
        "accuracy": correct / total,
        "balanced_accuracy": balanced_accuracy_from_confusion(confusion),
        "weighted_accuracy": weighted_correct / weighted_total if weighted_total else 0.0,
        "errors": total - correct,
        "star_to_galaxy": int(confusion[CLASS_TO_INT["STAR"], CLASS_TO_INT["GALAXY"]]),
        "galaxy_to_star": int(confusion[CLASS_TO_INT["GALAXY"], CLASS_TO_INT["STAR"]]),
        "star_galaxy_swaps": star_galaxy_swaps,
        "qso_to_galaxy": int(confusion[CLASS_TO_INT["QSO"], CLASS_TO_INT["GALAXY"]]),
        "galaxy_to_qso": int(confusion[CLASS_TO_INT["GALAXY"], CLASS_TO_INT["QSO"]]),
        "qso_to_star": int(confusion[CLASS_TO_INT["QSO"], CLASS_TO_INT["STAR"]]),
        "star_to_qso": int(confusion[CLASS_TO_INT["STAR"], CLASS_TO_INT["QSO"]]),
        "confusion": confusion,
    }


def metrics_from_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_weights: dict[str, float] | None = None,
) -> dict:
    return metrics_from_confusion(confusion_from_pred(y_true, y_pred), class_weights)


def confusion_delta_for_replacements(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    staged_pred: np.ndarray,
    replace_mask: np.ndarray,
) -> np.ndarray:
    if not replace_mask.any():
        return np.zeros((len(CLASSES), len(CLASSES)), dtype="int64")

    y = y_true[replace_mask].astype("int64")
    base = base_pred[replace_mask].astype("int64")
    staged = staged_pred[replace_mask].astype("int64")
    remove = np.bincount(
        y * len(CLASSES) + base,
        minlength=len(CLASSES) ** 2,
    ).reshape(len(CLASSES), len(CLASSES))
    add = np.bincount(
        y * len(CLASSES) + staged,
        minlength=len(CLASSES) ** 2,
    ).reshape(len(CLASSES), len(CLASSES))
    return add - remove


def two_stage_predict(
    proba: np.ndarray,
    qso_threshold: float,
    star_threshold: float,
    replace_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_pred = proba.argmax(axis=1).astype("int8")
    p_galaxy = proba[:, CLASS_TO_INT["GALAXY"]]
    p_qso = proba[:, CLASS_TO_INT["QSO"]]
    p_star = proba[:, CLASS_TO_INT["STAR"]]

    other_total = np.clip(p_galaxy + p_star, 1e-12, None)
    star_given_other = p_star / other_total
    galaxy_given_other = p_galaxy / other_total

    staged_pred = np.where(
        p_qso >= qso_threshold,
        CLASS_TO_INT["QSO"],
        np.where(star_given_other >= star_threshold, CLASS_TO_INT["STAR"], CLASS_TO_INT["GALAXY"]),
    ).astype("int8")
    staged_confidence = np.where(
        staged_pred == CLASS_TO_INT["QSO"],
        p_qso,
        np.where(staged_pred == CLASS_TO_INT["STAR"], star_given_other, galaxy_given_other),
    )
    replace_mask = (staged_pred != base_pred) & (staged_confidence >= replace_threshold)
    final_pred = base_pred.copy()
    final_pred[replace_mask] = staged_pred[replace_mask]
    return final_pred, staged_pred, replace_mask


def staged_confidence_for_pred(proba: np.ndarray, staged_pred: np.ndarray) -> np.ndarray:
    p_galaxy = proba[:, CLASS_TO_INT["GALAXY"]]
    p_qso = proba[:, CLASS_TO_INT["QSO"]]
    p_star = proba[:, CLASS_TO_INT["STAR"]]
    other_total = np.clip(p_galaxy + p_star, 1e-12, None)
    star_given_other = p_star / other_total
    galaxy_given_other = p_galaxy / other_total
    return np.where(
        staged_pred == CLASS_TO_INT["QSO"],
        p_qso,
        np.where(staged_pred == CLASS_TO_INT["STAR"], star_given_other, galaxy_given_other),
    )


def search_grid(
    proba: np.ndarray,
    y_true: np.ndarray,
    qso_thresholds: np.ndarray,
    star_thresholds: np.ndarray,
    replace_thresholds: np.ndarray,
    class_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
    base_pred = proba.argmax(axis=1).astype("int8")
    base_confusion = confusion_from_pred(y_true, base_pred)
    baseline = metrics_from_pred(y_true, base_pred, class_weights)
    rows = []
    total_rows = len(y_true)

    for qso_threshold in qso_thresholds:
        for star_threshold in star_thresholds:
            staged_pred_no_replace, staged_pred, _ = two_stage_predict(
                proba,
                qso_threshold=float(qso_threshold),
                star_threshold=float(star_threshold),
                replace_threshold=0.0,
            )
            staged_metrics = metrics_from_pred(y_true, staged_pred_no_replace, class_weights)
            staged_diff = staged_pred != base_pred
            staged_confidence = staged_confidence_for_pred(proba, staged_pred)

            for replace_threshold in replace_thresholds:
                replace_mask = staged_diff & (staged_confidence >= replace_threshold)
                confusion = base_confusion + confusion_delta_for_replacements(
                    y_true,
                    base_pred,
                    staged_pred,
                    replace_mask,
                )
                metrics = metrics_from_confusion(confusion, class_weights)
                replaced = int(replace_mask.sum())
                rows.append(
                    {
                        "qso_threshold": float(qso_threshold),
                        "star_threshold": float(star_threshold),
                        "replace_threshold": float(replace_threshold),
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "weighted_accuracy": metrics["weighted_accuracy"],
                        "errors": metrics["errors"],
                        "delta_accuracy": metrics["accuracy"] - baseline["accuracy"],
                        "delta_balanced_accuracy": metrics["balanced_accuracy"]
                        - baseline["balanced_accuracy"],
                        "delta_weighted_accuracy": metrics["weighted_accuracy"]
                        - baseline["weighted_accuracy"],
                        "delta_errors": metrics["errors"] - baseline["errors"],
                        "star_to_galaxy": metrics["star_to_galaxy"],
                        "galaxy_to_star": metrics["galaxy_to_star"],
                        "star_galaxy_swaps": metrics["star_galaxy_swaps"],
                        "delta_star_galaxy_swaps": metrics["star_galaxy_swaps"]
                        - baseline["star_galaxy_swaps"],
                        "qso_to_galaxy": metrics["qso_to_galaxy"],
                        "galaxy_to_qso": metrics["galaxy_to_qso"],
                        "qso_to_star": metrics["qso_to_star"],
                        "star_to_qso": metrics["star_to_qso"],
                        "replaced": replaced,
                        "replacement_rate": replaced / total_rows,
                        "staged_accuracy_without_replace_threshold": staged_metrics["accuracy"],
                        "staged_balanced_accuracy_without_replace_threshold": staged_metrics[
                            "balanced_accuracy"
                        ],
                        "staged_diff_from_argmax": int(staged_diff.sum()),
                    }
                )

    result = pd.DataFrame(rows)
    return result, baseline


def save_confusion(path: Path, confusion: np.ndarray) -> None:
    pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(path)


def save_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    pred: np.ndarray,
) -> None:
    submission = sample_submission.copy()
    submission[ID_COL] = test_ids.values
    submission[LABEL_COL] = INT_TO_CLASS[pred]
    submission.to_csv(path, index=False)


def predict_from_threshold_row(proba: np.ndarray, row: pd.Series | dict) -> np.ndarray:
    pred, _, _ = two_stage_predict(
        proba,
        qso_threshold=float(row["qso_threshold"]),
        star_threshold=float(row["star_threshold"]),
        replace_threshold=float(row["replace_threshold"]),
    )
    return pred


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    train, prefix = load_train_frame(args.train_proba, args.label_source, args.proba_prefix)
    cols = probability_columns(prefix)
    proba = normalize_proba(train[cols].to_numpy())
    y_true = encode_labels(train[LABEL_COL])

    qso_thresholds = parse_range(args.qso_thresholds)
    star_thresholds = parse_range(args.star_thresholds)
    replace_thresholds = parse_range(args.replace_thresholds)
    class_weights = parse_class_metric_weights(args.class_metric_weights)

    results, baseline = search_grid(
        proba,
        y_true,
        qso_thresholds=qso_thresholds,
        star_thresholds=star_thresholds,
        replace_thresholds=replace_thresholds,
        class_weights=class_weights,
    )
    results.to_csv(run_dir / "grid_results.csv", index=False)

    top_balanced = results.sort_values(
        ["balanced_accuracy", "accuracy", "star_galaxy_swaps"],
        ascending=[False, False, True],
    ).head(args.top_n)
    top_accuracy = results.sort_values(
        ["accuracy", "balanced_accuracy", "star_galaxy_swaps"],
        ascending=[False, False, True],
    ).head(args.top_n)
    top_weighted = results.sort_values(
        ["weighted_accuracy", "balanced_accuracy", "accuracy", "star_galaxy_swaps"],
        ascending=[False, False, False, True],
    ).head(args.top_n)
    top_star_galaxy = results.sort_values(
        ["star_galaxy_swaps", "balanced_accuracy", "accuracy"],
        ascending=[True, False, False],
    ).head(args.top_n)
    top_balanced.to_csv(run_dir / "top_by_balanced_accuracy.csv", index=False)
    top_accuracy.to_csv(run_dir / "top_by_accuracy.csv", index=False)
    top_weighted.to_csv(run_dir / "top_by_weighted_accuracy.csv", index=False)
    top_star_galaxy.to_csv(run_dir / "top_by_star_galaxy_swaps.csv", index=False)

    best = top_balanced.iloc[0].to_dict()
    best_accuracy = top_accuracy.iloc[0].to_dict()
    best_weighted = top_weighted.iloc[0].to_dict()
    best_pred = predict_from_threshold_row(proba, best)
    _, _, best_replace_mask = two_stage_predict(
        proba,
        qso_threshold=float(best["qso_threshold"]),
        star_threshold=float(best["star_threshold"]),
        replace_threshold=float(best["replace_threshold"]),
    )
    best_metrics = metrics_from_pred(y_true, best_pred, class_weights)
    save_confusion(run_dir / "baseline_confusion_matrix.csv", baseline["confusion"])
    save_confusion(run_dir / "best_confusion_matrix.csv", best_metrics["confusion"])

    replacement_audit = train[[ID_COL, LABEL_COL]].copy()
    replacement_audit["base_pred"] = INT_TO_CLASS[proba.argmax(axis=1)]
    replacement_audit["best_pred"] = INT_TO_CLASS[best_pred]
    replacement_audit["replaced"] = best_replace_mask
    replacement_audit[cols] = train[cols]
    replacement_audit[replacement_audit["replaced"]].to_csv(
        run_dir / "best_replacements_oof.csv", index=False
    )

    manifest = {
        "args": vars(args),
        "proba_prefix": prefix,
        "train_shape": list(train.shape),
        "classes": CLASSES,
        "class_metric_weights": class_weights,
        "grid": {
            "qso_thresholds": qso_thresholds.tolist(),
            "star_thresholds": star_thresholds.tolist(),
            "replace_thresholds": replace_thresholds.tolist(),
            "rows": int(len(results)),
        },
        "baseline": {
            key: value
            for key, value in baseline.items()
            if key != "confusion"
        },
        "best_by_balanced_accuracy": {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
            for key, value in best.items()
        },
        "best_by_accuracy": {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
            for key, value in best_accuracy.items()
        },
        "best_by_weighted_accuracy": {
            key: (float(value) if isinstance(value, (np.floating, float)) else int(value))
            for key, value in best_weighted.items()
        },
    }
    save_json(run_dir / "manifest.json", manifest)

    summary_lines = [
        "# Two-Stage Threshold Search",
        "",
        f"Train probability file: `{args.train_proba}`",
        f"Probability prefix: `{prefix}`",
        "",
        "## Baseline Argmax",
        "",
        f"- accuracy={baseline['accuracy']:.9f}",
        f"- balanced_accuracy={baseline['balanced_accuracy']:.9f}",
        f"- weighted_accuracy={baseline['weighted_accuracy']:.9f}",
        f"- errors={baseline['errors']}",
        f"- STAR->GALAXY={baseline['star_to_galaxy']}",
        f"- GALAXY->STAR={baseline['galaxy_to_star']}",
        "",
        "## Best Balanced Accuracy",
        "",
        f"- qso_threshold={best['qso_threshold']:.6f}",
        f"- star_threshold={best['star_threshold']:.6f}",
        f"- replace_threshold={best['replace_threshold']:.6f}",
        f"- accuracy={best['accuracy']:.9f} ({best['delta_accuracy']:+.9f})",
        f"- balanced_accuracy={best['balanced_accuracy']:.9f} ({best['delta_balanced_accuracy']:+.9f})",
        f"- weighted_accuracy={best['weighted_accuracy']:.9f} ({best['delta_weighted_accuracy']:+.9f})",
        f"- errors={int(best['errors'])} ({int(best['delta_errors']):+d})",
        f"- STAR/GALAXY swaps={int(best['star_galaxy_swaps'])} ({int(best['delta_star_galaxy_swaps']):+d})",
        f"- replaced={int(best['replaced'])} ({best['replacement_rate']:.4%})",
        "",
        "## Best Accuracy",
        "",
        f"- qso_threshold={best_accuracy['qso_threshold']:.6f}",
        f"- star_threshold={best_accuracy['star_threshold']:.6f}",
        f"- replace_threshold={best_accuracy['replace_threshold']:.6f}",
        f"- accuracy={best_accuracy['accuracy']:.9f} ({best_accuracy['delta_accuracy']:+.9f})",
        f"- balanced_accuracy={best_accuracy['balanced_accuracy']:.9f} ({best_accuracy['delta_balanced_accuracy']:+.9f})",
        f"- weighted_accuracy={best_accuracy['weighted_accuracy']:.9f} ({best_accuracy['delta_weighted_accuracy']:+.9f})",
        f"- errors={int(best_accuracy['errors'])} ({int(best_accuracy['delta_errors']):+d})",
        f"- STAR/GALAXY swaps={int(best_accuracy['star_galaxy_swaps'])} ({int(best_accuracy['delta_star_galaxy_swaps']):+d})",
        f"- replaced={int(best_accuracy['replaced'])} ({best_accuracy['replacement_rate']:.4%})",
        "",
        "## Best Weighted Accuracy",
        "",
        f"- qso_threshold={best_weighted['qso_threshold']:.6f}",
        f"- star_threshold={best_weighted['star_threshold']:.6f}",
        f"- replace_threshold={best_weighted['replace_threshold']:.6f}",
        f"- accuracy={best_weighted['accuracy']:.9f} ({best_weighted['delta_accuracy']:+.9f})",
        f"- balanced_accuracy={best_weighted['balanced_accuracy']:.9f} ({best_weighted['delta_balanced_accuracy']:+.9f})",
        f"- weighted_accuracy={best_weighted['weighted_accuracy']:.9f} ({best_weighted['delta_weighted_accuracy']:+.9f})",
        f"- errors={int(best_weighted['errors'])} ({int(best_weighted['delta_errors']):+d})",
        f"- STAR/GALAXY swaps={int(best_weighted['star_galaxy_swaps'])} ({int(best_weighted['delta_star_galaxy_swaps']):+d})",
        f"- replaced={int(best_weighted['replaced'])} ({best_weighted['replacement_rate']:.4%})",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    if args.save_submission:
        test = pd.read_csv(args.test_proba)
        missing = [col for col in cols + [ID_COL] if col not in test.columns]
        if missing:
            raise ValueError(f"Missing columns in {args.test_proba}: {missing}")
        test_proba = normalize_proba(test[cols].to_numpy())
        test_pred = predict_from_threshold_row(test_proba, best)
        sample_submission = pd.read_csv(args.sample_submission)
        save_submission(
            run_dir / "best_balanced_submission.csv",
            sample_submission,
            test[ID_COL],
            test_pred,
        )
        register_submission(
            run_dir / "best_balanced_submission.csv",
            run_dir=run_dir,
            script=Path(__file__).name,
            submission_type="two_stage_threshold",
            model_name="best_balanced_accuracy",
            metrics={
                key: best[key]
                for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "delta_accuracy",
                    "delta_balanced_accuracy",
                    "errors",
                    "delta_errors",
                    "replacement_rate",
                    "star_galaxy_swaps",
                ]
            },
            params={
                "proba_prefix": prefix,
                "train_proba": args.train_proba,
                "test_proba": args.test_proba,
                "qso_threshold": best["qso_threshold"],
                "star_threshold": best["star_threshold"],
                "replace_threshold": best["replace_threshold"],
            },
            extra={"top_results_path": "top_by_balanced_accuracy.csv"},
        )
        test_accuracy_pred = predict_from_threshold_row(test_proba, best_accuracy)
        save_submission(
            run_dir / "best_accuracy_submission.csv",
            sample_submission,
            test[ID_COL],
            test_accuracy_pred,
        )
        register_submission(
            run_dir / "best_accuracy_submission.csv",
            run_dir=run_dir,
            script=Path(__file__).name,
            submission_type="two_stage_threshold",
            model_name="best_accuracy",
            metrics={
                key: best_accuracy[key]
                for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "delta_accuracy",
                    "delta_balanced_accuracy",
                    "errors",
                    "delta_errors",
                    "replacement_rate",
                    "star_galaxy_swaps",
                ]
            },
            params={
                "proba_prefix": prefix,
                "train_proba": args.train_proba,
                "test_proba": args.test_proba,
                "qso_threshold": best_accuracy["qso_threshold"],
                "star_threshold": best_accuracy["star_threshold"],
                "replace_threshold": best_accuracy["replace_threshold"],
            },
            extra={"top_results_path": "top_by_accuracy.csv"},
        )
        test_weighted_pred = predict_from_threshold_row(test_proba, best_weighted)
        save_submission(
            run_dir / "best_weighted_submission.csv",
            sample_submission,
            test[ID_COL],
            test_weighted_pred,
        )
        register_submission(
            run_dir / "best_weighted_submission.csv",
            run_dir=run_dir,
            script=Path(__file__).name,
            submission_type="two_stage_threshold",
            model_name="best_weighted_accuracy",
            metrics={
                key: best_weighted[key]
                for key in [
                    "accuracy",
                    "balanced_accuracy",
                    "weighted_accuracy",
                    "delta_accuracy",
                    "delta_balanced_accuracy",
                    "delta_weighted_accuracy",
                    "errors",
                    "delta_errors",
                    "replacement_rate",
                    "star_galaxy_swaps",
                ]
            },
            params={
                "proba_prefix": prefix,
                "train_proba": args.train_proba,
                "test_proba": args.test_proba,
                "class_metric_weights": class_weights,
                "qso_threshold": best_weighted["qso_threshold"],
                "star_threshold": best_weighted["star_threshold"],
                "replace_threshold": best_weighted["replace_threshold"],
            },
            extra={"top_results_path": "top_by_weighted_accuracy.csv"},
        )

    print("Baseline:")
    print(json.dumps({k: v for k, v in baseline.items() if k != "confusion"}, indent=2))
    print("Best balanced-accuracy setting:")
    print(top_balanced.head(10).to_string(index=False))
    print("Best weighted-accuracy setting:")
    print(top_weighted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
