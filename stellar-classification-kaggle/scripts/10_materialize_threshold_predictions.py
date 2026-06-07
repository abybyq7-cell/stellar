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


CLASSES = ["GALAXY", "QSO", "STAR"]
INT_TO_CLASS = np.asarray(CLASSES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize train/test class predictions from a threshold-search run."
    )
    parser.add_argument("--threshold-run", type=Path, required=True)
    parser.add_argument(
        "--which",
        choices=["weighted", "balanced", "accuracy"],
        default="weighted",
    )
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    return parser.parse_args()


def detect_prefix(df: pd.DataFrame) -> str:
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
        raise ValueError(f"Could not detect one probability prefix. Found {valid}")
    return valid[0]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype("float64"), 1e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    return values


def two_stage_predict(proba: np.ndarray, qso_threshold: float, star_threshold: float, replace_threshold: float) -> np.ndarray:
    base_pred = proba.argmax(axis=1).astype("int8")
    p_galaxy = proba[:, 0]
    p_qso = proba[:, 1]
    p_star = proba[:, 2]
    other_total = np.clip(p_galaxy + p_star, 1e-12, None)
    star_given_other = p_star / other_total
    galaxy_given_other = p_galaxy / other_total
    staged_pred = np.where(
        p_qso >= qso_threshold,
        1,
        np.where(star_given_other >= star_threshold, 2, 0),
    ).astype("int8")
    staged_confidence = np.where(
        staged_pred == 1,
        p_qso,
        np.where(staged_pred == 2, star_given_other, galaxy_given_other),
    )
    replace_mask = (staged_pred != base_pred) & (staged_confidence >= replace_threshold)
    pred = base_pred.copy()
    pred[replace_mask] = staged_pred[replace_mask]
    return pred


def materialize(proba_path: Path, output_path: Path, row: dict) -> None:
    df = pd.read_csv(proba_path)
    prefix = detect_prefix(df)
    proba = normalize(df[[f"{prefix}__{cls}" for cls in CLASSES]].to_numpy())
    pred = two_stage_predict(
        proba,
        float(row["qso_threshold"]),
        float(row["star_threshold"]),
        float(row["replace_threshold"]),
    )
    out = pd.DataFrame({ID_COL: df[ID_COL].values, LABEL_COL: INT_TO_CLASS[pred]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.threshold_run / "manifest.json").read_text(encoding="utf-8"))
    row_key = {
        "weighted": "best_by_weighted_accuracy",
        "balanced": "best_by_balanced_accuracy",
        "accuracy": "best_by_accuracy",
    }[args.which]
    row = manifest[row_key]
    materialize(Path(manifest["args"]["train_proba"]), args.output_train, row)
    materialize(Path(manifest["args"]["test_proba"]), args.output_test, row)
    print(f"Saved train predictions: {args.output_train}")
    print(f"Saved test predictions: {args.output_test}")


if __name__ == "__main__":
    main()
