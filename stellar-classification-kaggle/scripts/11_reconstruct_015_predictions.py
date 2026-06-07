from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Reconstruct submission 015 OOF/test predictions.")
    parser.add_argument(
        "--pred005-train",
        type=Path,
        default=Path("outputs/two_stage_threshold/lr_balanced_fine_qso032_star025/best_balanced_oof_train_pred.csv"),
    )
    parser.add_argument(
        "--pred005-test",
        type=Path,
        default=Path("outputs/two_stage_threshold/lr_balanced_fine_qso032_star025/best_balanced_test_pred.csv"),
    )
    parser.add_argument(
        "--pred013-train",
        type=Path,
        default=Path("outputs/two_stage_threshold/full19_corr9995_lr_aggressive_grid_v2/best_balanced_oof_train_pred.csv"),
    )
    parser.add_argument(
        "--pred013-test",
        type=Path,
        default=Path("outputs/two_stage_threshold/full19_corr9995_lr_aggressive_grid_v2/best_balanced_test_pred.csv"),
    )
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    return parser.parse_args()


def read_pred(path: Path, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[[ID_COL, LABEL_COL]].rename(columns={LABEL_COL: col})


def reconstruct(df: pd.DataFrame) -> pd.Series:
    out = df["pred013"].copy()
    use_005 = (
        (df["pred005"].eq("QSO") & df["pred013"].eq("GALAXY"))
        | (df["pred005"].eq("STAR") & df["pred013"].eq("QSO"))
    )
    out.loc[use_005] = df.loc[use_005, "pred005"]
    return out


def write_reconstructed(path005: Path, path013: Path, output: Path) -> None:
    df = read_pred(path005, "pred005").merge(read_pred(path013, "pred013"), on=ID_COL, how="inner")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: df[ID_COL].values, LABEL_COL: reconstruct(df).values}).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    write_reconstructed(args.pred005_train, args.pred013_train, args.output_train)
    write_reconstructed(args.pred005_test, args.pred013_test, args.output_test)
    print(f"Saved 015 train predictions: {args.output_train}")
    print(f"Saved 015 test predictions: {args.output_test}")


if __name__ == "__main__":
    main()
