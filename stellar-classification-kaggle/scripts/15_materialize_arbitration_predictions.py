from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize train/test class predictions from a disagreement arbitration run."
    )
    parser.add_argument("--arbitration-run", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    return parser.parse_args()


def load_prediction(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"{path} must contain {ID_COL!r} and {LABEL_COL!r}.")
    return df[[ID_COL, LABEL_COL]].rename(columns={LABEL_COL: pred_col})


def materialize(path_a: Path, path_b: Path, output_path: Path, base: str, selected_flows: list[dict]) -> None:
    a = load_prediction(path_a, "pred_a")
    b = load_prediction(path_b, "pred_b")
    out = a.merge(b, on=ID_COL, how="inner")
    if len(out) != len(a) or len(out) != len(b):
        raise ValueError(f"Prediction alignment changed row count for {output_path}")

    base_col = "pred_a" if base == "a" else "pred_b"
    other_col = "pred_b" if base == "a" else "pred_a"
    out[LABEL_COL] = out[base_col]
    for flow in selected_flows:
        base_pred = str(flow["base_pred"])
        other_pred = str(flow["other_pred"])
        mask = out[base_col].astype(str).eq(base_pred) & out[other_col].astype(str).eq(other_pred)
        out.loc[mask, LABEL_COL] = out.loc[mask, other_col]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out[[ID_COL, LABEL_COL]].to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.arbitration_run / "manifest.json").read_text(encoding="utf-8"))
    run_args = manifest["args"]
    selected_flows = manifest.get("selected_flows", [])
    base = run_args["base"]
    materialize(
        Path(run_args["candidate_a_train"]),
        Path(run_args["candidate_b_train"]),
        args.output_train,
        base,
        selected_flows,
    )
    materialize(
        Path(run_args["candidate_a_test"]),
        Path(run_args["candidate_b_test"]),
        args.output_test,
        base,
        selected_flows,
    )
    print(f"Saved train predictions: {args.output_train}")
    print(f"Saved test predictions: {args.output_test}")


if __name__ == "__main__":
    main()
