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

from stellar.paths import OUTPUTS_DIR
from stellar.submissions import infer_run_context, register_submission, submission_summary
from stellar.constants import ID_COL, LABEL_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register existing submission CSV files with sidecar metadata."
    )
    parser.add_argument("--search-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--pattern", type=str, default="*submission*.csv")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write sidecar metadata and update submissions/submission_registry.*.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include files that already have .metadata.json sidecars.",
    )
    parser.add_argument(
        "--include-nonstandard",
        action="store_true",
        help="Also include CSV files that do not have the standard id/class columns.",
    )
    return parser.parse_args()


def has_submission_columns(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return {ID_COL, LABEL_COL}.issubset(columns)


def candidate_files(
    search_dir: Path,
    pattern: str,
    include_metadata: bool,
    include_nonstandard: bool,
) -> list[Path]:
    paths = []
    for path in search_dir.rglob(pattern):
        if path.name.endswith(".metadata.json"):
            continue
        if not include_metadata and path.with_suffix(".metadata.json").exists():
            continue
        if not include_nonstandard and not has_submission_columns(path):
            continue
        paths.append(path)
    return sorted(paths)


def nearby_metrics(path: Path) -> dict:
    run_dir = path.parent
    if run_dir.name in {"ensembles", "single_model_submissions"}:
        run_dir = run_dir.parent

    metric_files = [
        run_dir / "valid_metrics.json",
        run_dir / "autogluon_train_metrics.json",
        run_dir / "summary_overall.csv",
        run_dir / "ensemble_scores.csv",
        run_dir / "stacker_scores.csv",
        run_dir / "top_by_balanced_accuracy.csv",
        run_dir / "top_by_accuracy.csv",
    ]
    for metric_path in metric_files:
        if not metric_path.exists():
            continue
        try:
            if metric_path.suffix == ".json":
                return json.loads(metric_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(metric_path)
            if frame.empty:
                continue
            row = frame.iloc[0].to_dict()
            return {
                key: row[key]
                for key in ["accuracy", "balanced_accuracy", "log_loss", "score"]
                if key in row
            }
        except Exception:
            continue
    return {}


def infer_submission_type(path: Path) -> tuple[str, str]:
    name = path.stem
    parent = path.parent.name
    if parent == "ensembles":
        return "historical_weighted_ensemble", name.removesuffix("_submission")
    if parent == "single_model_submissions":
        return "historical_single_model", name
    if "autogluon" in name:
        return "historical_autogluon", name
    if "threshold" in str(path).lower() or name.startswith("best_"):
        return "historical_threshold", name
    if "lr_logits" in name:
        return "historical_lr_logits", name
    return "historical_submission", name


def main() -> None:
    args = parse_args()
    paths = candidate_files(
        args.search_dir,
        args.pattern,
        args.include_metadata,
        args.include_nonstandard,
    )
    rows = []

    for path in paths:
        summary = submission_summary(path)
        context = infer_run_context(path)
        submission_type, model_name = infer_submission_type(path)
        row = {
            "path": str(path),
            "workflow": context.get("workflow"),
            "run_name": context.get("run_name"),
            "submission_type": submission_type,
            "model_name": model_name,
            "row_count": summary.get("row_count"),
            "duplicate_id_count": summary.get("duplicate_id_count"),
            "sha256": summary.get("sha256"),
        }
        rows.append(row)

        if args.write:
            register_submission(
                path,
                run_dir=Path(context["run_dir"]) if context.get("run_dir") else None,
                script=Path(__file__).name,
                submission_type=submission_type,
                model_name=model_name,
                metrics=nearby_metrics(path),
                notes="Registered by historical submission scanner.",
            )

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("No matching submission files found.")

    if not args.write:
        print("Dry run only. Re-run with --write to update the registry.")


if __name__ == "__main__":
    main()
