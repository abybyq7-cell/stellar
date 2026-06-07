from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL
from stellar.features import add_astronomy_features
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoGluon baseline for the stellar classification Kaggle task."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "autogluon_baseline")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--target", type=str, default=LABEL_COL)
    parser.add_argument("--id-col", type=str, default=ID_COL)
    parser.add_argument("--eval-metric", type=str, default="accuracy")
    parser.add_argument("--presets", type=str, default="medium_quality")
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--num-cpus", default="auto")
    parser.add_argument("--num-gpus", default=0)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--skip-refit-full", action="store_true")
    parser.add_argument("--no-feature-engineering", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "sample_submission.csv"

    missing = [path for path in [train_path, test_path, submission_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(submission_path)
    return train, test, sample_submission


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    target: str,
    id_col: str,
) -> None:
    if target not in train.columns:
        raise ValueError(f"Target column '{target}' not found in train.csv")
    if id_col not in train.columns or id_col not in test.columns:
        raise ValueError(f"ID column '{id_col}' must exist in train.csv and test.csv")
    if len(test) != len(sample_submission):
        raise ValueError(
            f"test.csv has {len(test)} rows but sample_submission.csv has "
            f"{len(sample_submission)} rows"
        )
    if target not in sample_submission.columns:
        raise ValueError(f"Submission target column '{target}' not found")


def stratified_sample(
    train: pd.DataFrame,
    target: str,
    sample_rows: int,
    seed: int,
) -> pd.DataFrame:
    if sample_rows <= 0 or sample_rows >= len(train):
        return train

    if sample_rows < train[target].nunique():
        raise ValueError(
            f"--sample-rows={sample_rows} is smaller than the number of classes "
            f"({train[target].nunique()})"
        )

    sampled, _ = train_test_split(
        train,
        train_size=sample_rows,
        stratify=train[target],
        random_state=seed,
    )
    return sampled.reset_index(drop=True)


def build_model_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_col: str,
    use_feature_engineering: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_feature_engineering:
        train = add_astronomy_features(train)
        test = add_astronomy_features(test)
    else:
        train = train.copy()
        test = test.copy()

    train_model = train.drop(columns=[id_col])
    test_model = test.drop(columns=[id_col])
    return train_model, test_model


def fit_autogluon(
    train_data: pd.DataFrame,
    tuning_data: pd.DataFrame | None,
    target: str,
    model_path: Path,
    args: argparse.Namespace,
):
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor(
        label=target,
        problem_type="multiclass",
        eval_metric=args.eval_metric,
        path=str(model_path),
        verbosity=args.verbosity,
    )
    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        time_limit=args.time_limit if args.time_limit > 0 else None,
        presets=args.presets,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
    )
    return predictor


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    train, test, sample_submission = read_inputs(args.data_dir)
    validate_inputs(train, test, sample_submission, args.target, args.id_col)

    train = stratified_sample(train, args.target, args.sample_rows, args.seed)
    use_feature_engineering = not args.no_feature_engineering
    train_model, test_model = build_model_frames(
        train=train,
        test=test,
        target=args.target,
        id_col=args.id_col,
        use_feature_engineering=use_feature_engineering,
    )

    class_counts = train_model[args.target].value_counts().to_dict()
    metadata = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "model_train_shape": list(train_model.shape),
        "model_test_shape": list(test_model.shape),
        "target": args.target,
        "id_col": args.id_col,
        "class_counts": class_counts,
        "feature_engineering": use_feature_engineering,
        "eval_metric": args.eval_metric,
        "presets": args.presets,
        "time_limit": args.time_limit,
        "valid_size": args.valid_size,
        "sample_rows": args.sample_rows,
        "seed": args.seed,
    }
    save_json(run_dir / "metadata.json", metadata)

    print("Data summary:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete. No model was trained.")
        return

    train_part, valid_part = train_test_split(
        train_model,
        test_size=args.valid_size,
        stratify=train_model[args.target],
        random_state=args.seed,
    )
    train_part = train_part.reset_index(drop=True)
    valid_part = valid_part.reset_index(drop=True)

    predictor = fit_autogluon(
        train_data=train_part,
        tuning_data=valid_part,
        target=args.target,
        model_path=run_dir / "ag_model",
        args=args,
    )

    leaderboard = predictor.leaderboard(valid_part, silent=True)
    leaderboard.to_csv(run_dir / "leaderboard_valid.csv", index=False)

    valid_metrics = predictor.evaluate(valid_part, silent=True)
    save_json(run_dir / "valid_metrics.json", valid_metrics)
    print("Validation metrics:")
    print(json.dumps(valid_metrics, indent=2, ensure_ascii=False, default=str))

    if not args.skip_refit_full:
        refit_map = predictor.refit_full(
            set_best_to_refit_full=True,
            train_data_extra=valid_part,
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
        )
        save_json(run_dir / "refit_full_map.json", refit_map)
        print("Refit full complete.")

    test_pred = predictor.predict(test_model)
    submission = sample_submission.copy()
    submission[args.id_col] = test[args.id_col].values
    submission[args.target] = test_pred.values
    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    register_submission(
        submission_path,
        run_dir=run_dir,
        script=Path(__file__).name,
        submission_type="autogluon_baseline",
        model_name="autogluon_tabular",
        metrics=valid_metrics,
        params={
            "eval_metric": args.eval_metric,
            "presets": args.presets,
            "time_limit": args.time_limit,
            "valid_size": args.valid_size,
            "sample_rows": args.sample_rows,
            "seed": args.seed,
            "feature_engineering": use_feature_engineering,
        },
        extra={"metadata_path": "metadata.json"},
    )

    print(f"Saved submission: {submission_path.resolve()}")
    print(submission.head())


if __name__ == "__main__":
    main()
