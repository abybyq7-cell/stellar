from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
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


CLASSES = ["GALAXY", "QSO", "STAR"]


def load_layer1_helpers():
    path = PROJECT_ROOT / "scripts" / "03_layer1_oof_experiments.py"
    spec = importlib.util.spec_from_file_location("layer1_helpers_for_sweep", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["layer1_helpers_for_sweep"] = module
    spec.loader.exec_module(module)
    return module


def package_available(package_name: str) -> bool:
    try:
        __import__(package_name)
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small OOF hyperparameter sweep for LGBM/CatBoost/XGBoost variants."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "hyperparam_sweep",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=60000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["targeted", "targeted_wide"],
        choices=["raw", "baseline", "extra", "targeted", "autofe", "groupagg", "wide", "targeted_wide"],
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["lgbm", "xgboost", "catboost"],
        choices=["lgbm", "xgboost", "catboost"],
    )
    parser.add_argument(
        "--autofe-dir",
        type=Path,
        default=OUTPUTS_DIR / "feature_exploration" / "medium_autofe_groupby",
    )
    parser.add_argument("--groupby-top-n", type=int, default=16)
    parser.add_argument(
        "--weighting",
        choices=["none", "star_lowz", "star_lowz_hard"],
        default="none",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=OUTPUTS_DIR / "layer1_oof" / "layer1_50k_balanced",
    )
    parser.add_argument("--star-weight", type=float, default=1.15)
    parser.add_argument("--star-m-weight", type=float, default=1.45)
    parser.add_argument("--star-hard-color-weight", type=float, default=1.65)
    parser.add_argument("--lowz-weight", type=float, default=1.20)
    parser.add_argument("--star-lowz-weight", type=float, default=1.75)
    parser.add_argument("--hard-case-weight", type=float, default=1.90)
    parser.add_argument("--all-wrong-weight", type=float, default=2.30)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu", "auto"],
        default="cpu",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Optional cap after building all family/feature combinations.",
    )
    parser.add_argument("--skip-unavailable", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def lgbm_variants(layer1, seed: int) -> list:
    variants = []

    base = layer1.lgbm_params(seed, extra_trees=False, n_estimators=900)
    variants.append(("base", base))

    conservative = layer1.lgbm_params(seed + 1, extra_trees=False, n_estimators=1100)
    conservative.update(
        {
            "learning_rate": 0.025,
            "num_leaves": 48,
            "min_child_samples": 55,
            "colsample_bytree": 0.82,
            "reg_lambda": 2.2,
        }
    )
    variants.append(("conservative", conservative))

    expressive = layer1.lgbm_params(seed + 2, extra_trees=False, n_estimators=800)
    expressive.update(
        {
            "learning_rate": 0.045,
            "num_leaves": 96,
            "min_child_samples": 22,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
        }
    )
    variants.append(("expressive", expressive))

    xt = layer1.lgbm_params(seed + 3, extra_trees=True, n_estimators=1000)
    xt.update(
        {
            "learning_rate": 0.030,
            "num_leaves": 80,
            "min_child_samples": 35,
            "subsample": 0.92,
            "reg_lambda": 1.8,
        }
    )
    variants.append(("extra_trees", xt))

    star_recall = layer1.lgbm_params(seed + 4, extra_trees=False, n_estimators=950)
    star_recall.update(
        {
            "learning_rate": 0.032,
            "num_leaves": 72,
            "min_child_samples": 18,
            "min_split_gain": 0.0,
            "reg_alpha": 0.04,
            "reg_lambda": 1.0,
        }
    )
    variants.append(("star_recall", star_recall))
    return variants


def xgb_variants(layer1, seed: int) -> list:
    variants = []

    base = layer1.xgb_params(seed, n_estimators=750)
    variants.append(("base", base))

    shallow = layer1.xgb_params(seed + 1, n_estimators=900)
    shallow.update(
        {
            "learning_rate": 0.028,
            "max_depth": 4,
            "min_child_weight": 3.5,
            "subsample": 0.90,
            "colsample_bytree": 0.82,
            "reg_lambda": 2.2,
        }
    )
    variants.append(("shallow_regularized", shallow))

    deeper = layer1.xgb_params(seed + 2, n_estimators=650)
    deeper.update(
        {
            "learning_rate": 0.045,
            "max_depth": 6,
            "min_child_weight": 1.2,
            "subsample": 0.86,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.2,
        }
    )
    variants.append(("deeper", deeper))

    star_recall = layer1.xgb_params(seed + 3, n_estimators=800)
    star_recall.update(
        {
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.02,
            "reg_lambda": 1.0,
        }
    )
    variants.append(("star_recall", star_recall))
    return variants


def cat_variants(layer1, seed: int) -> list:
    variants = []

    base = layer1.cat_params(seed, depth=6, iterations=800)
    variants.append(("base", base))

    depth5 = layer1.cat_params(seed + 1, depth=5, iterations=950)
    depth5.update({"learning_rate": 0.040, "l2_leaf_reg": 7.0, "random_strength": 1.1})
    variants.append(("depth5_regularized", depth5))

    depth7 = layer1.cat_params(seed + 2, depth=7, iterations=700)
    depth7.update({"learning_rate": 0.050, "l2_leaf_reg": 4.0, "random_strength": 0.6})
    variants.append(("depth7_expressive", depth7))
    return variants


def build_sweep_experiments(args: argparse.Namespace, layer1) -> list:
    family_builders = {
        "lgbm": lgbm_variants,
        "xgboost": xgb_variants,
        "catboost": cat_variants,
    }
    availability = {
        "lgbm": package_available("lightgbm"),
        "xgboost": package_available("xgboost"),
        "catboost": package_available("catboost"),
    }
    experiments = []
    base_seed = 2700

    for family in args.families:
        if not availability[family]:
            message = f"{family} is not available in this Python environment."
            if args.skip_unavailable:
                print(f"Skipping: {message}")
                continue
            raise ImportError(message)

        for feature_set in args.feature_sets:
            for variant_name, params in family_builders[family](layer1, base_seed):
                exp_name = f"{family}_{variant_name}_{feature_set}_s{params.get('random_state', params.get('random_seed', base_seed))}"
                experiments.append(
                    layer1.Experiment(
                        name=exp_name,
                        model_type=family,
                        feature_set=feature_set,
                        seed=int(params.get("random_state", params.get("random_seed", base_seed))),
                        params=params,
                    )
                )
                base_seed += 11

    if args.max_experiments > 0:
        experiments = experiments[: args.max_experiments]
    return experiments


def class_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    rows = {}
    for idx, cls in enumerate(classes):
        support = int(np.sum(y_true == idx))
        pred_count = int(np.sum(pred == idx))
        tp = int(np.sum((y_true == idx) & (pred == idx)))
        fp = int(np.sum((y_true != idx) & (pred == idx)))
        fn = int(np.sum((y_true == idx) & (pred != idx)))
        rows[f"recall_{cls}"] = tp / support if support else 0.0
        rows[f"precision_{cls}"] = tp / pred_count if pred_count else 0.0
        rows[f"pred_count_{cls}"] = pred_count
        rows[f"fn_{cls}"] = fn
        rows[f"fp_{cls}"] = fp
    return rows


def run_one_experiment(
    exp,
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    y_enc: np.ndarray,
    sample_weights: np.ndarray,
    classes: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    run_dir: Path,
    layer1,
) -> tuple[dict, pd.DataFrame]:
    X, X_test = feature_sets[exp.feature_set]
    n_classes = len(classes)
    oof = np.zeros((len(X), n_classes), dtype="float32")
    fold_rows = []
    started = time.time()

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"[{exp.name}] fold {fold}/{len(folds)}")
        valid_proba, _, _, metrics = layer1.fit_predict_fold(
            exp=exp,
            X_train=X.iloc[train_idx].reset_index(drop=True),
            y_train=y_enc[train_idx],
            w_train=sample_weights[train_idx] if sample_weights is not None else None,
            X_valid=X.iloc[valid_idx].reset_index(drop=True),
            y_valid=y_enc[valid_idx],
            w_valid=sample_weights[valid_idx] if sample_weights is not None else None,
            X_test=None,
            args=args,
        )
        oof[valid_idx] = valid_proba.astype("float32")
        fold_rows.append(
            {
                "experiment": exp.name,
                "model_type": exp.model_type,
                "feature_set": exp.feature_set,
                "fold": fold,
                **metrics,
            }
        )

    pred = oof.argmax(axis=1)
    summary = {
        "experiment": exp.name,
        "model_type": exp.model_type,
        "feature_set": exp.feature_set,
        "variant": variant_from_name(exp.name, exp.model_type, exp.feature_set),
        "accuracy": accuracy_score(y_enc, pred),
        "balanced_accuracy": balanced_accuracy_score(y_enc, pred),
        "log_loss": log_loss(y_enc, oof, labels=np.arange(n_classes)),
        "fit_predict_seconds": time.time() - started,
        **class_metrics(y_enc, oof, classes),
    }
    exp_dir = run_dir / "experiments" / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COL: np.arange(len(oof)),
            LABEL_COL: classes[y_enc],
            **{f"proba__{cls}": oof[:, i] for i, cls in enumerate(classes)},
        }
    ).to_csv(exp_dir / "oof_train.csv", index=False)
    save_json(exp_dir / "config.json", asdict(exp))
    pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)
    return summary, pd.DataFrame(fold_rows)


def variant_from_name(name: str, model_type: str, feature_set: str) -> str:
    prefix = f"{model_type}_"
    suffix = f"_{feature_set}"
    out = name
    if out.startswith(prefix):
        out = out[len(prefix) :]
    if suffix in out:
        out = out.split(suffix)[0]
    return out


def write_summary_markdown(run_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    top_balanced = summary.sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=False,
    ).head(10)
    top_star = summary.sort_values(
        ["recall_STAR", "balanced_accuracy"],
        ascending=False,
    ).head(10)
    lines = [
        "# Small Hyperparameter Sweep",
        "",
        f"- sample_rows: {args.sample_rows}",
        f"- n_splits: {args.n_splits}",
        f"- feature_sets: {args.feature_sets}",
        f"- families: {args.families}",
        f"- weighting: {args.weighting}",
        "",
        "## Top Balanced Accuracy",
        "",
    ]
    for _, row in top_balanced.iterrows():
        lines.append(
            f"- `{row['experiment']}`: balanced={row['balanced_accuracy']:.6f}, "
            f"acc={row['accuracy']:.6f}, logloss={row['log_loss']:.6f}, "
            f"STAR recall={row['recall_STAR']:.6f}, QSO recall={row['recall_QSO']:.6f}"
        )
    lines.extend(["", "## Top STAR Recall", ""])
    for _, row in top_star.iterrows():
        lines.append(
            f"- `{row['experiment']}`: STAR recall={row['recall_STAR']:.6f}, "
            f"balanced={row['balanced_accuracy']:.6f}, acc={row['accuracy']:.6f}"
        )
    lines.append("")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    layer1 = load_layer1_helpers()

    if args.sample_rows <= 0:
        raise ValueError("Use a positive --sample-rows for a small sweep.")

    run_dir = make_run_dir(args.output_dir, args.run_name)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")

    train, test, _ = layer1.read_inputs(args.data_dir)
    train_sample = layer1.stratified_sample_train(
        train,
        LABEL_COL,
        args.sample_rows,
        args.sample_seed,
    )
    y = train_sample[LABEL_COL].copy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    classes = label_encoder.classes_

    feature_sets, feature_metadata = layer1.build_feature_sets(train_sample, test, args)
    feature_sets = {name: value for name, value in feature_sets.items() if name in set(args.feature_sets)}
    experiments = build_sweep_experiments(args, layer1)
    experiments = [exp for exp in experiments if exp.feature_set in feature_sets]
    sample_weights = layer1.compute_sample_weights(train_sample, args)

    metadata = {
        "args": vars(args),
        "train_full_shape": list(train.shape),
        "train_sample_shape": list(train_sample.shape),
        "test_shape": list(test.shape),
        "classes": classes.tolist(),
        "feature_sets": {
            name: {
                "train_shape": feature_metadata[name]["train_shape"],
                "test_shape": feature_metadata[name]["test_shape"],
                "n_columns": len(feature_metadata[name]["columns"]),
            }
            for name in feature_sets
        },
        "experiments": [asdict(exp) for exp in experiments],
    }
    save_json(run_dir / "manifest.json", metadata)

    print("Experiments:")
    for exp in experiments:
        print(f"- {exp.name} ({exp.model_type}, {exp.feature_set})")

    if args.dry_run:
        print("Dry run complete.")
        return

    folds = list(
        StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.fold_seed,
        ).split(np.zeros(len(y_enc)), y_enc)
    )

    summaries = []
    all_fold_rows = []
    for exp in experiments:
        summary, fold_rows = run_one_experiment(
            exp,
            feature_sets,
            y_enc,
            sample_weights,
            classes,
            folds,
            args,
            run_dir,
            layer1,
        )
        summaries.append(summary)
        all_fold_rows.append(fold_rows)
        summary_df = pd.DataFrame(summaries).sort_values(
            ["balanced_accuracy", "accuracy"],
            ascending=False,
        )
        summary_df.to_csv(run_dir / "summary_by_experiment.csv", index=False)
        pd.concat(all_fold_rows, ignore_index=True).to_csv(run_dir / "fold_metrics.csv", index=False)
        print(
            f"[{exp.name}] balanced={summary['balanced_accuracy']:.6f}, "
            f"acc={summary['accuracy']:.6f}, STAR recall={summary['recall_STAR']:.6f}, "
            f"QSO recall={summary['recall_QSO']:.6f}"
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=False,
    )
    summary_df.to_csv(run_dir / "summary_by_experiment.csv", index=False)
    write_summary_markdown(run_dir, summary_df, args)
    print("Sweep complete.")
    print(summary_df[["experiment", "balanced_accuracy", "accuracy", "log_loss", "recall_STAR", "recall_QSO"]].head(20))


if __name__ == "__main__":
    main()
