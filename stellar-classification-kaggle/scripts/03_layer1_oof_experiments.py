from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import CATEGORICAL_COLS, ID_COL, LABEL_COL, MAG_COLS
from stellar.features import (
    add_astronomy_features,
    add_local_guard_features,
    add_targeted_features,
)
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR
from stellar.submissions import register_submission


RAW_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_COLS = [
    "u_minus_g",
    "g_minus_r",
    "r_minus_i",
    "i_minus_z",
    "u_minus_r",
    "g_minus_i",
    "r_minus_z",
    "u_minus_z",
]


@dataclass(frozen=True)
class Experiment:
    name: str
    model_type: str
    feature_set: str
    seed: int
    params: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 5-fold layer-1 OOF predictions for stacking experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "layer1_oof")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--sample-rows", type=int, default=50000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=[
            "balanced",
            "lgbm_only",
            "full",
            "targeted",
            "selected_full",
            "winner_lgbm",
            "winner_xgb",
            "smoke",
        ],
        default="balanced",
    )
    parser.add_argument("--experiments", nargs="*", default=None)
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
    parser.add_argument(
        "--ensemble-strategies",
        nargs="*",
        default=["equal", "reduce_cat_star_lgbm"],
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "gpu", "auto"],
        default="cpu",
        help="Use GPU tree learners when available. Auto currently tries GPU settings.",
    )
    parser.add_argument("--save-single-submissions", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--skip-test-pred", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


def stratified_sample_train(
    train: pd.DataFrame,
    target: str,
    sample_rows: int,
    seed: int,
) -> pd.DataFrame:
    train = train.reset_index().rename(columns={"index": "source_index"})
    if sample_rows <= 0 or sample_rows >= len(train):
        return train.reset_index(drop=True)
    sampled, _ = train_test_split(
        train,
        train_size=sample_rows,
        stratify=train[target],
        random_state=seed,
    )
    return sampled.sort_values("source_index").reset_index(drop=True)


def harmonize_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    if cat_cols is None:
        cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]

    for col in cat_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        values = pd.concat([train_df[col], test_df[col]], axis=0).astype("string")
        values = values.fillna("__NA__")
        cats = sorted(values.unique().tolist())
        train_df[col] = pd.Categorical(
            train_df[col].astype("string").fillna("__NA__"), categories=cats
        )
        test_df[col] = pd.Categorical(
            test_df[col].astype("string").fillna("__NA__"), categories=cats
        )
    return train_df, test_df


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return df


def compute_sample_weights(
    train_sample: pd.DataFrame,
    args: argparse.Namespace,
) -> np.ndarray:
    weights = np.ones(len(train_sample), dtype="float32")
    if args.weighting == "none":
        return weights

    base = add_astronomy_features(train_sample.drop(columns=["source_index"]))
    if args.weighting in {"star_lowz", "star_lowz_hard"}:
        is_star = base[LABEL_COL].eq("STAR")
        is_m_star = is_star & base["spectral_type"].astype("string").eq("M")
        lowz_mid = base["redshift"].between(0.0497, 0.127, inclusive="right")
        hard_star_color = (
            is_star
            & (base["g_minus_r"] > 1.037)
            & (base["u_minus_z"] > 3.834)
            & ((base["mag_range"] > 3.949) | (base["mag_std"] > 1.624))
        )

        weights[is_star.to_numpy()] *= args.star_weight
        weights[is_m_star.to_numpy()] *= args.star_m_weight
        weights[lowz_mid.to_numpy()] *= args.lowz_weight
        weights[(is_star & lowz_mid).to_numpy()] *= args.star_lowz_weight
        weights[hard_star_color.to_numpy()] *= args.star_hard_color_weight

    if args.weighting == "star_lowz_hard":
        hard_ids = read_diagnostic_ids(args.diagnostic_dir / "diagnostic_hard_cases_top1000.csv")
        all_wrong_ids = read_all_wrong_ids(args.diagnostic_dir / "diagnostic_ensemble_error_rows.csv")
        if hard_ids:
            weights[train_sample[ID_COL].isin(hard_ids).to_numpy()] *= args.hard_case_weight
        if all_wrong_ids:
            weights[train_sample[ID_COL].isin(all_wrong_ids).to_numpy()] *= args.all_wrong_weight

    weights = np.clip(weights, 0.25, 8.0)
    weights /= float(np.mean(weights))
    return weights.astype("float32")


def read_diagnostic_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=[ID_COL])[ID_COL].astype(int).tolist())
    except Exception:
        return set()


def read_all_wrong_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        cols = [ID_COL, "model_correct_votes"]
        df = pd.read_csv(path, usecols=lambda col: col in cols)
    except Exception:
        return set()
    if "model_correct_votes" in df.columns:
        df = df[df["model_correct_votes"].eq(0)]
    return set(df[ID_COL].astype(int).tolist())


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-3

    if all(col in df.columns for col in MAG_COLS) and "mag_mean" in df.columns:
        for col in MAG_COLS:
            df[f"{col}_minus_mag_mean"] = df[col] - df["mag_mean"]
            df[f"{col}_over_mag_mean"] = df[col] / (df["mag_mean"].abs() + eps)

    for left, right in [
        ("u_minus_g", "g_minus_r"),
        ("g_minus_r", "r_minus_i"),
        ("r_minus_i", "i_minus_z"),
        ("u_minus_r", "r_minus_z"),
        ("g_minus_i", "i_minus_z"),
    ]:
        if left in df.columns and right in df.columns:
            df[f"{left}_over_{right}"] = df[left] / (df[right].abs() + eps)
            df[f"{left}_x_{right}"] = df[left] * df[right]

    if "redshift" in df.columns:
        redshift_abs = df["redshift"].abs()
        df["redshift_log1p_abs"] = np.log1p(redshift_abs)
        df["redshift_sq"] = df["redshift"] ** 2
        df["redshift_sin"] = np.sin(df["redshift"])
        df["redshift_cos"] = np.cos(df["redshift"])

    if "g_minus_r" in df.columns:
        df["g_minus_r_sin"] = np.sin(df["g_minus_r"])
        df["g_minus_r_cos"] = np.cos(df["g_minus_r"])

    if "mag_range" in df.columns and "redshift" in df.columns:
        df["mag_range_x_redshift"] = df["mag_range"] * df["redshift"]
        df["mag_range_over_redshift"] = df["mag_range"] / (df["redshift"].abs() + eps)

    return df


def load_autofe_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autofe_dir: Path,
    id_col: str,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_path = autofe_dir / "autofepg_selected_train.csv"
    test_path = autofe_dir / "autofepg_selected_test.csv"
    if not train_path.exists() or not test_path.exists():
        return train_df.copy(), test_df.copy(), []

    auto_train = pd.read_csv(train_path)
    auto_test = pd.read_csv(test_path)
    auto_cols = [c for c in auto_train.columns if c not in {id_col, target}]

    merged_train = train_df.merge(auto_train[[id_col] + auto_cols], on=id_col, how="left")
    merged_test = test_df.merge(auto_test[[id_col] + auto_cols], on=id_col, how="left")
    for col in auto_cols:
        merged_train[col] = merged_train[col].replace([np.inf, -np.inf], np.nan)
        merged_test[col] = merged_test[col].replace([np.inf, -np.inf], np.nan)
        fill_value = float(merged_train[col].median()) if merged_train[col].notna().any() else 0.0
        merged_train[col] = merged_train[col].fillna(fill_value).astype("float32")
        merged_test[col] = merged_test[col].fillna(fill_value).astype("float32")
    return merged_train, merged_test, auto_cols


def parse_group_key(
    key: str,
    all_X: pd.DataFrame,
    cache: dict[str, pd.Series],
) -> pd.Series | None:
    if key in cache:
        return cache[key]

    if key in all_X.columns:
        values = all_X[key]
        if pd.api.types.is_numeric_dtype(values):
            out = values.replace([np.inf, -np.inf], np.nan).fillna(-999999).astype("float32")
        else:
            out = values.astype("string").fillna("__NA__")
        cache[key] = out
        return out

    qbin_match = re.match(r"^qbin__(.+)__([0-9]+)$", key)
    if qbin_match:
        col = qbin_match.group(1)
        bins = int(qbin_match.group(2))
        if col not in all_X.columns:
            return None
        values = all_X[col].replace([np.inf, -np.inf], np.nan)
        try:
            binned = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        except Exception:
            return None
        out = pd.Series(binned, index=all_X.index).fillna(-1).astype("int16")
        cache[key] = out
        return out

    round_match = re.match(r"^round__(.+)__(-?[0-9]+)$", key)
    if round_match:
        col = round_match.group(1)
        decimals = int(round_match.group(2))
        if col not in all_X.columns:
            return None
        values = all_X[col].replace([np.inf, -np.inf], np.nan).fillna(-999999)
        out = values.round(decimals).astype("float32")
        cache[key] = out
        return out

    return None


def safe_group_feature_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:190]


def add_group_aggregate_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autofe_dir: Path,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    score_path = autofe_dir / "groupby_aggregate_feature_scores.csv"
    if top_n <= 0 or not score_path.exists():
        return train_df.copy(), test_df.copy(), []

    specs = pd.read_csv(score_path).head(top_n)
    all_X = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    n_train = len(train_df)
    cache: dict[str, pd.Series] = {}
    new_cols: list[str] = []
    derived = pd.DataFrame(index=all_X.index)

    for _, row in specs.iterrows():
        keys = str(row["keys"]).split("|")
        value_col = str(row["value_col"])
        stat = str(row["stat"])
        if value_col not in all_X.columns:
            continue

        key_series = []
        valid = True
        for key in keys:
            parsed = parse_group_key(key, all_X, cache)
            if parsed is None:
                valid = False
                break
            key_series.append(parsed.reset_index(drop=True))
        if not valid:
            continue

        feature_name = safe_group_feature_name(str(row["feature_name"]))
        if feature_name in all_X.columns or feature_name in derived.columns:
            feature_name = f"{feature_name}__gb"

        key_frame = pd.concat(key_series, axis=1)
        key_frame.columns = [f"k{i}" for i in range(len(keys))]
        train_keys = key_frame.iloc[:n_train].copy()
        train_values = (
            all_X[value_col].iloc[:n_train].replace([np.inf, -np.inf], np.nan)
        )
        global_value = getattr(train_values, stat)()
        if not np.isfinite(global_value):
            global_value = 0.0

        tmp = train_keys.copy()
        tmp["_value"] = train_values.values
        try:
            grouped = tmp.groupby(list(train_keys.columns), observed=True, dropna=False)[
                "_value"
            ].agg(stat)
        except Exception:
            continue

        if len(keys) == 1:
            mapped = key_frame.iloc[:, 0].map(grouped)
        else:
            index = pd.MultiIndex.from_frame(key_frame)
            mapped = pd.Series(index.map(grouped), index=all_X.index)

        values = (
            pd.Series(mapped, index=all_X.index)
            .astype("float64")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(float(global_value))
            .astype("float32")
        )
        if values.iloc[:n_train].nunique(dropna=False) <= 1:
            continue
        derived[feature_name] = values
        new_cols.append(feature_name)

    if not new_cols:
        return train_df.copy(), test_df.copy(), []

    train_out = pd.concat(
        [train_df.reset_index(drop=True), derived.iloc[:n_train].reset_index(drop=True)],
        axis=1,
    )
    test_out = pd.concat(
        [test_df.reset_index(drop=True), derived.iloc[n_train:].reset_index(drop=True)],
        axis=1,
    )
    return train_out, test_out, new_cols


def feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = [target, id_col, "source_index"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in [id_col] if c in test_df.columns])
    X_train, X_test = harmonize_categoricals(X_train, X_test)
    X_train = clean_numeric_frame(X_train)
    X_test = clean_numeric_frame(X_test)
    return X_train, X_test


def build_feature_sets(
    train_sample: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], dict]:
    raw_train = train_sample.copy()
    raw_test = test.copy()
    raw_X_train, raw_X_test = feature_matrix(raw_train, raw_test, LABEL_COL, ID_COL)

    baseline_train = add_astronomy_features(train_sample.drop(columns=["source_index"]))
    baseline_train["source_index"] = train_sample["source_index"].values
    baseline_test = add_astronomy_features(test)
    baseline_X_train, baseline_X_test = feature_matrix(
        baseline_train, baseline_test, LABEL_COL, ID_COL
    )

    extra_train = add_extra_features(baseline_train)
    extra_test = add_extra_features(baseline_test)
    extra_X_train, extra_X_test = feature_matrix(extra_train, extra_test, LABEL_COL, ID_COL)

    targeted_train = add_targeted_features(extra_train)
    targeted_test = add_targeted_features(extra_test)
    targeted_X_train, targeted_X_test = feature_matrix(
        targeted_train, targeted_test, LABEL_COL, ID_COL
    )

    local_guard_train = add_local_guard_features(targeted_train)
    local_guard_test = add_local_guard_features(targeted_test)
    local_guard_X_train, local_guard_X_test = feature_matrix(
        local_guard_train, local_guard_test, LABEL_COL, ID_COL
    )

    auto_train, auto_test, auto_cols = load_autofe_features(
        baseline_train,
        baseline_test,
        args.autofe_dir,
        ID_COL,
        LABEL_COL,
    )
    auto_X_train, auto_X_test = feature_matrix(auto_train, auto_test, LABEL_COL, ID_COL)

    group_train, group_test, group_cols = add_group_aggregate_features(
        baseline_train,
        baseline_test,
        args.autofe_dir,
        args.groupby_top_n,
    )
    group_X_train, group_X_test = feature_matrix(group_train, group_test, LABEL_COL, ID_COL)

    wide_train = extra_train.copy()
    wide_test = extra_test.copy()
    targeted_wide_train = targeted_train.copy()
    targeted_wide_test = targeted_test.copy()
    if auto_cols:
        auto_only_train = auto_train[[ID_COL] + auto_cols]
        auto_only_test = auto_test[[ID_COL] + auto_cols]
        wide_train = wide_train.merge(auto_only_train, on=ID_COL, how="left")
        wide_test = wide_test.merge(auto_only_test, on=ID_COL, how="left")
        targeted_wide_train = targeted_wide_train.merge(
            auto_only_train, on=ID_COL, how="left"
        )
        targeted_wide_test = targeted_wide_test.merge(
            auto_only_test, on=ID_COL, how="left"
        )
    if group_cols:
        group_only_train = group_train[[ID_COL] + group_cols]
        group_only_test = group_test[[ID_COL] + group_cols]
        wide_train = wide_train.merge(group_only_train, on=ID_COL, how="left")
        wide_test = wide_test.merge(group_only_test, on=ID_COL, how="left")
        targeted_wide_train = targeted_wide_train.merge(
            group_only_train, on=ID_COL, how="left"
        )
        targeted_wide_test = targeted_wide_test.merge(
            group_only_test, on=ID_COL, how="left"
        )
    wide_X_train, wide_X_test = feature_matrix(wide_train, wide_test, LABEL_COL, ID_COL)
    targeted_wide_X_train, targeted_wide_X_test = feature_matrix(
        targeted_wide_train, targeted_wide_test, LABEL_COL, ID_COL
    )

    feature_sets = {
        "raw": (raw_X_train, raw_X_test),
        "baseline": (baseline_X_train, baseline_X_test),
        "extra": (extra_X_train, extra_X_test),
        "targeted": (targeted_X_train, targeted_X_test),
        "local_guard": (local_guard_X_train, local_guard_X_test),
        "autofe": (auto_X_train, auto_X_test),
        "groupagg": (group_X_train, group_X_test),
        "wide": (wide_X_train, wide_X_test),
        "targeted_wide": (targeted_wide_X_train, targeted_wide_X_test),
    }
    metadata = {
        name: {
            "train_shape": list(frames[0].shape),
            "test_shape": list(frames[1].shape),
            "columns": frames[0].columns.tolist(),
        }
        for name, frames in feature_sets.items()
    }
    metadata["generated"] = {
        "autofe_columns": auto_cols,
        "groupagg_columns": group_cols,
        "targeted_columns": [
            c for c in targeted_X_train.columns if c not in extra_X_train.columns
        ],
        "local_guard_columns": [
            c for c in local_guard_X_train.columns if c not in targeted_X_train.columns
        ],
    }
    return feature_sets, metadata


def lgbm_params(seed: int, extra_trees: bool = False, n_estimators: int = 1500) -> dict:
    return {
        "objective": "multiclass",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 35,
        "subsample": 0.88,
        "subsample_freq": 1,
        "colsample_bytree": 0.86,
        "reg_alpha": 0.08,
        "reg_lambda": 1.4,
        "extra_trees": extra_trees,
        "random_state": seed,
        "verbosity": -1,
    }


def cat_params(seed: int, depth: int = 6, iterations: int = 1400) -> dict:
    return {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": iterations,
        "learning_rate": 0.045,
        "depth": depth,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.8,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.88,
        "random_seed": seed,
        "allow_writing_files": False,
    }


def xgb_params(seed: int, n_estimators: int = 1200) -> dict:
    return {
        "objective": "multi:softprob",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "max_depth": 5,
        "min_child_weight": 2.0,
        "subsample": 0.88,
        "colsample_bytree": 0.86,
        "reg_alpha": 0.08,
        "reg_lambda": 1.6,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": seed,
    }


def lgbm_conservative_params(seed: int) -> dict:
    params = lgbm_params(seed, extra_trees=False, n_estimators=1100)
    params.update(
        {
            "learning_rate": 0.025,
            "num_leaves": 48,
            "min_child_samples": 55,
            "colsample_bytree": 0.82,
            "reg_lambda": 2.2,
        }
    )
    return params


def xgb_star_recall_params(seed: int) -> dict:
    params = xgb_params(seed, n_estimators=800)
    params.update(
        {
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 0.8,
            "reg_alpha": 0.02,
            "reg_lambda": 1.0,
        }
    )
    return params


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    if args.suite == "smoke":
        experiments = [
            Experiment("lgbm_gbdt_s42_baseline", "lgbm", "baseline", 42, lgbm_params(42, False, 80)),
            Experiment("cat_s42_autofe", "catboost", "autofe", 42, cat_params(42, 6, 80)),
        ]
    elif args.suite == "lgbm_only":
        experiments = [
            Experiment("lgbm_gbdt_s42_baseline", "lgbm", "baseline", 42, lgbm_params(42)),
            Experiment("lgbm_xt_s2026_baseline", "lgbm", "baseline", 2026, lgbm_params(2026, True)),
            Experiment("lgbm_gbdt_s777_autofe", "lgbm", "autofe", 777, lgbm_params(777)),
            Experiment("lgbm_xt_s314_groupagg", "lgbm", "groupagg", 314, lgbm_params(314, True)),
            Experiment("lgbm_gbdt_s2027_wide", "lgbm", "wide", 2027, lgbm_params(2027)),
            Experiment("lgbm_gbdt_s919_targeted", "lgbm", "targeted", 919, lgbm_params(919)),
            Experiment("lgbm_gbdt_s920_targeted_wide", "lgbm", "targeted_wide", 920, lgbm_params(920)),
        ]
    elif args.suite == "full":
        experiments = [
            Experiment("lgbm_gbdt_s42_raw", "lgbm", "raw", 42, lgbm_params(42)),
            Experiment("lgbm_gbdt_s42_baseline", "lgbm", "baseline", 42, lgbm_params(42)),
            Experiment("lgbm_xt_s2026_baseline", "lgbm", "baseline", 2026, lgbm_params(2026, True)),
            Experiment("lgbm_gbdt_s777_autofe", "lgbm", "autofe", 777, lgbm_params(777)),
            Experiment("lgbm_xt_s314_groupagg", "lgbm", "groupagg", 314, lgbm_params(314, True)),
            Experiment("lgbm_gbdt_s2027_wide", "lgbm", "wide", 2027, lgbm_params(2027)),
            Experiment("lgbm_gbdt_s919_targeted", "lgbm", "targeted", 919, lgbm_params(919)),
            Experiment("lgbm_gbdt_s920_targeted_wide", "lgbm", "targeted_wide", 920, lgbm_params(920)),
            Experiment("cat_s42_baseline", "catboost", "baseline", 42, cat_params(42, 6)),
            Experiment("cat_s2026_autofe", "catboost", "autofe", 2026, cat_params(2026, 7)),
            Experiment("cat_s777_groupagg", "catboost", "groupagg", 777, cat_params(777, 6)),
            Experiment("cat_s920_targeted", "catboost", "targeted", 920, cat_params(920, 6)),
            Experiment("xgb_s42_baseline", "xgboost", "baseline", 42, xgb_params(42)),
            Experiment("xgb_s2026_wide", "xgboost", "wide", 2026, xgb_params(2026)),
            Experiment("xgb_s920_targeted_wide", "xgboost", "targeted_wide", 920, xgb_params(920)),
        ]
    elif args.suite == "targeted":
        experiments = [
            Experiment("lgbm_gbdt_s919_targeted", "lgbm", "targeted", 919, lgbm_params(919)),
            Experiment("lgbm_xt_s921_targeted", "lgbm", "targeted", 921, lgbm_params(921, True)),
            Experiment("lgbm_gbdt_s920_targeted_wide", "lgbm", "targeted_wide", 920, lgbm_params(920)),
            Experiment("cat_s920_targeted", "catboost", "targeted", 920, cat_params(920, 6)),
            Experiment("cat_s921_targeted_wide", "catboost", "targeted_wide", 921, cat_params(921, 7)),
            Experiment("xgb_s921_targeted", "xgboost", "targeted", 921, xgb_params(921)),
            Experiment("xgb_s920_targeted_wide", "xgboost", "targeted_wide", 920, xgb_params(920)),
        ]
    elif args.suite == "selected_full":
        experiments = [
            Experiment("xgb_s42_baseline", "xgboost", "baseline", 42, xgb_params(42)),
            Experiment("lgbm_gbdt_s777_autofe", "lgbm", "autofe", 777, lgbm_params(777)),
            Experiment("xgb_s2026_wide", "xgboost", "wide", 2026, xgb_params(2026)),
            Experiment("cat_s2026_autofe", "catboost", "autofe", 2026, cat_params(2026, 7)),
            Experiment("lgbm_gbdt_s920_targeted_wide", "lgbm", "targeted_wide", 920, lgbm_params(920)),
            Experiment("cat_s921_targeted_wide", "catboost", "targeted_wide", 921, cat_params(921, 7)),
            Experiment("xgb_s920_targeted_wide", "xgboost", "targeted_wide", 920, xgb_params(920)),
        ]
    elif args.suite == "winner_lgbm":
        experiments = [
            Experiment(
                "lgbm_conservative_targeted_wide_s2756",
                "lgbm",
                "targeted_wide",
                2756,
                lgbm_conservative_params(2756),
            ),
        ]
    elif args.suite == "winner_xgb":
        experiments = [
            Experiment(
                "xgboost_star_recall_targeted_s2703",
                "xgboost",
                "targeted",
                2703,
                xgb_star_recall_params(2703),
            ),
        ]
    else:
        experiments = [
            Experiment("lgbm_gbdt_s42_baseline", "lgbm", "baseline", 42, lgbm_params(42)),
            Experiment("lgbm_xt_s2026_baseline", "lgbm", "baseline", 2026, lgbm_params(2026, True)),
            Experiment("lgbm_gbdt_s777_autofe", "lgbm", "autofe", 777, lgbm_params(777)),
            Experiment("lgbm_xt_s314_groupagg", "lgbm", "groupagg", 314, lgbm_params(314, True)),
            Experiment("lgbm_gbdt_s919_targeted", "lgbm", "targeted", 919, lgbm_params(919)),
            Experiment("lgbm_gbdt_s920_targeted_wide", "lgbm", "targeted_wide", 920, lgbm_params(920)),
            Experiment("cat_s42_baseline", "catboost", "baseline", 42, cat_params(42, 6)),
            Experiment("cat_s2026_autofe", "catboost", "autofe", 2026, cat_params(2026, 7)),
            Experiment("xgb_s42_baseline", "xgboost", "baseline", 42, xgb_params(42)),
            Experiment("xgb_s2026_wide", "xgboost", "wide", 2026, xgb_params(2026)),
            Experiment("xgb_s920_targeted_wide", "xgboost", "targeted_wide", 920, xgb_params(920)),
        ]

    if args.experiments:
        wanted = set(args.experiments)
        experiments = [exp for exp in experiments if exp.name in wanted]
    return experiments


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(df[c].dtype) == "category" or c in CATEGORICAL_COLS]


def prepare_xgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if str(out[col].dtype) == "category":
            out[col] = out[col].cat.codes.astype("int16")
        elif pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype("string").fillna("__NA__"))[0].astype("int16")
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_catboost_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__NA__").astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_predict_fold_impl(
    exp: Experiment,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    w_valid: np.ndarray | None,
    X_test: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, object, dict]:
    started = time.time()
    model = None

    if exp.model_type == "lgbm":
        import lightgbm as lgb

        params = exp.params.copy()
        params["n_jobs"] = args.n_jobs
        if args.accelerator in {"gpu", "auto"}:
            params["device_type"] = "gpu"
        model = lgb.LGBMClassifier(**params)
        cat_cols = categorical_columns(X_train)
        callbacks = [
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid] if w_valid is not None else None,
            eval_metric="multi_logloss",
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=callbacks,
        )
        valid_proba = model.predict_proba(X_valid)
        test_proba = model.predict_proba(X_test) if X_test is not None else None
        best_iteration = getattr(model, "best_iteration_", None)

    elif exp.model_type == "catboost":
        from catboost import CatBoostClassifier

        params = exp.params.copy()
        params["thread_count"] = args.n_jobs
        params["od_type"] = "Iter"
        params["od_wait"] = args.early_stopping_rounds
        if args.accelerator in {"gpu", "auto"}:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        model = CatBoostClassifier(**params)
        cat_cols = categorical_columns(X_train)
        X_train_cat = prepare_catboost_frame(X_train, cat_cols)
        X_valid_cat = prepare_catboost_frame(X_valid, cat_cols)
        X_test_cat = prepare_catboost_frame(X_test, cat_cols) if X_test is not None else None
        model.fit(
            X_train_cat,
            y_train,
            sample_weight=w_train,
            eval_set=(X_valid_cat, y_valid),
            cat_features=cat_cols,
            use_best_model=True,
            verbose=False,
        )
        valid_proba = model.predict_proba(X_valid_cat)
        test_proba = model.predict_proba(X_test_cat) if X_test_cat is not None else None
        best_iteration = model.get_best_iteration()

    elif exp.model_type == "xgboost":
        from xgboost import XGBClassifier

        params = exp.params.copy()
        params["n_jobs"] = args.n_jobs
        params["early_stopping_rounds"] = args.early_stopping_rounds
        if args.accelerator in {"gpu", "auto"}:
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        model = XGBClassifier(**params)
        X_train_xgb = prepare_xgb_frame(X_train)
        X_valid_xgb = prepare_xgb_frame(X_valid)
        X_test_xgb = prepare_xgb_frame(X_test) if X_test is not None else None
        model.fit(
            X_train_xgb,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_valid_xgb, y_valid)],
            sample_weight_eval_set=[w_valid] if w_valid is not None else None,
            verbose=False,
        )
        valid_proba = model.predict_proba(X_valid_xgb)
        test_proba = model.predict_proba(X_test_xgb) if X_test_xgb is not None else None
        best_iteration = getattr(model, "best_iteration", None)

    else:
        raise ValueError(f"Unknown model_type: {exp.model_type}")

    elapsed = time.time() - started
    valid_pred = valid_proba.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_valid, valid_pred),
        "balanced_accuracy": balanced_accuracy_score(y_valid, valid_pred),
        "log_loss": log_loss(y_valid, valid_proba, labels=np.arange(valid_proba.shape[1])),
        "fit_predict_seconds": elapsed,
        "best_iteration": best_iteration,
    }
    return valid_proba, test_proba, model, metrics


def fit_predict_fold(
    exp: Experiment,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    w_valid: np.ndarray | None,
    X_test: pd.DataFrame | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, object, dict]:
    try:
        return _fit_predict_fold_impl(
            exp,
            X_train,
            y_train,
            w_train,
            X_valid,
            y_valid,
            w_valid,
            X_test,
            args,
        )
    except Exception as exc:
        if args.accelerator != "auto":
            raise
        print(
            f"[{exp.name}] GPU/auto fit failed with {type(exc).__name__}: {exc}. "
            "Retrying this fold on CPU."
        )
        cpu_args = argparse.Namespace(**vars(args))
        cpu_args.accelerator = "cpu"
        return _fit_predict_fold_impl(
            exp,
            X_train,
            y_train,
            w_train,
            X_valid,
            y_valid,
            w_valid,
            X_test,
            cpu_args,
        )


def save_single_submission(
    path: Path,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    test_proba: np.ndarray,
    classes: np.ndarray,
) -> None:
    submission = sample_submission.copy()
    submission[ID_COL] = test_ids.values
    submission[LABEL_COL] = classes[test_proba.argmax(axis=1)]
    submission.to_csv(path, index=False)


def overall_metrics_from_rows(rows: list[dict]) -> dict:
    for row in rows:
        if str(row.get("fold")) == "overall":
            return {
                key: row[key]
                for key in ["accuracy", "balanced_accuracy", "log_loss"]
                if key in row
            }
    return {}


def ensemble_model_weights(strategy: str, experiments: list[Experiment]) -> dict[str, float]:
    weights = {exp.name: 1.0 for exp in experiments}
    if strategy == "equal":
        return weights
    if strategy == "reduce_cat_star_lgbm":
        for exp in experiments:
            if exp.model_type == "catboost":
                weights[exp.name] = 0.55
            elif exp.name in {"lgbm_gbdt_s777_autofe", "lgbm_gbdt_s42_baseline"}:
                weights[exp.name] = 1.35
            elif "targeted" in exp.feature_set and exp.model_type == "lgbm":
                weights[exp.name] = 1.25
            elif exp.model_type == "xgboost":
                weights[exp.name] = 1.05
    return weights


def build_probability_matrix(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    classes: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    total_weight = 0.0
    proba = np.zeros((len(frame), len(classes)), dtype="float64")
    for exp in experiments:
        weight = float(weights.get(exp.name, 1.0))
        cols = [f"l1__{exp.name}__{cls}" for cls in classes]
        if not all(col in frame.columns for col in cols):
            continue
        proba += frame[cols].to_numpy(dtype="float64") * weight
        total_weight += weight
    if total_weight == 0:
        raise ValueError("No layer-1 probability columns found for ensemble.")
    proba /= total_weight
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype("float32")


def save_weighted_ensembles(
    run_dir: Path,
    oof_train: pd.DataFrame,
    oof_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    y_enc: np.ndarray,
    classes: np.ndarray,
    experiments: list[Experiment],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    ensemble_dir = run_dir / "ensembles"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for strategy in args.ensemble_strategies:
        weights = ensemble_model_weights(strategy, experiments)
        train_proba = build_probability_matrix(oof_train, experiments, classes, weights)
        train_pred = train_proba.argmax(axis=1)
        row = {
            "strategy": strategy,
            "accuracy": accuracy_score(y_enc, train_pred),
            "balanced_accuracy": balanced_accuracy_score(y_enc, train_pred),
            "log_loss": log_loss(y_enc, train_proba, labels=np.arange(len(classes))),
            "weights": json.dumps(weights, ensure_ascii=False),
        }
        rows.append(row)

        pd.DataFrame(
            {
                ID_COL: oof_train[ID_COL].values,
                LABEL_COL: oof_train[LABEL_COL].values,
                **{f"ens__{strategy}__{cls}": train_proba[:, i] for i, cls in enumerate(classes)},
            }
        ).to_csv(ensemble_dir / f"{strategy}_oof_train.csv", index=False)

        if not args.skip_test_pred:
            test_proba = build_probability_matrix(oof_test, experiments, classes, weights)
            pd.DataFrame(
                {
                    ID_COL: test_ids.values,
                    **{f"ens__{strategy}__{cls}": test_proba[:, i] for i, cls in enumerate(classes)},
                }
            ).to_csv(ensemble_dir / f"{strategy}_oof_test.csv", index=False)
            save_single_submission(
                ensemble_dir / f"{strategy}_submission.csv",
                sample_submission,
                test_ids,
                test_proba,
                classes,
            )
            register_submission(
                ensemble_dir / f"{strategy}_submission.csv",
                run_dir=run_dir,
                script=Path(__file__).name,
                submission_type="layer1_weighted_ensemble",
                model_name=strategy,
                metrics={
                    "accuracy": row["accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "log_loss": row["log_loss"],
                },
                params={
                    "weights": weights,
                    "suite": args.suite,
                    "sample_rows": args.sample_rows,
                    "n_splits": args.n_splits,
                    "weighting": args.weighting,
                    "accelerator": args.accelerator,
                },
                extra={"ensemble_scores_path": "ensemble_scores.csv"},
            )

    result = pd.DataFrame(rows).sort_values(
        ["accuracy", "balanced_accuracy"], ascending=False
    )
    result.to_csv(run_dir / "ensemble_scores.csv", index=False)
    return result


def write_run_summary(run_dir: Path, metadata: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# Layer-1 OOF Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Data",
        "",
        f"- Train sample shape: {metadata['train_sample_shape']}",
        f"- Test shape: {metadata['test_shape']}",
        f"- Classes: {metadata['classes']}",
        f"- Folds: {metadata['args']['n_splits']}",
        "",
        "## Outputs",
        "",
        "- `oof_train.csv`: first-layer OOF probabilities for the sampled train rows.",
        "- `oof_test.csv`: test probabilities averaged across the 5 folds of each experiment.",
        "- `summary_overall.csv`: overall OOF metrics per experiment.",
        "- `experiment_scores.csv`: fold-level and overall metrics.",
        "- `experiments/<name>/`: per-experiment OOF files, config, and fold metrics.",
        "",
        "## Overall Scores",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['experiment']}` ({row['feature_set']}): "
            f"accuracy={row['accuracy']:.6f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}, "
            f"log_loss={row['log_loss']:.6f}"
        )
    lines.append("")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    exp: Experiment,
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    y_enc: np.ndarray,
    sample_weights: np.ndarray,
    classes: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    oof_train: pd.DataFrame,
    oof_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    test_ids: pd.Series,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict]:
    X, X_test = feature_sets[exp.feature_set]
    n_classes = len(classes)
    oof_proba = np.zeros((len(X), n_classes), dtype="float32")
    test_proba_sum = np.zeros((len(X_test), n_classes), dtype="float32")
    fold_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        print(f"[{exp.name}] fold {fold}/{len(folds)}")
        valid_proba, test_proba, model, metrics = fit_predict_fold(
            exp=exp,
            X_train=X.iloc[train_idx].reset_index(drop=True),
            y_train=y_enc[train_idx],
            w_train=sample_weights[train_idx] if sample_weights is not None else None,
            X_valid=X.iloc[valid_idx].reset_index(drop=True),
            y_valid=y_enc[valid_idx],
            w_valid=sample_weights[valid_idx] if sample_weights is not None else None,
            X_test=X_test if not args.skip_test_pred else None,
            args=args,
        )
        oof_proba[valid_idx] = valid_proba.astype("float32")
        if test_proba is not None:
            test_proba_sum += test_proba.astype("float32") / len(folds)

        row = {
            "experiment": exp.name,
            "model_type": exp.model_type,
            "feature_set": exp.feature_set,
            "seed": exp.seed,
            "fold": fold,
            **metrics,
        }
        fold_rows.append(row)

        if args.save_fold_models:
            import joblib

            model_dir = run_dir / "fold_models" / exp.name
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / f"fold{fold}.joblib")

    prefix = f"l1__{exp.name}"
    for i, cls in enumerate(classes):
        col = f"{prefix}__{cls}"
        oof_train[col] = oof_proba[:, i]
        if not args.skip_test_pred:
            oof_test[col] = test_proba_sum[:, i]

    overall_pred = oof_proba.argmax(axis=1)
    overall = {
        "experiment": exp.name,
        "model_type": exp.model_type,
        "feature_set": exp.feature_set,
        "seed": exp.seed,
        "fold": "overall",
        "accuracy": accuracy_score(y_enc, overall_pred),
        "balanced_accuracy": balanced_accuracy_score(y_enc, overall_pred),
        "log_loss": log_loss(y_enc, oof_proba, labels=np.arange(n_classes)),
        "fit_predict_seconds": sum(row["fit_predict_seconds"] for row in fold_rows),
        "best_iteration": None,
    }
    fold_rows.append(overall)

    exp_dir = run_dir / "experiments" / exp.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COL: oof_train[ID_COL].values,
            LABEL_COL: oof_train[LABEL_COL].values,
            **{f"{prefix}__{cls}": oof_proba[:, i] for i, cls in enumerate(classes)},
        }
    ).to_csv(exp_dir / "oof_train.csv", index=False)
    if not args.skip_test_pred:
        pd.DataFrame(
            {
                ID_COL: test_ids.values,
                **{f"{prefix}__{cls}": test_proba_sum[:, i] for i, cls in enumerate(classes)},
            }
        ).to_csv(exp_dir / "oof_test.csv", index=False)
        if args.save_single_submissions:
            submission_dir = run_dir / "single_model_submissions"
            submission_dir.mkdir(parents=True, exist_ok=True)
            save_single_submission(
                submission_dir / f"{exp.name}.csv",
                sample_submission,
                test_ids,
                test_proba_sum,
                classes,
            )
            register_submission(
                submission_dir / f"{exp.name}.csv",
                run_dir=run_dir,
                script=Path(__file__).name,
                submission_type="layer1_single_model",
                model_name=exp.name,
                metrics=overall_metrics_from_rows(fold_rows),
                params={
                    "experiment": asdict(exp),
                    "suite": args.suite,
                    "sample_rows": args.sample_rows,
                    "n_splits": args.n_splits,
                    "weighting": args.weighting,
                    "accelerator": args.accelerator,
                },
                extra={"fold_metrics_path": f"experiments/{exp.name}/fold_metrics.csv"},
            )

    save_json(exp_dir / "config.json", asdict(exp))
    pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)
    print(
        f"[{exp.name}] OOF accuracy={overall['accuracy']:.6f}, "
        f"balanced={overall['balanced_accuracy']:.6f}, log_loss={overall['log_loss']:.6f}"
    )
    return fold_rows


def main() -> None:
    args = parse_args()
    if args.suite == "smoke":
        args.sample_rows = min(args.sample_rows, 2000)
        args.groupby_top_n = min(args.groupby_top_n, 4)
        args.early_stopping_rounds = min(args.early_stopping_rounds, 20)

    run_dir = make_run_dir(args.output_dir, args.run_name)
    (run_dir / "experiments").mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir.resolve()}")

    train, test, sample_submission = read_inputs(args.data_dir)
    train_sample = stratified_sample_train(
        train, LABEL_COL, args.sample_rows, args.sample_seed
    )
    y = train_sample[LABEL_COL].copy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    classes = label_encoder.classes_

    feature_sets, feature_metadata = build_feature_sets(train_sample, test, args)
    experiments = build_experiments(args)
    experiments = [exp for exp in experiments if exp.feature_set in feature_sets]
    sample_weights = compute_sample_weights(train_sample, args)

    metadata = {
        "args": vars(args),
        "train_full_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_sample_shape": list(train_sample.shape),
        "class_counts_sample": y.value_counts().to_dict(),
        "classes": classes.tolist(),
        "feature_sets": feature_metadata,
        "sample_weight_summary": {
            "min": float(sample_weights.min()),
            "mean": float(sample_weights.mean()),
            "max": float(sample_weights.max()),
            "std": float(sample_weights.std()),
            "weighting": args.weighting,
        },
        "experiments": [asdict(exp) for exp in experiments],
    }
    save_json(run_dir / "manifest.json", metadata)
    save_json(
        run_dir / "feature_set_columns.json",
        {name: frames[0].columns.tolist() for name, frames in feature_sets.items()},
    )

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str)[:6000])
        print("Dry run complete. No models were trained.")
        return

    folds = list(
        StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.fold_seed,
        ).split(np.zeros(len(y_enc)), y_enc)
    )

    oof_train = pd.DataFrame(
        {
            ID_COL: train_sample[ID_COL].values,
            "source_index": train_sample["source_index"].values,
            LABEL_COL: y.values,
        }
    )
    oof_test = pd.DataFrame({ID_COL: test[ID_COL].values})
    all_rows: list[dict] = []

    for exp in experiments:
        started = time.time()
        print(f"Starting experiment: {exp.name} ({exp.model_type}, {exp.feature_set})")
        rows = run_experiment(
            exp,
            feature_sets,
            y_enc,
            sample_weights,
            classes,
            folds,
            oof_train,
            oof_test,
            sample_submission,
            test[ID_COL],
            args,
            run_dir,
        )
        for row in rows:
            row["experiment_wall_seconds"] = time.time() - started
        all_rows.extend(rows)

        oof_train.to_csv(run_dir / "oof_train.csv", index=False)
        if not args.skip_test_pred:
            oof_test.to_csv(run_dir / "oof_test.csv", index=False)
        pd.DataFrame(all_rows).to_csv(run_dir / "experiment_scores.csv", index=False)

    summary = (
        pd.DataFrame(all_rows)
        .query("fold == 'overall'")
        .sort_values(["accuracy", "balanced_accuracy"], ascending=False)
    )
    summary.to_csv(run_dir / "summary_overall.csv", index=False)
    ensemble_summary = save_weighted_ensembles(
        run_dir,
        oof_train,
        oof_test,
        sample_submission,
        test[ID_COL],
        y_enc,
        classes,
        experiments,
        args,
    )
    write_run_summary(run_dir, metadata, summary)
    print("Layer-1 OOF complete.")
    print(summary[["experiment", "feature_set", "accuracy", "balanced_accuracy", "log_loss"]])
    print("Weighted ensembles:")
    print(ensemble_summary[["strategy", "accuracy", "balanced_accuracy", "log_loss"]])
    print(f"OOF train: {(run_dir / 'oof_train.csv').resolve()}")
    if not args.skip_test_pred:
        print(f"OOF test: {(run_dir / 'oof_test.csv').resolve()}")


if __name__ == "__main__":
    main()
