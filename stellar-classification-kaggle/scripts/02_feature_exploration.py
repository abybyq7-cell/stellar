from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import CATEGORICAL_COLS, ID_COL, LABEL_COL
from stellar.features import add_astronomy_features
from stellar.io import make_run_dir, save_json
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR


RAW_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
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
MAG_STAT_COLS = ["mag_mean", "mag_std", "mag_min", "mag_max", "mag_range"]
REDSHIFT_INTERACTION_COLS = [
    "redshift_x_mag_mean",
    "u_redshift_ratio",
    "g_redshift_ratio",
    "r_redshift_ratio",
    "i_redshift_ratio",
    "z_redshift_ratio",
]
ANGLE_CYCLE_COLS = ["alpha_sin", "alpha_cos", "delta_sin", "delta_cos"]
PREFERRED_NUM_COLS = (
    RAW_NUM_COLS
    + COLOR_COLS
    + MAG_STAT_COLS
    + REDSHIFT_INTERACTION_COLS
    + ANGLE_CYCLE_COLS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore stellar features with autofepg AutoFE and groupby scans."
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR / "feature_exploration"
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", type=str, default=LABEL_COL)
    parser.add_argument("--id-col", type=str, default=ID_COL)
    parser.add_argument("--no-baseline-features", action="store_true")
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--skip-autofe", action="store_true")
    parser.add_argument("--autofe-sample", type=int, default=30000)
    parser.add_argument("--autofe-time-limit", type=int, default=300)
    parser.add_argument("--autofe-folds", type=int, default=3)
    parser.add_argument("--autofe-max-pair-cols", type=int, default=8)
    parser.add_argument("--autofe-max-num-cols", type=int, default=18)
    parser.add_argument("--autofe-max-digit-positions", type=int, default=2)
    parser.add_argument("--autofe-max-decimal-positions", type=int, default=1)
    parser.add_argument("--autofe-max-digit-interaction-order", type=int, default=2)
    parser.add_argument("--autofe-quantile-bins", type=int, nargs="+", default=[5, 10])
    parser.add_argument(
        "--autofe-rounding-decimals", type=int, nargs="+", default=[-1, 0, 1]
    )
    parser.add_argument("--autofe-gp-components", type=int, default=1)
    parser.add_argument("--autofe-gp-generations", type=int, default=1)
    parser.add_argument("--xgb-n-estimators", type=int, default=80)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--no-save-autofe-matrices", action="store_true")

    parser.add_argument("--skip-groupby", action="store_true")
    parser.add_argument("--groupby-max-num-bin-cols", type=int, default=26)
    parser.add_argument("--groupby-bin-counts", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--groupby-max-degree", type=int, default=3)
    parser.add_argument("--groupby-cross-bin-limit", type=int, default=24)
    parser.add_argument("--groupby-max-candidates", type=int, default=1200)
    parser.add_argument("--groupby-min-train-count", type=int, default=30)
    parser.add_argument("--top-groups-per-key", type=int, default=20)
    parser.add_argument("--agg-top-keys", type=int, default=25)
    parser.add_argument("--agg-max-value-cols", type=int, default=22)
    parser.add_argument("--agg-stats", nargs="+", default=["mean", "std", "count"])
    return parser.parse_args()


def safe_name(parts: Iterable[str], max_len: int = 180) -> str:
    raw = "__".join(parts)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return cleaned[:max_len]


def read_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def build_base_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    id_col: str,
    use_baseline_features: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if use_baseline_features:
        train = add_astronomy_features(train)
        test = add_astronomy_features(test)
    else:
        train = train.copy()
        test = test.copy()

    y = train[target].copy()
    X_train = train.drop(columns=[id_col, target])
    X_test = test.drop(columns=[id_col])

    for col in CATEGORICAL_COLS:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    return X_train, X_test, y


def existing_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != LABEL_COL and c != ID_COL
    ]


def ordered_subset(existing: list[str], preferred: list[str], max_cols: int) -> list[str]:
    ordered = [c for c in preferred if c in existing]
    ordered.extend(c for c in existing if c not in ordered)
    if max_cols > 0:
        return ordered[:max_cols]
    return ordered


def multiclass_accuracy_metric(y_true, y_pred, **_: object) -> float:
    pred = np.asarray(y_pred)
    if pred.ndim == 2:
        pred = np.argmax(pred, axis=1)
    return accuracy_score(y_true, pred)


def run_autofe(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    train_ids: pd.Series,
    test_ids: pd.Series,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    from autofepg import AutoFE

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    X_train_autofe = X_train.copy()
    X_test_autofe = X_test.copy()
    for col in cat_cols:
        X_train_autofe[col] = X_train_autofe[col].astype("object")
        X_test_autofe[col] = X_test_autofe[col].astype("object")

    numeric_cols = existing_numeric_cols(X_train)
    autofe_num_cols = ordered_subset(
        numeric_cols, PREFERRED_NUM_COLS, args.autofe_max_num_cols
    )

    xgb_params = {
        "n_estimators": args.xgb_n_estimators,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_learning_rate,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "early_stopping_rounds": 20,
    }

    started = time.time()
    autofe = AutoFE(
        task="classification",
        n_folds=args.autofe_folds,
        time_budget=args.autofe_time_limit,
        random_state=args.seed,
        metric_fn=multiclass_accuracy_metric,
        metric_direction="maximize",
        max_pair_cols=args.autofe_max_pair_cols,
        max_digit_positions=args.autofe_max_digit_positions,
        max_decimal_positions=args.autofe_max_decimal_positions,
        max_digit_interaction_order=args.autofe_max_digit_interaction_order,
        rounding_decimals=args.autofe_rounding_decimals,
        quantile_bins=args.autofe_quantile_bins,
        gp_n_components=args.autofe_gp_components,
        gp_generations=args.autofe_gp_generations,
        verbose=True,
        xgb_params=xgb_params,
        improvement_threshold=1e-5,
        sample=args.autofe_sample,
        backward_selection=False,
        report_path=str(run_dir / "autofepg_report.txt"),
    )
    X_train_new, X_test_new = autofe.fit_select(
        X_train_autofe,
        y,
        X_test_autofe,
        cat_cols=cat_cols,
        num_cols=autofe_num_cols,
    )
    elapsed = time.time() - started

    base_cols = set(X_train.columns)
    selected_cols = [c for c in X_train_new.columns if c not in base_cols]
    history = pd.DataFrame(autofe.history_)
    details = pd.DataFrame(autofe.selection_details_)
    history.to_csv(run_dir / "autofepg_history.csv", index=False)
    details.to_csv(run_dir / "autofepg_selected_details.csv", index=False)

    if selected_cols and not args.no_save_autofe_matrices:
        train_selected = pd.concat(
            [train_ids.rename(ID_COL), y.rename(LABEL_COL), X_train_new[selected_cols]],
            axis=1,
        )
        test_selected = pd.concat(
            [test_ids.rename(ID_COL), X_test_new[selected_cols]],
            axis=1,
        )
        train_selected.to_csv(run_dir / "autofepg_selected_train.csv", index=False)
        test_selected.to_csv(run_dir / "autofepg_selected_test.csv", index=False)

    summary = {
        "elapsed_seconds": elapsed,
        "base_score": autofe.base_score_,
        "base_score_std": autofe.base_score_std_,
        "best_score": autofe.best_score_,
        "best_score_std": autofe.best_score_std_,
        "selected_feature_count": len(selected_cols),
        "selected_feature_columns": selected_cols,
        "selected_generator_names": [g.name for g in autofe.selected_generators_],
        "cat_cols": cat_cols,
        "candidate_num_cols": autofe_num_cols,
        "history_rows": len(history),
    }
    save_json(run_dir / "autofepg_summary.json", summary)
    return summary


def make_binned_keys(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    n_train = len(X_train)
    key_df = pd.DataFrame(index=all_X.index)

    cat_keys = [c for c in CATEGORICAL_COLS if c in all_X.columns]
    for col in cat_keys:
        key_df[col] = all_X[col].astype("string").fillna("__NA__").astype("category")

    numeric_cols = existing_numeric_cols(all_X)
    num_key_cols = ordered_subset(
        numeric_cols, PREFERRED_NUM_COLS, args.groupby_max_num_bin_cols
    )

    bin_keys: list[str] = []
    for col in num_key_cols:
        vals = all_X[col].replace([np.inf, -np.inf], np.nan)
        for bins in args.groupby_bin_counts:
            name = f"qbin__{col}__{bins}"
            try:
                binned = pd.qcut(vals, q=bins, labels=False, duplicates="drop")
                key_df[name] = (
                    pd.Series(binned, index=all_X.index)
                    .astype("float64")
                    .fillna(-1)
                    .astype("int16")
                )
                bin_keys.append(name)
            except Exception:
                continue

    rounded_specs = {
        "alpha": [0, -1],
        "delta": [0],
        "redshift": [1, 2],
        "u": [0, 1],
        "g": [0, 1],
        "r": [0, 1],
        "i": [0, 1],
        "z": [0, 1],
        "u_minus_g": [1],
        "g_minus_r": [1],
        "r_minus_i": [1],
        "i_minus_z": [1],
        "mag_mean": [1],
        "mag_range": [1],
    }
    for col, decimals_list in rounded_specs.items():
        if col not in all_X.columns:
            continue
        vals = all_X[col].replace([np.inf, -np.inf], np.nan).fillna(-999999)
        for decimals in decimals_list:
            name = f"round__{col}__{decimals}"
            key_df[name] = vals.round(decimals).astype("float32")
            bin_keys.append(name)

    key_train = key_df.iloc[:n_train].reset_index(drop=True)
    key_test = key_df.iloc[n_train:].reset_index(drop=True)
    return key_train, key_test, cat_keys, bin_keys


def build_group_candidates(
    cat_keys: list[str],
    bin_keys: list[str],
    args: argparse.Namespace,
) -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = []

    def add(combo: Iterable[str]) -> None:
        item = tuple(combo)
        if item not in candidates:
            candidates.append(item)

    for key in cat_keys + bin_keys:
        add([key])

    if args.groupby_max_degree >= 2:
        for combo in combinations(cat_keys, 2):
            add(combo)
        for cat in cat_keys:
            for b in bin_keys:
                add([cat, b])
        for combo in combinations(bin_keys[: args.groupby_cross_bin_limit], 2):
            add(combo)

    if args.groupby_max_degree >= 3:
        for combo in combinations(cat_keys, 2):
            for b in bin_keys:
                add([combo[0], combo[1], b])
        for cat in cat_keys:
            for combo in combinations(bin_keys[: max(args.groupby_cross_bin_limit // 2, 2)], 2):
                add([cat, combo[0], combo[1]])

    return candidates[: args.groupby_max_candidates]


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totals, where=totals > 0)
    probs = np.where(probs > 0, probs, 1.0)
    entropy = -(probs * np.log2(probs)).sum(axis=1)
    return entropy


def score_group_key(
    keys: tuple[str, ...],
    key_train: pd.DataFrame,
    key_test: pd.DataFrame,
    y: pd.Series,
    classes: list[str],
    min_train_count: int,
) -> tuple[dict, pd.DataFrame]:
    tmp = key_train.loc[:, list(keys)].copy()
    tmp["_target"] = y.values
    target_counts = (
        tmp.groupby(list(keys), observed=True, dropna=False)["_target"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
    )
    train_counts = target_counts.sum(axis=1)
    test_counts = key_test.groupby(list(keys), observed=True, dropna=False).size()

    train_freq = train_counts / len(key_train)
    test_freq = test_counts / len(key_test)
    aligned_train, aligned_test = train_freq.align(test_freq, join="outer", fill_value=0)
    freq_tvd = 0.5 * np.abs(aligned_train.values - aligned_test.values).sum()
    unseen_idx = test_counts.index.difference(train_counts.index)
    test_unseen_rate = (
        float(test_counts.loc[unseen_idx].sum() / len(key_test))
        if len(unseen_idx) > 0
        else 0.0
    )

    counts_arr = target_counts.to_numpy(dtype=float)
    group_sizes = counts_arr.sum(axis=1)
    valid_group_mask = group_sizes >= min_train_count
    class_probs = np.divide(
        counts_arr,
        group_sizes.reshape(-1, 1),
        where=group_sizes.reshape(-1, 1) > 0,
    )
    max_rates = class_probs.max(axis=1)
    dominant_idx = class_probs.argmax(axis=1)
    entropy = entropy_from_counts(counts_arr)
    normalizer = math.log2(len(classes)) if len(classes) > 1 else 1.0
    weighted_purity = float(np.average(max_rates, weights=group_sizes))
    weighted_entropy = float(np.average(entropy / normalizer, weights=group_sizes))
    global_counts = np.asarray([y.eq(cls).sum() for cls in classes], dtype=float)
    global_probs = global_counts / global_counts.sum()
    global_entropy = -np.sum(np.where(global_probs > 0, global_probs * np.log(global_probs), 0))
    mi = float(mutual_info_score(None, None, contingency=counts_arr))
    normalized_mi = float(mi / global_entropy) if global_entropy > 0 else 0.0
    small_group_row_rate = float(
        group_sizes[~valid_group_mask].sum() / group_sizes.sum()
    )

    key_name = safe_name(keys)
    result = {
        "key_name": key_name,
        "keys": "|".join(keys),
        "degree": len(keys),
        "n_train_groups": int(len(train_counts)),
        "n_test_groups": int(len(test_counts)),
        "median_train_group_size": float(np.median(group_sizes)),
        "min_train_group_size": float(np.min(group_sizes)),
        "max_train_group_size": float(np.max(group_sizes)),
        "small_group_row_rate": small_group_row_rate,
        "test_unseen_row_rate": test_unseen_rate,
        "train_test_frequency_tvd": float(freq_tvd),
        "weighted_purity": weighted_purity,
        "weighted_entropy": weighted_entropy,
        "mi_target": mi,
        "normalized_mi": normalized_mi,
        "purity_lift": weighted_purity - float(global_probs.max()),
    }

    profile = target_counts.copy()
    profile["train_count"] = train_counts.values
    for idx, cls in enumerate(classes):
        profile[f"rate_{cls}"] = class_probs[:, idx]
    profile["dominant_class"] = [classes[i] for i in dominant_idx]
    profile["dominant_rate"] = max_rates
    profile["target_entropy_norm"] = entropy / normalizer
    profile = profile.reset_index()
    profile.insert(0, "key_name", key_name)
    profile.insert(1, "keys", "|".join(keys))
    return result, profile


def run_groupby_exploration(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    started = time.time()
    key_train, key_test, cat_keys, bin_keys = make_binned_keys(X_train, X_test, args)
    candidates = build_group_candidates(cat_keys, bin_keys, args)
    classes = list(y.value_counts().index)

    scores: list[dict] = []
    profiles: list[pd.DataFrame] = []
    for i, keys in enumerate(candidates, start=1):
        try:
            score, profile = score_group_key(
                keys,
                key_train,
                key_test,
                y,
                classes,
                args.groupby_min_train_count,
            )
            scores.append(score)
            if i <= 25:
                profiles.append(
                    profile.sort_values(
                        ["dominant_rate", "train_count"], ascending=[False, False]
                    ).head(args.top_groups_per_key)
                )
        except Exception as exc:
            scores.append(
                {
                    "key_name": safe_name(keys),
                    "keys": "|".join(keys),
                    "degree": len(keys),
                    "error": str(exc),
                }
            )

    score_df = pd.DataFrame(scores)
    score_df = score_df.sort_values(
        ["normalized_mi", "test_unseen_row_rate", "train_test_frequency_tvd"],
        ascending=[False, True, True],
    )
    score_df.to_csv(run_dir / "groupby_key_scores.csv", index=False)

    if profiles:
        pd.concat(profiles, axis=0, ignore_index=True).to_csv(
            run_dir / "groupby_top_profiles_initial.csv", index=False
        )

    top_keys = []
    for keys_text in score_df.head(args.agg_top_keys)["keys"].dropna():
        top_keys.append(tuple(keys_text.split("|")))

    agg_scores = score_aggregate_features(
        top_keys,
        key_train,
        key_test,
        X_train,
        X_test,
        y,
        args,
    )
    agg_scores.to_csv(run_dir / "groupby_aggregate_feature_scores.csv", index=False)

    top_profiles = []
    for keys_text in score_df.head(15)["keys"].dropna():
        keys = tuple(keys_text.split("|"))
        try:
            _, profile = score_group_key(
                keys,
                key_train,
                key_test,
                y,
                classes,
                args.groupby_min_train_count,
            )
            top_profiles.append(
                profile.sort_values(
                    ["dominant_rate", "train_count"], ascending=[False, False]
                ).head(args.top_groups_per_key)
            )
        except Exception:
            continue
    if top_profiles:
        pd.concat(top_profiles, axis=0, ignore_index=True).to_csv(
            run_dir / "groupby_top_profiles.csv", index=False
        )

    summary = {
        "elapsed_seconds": time.time() - started,
        "key_candidate_count": len(candidates),
        "key_column_count": len(cat_keys) + len(bin_keys),
        "cat_keys": cat_keys,
        "bin_key_count": len(bin_keys),
        "top_key_rows": score_df.head(30).to_dict(orient="records"),
        "top_aggregate_rows": agg_scores.head(30).to_dict(orient="records"),
    }
    save_json(run_dir / "groupby_summary.json", summary)
    return summary


def score_aggregate_features(
    top_keys: list[tuple[str, ...]],
    key_train: pd.DataFrame,
    key_test: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not top_keys:
        return pd.DataFrame()

    all_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    all_keys = pd.concat([key_train, key_test], axis=0, ignore_index=True)
    numeric_cols = existing_numeric_cols(X_train)
    value_cols = ordered_subset(numeric_cols, PREFERRED_NUM_COLS, args.agg_max_value_cols)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    rows: list[dict] = []
    n_train = len(X_train)

    for keys in top_keys:
        available_keys = [k for k in keys if k in all_keys.columns]
        if len(available_keys) != len(keys):
            continue
        key_name = safe_name(keys)
        key_frame = all_keys.loc[:, available_keys]
        for value_col in value_cols:
            if value_col not in all_X.columns:
                continue
            values = all_X[value_col].replace([np.inf, -np.inf], np.nan)
            tmp = key_frame.copy()
            tmp["_value"] = values.values
            grouped = tmp.groupby(available_keys, observed=True, dropna=False)["_value"]
            for stat in args.agg_stats:
                try:
                    mapped = grouped.transform(stat)
                except Exception:
                    continue
                train_values = (
                    mapped.iloc[:n_train]
                    .astype("float64")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy()
                )
                if np.unique(train_values).size <= 1:
                    continue
                try:
                    f_val, p_val = f_classif(train_values.reshape(-1, 1), y_enc)
                    f_score = float(f_val[0])
                    p_score = float(p_val[0])
                except Exception:
                    continue
                rows.append(
                    {
                        "feature_name": f"gb__{key_name}__{value_col}__{stat}",
                        "keys": "|".join(keys),
                        "value_col": value_col,
                        "stat": stat,
                        "f_score": f_score,
                        "p_value": p_score,
                        "n_unique_train": int(np.unique(train_values).size),
                        "missing_rate_train": float(np.isnan(train_values).mean()),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["f_score", "n_unique_train"], ascending=[False, False]
    )


def write_summary(run_dir: Path, metadata: dict) -> None:
    lines = [
        "# Feature Exploration Summary",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Data",
        "",
        f"- Train shape: {metadata['train_shape']}",
        f"- Test shape: {metadata['test_shape']}",
        f"- Model train shape: {metadata['model_train_shape']}",
        f"- Baseline astronomy features: {metadata['use_baseline_features']}",
        "",
    ]
    if metadata.get("autofe"):
        autofe = metadata["autofe"]
        lines.extend(
            [
                "## AutoFE-PG",
                "",
                f"- Base score: {autofe.get('base_score')}",
                f"- Best score: {autofe.get('best_score')}",
                f"- Selected generated columns: {autofe.get('selected_feature_count')}",
                f"- Selected generators: {autofe.get('selected_generator_names')}",
                "",
            ]
        )
    if metadata.get("groupby"):
        groupby = metadata["groupby"]
        top_keys = groupby.get("top_key_rows", [])[:10]
        top_aggs = groupby.get("top_aggregate_rows", [])[:10]
        lines.extend(
            [
                "## Groupby Key Scan",
                "",
                f"- Key candidates scored: {groupby.get('key_candidate_count')}",
                f"- Derived key columns: {groupby.get('key_column_count')}",
                "",
                "Top keys by target mutual information:",
            ]
        )
        for row in top_keys:
            lines.append(
                f"- `{row.get('keys')}`: normalized_mi={row.get('normalized_mi')}, "
                f"purity_lift={row.get('purity_lift')}, "
                f"test_unseen={row.get('test_unseen_row_rate')}"
            )
        lines.extend(["", "Top aggregate feature candidates:"])
        for row in top_aggs:
            lines.append(
                f"- `{row.get('feature_name')}`: f_score={row.get('f_score')}"
            )
        lines.append("")

    (run_dir / "feature_exploration_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.smoke:
        args.autofe_sample = min(args.autofe_sample, 5000)
        args.autofe_time_limit = min(args.autofe_time_limit, 60)
        args.groupby_max_num_bin_cols = min(args.groupby_max_num_bin_cols, 8)
        args.groupby_bin_counts = [5]
        args.groupby_cross_bin_limit = min(args.groupby_cross_bin_limit, 8)
        args.groupby_max_candidates = min(args.groupby_max_candidates, 120)
        args.agg_top_keys = min(args.agg_top_keys, 5)
        args.agg_max_value_cols = min(args.agg_max_value_cols, 8)

    run_dir = make_run_dir(args.output_dir, args.run_name)
    print(f"Run directory: {run_dir.resolve()}")

    train, test = read_inputs(args.data_dir)
    X_train, X_test, y = build_base_frames(
        train,
        test,
        target=args.target,
        id_col=args.id_col,
        use_baseline_features=not args.no_baseline_features,
    )

    metadata = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "model_train_shape": list(X_train.shape),
        "model_test_shape": list(X_test.shape),
        "target": args.target,
        "id_col": args.id_col,
        "use_baseline_features": not args.no_baseline_features,
        "class_counts": y.value_counts().to_dict(),
        "args": vars(args),
    }
    save_json(run_dir / "metadata.json", metadata)

    if not args.skip_autofe:
        print("Starting AutoFE-PG exploration...")
        metadata["autofe"] = run_autofe(
            X_train,
            y,
            X_test,
            train[args.id_col],
            test[args.id_col],
            args,
            run_dir,
        )
        save_json(run_dir / "metadata.json", metadata)

    if not args.skip_groupby:
        print("Starting groupby exploration...")
        metadata["groupby"] = run_groupby_exploration(
            X_train,
            y,
            X_test,
            args,
            run_dir,
        )
        save_json(run_dir / "metadata.json", metadata)

    write_summary(run_dir, metadata)
    print(f"Saved summary: {(run_dir / 'feature_exploration_summary.md').resolve()}")


if __name__ == "__main__":
    main()
