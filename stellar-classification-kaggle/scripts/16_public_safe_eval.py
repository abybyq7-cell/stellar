from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stellar.constants import ID_COL, LABEL_COL, MAG_COLS
from stellar.paths import OUTPUTS_DIR, RAW_DATA_DIR


CLASSES = ["GALAXY", "QSO", "STAR"]
DEFAULT_CLASS_WEIGHTS = {"GALAXY": 1.0, "QSO": 3.2, "STAR": 4.6}
DEFAULT_CANDIDATES = {
    "018_full21_weighted_vs_015_arbitration": (
        OUTPUTS_DIR / "disagreement_arbitration" / "full21_weighted_vs_015_arbitration" / "arbitrated_oof_train_pred.csv",
        OUTPUTS_DIR / "disagreement_arbitration" / "full21_weighted_vs_015_arbitration" / "arbitrated_test_pred.csv",
    ),
    "023_full25_weighted_vs_015_arbitration": (
        OUTPUTS_DIR / "disagreement_arbitration" / "full25_weighted_vs_015_arbitration" / "arbitrated_oof_train_pred.csv",
        OUTPUTS_DIR / "disagreement_arbitration" / "full25_weighted_vs_015_arbitration" / "arbitrated_test_pred.csv",
    ),
    "025_full26_nn_balanced_threshold": (
        OUTPUTS_DIR / "two_stage_threshold" / "full26_nn_balanced_lr_threshold_grid" / "best_weighted_oof_train_pred.csv",
        OUTPUTS_DIR / "two_stage_threshold" / "full26_nn_balanced_lr_threshold_grid" / "best_weighted_test_pred.csv",
    ),
    "026_full26_nn_weighted_vs_015_arbitration": (
        OUTPUTS_DIR / "disagreement_arbitration" / "full26_nn_weighted_vs_015_arbitration" / "arbitrated_oof_train_pred.csv",
        OUTPUTS_DIR / "disagreement_arbitration" / "full26_nn_weighted_vs_015_arbitration" / "arbitrated_test_pred.csv",
    ),
}
DEFAULT_REFERENCE_SUBMISSIONS = {
    "018_full21_weighted_vs_015_arbitration": PROJECT_ROOT
    / "submissions"
    / "018_full21_weighted_vs_015_arbitration.csv",
    "023_full25_weighted_vs_015_arbitration": PROJECT_ROOT
    / "submissions"
    / "023_full25_weighted_vs_015_arbitration.csv",
    "025_full26_nn_balanced_threshold": PROJECT_ROOT
    / "submissions"
    / "025_full26_nn_balanced_threshold.csv",
    "026_full26_nn_weighted_vs_015_arbitration": PROJECT_ROOT
    / "submissions"
    / "026_full26_nn_weighted_vs_015_arbitration.csv",
}


def parse_named_path_pair(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Candidate must be name=train.csv,test.csv, got {spec!r}")
    name, paths = parts
    train_path, test_path = [Path(item.strip()) for item in paths.split(",", 1)]
    return name.strip(), train_path, test_path


def parse_named_path(spec: str) -> tuple[str, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Reference must be name=path.csv, got {spec!r}")
    return parts[0].strip(), Path(parts[1].strip())


def parse_class_weights(spec: str) -> dict[str, float]:
    weights = DEFAULT_CLASS_WEIGHTS.copy()
    if spec:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            name, value = item.split(":", 1)
            weights[name.strip()] = float(value)
    missing = [cls for cls in CLASSES if cls not in weights]
    if missing:
        raise ValueError(f"Missing class weights for: {missing}")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Public-safe evaluator for candidate train/test predictions. Train metrics use "
            "raw train labels only; test outputs are disagreement/crosstab audits only."
        )
    )
    parser.add_argument("--raw-train", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--raw-test", type=Path, default=RAW_DATA_DIR / "test.csv")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Add candidate as name=train_pred.csv,test_pred.csv. Can be repeated.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Add test reference submission as name=submission.csv. Can be repeated.",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="GALAXY:1,QSO:3.2,STAR:4.6",
        help="Weights for weighted accuracy and weighted error summaries.",
    )
    parser.add_argument("--lowz-redshift-abs-max", type=float, default=0.144)
    parser.add_argument("--compact-color-score-max", type=float, default=1.30)
    parser.add_argument("--low-mag-std-max", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "diagnostics" / "public_safe_eval")
    return parser.parse_args()


def read_prediction(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in [ID_COL, LABEL_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[[ID_COL, LABEL_COL]].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(str)
    unknown = sorted(set(out[LABEL_COL].dropna()) - set(CLASSES))
    if unknown:
        raise ValueError(f"{path} has unknown classes: {unknown}")
    return out.rename(columns={LABEL_COL: pred_col})


def existing_default_candidates() -> dict[str, tuple[Path, Path]]:
    return {
        name: paths
        for name, paths in DEFAULT_CANDIDATES.items()
        if paths[0].exists() and paths[1].exists()
    }


def existing_default_references() -> dict[str, Path]:
    return {name: path for name, path in DEFAULT_REFERENCE_SUBMISSIONS.items() if path.exists()}


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if all(col in out.columns for col in MAG_COLS):
        out["u_minus_g"] = out["u"] - out["g"]
        out["g_minus_r"] = out["g"] - out["r"]
        out["r_minus_i"] = out["r"] - out["i"]
        out["i_minus_z"] = out["i"] - out["z"]
        out["u_minus_z"] = out["u"] - out["z"]
        mag_values = out[MAG_COLS]
        out["mag_std"] = mag_values.std(axis=1)
        out["mag_range"] = mag_values.max(axis=1) - mag_values.min(axis=1)
        color_values = out[["u_minus_g", "g_minus_r", "r_minus_i", "i_minus_z"]].astype("float64")
        out["compact_color_score"] = np.sqrt(np.square(color_values).mean(axis=1))
        out["wide_color_flag"] = (
            (out["g_minus_r"] > 1.037)
            & (out["u_minus_z"] > 3.834)
            & ((out["mag_range"] > 3.949) | (out["mag_std"] > 1.624))
        )
    return out


def build_slice_masks(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series]:
    feat = add_feature_columns(df)
    base_masks = {
        "lowz": feat["redshift"].abs().le(args.lowz_redshift_abs_max),
        "Blue_Cloud": feat["galaxy_population"].astype("string").eq("Blue_Cloud"),
        "compact_color": feat["compact_color_score"].le(args.compact_color_score_max),
        "low_mag_std": feat["mag_std"].le(args.low_mag_std_max),
        "wide_color_false": ~feat["wide_color_flag"],
    }
    masks: dict[str, pd.Series] = {"ALL": pd.Series(True, index=feat.index)}
    names = list(base_masks)
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            mask = pd.Series(True, index=feat.index)
            for name in combo:
                mask = mask & base_masks[name]
            masks["&".join(combo)] = mask
    return masks


def class_weight_array(labels: pd.Series, weights: dict[str, float]) -> np.ndarray:
    return labels.astype(str).map(weights).to_numpy(dtype="float64")


def balanced_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = []
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    for cls in CLASSES:
        cls_mask = yt.eq(cls)
        values.append(float(yp[cls_mask].eq(cls).mean()) if cls_mask.any() else np.nan)
    valid = [value for value in values if not np.isnan(value)]
    return float(np.mean(valid)) if valid else np.nan


def metrics_row(
    candidate: str,
    slice_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: dict[str, float],
) -> dict:
    correct = y_true.astype(str).eq(y_pred.astype(str))
    sample_weight = class_weight_array(y_true, weights)
    total_weight = float(sample_weight.sum())
    weighted_correct = float(np.sum(sample_weight * correct.to_numpy(dtype="float64")))
    return {
        "candidate": candidate,
        "slice": slice_name,
        "n": int(len(y_true)),
        "accuracy": float(correct.mean()) if len(y_true) else np.nan,
        "balanced_accuracy": balanced_accuracy(y_true, y_pred) if len(y_true) else np.nan,
        "weighted_accuracy": weighted_correct / total_weight if total_weight else np.nan,
        "errors": int((~correct).sum()),
        "weighted_errors": total_weight - weighted_correct,
    }


def error_flow_rows(
    candidate: str,
    slice_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    weights: dict[str, float],
) -> list[dict]:
    rows = []
    frame = pd.DataFrame({"true_class": y_true.astype(str), "pred_class": y_pred.astype(str)})
    frame["sample_weight"] = class_weight_array(frame["true_class"], weights)
    for (true_class, pred_class), group in frame.groupby(["true_class", "pred_class"], dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "slice": slice_name,
                "true_class": true_class,
                "pred_class": pred_class,
                "n": int(len(group)),
                "errors": int(true_class != pred_class) * int(len(group)),
                "weighted_n": float(group["sample_weight"].sum()),
                "weighted_errors": float(group["sample_weight"].sum()) if true_class != pred_class else 0.0,
            }
        )
    return rows


def evaluate_train(
    raw_train: pd.DataFrame,
    train_masks: dict[str, pd.Series],
    candidates: dict[str, tuple[Path, Path]],
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    flow_rows = []
    labels = raw_train[[ID_COL, LABEL_COL]].copy().rename(columns={LABEL_COL: "true_class"})
    for name, (train_path, _) in candidates.items():
        pred = read_prediction(train_path, "pred").merge(labels, on=ID_COL, how="right", validate="one_to_one")
        if pred["pred"].isna().any():
            raise ValueError(f"{train_path} does not cover all train ids.")
        for slice_name, mask in train_masks.items():
            part = pred.loc[mask.to_numpy()]
            metric_rows.append(metrics_row(name, slice_name, part["true_class"], part["pred"], weights))
            flow_rows.extend(error_flow_rows(name, slice_name, part["true_class"], part["pred"], weights))
    return pd.DataFrame(metric_rows), pd.DataFrame(flow_rows)


def class_distribution_rows(name: str, pred: pd.Series, split: str) -> list[dict]:
    counts = pred.astype(str).value_counts()
    total = int(counts.sum())
    return [
        {
            "name": name,
            "split": split,
            "class": cls,
            "n": int(counts.get(cls, 0)),
            "rate": float(counts.get(cls, 0) / total) if total else np.nan,
        }
        for cls in CLASSES
    ]


def test_disagreement_rows(
    candidate_name: str,
    reference_name: str,
    candidate_pred: pd.Series,
    reference_pred: pd.Series,
    masks: dict[str, pd.Series],
) -> list[dict]:
    rows = []
    disagree = candidate_pred.astype(str).ne(reference_pred.astype(str))
    for slice_name, mask in masks.items():
        part = disagree[mask.to_numpy()]
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                "slice": slice_name,
                "n": int(len(part)),
                "disagreements": int(part.sum()),
                "disagreement_rate": float(part.mean()) if len(part) else np.nan,
                "agreement_rate": float(1.0 - part.mean()) if len(part) else np.nan,
            }
        )
    return rows


def crosstab_rows(
    candidate_name: str,
    reference_name: str,
    candidate_pred: pd.Series,
    reference_pred: pd.Series,
) -> list[dict]:
    rows = []
    table = pd.crosstab(
        reference_pred.astype(str),
        candidate_pred.astype(str),
        rownames=["reference_class"],
        colnames=["candidate_class"],
        dropna=False,
    )
    for ref_cls in CLASSES:
        for cand_cls in CLASSES:
            rows.append(
                {
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "reference_class": ref_cls,
                    "candidate_class": cand_cls,
                    "n": int(table.loc[ref_cls, cand_cls]) if ref_cls in table.index and cand_cls in table.columns else 0,
                }
            )
    return rows


def evaluate_test(
    raw_test: pd.DataFrame,
    test_masks: dict[str, pd.Series],
    candidates: dict[str, tuple[Path, Path]],
    references: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_rows = []
    disagreement = []
    crosstabs = []
    test_ids = raw_test[[ID_COL]].copy()

    loaded_references = {
        name: read_prediction(path, "reference_pred").merge(test_ids, on=ID_COL, how="right", validate="one_to_one")
        for name, path in references.items()
    }
    for ref_name, ref in loaded_references.items():
        if ref["reference_pred"].isna().any():
            raise ValueError(f"{references[ref_name]} does not cover all test ids.")
        dist_rows.extend(class_distribution_rows(ref_name, ref["reference_pred"], "reference_test"))

    for cand_name, (_, test_path) in candidates.items():
        cand = read_prediction(test_path, "candidate_pred").merge(test_ids, on=ID_COL, how="right", validate="one_to_one")
        if cand["candidate_pred"].isna().any():
            raise ValueError(f"{test_path} does not cover all test ids.")
        dist_rows.extend(class_distribution_rows(cand_name, cand["candidate_pred"], "candidate_test"))
        for ref_name, ref in loaded_references.items():
            disagreement.extend(
                test_disagreement_rows(
                    cand_name,
                    ref_name,
                    cand["candidate_pred"],
                    ref["reference_pred"],
                    test_masks,
                )
            )
            crosstabs.extend(
                crosstab_rows(cand_name, ref_name, cand["candidate_pred"], ref["reference_pred"])
            )
    return pd.DataFrame(dist_rows), pd.DataFrame(disagreement), pd.DataFrame(crosstabs)


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    candidates: dict[str, tuple[Path, Path]],
    references: dict[str, Path],
    weights: dict[str, float],
) -> None:
    manifest = {
        "script": Path(__file__).name,
        "public_safe_note": "Train metrics use raw train labels only. Test outputs contain only prediction/reference disagreement summaries.",
        "raw_train": str(args.raw_train),
        "raw_test": str(args.raw_test),
        "class_weights": weights,
        "slice_thresholds": {
            "lowz_redshift_abs_max": args.lowz_redshift_abs_max,
            "compact_color_score_max": args.compact_color_score_max,
            "low_mag_std_max": args.low_mag_std_max,
            "wide_color_false": "not ((g_minus_r > 1.037) and (u_minus_z > 3.834) and (mag_range > 3.949 or mag_std > 1.624))",
        },
        "candidates": {
            name: {"train": str(paths[0]), "test": str(paths[1])}
            for name, paths in candidates.items()
        },
        "references": {name: str(path) for name, path in references.items()},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    weights = parse_class_weights(args.class_weights)
    candidates = existing_default_candidates()
    references = existing_default_references()
    for spec in args.candidate:
        name, train_path, test_path = parse_named_path_pair(spec)
        candidates[name] = (train_path, test_path)
    for spec in args.reference:
        name, path = parse_named_path(spec)
        references[name] = path
    if not candidates:
        raise ValueError("No candidates found. Pass --candidate name=train.csv,test.csv.")
    if not references:
        raise ValueError("No references found. Pass --reference name=submission.csv.")

    raw_train = pd.read_csv(args.raw_train)
    raw_test = pd.read_csv(args.raw_test)
    train_masks = build_slice_masks(raw_train, args)
    test_masks = build_slice_masks(raw_test, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics, train_error_flows = evaluate_train(raw_train, train_masks, candidates, weights)
    test_distribution, test_disagreement, test_crosstab = evaluate_test(raw_test, test_masks, candidates, references)

    overall = train_metrics[train_metrics["slice"].eq("ALL")].copy()
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    train_metrics.to_csv(args.output_dir / "slice_metrics.csv", index=False)
    train_error_flows.to_csv(args.output_dir / "slice_error_flows.csv", index=False)
    test_distribution.to_csv(args.output_dir / "test_class_distribution.csv", index=False)
    test_disagreement.to_csv(args.output_dir / "test_reference_disagreement_by_slice.csv", index=False)
    test_crosstab.to_csv(args.output_dir / "test_reference_crosstab.csv", index=False)
    write_manifest(args.output_dir / "manifest.json", args, candidates, references, weights)

    print(f"Saved diagnostics to: {args.output_dir.resolve()}")
    print("Overall metrics:")
    print(overall.sort_values("weighted_accuracy", ascending=False).to_string(index=False))
    print(f"Candidates: {', '.join(candidates)}")
    print(f"References: {', '.join(references)}")


if __name__ == "__main__":
    main()
