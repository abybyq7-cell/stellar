"""Submission auditing and registry helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stellar.constants import ID_COL, LABEL_COL
from stellar.paths import OUTPUTS_DIR, PROJECT_ROOT, SUBMISSIONS_DIR


REGISTRY_CSV = "submission_registry.csv"
REGISTRY_JSONL = "submission_registry.jsonl"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return relpath(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def relpath(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_run_context(submission_path: Path, run_dir: Path | None = None) -> dict[str, str | None]:
    submission_path = submission_path.resolve()
    if run_dir is None:
        run_dir = submission_path.parent
        if run_dir.name in {"ensembles", "single_model_submissions"}:
            run_dir = run_dir.parent

    context = {
        "workflow": None,
        "run_name": None,
        "run_dir": relpath(run_dir),
    }
    try:
        parts = run_dir.resolve().relative_to(OUTPUTS_DIR).parts
    except Exception:
        return context

    if len(parts) >= 1:
        context["workflow"] = parts[0]
    if len(parts) >= 2:
        context["run_name"] = parts[1]
    return context


def submission_summary(
    path: Path,
    id_col: str = ID_COL,
    label_col: str = LABEL_COL,
) -> dict[str, Any]:
    df = pd.read_csv(path)
    summary: dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": df.columns.tolist(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    if id_col in df.columns:
        id_null_count = int(df[id_col].isna().sum())
        unique_id_count = int(df[id_col].nunique(dropna=True))
        summary.update(
            {
                "id_col": id_col,
                "id_null_count": id_null_count,
                "unique_id_count": unique_id_count,
                "duplicate_id_count": int(len(df) - unique_id_count - id_null_count),
            }
        )
    if label_col in df.columns:
        counts = df[label_col].astype("string").value_counts(dropna=False).to_dict()
        counts = {str(key): int(value) for key, value in counts.items()}
        summary.update(
            {
                "label_col": label_col,
                "target_null_count": int(df[label_col].isna().sum()),
                "class_distribution": counts,
                "predicted_classes": sorted(counts),
            }
        )
    return summary


def _primary_metric(metrics: dict[str, Any] | None) -> tuple[str | None, Any | None]:
    if not metrics:
        return None, None
    for key in ["accuracy", "balanced_accuracy", "log_loss", "score"]:
        if key in metrics:
            return key, metrics[key]
    first_key = next(iter(metrics))
    return str(first_key), metrics[first_key]


def _registry_row(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    source = record["source"]
    primary_name, primary_value = _primary_metric(record.get("metrics"))
    return {
        "registered_at": record["registered_at"],
        "submission_relpath": record["submission_relpath"],
        "sidecar_relpath": record["sidecar_relpath"],
        "sha256": summary.get("sha256"),
        "size_bytes": summary.get("size_bytes"),
        "row_count": summary.get("row_count"),
        "duplicate_id_count": summary.get("duplicate_id_count"),
        "id_null_count": summary.get("id_null_count"),
        "target_null_count": summary.get("target_null_count"),
        "predicted_classes": "|".join(summary.get("predicted_classes", [])),
        "class_distribution": json.dumps(
            summary.get("class_distribution", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "workflow": source.get("workflow"),
        "run_name": source.get("run_name"),
        "run_dir": source.get("run_dir"),
        "script": source.get("script"),
        "submission_type": source.get("submission_type"),
        "model_name": source.get("model_name"),
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "metrics_json": json.dumps(record.get("metrics") or {}, ensure_ascii=False, sort_keys=True),
        "params_json": json.dumps(record.get("params") or {}, ensure_ascii=False, sort_keys=True),
        "notes": record.get("notes"),
    }


def _write_registry(registry_dir: Path, record: dict[str, Any], row: dict[str, Any]) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = registry_dir / REGISTRY_JSONL
    records: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records = [
        existing
        for existing in records
        if existing.get("submission_relpath") != record["submission_relpath"]
    ]
    records.append(record)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    csv_path = registry_dir / REGISTRY_CSV
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [
        existing
        for existing in rows
        if existing.get("submission_relpath") != row["submission_relpath"]
    ]
    rows.append(row)

    fieldnames = list(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def register_submission(
    path: Path,
    *,
    run_dir: Path | None = None,
    script: str | None = None,
    submission_type: str | None = None,
    model_name: str | None = None,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
    registry_dir: Path = SUBMISSIONS_DIR,
) -> dict[str, Any]:
    """Write sidecar metadata and update the central submission registry."""
    path = Path(path)
    context = infer_run_context(path, run_dir)
    summary = submission_summary(path)
    sidecar_path = path.with_suffix(".metadata.json")
    record = {
        "registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "submission_path": str(path.resolve()),
        "submission_relpath": relpath(path),
        "sidecar_relpath": relpath(sidecar_path),
        "summary": summary,
        "source": {
            **context,
            "script": script,
            "submission_type": submission_type,
            "model_name": model_name,
        },
        "metrics": _jsonable(metrics or {}),
        "params": _jsonable(params or {}),
        "notes": notes,
        "extra": _jsonable(extra or {}),
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, sort_keys=True)

    _write_registry(registry_dir, record, _registry_row(record))
    return record
