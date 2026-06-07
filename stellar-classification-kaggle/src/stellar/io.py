"""Small IO helpers shared by experiment scripts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    """Create a timestamped run directory without overwriting old outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = output_dir / run_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    suffix = 1
    while True:
        candidate = output_dir / f"{run_name}_{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def save_json(path: Path, payload: dict) -> None:
    """Write a UTF-8 JSON file with stable pretty formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

