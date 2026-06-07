"""Project paths used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PACKAGES_DIR = PROJECT_ROOT / "packages"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

