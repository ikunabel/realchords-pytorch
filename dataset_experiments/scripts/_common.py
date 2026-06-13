"""Shared paths and helpers for dataset exploration scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"


def setup_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def ensure_output(subdir: str) -> Path:
    path = OUTPUT_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(subdir: str, filename: str, content: str) -> Path:
    out_dir = ensure_output(subdir)
    path = out_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def write_json(subdir: str, filename: str, data: Any, indent: int = 2) -> Path:
    out_dir = ensure_output(subdir)
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")
    return path
