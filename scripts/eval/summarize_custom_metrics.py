#!/usr/bin/env python3
"""Aggregate per-dataset `means.json` files from `paired_gt_all` into
one cross-dataset comparison table.

`scripts/eval/custom_eval.sh`'s `paired_gt_*` functions each write to
`logs/paired_eval/gt/<dataset>_<split>/{cropped_songs,full_songs}/means.json`
(produced by `realchords/utils/custom_evaluation.py`'s `_save_mean_metrics`/`_compare_to_gt`).
Each of those files is already a per-dataset summary; this script just walks
all of them and puts the GT-side stats side by side so datasets can be
compared directly (e.g. "which dataset has the highest melody silence
ratio", "does WJD's rhythmic diversity really look different from
FiloBass's").

Usage::

    python scripts/eval/summarize_custom_metrics.py
    python scripts/eval/summarize_custom_metrics.py --gt_root logs/paired_eval/gt \\
        --out_dir logs/paired_eval/gt_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_MODES = ("cropped_songs", "full_songs")


def _load_gt_row(means_path: Path) -> Optional[Dict[str, float]]:
    if not means_path.exists():
        return None
    with open(means_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("gt")


def _load_model_comparison_rows(
    means_path: Path,
) -> Dict[str, Dict[str, float]]:
    """Optional bonus: model-vs-GT comparison metrics, if this run had models."""
    if not means_path.exists():
        return {}
    with open(means_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("models", {})


def collect_gt_summary(gt_root: Path) -> Dict[str, pd.DataFrame]:
    """Returns {"cropped_songs": df, "full_songs": df}, one row per dataset dir."""
    tables: Dict[str, Dict[str, Dict[str, float]]] = {mode: {} for mode in _MODES}

    for dataset_dir in sorted(gt_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for mode in _MODES:
            row = _load_gt_row(dataset_dir / mode / "means.json")
            if row is not None:
                tables[mode][dataset_dir.name] = row

    return {
        mode: pd.DataFrame.from_dict(rows, orient="index").sort_index()
        for mode, rows in tables.items()
        if rows
    }


def collect_model_comparison_summary(gt_root: Path) -> pd.DataFrame:
    """Model-vs-GT rows (sync EMD, chord-type JS distance), if any runs had models."""
    rows: Dict[str, Dict[str, float]] = {}
    for dataset_dir in sorted(gt_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for mode in _MODES:
            models = _load_model_comparison_rows(dataset_dir / mode / "means.json")
            for model_label, metrics in models.items():
                rows[f"{dataset_dir.name}/{mode}/{model_label}"] = metrics
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt_root",
        type=str,
        default="logs/paired_eval/gt",
        help="Root directory containing <dataset>_<split>/{cropped_songs,full_songs}/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="If set, also write cropped_songs.csv / full_songs.csv / model_comparison.csv here",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    if not gt_root.exists():
        raise SystemExit(f"{gt_root} does not exist -- run paired_gt_all first")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    tables = collect_gt_summary(gt_root)
    for mode, df in tables.items():
        print(f"\n=== GT stats: {mode} ===")
        print(df.to_string())

    model_df = collect_model_comparison_summary(gt_root)
    if not model_df.empty:
        print("\n=== Model-vs-GT comparison (runs that included models) ===")
        print(model_df.to_string())

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode, df in tables.items():
            df.to_csv(out_dir / f"{mode}.csv")
        if not model_df.empty:
            model_df.to_csv(out_dir / "model_comparison.csv")
        print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
