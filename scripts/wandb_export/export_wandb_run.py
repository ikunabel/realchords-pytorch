#!/usr/bin/env python3
"""Export W&B run history/config/summary to local CSV/JSON for offline analysis.

Thin CLI wrapper around realchords.utils.wandb_export -- the same code path
Trainer uses to auto-export a run right after training finishes (see
realchords/base_trainer.py). Use this script to backfill runs that predate
that hook, or to re-export/pull runs on demand.

Writes into logs/<run_name>/wandb_export/, alongside that run's checkpoints,
args.yml, and wandb/ folder -- the same convention the auto-export hook uses.

Usage::

    python scripts/wandb/export_wandb_run.py RUN_ID
    python scripts/wandb/export_wandb_run.py RUN_ID_1 RUN_ID_2
    python scripts/wandb/export_wandb_run.py entity/project/RUN_ID
    python scripts/wandb/export_wandb_run.py --all
    python scripts/wandb/export_wandb_run.py --all --logs_dir logs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import wandb

from realchords.utils.wandb_export import export_run, resolve_run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run IDs, or full 'entity/project/run_id' paths (mix and match).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every run in --entity/--project instead of specific runs.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "ikunabel"),
        help="Used for bare run IDs and --all (default: $WANDB_ENTITY or 'ikunabel').",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "realchords"),
        help="Used for bare run IDs and --all (default: $WANDB_PROJECT or 'realchords').",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Root logs directory; writes into <logs_dir>/<run_name>/wandb_export/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and not args.runs:
        raise SystemExit("Pass one or more run IDs, or --all to export every run in the project.")

    logs_dir = Path(args.logs_dir)
    api = wandb.Api()

    if args.all:
        runs = api.runs(f"{args.entity}/{args.project}")
        run_paths = [f"{args.entity}/{args.project}/{r.id}" for r in runs]
        print(f"Found {len(run_paths)} runs in {args.entity}/{args.project}")
    else:
        run_paths = [resolve_run_path(r, args.entity, args.project) for r in args.runs]

    for run_path in run_paths:
        run = api.run(run_path)
        out_dir = logs_dir / run.name / "wandb_export"
        export_run(api, run_path, out_dir)
        print(f"Exported {run_path} ({run.name}) -> {out_dir}")


if __name__ == "__main__":
    main()
