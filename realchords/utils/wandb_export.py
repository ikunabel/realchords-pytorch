"""Core logic for dumping a W&B run's history/config/summary to local disk.

Used both by the standalone CLI (scripts/wandb/export_run.py) and by
Trainer's automatic post-training export (realchords/base_trainer.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import wandb


def resolve_run_path(run: str, entity: str, project: str) -> str:
    """Accept either a bare run ID or a full "entity/project/run_id" path."""
    if run.count("/") == 2:
        return run
    return f"{entity}/{project}/{run}"


def export_run(api: wandb.Api, run_path: str, out_dir: str | Path) -> Path:
    """Dump one run's full history/config/summary to ``<out_dir>/<run_id>/``.

    Uses ``run.scan_history()`` rather than ``run.history()``, which the W&B
    API subsamples to ~500 points to match the dashboard's plot resolution.
    """
    run = api.run(run_path)
    run_dir = Path(out_dir) / run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(list(run.scan_history()))
    history.to_csv(run_dir / "history.csv", index=False)

    with open(run_dir / "config.json", "w") as f:
        json.dump(dict(run.config), f, indent=2, default=str)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(dict(run.summary), f, indent=2, default=str)

    return run_dir
