"""Plain summary of one custom-eval experiment: metric -> model -> condition (dataset/split).

Collects every means.json under an experiment folder in logs/custom_eval/ -- the folder itself
(one eval run) or its subfolders (one run per condition, e.g. dataset_loo/<test set>/) -- into a
single JSON, with model labels as given in the eval config and GT as "gt". No derived statistics.

Output (<experiment dir>/summary.json), one table per metric -- rows = models, columns = conditions:
    {
      "experiment": "dataset_loo",
      "conditions": {"pop909": {"dataset": "pop909", "split": "test"}, ...},
      "metrics": {
        "note_in_chord_ratio_mean": {
          "7sets": {"hooktheory": 0.382, "pop909": 0.409, ...},
          ...,
          "gt":    {"hooktheory": 0.724, "pop909": 0.705, ...}
        }, ...
      }
    }
Condition keys are the run subfolders ('.' for a single-run experiment). A model/metric missing
from a condition is simply absent from that row.

Alongside it, <experiment dir>/summary.md has the same tables in markdown, with the best model per
condition in bold (GT excluded): highest or lowest value, or closest to GT for statistics that
should match the reference (see METRIC_DIRECTION; unlisted metrics are not highlighted).
Onset-weighted chord distribution distances are left out of the markdown (frame weighting is
reported); they remain in summary.json.

Written automatically at the end of every custom_evaluation.py run (and after
run_multi_model_eval.py merges its sub-runs); runs belonging to one experiment each rewrite the
summary, so it's complete once the last one finishes. By hand:
    python scripts/eval/custom_eval/summarize_experiment.py logs/custom_eval/<experiment>
"""

from __future__ import annotations

import json
import os
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Sequence

import yaml

STAGING_DIR = "_staging"

# How "best" is determined per metric in summary.md: "max", "min", or "gt" (closest to GT).
METRIC_DIRECTION = {
    "note_in_chord_ratio_mean": "max",
    "note_in_chord_ratio_per_song_mean": "max",
    "note_in_mode_ratio_mean": "max",
    "note_in_mode_ratio_per_song_mean": "max",
    "vendi_score": "max",
    # frozen reward models used as metrics: closest to GT is best -- scoring *above* GT means the
    # policy exploits the reward model rather than being better than real data
    "reward_contrastive_score": "gt",
    "reward_discriminative_score": "gt",
    "clamp2_cos_per_song_chords_vs_gt": "max",
    "clamp2_cos_per_song_full_vs_gt": "max",
    "sync_emd_vs_gt": "min",
    "chord_dist_wasserstein_per_song_onset_vs_gt": "min",
    "chord_dist_wasserstein_per_song_frame_vs_gt": "min",
    "chord_dist_wasserstein_key_relative_onset_vs_gt": "min",
    "chord_dist_wasserstein_key_relative_frame_vs_gt": "min",
    "chord_dist_wasserstein_onset_vs_gt": "min",
    "chord_dist_wasserstein_frame_vs_gt": "min",
    "chord_type_js_distance_onset_vs_gt": "min",
    "chord_type_js_distance_frame_vs_gt": "min",
    "clamp2_fmd_chords_vs_gt": "min",
    "clamp2_fmd_full_vs_gt": "min",
    "clamp2_fmd_chords_vs_gt_half": "min",
    "clamp2_fmd_full_vs_gt_half": "min",
    "chord_duration_entropy": "gt",
    "chord_complexity_mean": "gt",
    "chord_silence_ratio_mean": "gt",
}
# Model-vs-GT distances whose GT row is 0 by definition (not stored in means.json).
GT_ZERO_PREFIXES = ("chord_dist_wasserstein_", "chord_type_js_distance_")

# Still computed and kept in summary.json, but not shown in summary.md: the onset-weighted chord
# distribution distances (frame weighting is the reported variant).
def _hidden_in_markdown(metric: str) -> bool:
    return metric.startswith(GT_ZERO_PREFIXES) and "_onset_" in metric


_DIRECTION_NOTE = {"max": "higher is better", "min": "lower is better", "gt": "closest to GT is best"}

# summary.md section order and headings; metrics not listed here follow, in means.json order.
METRIC_ORDER = [
    ("note_in_chord_ratio_per_song_mean", "Harmony"),
    ("sync_emd_vs_gt", "Synchronization"),
    ("chord_duration_entropy", "Rhythm Diversity"),
    ("chord_silence_ratio_mean", "Chord silence ratio"),
]
_METRIC_LABEL = dict(METRIC_ORDER)


def _ordered_metrics(metrics) -> list:
    """METRIC_ORDER first (those that exist), then everything else as it comes."""
    lead = [m for m, _ in METRIC_ORDER if m in metrics]
    return lead + [m for m in metrics if m not in lead]


def experiment_root(save_dir: Path) -> Optional[Path]:
    """The experiment folder a run's save_dir belongs to: logs/custom_eval/<experiment> for a
    save_dir anywhere below logs/custom_eval/, else save_dir itself. None for staging sub-runs
    of run_multi_model_eval.py (their merged experiment is summarized after merging)."""
    save_dir = Path(save_dir).resolve()
    parts = save_dir.parts
    if STAGING_DIR in parts:
        return None
    if "custom_eval" in parts:
        i = len(parts) - 1 - parts[::-1].index("custom_eval")
        if i + 1 < len(parts):
            return Path(*parts[: i + 2])
    return save_dir


def _summarize_run(run_dir: Path) -> dict:
    means = json.loads((run_dir / "means.json").read_text())
    info = _load_nearby(run_dir, "model_labels.json", json.loads) or {}
    slug_to_label = {slug: label for label, slug in info.get("labels", {}).items()}
    args = _load_nearby(run_dir, "args.yml", yaml.safe_load) or {}
    run_models = means.get("models", {})
    # config order (model_labels.json), then anything not listed there; means.json is sorted
    order = [slug for slug in info.get("labels", {}).values() if slug in run_models]
    order += [slug for slug in run_models if slug not in order]
    models = {slug_to_label.get(slug, slug): run_models[slug] for slug in order}
    gt = dict(means["gt"])
    # chord distribution distances vs GT aren't stored for GT itself; GT vs GT is 0 by definition
    for metric in {k for m in run_models.values() for k in m if k.startswith(GT_ZERO_PREFIXES)}:
        if metric not in gt:
            present = any(m.get(metric) is not None for m in run_models.values())
            gt[metric] = 0.0 if present else None
    models["gt"] = gt
    dataset = args.get("dataset_name") or info.get("dataset_name") or _dataset_from_metadata(run_dir)
    return {"dataset": dataset, "split": args.get("dataset_split"), "models": models}


def _load_nearby(run_dir: Path, name: str, parse):
    """Read <name> from run_dir or, failing that, its parents (gt_only runs keep args.yml one
    level up, next to cropped_songs/ and full_songs/). None if there is none."""
    for d in (run_dir, *run_dir.parents):
        path = d / name
        if path.exists():
            return parse(path.read_text())
        if d.name == "custom_eval":
            break
    return None


def _dataset_from_metadata(run_dir: Path) -> Optional[str]:
    path = run_dir / "metadata.jsonl"
    if not path.exists():
        return None
    with path.open() as fh:
        first = fh.readline()
    return json.loads(first).get("dataset_name") if first.strip() else None


def write_experiment_summary(experiment_dir: Path, include: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Write <experiment_dir>/summary.json from every means.json below it; None if there are none.

    include: optional fnmatch patterns on the condition name (the run's path relative to
    experiment_dir), e.g. ["*/full_songs"] to leave out the cropped-song variants."""
    root = Path(experiment_dir)
    run_dirs = sorted(p.parent for p in root.rglob("means.json") if STAGING_DIR not in p.parts)
    if include:
        run_dirs = [d for d in run_dirs
                    if any(fnmatch(str(d.relative_to(root)), pat) for pat in include)]
    if not run_dirs:
        return None
    runs = {str(d.relative_to(root)): _summarize_run(d) for d in run_dirs}
    metrics: dict = {}
    for cond, run in runs.items():
        for model, values in run["models"].items():
            for metric, value in values.items():
                metrics.setdefault(metric, {}).setdefault(model, {})[cond] = value
    for metric, rows in metrics.items():  # models in first-seen order, GT last
        if "gt" in rows:
            rows["gt"] = rows.pop("gt")
    summary = {
        "experiment": root.name,
        "conditions": {cond: {"dataset": r["dataset"], "split": r["split"]} for cond, r in runs.items()},
        "metrics": metrics,
    }
    out = root / "summary.json"
    # atomic replace: several jobs of one experiment may finish at the same time
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".summary.", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, out)
    _atomic_write(root / "summary.md", _markdown(summary))
    return out


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _fmt(value) -> str:
    if isinstance(value, bool) or value is None:
        return "–" if value is None else str(value)
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer() and abs(value) >= 100):
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _best_models(rows: dict, cond: str, direction: str) -> set:
    values = {m: r[cond] for m, r in rows.items()
              if m != "gt" and isinstance(r.get(cond), (int, float)) and not isinstance(r.get(cond), bool)}
    if len(values) < 2:
        return set()
    if direction == "gt":
        target = rows.get("gt", {}).get(cond)
        if not isinstance(target, (int, float)):
            return set()
        score = {m: abs(v - target) for m, v in values.items()}
        best = min(score.values())
        return {m for m, v in score.items() if v == best}
    best = max(values.values()) if direction == "max" else min(values.values())
    return {m for m, v in values.items() if v == best}


def _markdown(summary: dict) -> str:
    conds = list(summary["conditions"])
    datasets = [summary["conditions"][c]["dataset"] for c in conds]
    # label columns by dataset when that is unambiguous (one run per dataset), else by run folder
    if all(datasets) and len(set(datasets)) == len(conds):
        headers = list(datasets)
    else:
        headers = [summary["conditions"][c]["dataset"] or c if c == "." else c for c in conds]
    lines = [f"# {summary['experiment']}", ""]
    lines.append("Conditions: " + ", ".join(
        f"{h} ({summary['conditions'][c]['split']})" if summary["conditions"][c]["split"] else h
        for h, c in zip(headers, conds)) + ".")
    if any(len(set(rows) - {"gt"}) > 1 for rows in summary["metrics"].values()):
        lines.append("Bold = best model per condition (GT excluded).")
    for metric in _ordered_metrics(summary["metrics"]):
        rows = summary["metrics"][metric]
        if _hidden_in_markdown(metric):
            continue
        direction = METRIC_DIRECTION.get(metric)
        note = f" ({_DIRECTION_NOTE[direction]})" if direction else ""
        heading = f"{_METRIC_LABEL[metric]} ({metric})" if metric in _METRIC_LABEL else metric
        best = {c: _best_models(rows, c, direction) if direction else set() for c in conds}
        lines += ["", f"## {heading}{note}", "",
                  "| model | " + " | ".join(headers) + " |",
                  "|---|" + "---|" * len(conds)]
        for model, row in rows.items():
            cells = []
            for c in conds:
                cell = _fmt(row.get(c)) if c in row else ""
                cells.append(f"**{cell}**" if model in best[c] else cell)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
