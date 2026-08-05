#!/usr/bin/env python3
"""Run a multi-model custom-eval experiment from one yml, with results
combined into one experiment folder -- gt/, models/<slug>/, means.json,
midi/<song>/.

Why this isn't just one custom_evaluation.py --model list: checkpoints
being compared can use different chord vocabularies (confirmed, e.g.
decoder.chord.hooktheory.seed=1 uses a 5902-token vocab,
decoder.chord.7sets.seed=1.alpha=0.5 uses 6198 -- see
journal/SEQUENCE_EVALUATION.md) and/or different architectures
(decoder-only vs encoder-decoder). custom_evaluation.py's generation step
needs one correct tokenizer per run, so mixing differently-vocabbed
checkpoints in one --model list would silently misdecode whichever doesn't
match. Instead, each model here runs through its own single-model
custom_evaluation.py pass (already the safe, validated path -- each gets
its own correctly-matched vocab/tokenizer/architecture), into a private
staging directory, and this script merges the results afterward:

  - gt/: copied from the first model's staging run. Safe to treat as
    canonical across all models -- every model was conditioned on the
    exact same melody sequence (same dataset_name/split/batch_size/seed
    across all sub-runs => identical deterministic traversal order), and
    while the *chord* vocab differs numerically between models, chord
    metrics (entropy, durations, complexity, ...) are computed from
    decoded chord *names*, not raw ids, so they don't depend on which
    vocab encoded them.
  - models/<slug>/: copied directly from each model's own staging run.
  - means.json: each model's own "models" entry, computed correctly
    in its own sub-run, merged into one file; "gt" row from the first.
  - metadata.jsonl: merged by seq_idx (song_url/provenance identical
    across sub-runs; nicr/mode_fit combined from every model).
  - midi/<song>/: song-folder names are identical across sub-runs (same
    seed + same population size => select_midi_indices picks the same
    songs) -- gt.mid copied from the first, <slug>.mid from each.

Experiment yml schema:
    dataset_name: hooktheory
    dataset_split: test
    batch_size: 64
    num_batches: -1
    seed: 42
    midi_samples: 10
    save_dir: logs/custom_eval/four_way_hooktheory_vs_7sets   # optional,
        # defaults to logs/custom_eval/<yml filename stem>
    models:
      - label: Decoder_HT
        checkpoint: /path/to/step=11000.ckpt
        contrastive_checkpoint: /path/to/matching_reward/step=8000.ckpt  # optional
      - label: Decoder_7sets
        checkpoint: /path/to/step=13000.ckpt
        # contrastive_checkpoint omitted -- no reward model exists trained
        # on this checkpoint's dataset mix, so Vendi is simply not
        # computed for it (vendi_score: null) rather than computed wrong
        # against a mismatched checkpoint. Per-model, not shared across
        # the experiment -- see custom_evaluation.py's
        # contrastive_checkpoint docstring for why.

Usage:
    python scripts/eval/custom_eval/run_multi_model_eval.py \
        configs/custom_eval/experiments/four_way_hooktheory_vs_7sets.yml
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_SCRIPT = REPO_ROOT / "realchords" / "utils" / "custom_evaluation.py"


def _slugify(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "model"


def _run_one_model(
    label: str, checkpoint: str, contrastive_checkpoint: str, shared: dict, staging_dir: Path
) -> None:
    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--base_model", checkpoint,
        "--model", f"{label}=base",
        "--dataset_name", str(shared["dataset_name"]),
        "--dataset_split", str(shared["dataset_split"]),
        "--save_dir", str(staging_dir),
        "--batch_size", str(shared["batch_size"]),
        "--num_batches", str(shared["num_batches"]),
        "--seed", str(shared["seed"]),
        "--midi_samples", str(shared["midi_samples"]),
    ]
    if contrastive_checkpoint:
        cmd += ["--contrastive_checkpoint", contrastive_checkpoint]
    else:
        print(f"  (no contrastive_checkpoint for {label} -- Vendi score will not be computed for it)")
    print(f"\n=== {label} ({_slugify(label)}) -> {staging_dir} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed (exit {result.returncode}); see output above.")


def _merge_means(experiment_dir: Path, staging_dirs: Dict[str, Path]) -> None:
    combined = {"gt": None, "models": {}}
    for label, staging_dir in staging_dirs.items():
        with (staging_dir / "means.json").open(encoding="utf-8") as fh:
            means = json.load(fh)
        if combined["gt"] is None:
            combined["gt"] = means["gt"]
        slug = _slugify(label)
        combined["models"][slug] = means["models"][slug]
    with (experiment_dir / "means.json").open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, sort_keys=True)
    print(f"  means.json  ({experiment_dir / 'means.json'})")


def _merge_metadata(experiment_dir: Path, staging_dirs: Dict[str, Path]) -> List[Dict]:
    labels = list(staging_dirs.keys())
    first_label = labels[0]
    with (staging_dirs[first_label] / "metadata.jsonl").open(encoding="utf-8") as fh:
        merged = [json.loads(line) for line in fh]

    for label in labels[1:]:
        slug = _slugify(label)
        with (staging_dirs[label] / "metadata.jsonl").open(encoding="utf-8") as fh:
            other = [json.loads(line) for line in fh]
        for entry, other_entry in zip(merged, other):
            assert entry["seq_idx"] == other_entry["seq_idx"], (
                "metadata.jsonl rows out of alignment across sub-runs -- "
                "dataset_name/dataset_split/batch_size/num_batches/seed "
                "must be identical across every model in the experiment yml."
            )
            entry["nicr"][slug] = other_entry["nicr"][slug]
            entry["mode_fit"][slug] = other_entry["mode_fit"][slug]

    with (experiment_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for entry in merged:
            fh.write(json.dumps(entry) + "\n")
    print(f"  metadata.jsonl  ({len(merged)} rows, {len(labels)} model(s) merged)")
    return merged


def _merge_model_labels(experiment_dir: Path, staging_dirs: Dict[str, Path], dataset_name: str) -> None:
    with (experiment_dir / "model_labels.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_name": dataset_name,
                "labels": {label: _slugify(label) for label in staging_dirs},
            },
            fh,
            indent=2,
        )
    print(f"  model_labels.json  ({experiment_dir / 'model_labels.json'})")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _merge_midi(experiment_dir: Path, staging_dirs: Dict[str, Path]) -> None:
    labels = list(staging_dirs.keys())
    first_midi = staging_dirs[labels[0]] / "midi"
    if not first_midi.is_dir():
        return  # midi_samples was 0
    midi_dst = experiment_dir / "midi"
    midi_dst.mkdir(parents=True, exist_ok=True)

    song_folders = sorted(p.name for p in first_midi.iterdir() if p.is_dir())
    for song_folder in song_folders:
        out_song_dir = midi_dst / song_folder
        out_song_dir.mkdir(parents=True, exist_ok=True)
        gt_src = first_midi / song_folder / "gt.mid"
        if gt_src.exists():
            shutil.copy2(gt_src, out_song_dir / "gt.mid")
        for label in labels:
            slug = _slugify(label)
            src = staging_dirs[label] / "midi" / song_folder / f"{slug}.mid"
            if src.exists():
                shutil.copy2(src, out_song_dir / f"{slug}.mid")
    print(f"  midi/  ({len(song_folders)} song folder(s) x {1 + len(labels)} source(s))")


def run_experiment(experiment_yml: Path) -> Path:
    with experiment_yml.open(encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    model_specs = spec["models"]
    if not model_specs:
        raise ValueError(f"{experiment_yml}: 'models' list is empty.")

    shared = {
        "dataset_name": spec.get("dataset_name", "hooktheory"),
        "dataset_split": spec.get("dataset_split", "test"),
        "batch_size": spec.get("batch_size", 64),
        "num_batches": spec.get("num_batches", -1),
        "seed": spec.get("seed", 42),
        "midi_samples": spec.get("midi_samples", 10),
    }

    experiment_dir = Path(spec.get("save_dir", f"logs/custom_eval/{experiment_yml.stem}"))
    staging_root = REPO_ROOT / "logs" / "custom_eval" / "_staging" / experiment_yml.stem

    staging_dirs: Dict[str, Path] = {}
    for model_spec in model_specs:
        label = model_spec["label"]
        slug = _slugify(label)
        staging_dir = staging_root / slug
        _run_one_model(
            label, model_spec["checkpoint"], model_spec.get("contrastive_checkpoint", ""),
            shared, staging_dir,
        )
        staging_dirs[label] = staging_dir

    print(f"\n=== Merging {len(staging_dirs)} model(s) into {experiment_dir} ===")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    first_label = next(iter(staging_dirs))
    _copy_tree(staging_dirs[first_label] / "gt", experiment_dir / "gt")
    print(f"  gt/  (from {first_label})")

    models_dst = experiment_dir / "models"
    models_dst.mkdir(parents=True, exist_ok=True)
    for label, staging_dir in staging_dirs.items():
        slug = _slugify(label)
        _copy_tree(staging_dir / "models" / slug, models_dst / slug)
        print(f"  models/{slug}/")

    _merge_means(experiment_dir, staging_dirs)
    _merge_metadata(experiment_dir, staging_dirs)
    _merge_model_labels(experiment_dir, staging_dirs, shared["dataset_name"])

    # Vocab snapshot: same rationale as gt/ -- copy from the first model,
    # only used for re-decoding gt.pt (melody+chord *names* are what
    # matter, not which model happened to produce the snapshot).
    snapshot_src = staging_dirs[first_label] / "chord_names_augmented.json"
    if snapshot_src.exists():
        shutil.copy2(snapshot_src, experiment_dir / "chord_names_augmented.json")

    _merge_midi(experiment_dir, staging_dirs)

    print(f"\nDone. Combined experiment at {experiment_dir}")
    return experiment_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_yml", type=str, help="Path to the multi-model experiment yml.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(Path(args.experiment_yml))
