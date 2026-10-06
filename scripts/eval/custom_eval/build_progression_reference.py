#!/usr/bin/env python3
"""Build the chord-progression reference for a corpus from its TRAINING split.

Why training and not the test split's ground truth: the reference is a statement
about what progressions look like in this corpus, not about the particular melodies
being harmonised, and the training split has 5-12x more chord changes. The test
split's ground truth is still computed here, as the *floor* -- the divergence a
perfect model would show from finite sampling alone (see progression_metrics).

JAZZMUS has 579 chord changes in its whole test split against ~5,400 in training, so
without this the reference would be the sparser side of the comparison.

Writes data/cache/progression_reference/<corpus>.json, which records the chord vocab
it was built against so a vocab change invalidates it rather than silently
mismatching. Run once per corpus; every later evaluation reuses it.

Usage:
    python scripts/eval/custom_eval/build_progression_reference.py --datasets all
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from realchords.utils import progression_metrics as pm
from realchords.utils.experiment_utils import create_dataset_dataloaders

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data" / "cache" / "progression_reference"
DATASETS = [
    "hooktheory", "pop909", "emopia_plus", "nottingham",
    "wikifonia", "chord_melody_dataset", "jazzmus",
]


def _sequences(dataset_name: str, split: str, max_len: int):
    """Chord lanes for every non-overlapping window of every song in a split."""
    built = create_dataset_dataloaders(
        dataset_name=dataset_name, dataset_split=split, model_part="chord",
        batch_size=64, max_len=max_len, num_workers=0, all_windows=True,
    )
    loader = next(x for x in built if x is not None)
    dataset = loader.dataset
    tokenizer = dataset.tokenizer
    id_to_name = tokenizer.id_to_name
    for i in range(len(dataset)):
        item = dataset[i]
        # Not mask-filtered: dropping padded positions before the lane split would
        # shift the parity. Padding parses to None in chord_events and is skipped.
        yield pm.chord_lane(item["targets"].tolist()), id_to_name


def _collect(dataset_name: str, split: str, max_len: int, classes=None, use_cache: bool = True):
    """-> (list of event sequences, raw quality counts).

    Events are cached verbatim (ungrouped, so a later change to the class list does
    not invalidate them); grouping onto a class list happens on load.
    """
    meta = {"max_len": max_len, "split": split, "dataset_name": dataset_name}
    cache = pm.events_cache_path(dataset_name, split)
    sequences = pm.load_events(cache, meta) if use_cache else None
    if sequences is None:
        sequences = [
            pm.chord_events(lane, id_to_name)
            for lane, id_to_name in _sequences(dataset_name, split, max_len)
        ]
        pm.save_events(cache, sequences, meta)
        print(f"  cached {len(sequences):,} {split} sequences -> {cache}")
    else:
        print(f"  loaded {len(sequences):,} {split} sequences from cache")
    if classes is not None:
        allowed = set(classes)
        sequences = [[(r, q if q in allowed else pm.OTHER) for r, q in s] for s in sequences]
    quality_counts = Counter(q for s in sequences for _, q in s)
    return sequences, quality_counts


def build(dataset_name: str, max_len: int, coverage: float) -> dict:
    started = time.time()

    # Pass 1: raw quality counts on train, to derive this corpus's class list.
    train_raw, quality_counts = _collect(dataset_name, "train", max_len)
    classes = pm.derive_classes(quality_counts, coverage=coverage)

    allowed = set(classes)
    train = [
        [(r, q if q in allowed else pm.OTHER) for r, q in events] for events in train_raw
    ]
    test, _ = _collect(dataset_name, "test", max_len, classes=classes)

    ref_hist = pm.histograms(train, classes)
    ref_trans = pm.transitions(train)
    test_trans = pm.transitions(test)

    # Floor: held-out ground truth against the reference, at the test split's own
    # sample size. Every model's distance is read against this.
    floor = pm.compare(test_trans, ref_hist, classes, match_n=len(test_trans))

    return {
        "dataset_name": dataset_name,
        "classes": classes,
        "coverage_target": coverage,
        "quality_counts_raw": {k: v for k, v in quality_counts.most_common()},
        "num_classes_present": len(quality_counts),
        "train": {
            "num_sequences": len(train),
            "num_transitions": len(ref_trans),
            "histograms": {
                name: {json.dumps(k): v for k, v in hist.items()}
                for name, hist in ref_hist.items()
            },
        },
        "test_floor": {
            "num_sequences": len(test),
            "num_transitions": len(test_trans),
            "js_vs_train": floor,
            "conditionals": pm.conditionals(test_trans),
        },
        "train_conditionals": pm.conditionals(ref_trans),
        "max_len": max_len,
        "build_seconds": round(time.time() - started, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--max_len", type=int, default=512, help="tokens; 512 = 16 bars/lane")
    ap.add_argument("--coverage", type=float, default=0.99)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    names = DATASETS if args.datasets == ["all"] else args.datasets
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        ref = build(name, args.max_len, args.coverage)
        path = args.out_dir / f"{name}.json"
        path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        tr, te = ref["train"], ref["test_floor"]
        print(f"  classes ({len(ref['classes'])}): {', '.join(ref['classes'])}")
        print(f"  train {tr['num_sequences']:,} seqs / {tr['num_transitions']:,} transitions")
        print(f"  test  {te['num_sequences']:,} seqs / {te['num_transitions']:,} transitions")
        print(f"  floor: " + "  ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=--"
            for k, v in te["js_vs_train"].items()))
        print(f"  -> {path}  ({ref['build_seconds']}s)")


if __name__ == "__main__":
    main()
