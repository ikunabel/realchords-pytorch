#!/usr/bin/env python3
"""Distribution of annotated keys/modes in the Hooktheory dataset.

Hooktheory annotates every entry (one song *section*) with its key: a tonic
pitch class and the scale as step intervals, e.g. [2, 2, 1, 2, 2, 2] = major
(ionian), plus the beat at which each key starts (sections can change key).
This counts, per split, how many sections are in each mode, how many change
key, and how the tonics are distributed -- used to decide which mode groups
are large enough to report key-aware metrics for separately (see
journal/DATASET_SUMMARY.md, "Hooktheory keys and modes").

Reads the non-augmented cache files data/cache/hooktheory/{split}.jsonl (the
augmented ones contain key-transposed copies of the same sections). Mode
counts use single-key sections only, since a section with key changes has no
single mode.

Usage (from the repo root):
    python scripts/hooktheory_mode_statistics.py
    python scripts/hooktheory_mode_statistics.py --splits test
"""

import argparse
import collections
import json
from pathlib import Path

MODE_NAMES = {
    (2, 2, 1, 2, 2, 2): "ionian (major)",
    (2, 1, 2, 2, 1, 2): "aeolian (natural minor)",
    (2, 1, 2, 2, 2, 1): "dorian",
    (2, 2, 1, 2, 2, 1): "mixolydian",
    (1, 2, 2, 2, 1, 2): "phrygian",
    (2, 2, 2, 1, 2, 2): "lydian",
    (1, 2, 2, 1, 2, 2): "locrian",
    (2, 1, 2, 2, 1, 3): "harmonic minor",
    (1, 3, 1, 2, 1, 2): "phrygian dominant",
}
PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def count_split(path: Path):
    modes: collections.Counter = collections.Counter()
    tonics: collections.Counter = collections.Counter()
    num_sections = num_key_changes = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            keys = json.loads(line)["annotations"]["keys"]
            num_sections += 1
            if len(keys) != 1:
                num_key_changes += 1
                continue
            steps = tuple(keys[0]["scale_degree_intervals"])
            modes[MODE_NAMES.get(steps, f"other {list(steps)}")] += 1
            tonics[PITCH_CLASS_NAMES[keys[0]["tonic_pitch_class"] % 12]] += 1
    return num_sections, num_key_changes, modes, tonics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default="data/cache/hooktheory")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = parser.parse_args()

    for split in args.splits:
        num_sections, num_changes, modes, tonics = count_split(Path(args.cache_dir) / f"{split}.jsonl")
        single = num_sections - num_changes
        print(f"\n=== {split}: {num_sections} sections, {single} single-key, "
              f"{num_changes} with key changes ({100 * num_changes / num_sections:.1f}%)")
        print("  mode (single-key sections):")
        for mode, n in modes.most_common():
            print(f"    {mode:28s} {n:6d}  ({100 * n / single:5.1f}%)")
        print("  tonic (single-key sections): " + ", ".join(
            f"{t} {100 * n / single:.1f}%" for t, n in tonics.most_common()))


if __name__ == "__main__":
    main()
