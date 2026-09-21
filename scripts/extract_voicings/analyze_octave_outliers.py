#!/usr/bin/env python3
"""Measure how many cached voicings show suspected melody-bleed, WITHOUT
filtering anything -- a documentation/decision step before deciding whether
to actually apply the filter.

See extract_voicings.octave_outlier_split() for the detection rule: a
minority of notes sitting >= 1 octave away from an otherwise
tightly-clustered majority. Suspected cause: the onset-window chord detector
(extract_voicings.py's --onset_tolerance) groups any notes hitting within
~50ms into one "chord", so a left-hand chord + a high right-hand melody note
struck at the same instant get recorded as a single voicing.

Usage::

    python scripts/extract_voicings/analyze_octave_outliers.py \\
        [--voicings data/voicings/merged/all_voicings.json] \\
        [--examples 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_voicings import octave_outlier_split

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def pitch_name(p: int) -> str:
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicings", default="data/voicings/merged/all_voicings.json")
    parser.add_argument("--examples", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.voicings} ...")
    with open(args.voicings, encoding="utf-8") as f:
        all_voicings: List[dict] = json.load(f)
    print(f"  {len(all_voicings):,} raw (genre, performer, pitches) rows")

    flagged_rows = 0
    flagged_occurrences = 0
    total_occurrences = 0
    minority_size_hist: dict = {}
    examples = []

    # Split flagged rows further: is every outlier pitch just an octave
    # duplicate of a pitch class already in the majority (normal voicing
    # practice -- doubling a chord tone), or does it introduce a genuinely
    # new pitch class (more likely a real stray note from another line)?
    doubling_rows = doubling_occurrences = 0
    new_pc_rows = new_pc_occurrences = 0
    doubling_examples = []
    new_pc_examples = []
    direction_hist: dict = {}
    direction_occ: dict = {}
    above_examples = []
    below_examples = []

    for v in all_voicings:
        total_occurrences += v["count"]
        split = octave_outlier_split(v["pitches"])
        if split is None:
            continue
        majority, minority = split
        flagged_rows += 1
        flagged_occurrences += v["count"]
        minority_size_hist[len(minority)] = minority_size_hist.get(len(minority), 0) + 1
        if len(examples) < args.examples:
            examples.append((v, majority, minority))

        majority_pcs = {p % 12 for p in majority}
        is_pure_doubling = all(p % 12 in majority_pcs for p in minority)
        if is_pure_doubling:
            doubling_rows += 1
            doubling_occurrences += v["count"]
            if len(doubling_examples) < args.examples:
                doubling_examples.append((v, majority, minority))
        else:
            new_pc_rows += 1
            new_pc_occurrences += v["count"]
            if len(new_pc_examples) < args.examples:
                new_pc_examples.append((v, majority, minority))
            is_above = min(minority) > max(majority)
            key = "above" if is_above else "below"
            direction_hist[key] = direction_hist.get(key, 0) + 1
            direction_occ[key] = direction_occ.get(key, 0) + v["count"]
            if is_above and len(above_examples) < args.examples:
                above_examples.append((v, majority, minority))
            if not is_above and len(below_examples) < args.examples:
                below_examples.append((v, majority, minority))

    print()
    print("=" * 60)
    print("OCTAVE-OUTLIER ANALYSIS (no filtering applied)")
    print("=" * 60)
    print(f"Unique (genre, performer, pitches) rows : {len(all_voicings):,}")
    print(f"  Flagged                               : {flagged_rows:,} "
          f"({100 * flagged_rows / len(all_voicings):.2f}%)")
    print(f"Total occurrences (sum of counts)       : {total_occurrences:,}")
    print(f"  Flagged                               : {flagged_occurrences:,} "
          f"({100 * flagged_occurrences / total_occurrences:.2f}%)")
    print()
    print("Minority (outlier) group size, among flagged rows:")
    for size, n in sorted(minority_size_hist.items()):
        print(f"  {size} outlier note(s): {n:,} rows")

    print()
    print("-" * 60)
    print("Breakdown: octave-doubling (same pitch class) vs. new pitch class")
    print("-" * 60)
    print(f"Pure octave doubling (outlier pitch class already in majority   -- "
          f"likely NORMAL voicing, not melody-bleed):")
    print(f"  Rows       : {doubling_rows:,} ({100 * doubling_rows / len(all_voicings):.2f}% of all rows, "
          f"{100 * doubling_rows / max(flagged_rows, 1):.2f}% of flagged)")
    print(f"  Occurrences: {doubling_occurrences:,} "
          f"({100 * doubling_occurrences / total_occurrences:.2f}% of all occurrences)")
    print(f"New pitch class introduced (outlier NOT a pitch class already present -- "
          f"more likely genuine melody-bleed):")
    print(f"  Rows       : {new_pc_rows:,} ({100 * new_pc_rows / len(all_voicings):.2f}% of all rows, "
          f"{100 * new_pc_rows / max(flagged_rows, 1):.2f}% of flagged)")
    print(f"  Occurrences: {new_pc_occurrences:,} "
          f"({100 * new_pc_occurrences / total_occurrences:.2f}% of all occurrences)")

    print()
    print(f"Example octave-doubling rows (up to {args.examples}, suspected NOT melody-bleed):")
    for v, majority, minority in doubling_examples:
        maj_names = [pitch_name(p) for p in majority]
        min_names = [pitch_name(p) for p in minority]
        print(f"  genre={v['genre']:10s} performer={v['performer']:20s} count={v['count']:6d}  "
              f"majority={maj_names}  outlier={min_names}")

    print()
    print(f"Example new-pitch-class rows (up to {args.examples}, more likely REAL melody-bleed):")
    for v, majority, minority in new_pc_examples:
        maj_names = [pitch_name(p) for p in majority]
        min_names = [pitch_name(p) for p in minority]
        print(f"  genre={v['genre']:10s} performer={v['performer']:20s} count={v['count']:6d}  "
              f"majority={maj_names}  outlier={min_names}")

    print()
    print("-" * 60)
    print("Within new-pitch-class rows: is the outlier ABOVE or BELOW the cluster?")
    print("(above = your described scenario: RH melody note over an LH chord)")
    print("(below = a separate bass/root note under the chord -- normal voicing)")
    print("-" * 60)
    for key in ("above", "below"):
        n_rows = direction_hist.get(key, 0)
        n_occ = direction_occ.get(key, 0)
        print(f"  {key:6s}: {n_rows:,} rows ({100*n_rows/max(new_pc_rows,1):.1f}% of new-pitch-class), "
              f"{n_occ:,} occurrences ({100*n_occ/max(new_pc_occurrences,1):.1f}% of new-pitch-class occurrences)")

    print()
    print(f"Example ABOVE rows (up to {args.examples}, your described scenario):")
    for v, majority, minority in above_examples:
        maj_names = [pitch_name(p) for p in majority]
        min_names = [pitch_name(p) for p in minority]
        print(f"  genre={v['genre']:10s} performer={v['performer']:20s} count={v['count']:6d}  "
              f"majority={maj_names}  outlier={min_names}")

    print()
    print(f"Example BELOW rows (up to {args.examples}, likely normal bass+chord):")
    for v, majority, minority in below_examples:
        maj_names = [pitch_name(p) for p in majority]
        min_names = [pitch_name(p) for p in minority]
        print(f"  genre={v['genre']:10s} performer={v['performer']:20s} count={v['count']:6d}  "
              f"majority={maj_names}  outlier={min_names}")


if __name__ == "__main__":
    main()
