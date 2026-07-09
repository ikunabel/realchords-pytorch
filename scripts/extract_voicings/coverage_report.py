#!/usr/bin/env python3
"""Report voicing coverage statistics for a chord vocabulary.

Compares ``chord_names_augmented.json`` against a ``chord_voicings.json``
lookup table and prints a breakdown of which chords are covered, which are
missing, and why.

Usage::

    python scripts/extract_voicings/coverage_report.py \\
        [--chord_names data/cache/chord_names_augmented.json] \\
        [--chord_voicings data/voicings/pijama/chord_voicings.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chord_names",
        default="data/cache/chord_names_augmented.json",
    )
    parser.add_argument(
        "--chord_voicings",
        default="data/voicings/pijama/chord_voicings.json",
    )
    return parser.parse_args()


def chord_type(name: str) -> str:
    """Extract the bare chord type (strip root, slash bass, accidentals)."""
    # Remove slash bass
    name = name.split("/")[0]
    # Remove root note (A-G with optional # or b)
    name = re.sub(r"^[A-G][#b]?", "", name)
    return name or "major"


def main() -> None:
    args = parse_args()

    with open(args.chord_names, encoding="utf-8") as f:
        vocab: list = json.load(f)
    with open(args.chord_voicings, encoding="utf-8") as f:
        voicings: dict = json.load(f)

    covered = set(voicings.keys())
    vocab_set = set(vocab)
    missing = vocab_set - covered

    # --- overall ---
    print("=" * 60)
    print("VOICING COVERAGE REPORT")
    print("=" * 60)
    print(f"Chord vocab size          : {len(vocab_set):>6}")
    print(f"Chords with voicings      : {len(covered):>6}  ({100*len(covered)/len(vocab_set):.1f}%)")
    print(f"Chords without voicings   : {len(missing):>6}  ({100*len(missing)/len(vocab_set):.1f}%)")

    total_voicings = sum(len(v) for v in voicings.values())
    total_occurrences = sum(e["count"] for v in voicings.values() for e in v)
    print(f"Total voicing entries     : {total_voicings:>6,}")
    print(f"Total chord occurrences   : {total_occurrences:>6,}")
    print()

    # --- voicings per chord histogram ---
    counts = [len(v) for v in voicings.values()]
    buckets = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 500), (501, 10000)]
    print("Voicings per chord:")
    for lo, hi in buckets:
        n = sum(1 for c in counts if lo <= c <= hi)
        print(f"  {lo:>4}–{hi:<5}  {n:>5} chords")
    print()

    # --- missing: slash vs non-slash ---
    slash_missing = [c for c in missing if "/" in c]
    nonslash_missing = sorted(c for c in missing if "/" not in c)
    print(f"Missing — slash chords    : {len(slash_missing):>6}")
    print(f"Missing — non-slash chords: {len(nonslash_missing):>6}")
    print()

    # --- missing non-slash chords grouped by type ---
    type_counter: Counter = Counter()
    for name in nonslash_missing:
        type_counter[chord_type(name)] += 1

    print("Missing non-slash chords by type (top 20):")
    for ctype, n in type_counter.most_common(20):
        print(f"  {ctype or 'major':30}  {n:>4} chords missing")
    print()

    # --- covered chord stats: most and least voicings ---
    sorted_covered = sorted(voicings.items(), key=lambda x: -len(x[1]))
    print("Best covered chords (most voicings):")
    for name, vlist in sorted_covered[:10]:
        total_occ = sum(e["count"] for e in vlist)
        print(f"  {name:30}  {len(vlist):>4} voicings  {total_occ:>7,} occurrences")
    print()
    print("Sparsely covered chords (fewest voicings, covered > 0):")
    for name, vlist in sorted_covered[-10:]:
        total_occ = sum(e["count"] for e in vlist)
        print(f"  {name:30}  {len(vlist):>4} voicings  {total_occ:>7,} occurrences")


if __name__ == "__main__":
    main()
