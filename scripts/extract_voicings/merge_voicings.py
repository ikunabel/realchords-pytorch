#!/usr/bin/env python3
"""Merge multiple all_voicings.json files into one by summing counts.

Counts are summed per (genre, performer, pitches) triple, so both tags
survive the merge and can still be filtered on downstream (e.g. in
match_voicings_to_chords.py). ``count`` and ``num_songs`` are summed
independently — safe as long as the input files are disjoint sets of songs
(true for merging different source datasets). Entries missing "genre",
"performer" or "num_songs" (older-format files) default to "unknown" / 0.

Usage::

    python scripts/extract_voicings/merge_voicings.py \\
        data/voicings/pijama/all_voicings.json \\
        data/voicings/aria-midi-all/all_voicings.json \\
        --output data/voicings/merged/all_voicings.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to all_voicings.json files to merge.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for merged all_voicings.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counter: Counter = Counter()
    song_counter: Counter = Counter()

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            voicings = json.load(f)
        n = sum(v["count"] for v in voicings)
        print(f"  {path}: {len(voicings):,} unique voicings, {n:,} total events")
        for v in voicings:
            genre = v.get("genre", "unknown")
            performer = v.get("performer", "unknown")
            key = (genre, performer, tuple(v["pitches"]))
            counter[key] += v["count"]
            song_counter[key] += v.get("num_songs", 0)

    merged = [
        {
            "genre": genre,
            "performer": performer,
            "pitches": list(pitches),
            "count": count,
            "num_songs": song_counter[(genre, performer, pitches)],
        }
        for (genre, performer, pitches), count in counter.most_common()
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    total = sum(v["count"] for v in merged)
    print(f"\nMerged: {len(merged):,} unique voicings, {total:,} total events")

    genre_totals: Counter = Counter()
    performer_totals: Counter = Counter()
    for v in merged:
        genre_totals[v["genre"]] += v["count"]
        performer_totals[v["performer"]] += v["count"]
    print("Events by genre:")
    for genre, n in genre_totals.most_common():
        print(f"  {genre:12s}: {n:,}")
    print("Events by performer (top 15):")
    for performer, n in performer_totals.most_common(15):
        print(f"  {performer:20s}: {n:,}")

    print(f"Written to: {out}")


if __name__ == "__main__":
    main()
