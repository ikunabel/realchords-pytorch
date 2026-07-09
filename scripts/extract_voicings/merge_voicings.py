#!/usr/bin/env python3
"""Merge multiple all_voicings.json files into one by summing counts.

Usage::

    python scripts/extract_voicings/merge_voicings.py \\
        data/voicings/pijama/all_voicings.json \\
        data/voicings/aria-midi-jazz/all_voicings.json \\
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

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            voicings = json.load(f)
        n = sum(v["count"] for v in voicings)
        print(f"  {path}: {len(voicings):,} unique voicings, {n:,} total events")
        for v in voicings:
            counter[tuple(v["pitches"])] += v["count"]

    merged = [
        {"pitches": list(pitches), "count": count}
        for pitches, count in counter.most_common()
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    total = sum(v["count"] for v in merged)
    print(f"\nMerged: {len(merged):,} unique voicings, {total:,} total events")
    print(f"Written to: {out}")


if __name__ == "__main__":
    main()
