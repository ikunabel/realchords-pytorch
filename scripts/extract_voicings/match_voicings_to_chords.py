#!/usr/bin/env python3
"""Build a chord → voicings lookup table from extracted PIJAMA voicings.

For each voicing (a set of MIDI pitches), find the **most specific** chord in
our vocabulary whose pitch-class set is a subset of the voicing's pitch classes.
"Most specific" = the matching chord with the most required pitch classes (e.g.
a Cmaj9 voicing maps to Cmaj9, not to C or Cmaj7).

Output
------
``<output>``  (default: ``data/voicings/pijama/chord_voicings.json``)
    A JSON object mapping each chord name that has at least one matched
    voicing to a sorted list of voicing dicts::

        {
          "C":    [{"pitches": [48, 52, 55], "count": 500}, ...],
          "Cm7":  [{"pitches": [48, 51, 55, 58], "count": 300}, ...],
          ...
        }

    Voicings within each chord entry are sorted by descending ``count``
    (= number of times that exact pitch combination appeared in PIJAMA).

Usage::

    python scripts/extract_voicings/match_voicings_to_chords.py \\
        [--voicings data/voicings/pijama/all_voicings.json] \\
        [--chord_names data/cache/chord_names_augmented.json] \\
        [--output data/voicings/pijama/chord_voicings.json] \\
        [--min_count 3] \\
        [--min_notes 3] \\
        [--max_notes 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

CONVERT_DIR = str(Path(__file__).resolve().parents[1] / "convert_data_to_cache")
if CONVERT_DIR not in sys.path:
    sys.path.insert(0, CONVERT_DIR)

from convert_wikifonia_to_cache import parse_chord_symbol_with_noteseq


# ---------------------------------------------------------------------------
# Chord → pitch-class set
# ---------------------------------------------------------------------------

def chord_name_to_pcs(name: str) -> Optional[FrozenSet[int]]:
    """Return the (root-position) pitch-class set for a chord name."""
    try:
        root_pc, intervals, _ = parse_chord_symbol_with_noteseq(name)
    except Exception:
        return None
    pcs = {root_pc % 12}
    curr = root_pc
    for iv in intervals:
        curr += iv
        pcs.add(curr % 12)
    return frozenset(pcs)


def pitches_to_pcs_mask(pitches: List[int]) -> int:
    """Convert a list of MIDI pitches to a 12-bit bitmask of pitch classes."""
    mask = 0
    for p in pitches:
        mask |= (1 << (p % 12))
    return mask


def pcs_to_mask(pcs: FrozenSet[int]) -> int:
    mask = 0
    for pc in pcs:
        mask |= (1 << pc)
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voicings",
        default="data/voicings/pijama/all_voicings.json",
        help="Path to all_voicings.json from extract_pijama_voicings.py.",
    )
    parser.add_argument(
        "--chord_names",
        default="data/cache/chord_names_augmented.json",
        help="Path to chord_names_augmented.json (global vocabulary).",
    )
    parser.add_argument(
        "--output",
        default="data/voicings/pijama/chord_voicings.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Ignore voicings seen fewer than this many times across PIJAMA (default: 3).",
    )
    parser.add_argument(
        "--min_notes",
        type=int,
        default=3,
        help="Minimum note count per voicing (default: 3).",
    )
    parser.add_argument(
        "--max_notes",
        type=int,
        default=8,
        help="Maximum note count per voicing (default: 8; avoids dense clusters).",
    )
    parser.add_argument(
        "--max_voicings_per_chord",
        type=int,
        default=500,
        help="Cap on stored voicings per chord (default: 500, by descending count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load and parse chord vocabulary
    # ------------------------------------------------------------------
    print("Parsing chord vocabulary …")
    with open(args.chord_names, encoding="utf-8") as f:
        chord_names: List[str] = json.load(f)

    # chord_masks[i] = 12-bit bitmask for chord i
    # chord_n_pcs[i] = number of distinct pitch classes in chord i
    chord_masks: List[int] = []
    valid_names: List[str] = []
    parse_errors = 0

    for name in chord_names:
        pcs = chord_name_to_pcs(name)
        if pcs is None:
            parse_errors += 1
            continue
        chord_masks.append(pcs_to_mask(pcs))
        valid_names.append(name)

    chord_masks_np = np.array(chord_masks, dtype=np.int32)

    print(f"  Parsed {len(valid_names)}/{len(chord_names)} chord names "
          f"({parse_errors} failed)")

    # ------------------------------------------------------------------
    # 2. Load voicings and apply filters
    # ------------------------------------------------------------------
    print("Loading voicings …")
    with open(args.voicings, encoding="utf-8") as f:
        all_voicings = json.load(f)

    filtered = [
        v for v in all_voicings
        if v["count"] >= args.min_count
        and args.min_notes <= len(v["pitches"]) <= args.max_notes
    ]
    print(f"  {len(all_voicings)} total → {len(filtered)} after filtering "
          f"(min_count={args.min_count}, notes={args.min_notes}–{args.max_notes})")

    # ------------------------------------------------------------------
    # 3. Match each voicing to its most specific chord
    # ------------------------------------------------------------------
    print("Matching voicings to chords …")
    lookup: Dict[str, List[dict]] = defaultdict(list)
    unmatched = 0

    for v in tqdm(filtered, desc="Matching"):
        pitches = v["pitches"]
        count = v["count"]
        voicing_mask = pitches_to_pcs_mask(pitches)

        # A chord matches if its pitch-class set is EXACTLY the voicing's
        # pitch-class set (after collapsing octave doublings via mod 12).
        # Octave doublings are allowed (G3 and G4 both contribute the G pitch
        # class), but no extra or missing pitch classes are permitted.
        matches = chord_masks_np == voicing_mask  # bool array

        if not matches.any():
            unmatched += 1
            continue

        # Multiple chord names can share the same pitch-class set (e.g. slash
        # chords like "C/E" have the same PCs as "C"). Take the first match
        # (alphabetical order within the vocab).
        best_idx = int(np.where(matches)[0][0])
        best_name = valid_names[best_idx]

        lookup[best_name].append({"pitches": pitches, "count": count})

    print(f"  Matched: {len(filtered) - unmatched}  Unmatched: {unmatched}")
    print(f"  Distinct chords covered: {len(lookup)}")

    # ------------------------------------------------------------------
    # 4. Sort each chord's list by count desc and cap
    # ------------------------------------------------------------------
    result: Dict[str, List[dict]] = {}
    for chord_name in sorted(lookup.keys()):
        voicings = sorted(lookup[chord_name], key=lambda x: -x["count"])
        result[chord_name] = voicings[: args.max_voicings_per_chord]

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. Written to {out_path}")
    print(f"  Chords with voicings : {len(result)}")
    print(f"  Total voicing entries: {sum(len(v) for v in result.values())}")

    # Quick sample
    print("\nSample (top 3 voicings for a few chords):")
    sample_names = list(result.keys())[:8]
    NAMES = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    for name in sample_names:
        top = result[name][:3]
        print(f"  {name!r:20}")
        for entry in top:
            pcs = [NAMES[p % 12] for p in entry['pitches']]
            print(f"    count={entry['count']:5d}  pitches={entry['pitches']}  pc={pcs}")


if __name__ == "__main__":
    main()
