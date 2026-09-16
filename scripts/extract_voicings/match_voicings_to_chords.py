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
          "C":    [{"pitches": [48, 52, 55], "count": 500, "num_songs": 120}, ...],
          "Cm7":  [{"pitches": [48, 51, 55, 58], "count": 300, "num_songs": 80}, ...],
          ...
        }

    Voicings within each chord entry are sorted by descending ``count``
    (= number of times that exact pitch combination appeared in the source).
    ``num_songs`` is how many distinct source files it appeared in at all
    (deduplicated) — useful for spotting a voicing whose count is mostly one
    song's repeated figure rather than genuinely widespread use.

If the input voicings carry ``"genre"`` / ``"performer"`` tags (see
``extract_voicings.py``), pass ``--genres`` / ``--performers`` to restrict
matching to a subset — e.g. build a jazz-only lookup, a Keith-Jarrett-only
lookup, or an all-genres/all-performers lookup, all from the *same* cached
extraction, without re-scanning any MIDI.

Usage::

    python scripts/extract_voicings/match_voicings_to_chords.py \\
        [--voicings data/voicings/pijama/all_voicings.json] \\
        [--chord_names data/cache/chord_names_augmented.json] \\
        [--output data/voicings/pijama/chord_voicings.json] \\
        [--genres jazz pop] \\
        [--performers "keith jarrett"] \\
        [--min_count 3] \\
        [--min_songs 1] \\
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
        "--genres",
        nargs="+",
        default=None,
        help=(
            "If given, only use voicings whose 'genre' tag matches one of these "
            "(case-insensitive). Entries with no 'genre' field are treated as "
            "'unknown' and only included if 'unknown' is passed explicitly. "
            "Omit to use every genre (the previous, ungenred behaviour)."
        ),
    )
    parser.add_argument(
        "--performers",
        nargs="+",
        default=None,
        help=(
            "If given, only use voicings whose 'performer' tag matches one of these "
            "(case-insensitive). Entries with no 'performer' field are treated as "
            "'unknown' and only included if 'unknown' is passed explicitly. "
            "Omit to use every performer."
        ),
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Ignore voicings seen fewer than this many times in the source (default: 3).",
    )
    parser.add_argument(
        "--min_songs",
        type=int,
        default=1,
        help=(
            "Ignore voicings that appeared in fewer than this many distinct songs "
            "(default: 1, i.e. no extra filtering beyond --min_count). Raise this to "
            "guard against a single repetitive song inflating a voicing's count."
        ),
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


def load_chord_vocab(chord_names_path: Path) -> Tuple[np.ndarray, List[str], int, int]:
    """Load and parse the chord vocabulary. Returns (masks, names, n_total, n_failed)."""
    with open(chord_names_path, encoding="utf-8") as f:
        chord_names: List[str] = json.load(f)

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

    return np.array(chord_masks, dtype=np.int32), valid_names, len(chord_names), parse_errors


def build_chord_lookup(
    all_voicings: List[dict],
    chord_masks_np: np.ndarray,
    valid_names: List[str],
    genres: Optional[List[str]] = None,
    performers: Optional[List[str]] = None,
    min_count: int = 3,
    min_songs: int = 1,
    min_notes: int = 3,
    max_notes: int = 8,
    max_voicings_per_chord: int = 500,
    show_progress: bool = True,
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """Filter/select/match voicings into a chord-name lookup table.

    This is the reusable core of the CLI in this module — factored out so a
    caller (e.g. a bulk-generation script) can run it many times against an
    already-loaded ``all_voicings`` list without re-reading the file or
    re-parsing the chord vocabulary each time.

    Returns ``(lookup, stats)`` where ``stats`` has: raw_entries,
    unique_pitch_sets, after_filtering, matched, unmatched, distinct_chords,
    total_voicing_entries, total_occurrences.
    """
    selected = all_voicings
    if genres:
        genre_set = {g.lower() for g in genres}
        selected = [v for v in selected if v.get("genre", "unknown").lower() in genre_set]
    if performers:
        performer_set = {p.lower() for p in performers}
        selected = [v for v in selected if v.get("performer", "unknown").lower() in performer_set]

    # Re-aggregate by pitch set: a genre/performer-tagged source can list the
    # same pitch set once per (genre, performer), so sum counts (and songs)
    # across whichever subset was selected before applying the thresholds.
    pitch_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    pitch_songs: Dict[Tuple[int, ...], int] = defaultdict(int)
    for v in selected:
        key = tuple(v["pitches"])
        pitch_counts[key] += v["count"]
        pitch_songs[key] += v.get("num_songs", 0)

    filtered = [
        {"pitches": list(pitches), "count": count, "num_songs": pitch_songs[pitches]}
        for pitches, count in pitch_counts.items()
        if count >= min_count
        and pitch_songs[pitches] >= min_songs
        and min_notes <= len(pitches) <= max_notes
    ]

    lookup: Dict[str, List[dict]] = defaultdict(list)
    unmatched = 0
    iterator = tqdm(filtered, desc="Matching") if show_progress else filtered
    for v in iterator:
        pitches = v["pitches"]
        voicing_mask = pitches_to_pcs_mask(pitches)

        # A chord matches if its pitch-class set is EXACTLY the voicing's
        # pitch-class set (after collapsing octave doublings via mod 12).
        matches = chord_masks_np == voicing_mask
        if not matches.any():
            unmatched += 1
            continue

        # Multiple chord names can share the same pitch-class set (e.g. slash
        # chords like "C/E" have the same PCs as "C"). Take the first match
        # (alphabetical order within the vocab).
        best_idx = int(np.where(matches)[0][0])
        best_name = valid_names[best_idx]
        lookup[best_name].append({
            "pitches": pitches, "count": v["count"], "num_songs": v["num_songs"],
        })

    result: Dict[str, List[dict]] = {}
    for chord_name in sorted(lookup.keys()):
        voicings = sorted(lookup[chord_name], key=lambda x: -x["count"])
        result[chord_name] = voicings[:max_voicings_per_chord]

    stats = {
        "raw_entries": len(all_voicings),
        "selected_entries": len(selected),
        "unique_pitch_sets": len(pitch_counts),
        "after_filtering": len(filtered),
        "matched": len(filtered) - unmatched,
        "unmatched": unmatched,
        "distinct_chords": len(result),
        "total_voicing_entries": sum(len(v) for v in result.values()),
        "total_occurrences": sum(e["count"] for v in result.values() for e in v),
    }
    return result, stats


def main() -> None:
    args = parse_args()

    print("Parsing chord vocabulary …")
    chord_masks_np, valid_names, n_total, n_failed = load_chord_vocab(Path(args.chord_names))
    print(f"  Parsed {len(valid_names)}/{n_total} chord names ({n_failed} failed)")

    print("Loading voicings …")
    with open(args.voicings, encoding="utf-8") as f:
        all_voicings = json.load(f)

    if args.genres:
        print(f"  Genre filter: {args.genres}")
    if args.performers:
        print(f"  Performer filter: {args.performers}")

    print("Matching voicings to chords …")
    result, stats = build_chord_lookup(
        all_voicings, chord_masks_np, valid_names,
        genres=args.genres, performers=args.performers,
        min_count=args.min_count, min_songs=args.min_songs,
        min_notes=args.min_notes, max_notes=args.max_notes,
        max_voicings_per_chord=args.max_voicings_per_chord,
    )

    print(f"  {stats['raw_entries']} raw entries → "
          f"{stats['selected_entries']:,} selected → "
          f"{stats['unique_pitch_sets']:,} unique pitch sets → "
          f"{stats['after_filtering']:,} after filtering "
          f"(min_count={args.min_count}, min_songs={args.min_songs}, "
          f"notes={args.min_notes}–{args.max_notes})")
    print(f"  Matched: {stats['matched']}  Unmatched: {stats['unmatched']}")
    print(f"  Distinct chords covered: {stats['distinct_chords']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. Written to {out_path}")
    print(f"  Chords with voicings : {stats['distinct_chords']}")
    print(f"  Total voicing entries: {stats['total_voicing_entries']}")

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
