#!/usr/bin/env python3
"""Extract chord voicings from PIJAMA piano MIDI files.

A "voicing" is defined as a set of MIDI pitches whose note-on events fall
within a short time window (``--onset_tolerance``, default 50 ms).  Only
groups of at least ``--min_notes`` simultaneous notes are kept; isolated
single notes are discarded as melodic events.

No chord-name labelling is done here — the raw pitch sets are saved as-is.
Chord-name matching can be applied in a later step.

Output
------
``<output_dir>/per_file/<stem>.json``
    One file per input MIDI.  Each file contains a list of chord events::

        [
          {"onset": 0.575, "pitches": [61, 68, 72, 77]},
          ...
        ]

``<output_dir>/all_voicings.json``
    Aggregate of every unique pitch set across all files, sorted by
    descending occurrence count::

        [
          {"pitches": [61, 68, 72, 77], "count": 142},
          ...
        ]

Usage::

    python scripts/extract_voicings/extract_pijama_voicings.py \\
        [--pijama_dir data/pijama] \\
        [--output_dir data/voicings/pijama] \\
        [--onset_tolerance 0.05] \\
        [--min_notes 3] \\
        [--subdirs midi_hawthorne midi_kong]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pretty_midi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Chord extraction
# ---------------------------------------------------------------------------

def extract_chords_from_midi(
    midi_path: Path,
    onset_tolerance: float = 0.05,
    min_notes: int = 3,
) -> List[Dict]:
    """Extract simultaneous note groups from a single MIDI file.

    Args:
        midi_path: Path to the MIDI file.
        onset_tolerance: Maximum time gap (seconds) between the first and last
            note of a group for them to be considered simultaneous.
        min_notes: Minimum number of notes to count as a chord (groups
            smaller than this are discarded).

    Returns:
        List of chord events, each a dict with:
            - ``onset``  (float): onset time of the chord in seconds
            - ``pitches`` (list[int]): sorted MIDI pitches in the chord
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []

    # Collect all non-drum notes across all tracks
    all_notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return []

    # Sort by onset time, break ties by pitch (descending — bass first)
    all_notes.sort(key=lambda n: (n.start, -n.pitch))

    chords: List[Dict] = []
    group_start: float = all_notes[0].start
    group_pitches: List[int] = []

    for note in all_notes:
        if note.start - group_start <= onset_tolerance:
            # Still within the onset window of this group
            group_pitches.append(note.pitch)
        else:
            # Close the current group
            if len(group_pitches) >= min_notes:
                chords.append({
                    "onset": round(group_start, 4),
                    "pitches": sorted(set(group_pitches)),
                })
            # Start a new group
            group_start = note.start
            group_pitches = [note.pitch]

    # Close the final group
    if len(group_pitches) >= min_notes:
        chords.append({
            "onset": round(group_start, 4),
            "pitches": sorted(set(group_pitches)),
        })

    return chords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pijama_dir",
        type=str,
        default="data/pijama",
        help="Root directory of the PIJAMA dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/voicings/pijama",
        help="Directory where output JSON files are written.",
    )
    parser.add_argument(
        "--onset_tolerance",
        type=float,
        default=0.05,
        help="Max time gap (seconds) to group notes as a chord (default: 0.05 = 50 ms).",
    )
    parser.add_argument(
        "--min_notes",
        type=int,
        default=3,
        help="Minimum notes in a group to call it a chord (default: 3).",
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["midi_hawthorne", "midi_kong"],
        help="Sub-directories of pijama_dir to scan (default: midi_hawthorne midi_kong).",
    )
    parser.add_argument(
        "--no_per_file",
        action="store_true",
        help="Skip writing per-file JSON files (only write all_voicings.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pijama_dir = Path(args.pijama_dir)
    output_dir = Path(args.output_dir)
    per_file_dir = output_dir / "per_file"

    if not args.no_per_file:
        per_file_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all MIDI paths
    midi_files: List[Path] = []
    for subdir in args.subdirs:
        for ext in ("*.midi", "*.mid"):
            midi_files.extend((pijama_dir / subdir).rglob(ext))
    midi_files = sorted(set(midi_files))
    print(f"Found {len(midi_files)} MIDI files in {args.subdirs}")

    voicing_counter: Counter = Counter()
    total_chords = 0
    skipped = 0

    for midi_path in tqdm(midi_files, desc="Extracting voicings"):
        chords = extract_chords_from_midi(
            midi_path,
            onset_tolerance=args.onset_tolerance,
            min_notes=args.min_notes,
        )

        if not chords:
            skipped += 1
            continue

        total_chords += len(chords)

        # Count pitch sets for the aggregate
        for chord in chords:
            key: Tuple[int, ...] = tuple(chord["pitches"])
            voicing_counter[key] += 1

        # Write per-file JSON
        if not args.no_per_file:
            # Use a flat stem: replace path separators to avoid nesting issues
            rel = midi_path.relative_to(pijama_dir)
            stem = str(rel).replace("/", "__").replace(" ", "_")
            stem = stem.rsplit(".", 1)[0] + ".json"
            out_path = per_file_dir / stem
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(chords, f)

    # Write aggregate voicings sorted by frequency
    all_voicings = [
        {"pitches": list(pitches), "count": count}
        for pitches, count in voicing_counter.most_common()
    ]
    agg_path = output_dir / "all_voicings.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_voicings, f, indent=2)

    print(f"\nDone.")
    print(f"  MIDI files processed : {len(midi_files) - skipped}")
    print(f"  Files skipped        : {skipped}")
    print(f"  Total chord events   : {total_chords}")
    print(f"  Unique pitch sets    : {len(voicing_counter)}")
    print(f"  Aggregate written to : {agg_path}")


if __name__ == "__main__":
    main()
