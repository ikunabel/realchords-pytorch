#!/usr/bin/env python3
"""Extract chord voicings from a directory of MIDI files.

A "voicing" is a set of MIDI pitches whose note-on events fall within a short
time window (``--onset_tolerance``, default 50 ms). Only groups of at least
``--min_notes`` simultaneous notes are kept.

Supports optional metadata-based genre filtering (e.g. for aria-midi).

Output
------
``<output_dir>/all_voicings.json``
    List of ``{"pitches": [...], "count": N}`` dicts sorted by descending count.

``<midi_output_dir>/<relative_path>/<file>.mid``  (when --midi_output_dir is set)
    One MIDI file per input file containing only the detected chord voicings,
    placed at their original onset times.  The folder structure under
    ``--input_dir`` is mirrored under ``--midi_output_dir``.

Usage::

    # All files in a directory tree
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/pijama \\
        --output_dir data/voicings/pijama

    # Also write per-song chord MIDIs
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/pijama \\
        --output_dir data/voicings/pijama \\
        --midi_output_dir data/voicings/pijama/chord_midi

    # aria-midi, jazz genre only, with 8 parallel workers
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/aria-midi-v1-deduped-ext/data \\
        --metadata_json data/aria-midi-v1-deduped-ext/metadata.json \\
        --genres jazz \\
        --output_dir data/voicings/aria-midi-jazz \\
        --midi_output_dir data/voicings/aria-midi-jazz/chord_midi \\
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pretty_midi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core extraction (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

_ONSET_TOLERANCE: float = 0.05
_MIN_NOTES: int = 3

# (onset_seconds, sorted_pitch_tuple)
TimedChord = Tuple[float, Tuple[int, ...]]


def _extract_file(midi_path: Path) -> List[TimedChord]:
    """Extract timed chord events from one MIDI file.

    Returns a list of (onset_seconds, sorted_pitch_tuple) pairs, one per
    detected simultaneous note group.  Returns [] on parse error or if no
    chord groups are found.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return []

    all_notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return []

    all_notes.sort(key=lambda n: (n.start, -n.pitch))

    chords: List[TimedChord] = []
    group_start: float = all_notes[0].start
    group_pitches: List[int] = []

    for note in all_notes:
        if note.start - group_start <= _ONSET_TOLERANCE:
            group_pitches.append(note.pitch)
        else:
            if len(group_pitches) >= _MIN_NOTES:
                chords.append((group_start, tuple(sorted(set(group_pitches)))))
            group_start = note.start
            group_pitches = [note.pitch]

    if len(group_pitches) >= _MIN_NOTES:
        chords.append((group_start, tuple(sorted(set(group_pitches)))))

    return chords


def _extract_file_wrapper(args: Tuple) -> Tuple[Path, List[TimedChord]]:
    """Wrapper for multiprocessing: returns (path, timed_chords)."""
    path, onset_tol, min_notes = args
    global _ONSET_TOLERANCE, _MIN_NOTES
    _ONSET_TOLERANCE = onset_tol
    _MIN_NOTES = min_notes
    return path, _extract_file(path)


# ---------------------------------------------------------------------------
# Per-file MIDI writing
# ---------------------------------------------------------------------------

def write_chord_midi(
    timed_chords: List[TimedChord],
    out_path: Path,
    max_hold: float = 2.0,
    velocity: int = 80,
) -> None:
    """Write a MIDI file containing only the detected chord voicings.

    Each chord starts at its original onset time and ends at the next chord's
    onset (or onset + ``max_hold`` for the final chord).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI()
    instr = pretty_midi.Instrument(program=0, name="Chords")

    for i, (onset, pitches) in enumerate(timed_chords):
        if i + 1 < len(timed_chords):
            offset = timed_chords[i + 1][0]
        else:
            offset = onset + max_hold

        for pitch in pitches:
            instr.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=max(0, min(127, pitch)),
                start=onset,
                end=offset,
            ))

    pm.instruments.append(instr)
    pm.write(str(out_path))


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def load_genre_ids(
    metadata_json: Path,
    genres: List[str],
) -> Optional[Set[int]]:
    """Return a set of integer file IDs matching the requested genres.

    aria-midi metadata keys are plain integers (e.g. "31357") while file stems
    are zero-padded (e.g. "031357"). We normalise both to int.
    Returns None if no metadata_json / genres provided.
    """
    genre_set = {g.lower() for g in genres}
    with open(metadata_json, encoding="utf-8") as f:
        meta: Dict = json.load(f)

    ids: Set[int] = set()
    for file_id, entry in meta.items():
        g = entry.get("metadata", {}).get("genre", "").lower()
        if g in genre_set:
            try:
                ids.add(int(file_id))
            except ValueError:
                pass
    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory to scan recursively for .mid / .midi files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output files.",
    )
    parser.add_argument(
        "--metadata_json",
        default=None,
        help="Optional path to metadata.json for genre filtering.",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=None,
        help="If given, only process files whose metadata genre matches (e.g. jazz pop).",
    )
    parser.add_argument(
        "--onset_tolerance",
        type=float,
        default=0.05,
        help="Max time gap (seconds) to group notes as a chord (default: 0.05).",
    )
    parser.add_argument(
        "--min_notes",
        type=int,
        default=3,
        help="Minimum notes in a group to call it a chord (default: 3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Parallel worker processes (default: CPU count − 1).",
    )
    parser.add_argument(
        "--midi_output_dir",
        default=None,
        help=(
            "If set, write one chord MIDI per input file into this directory, "
            "mirroring the folder structure under --input_dir. "
            "The input files themselves are never modified."
        ),
    )
    parser.add_argument(
        "--max_hold",
        type=float,
        default=2.0,
        help="Max duration (seconds) for the last chord in each file (default: 2.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally pre-build genre ID filter (fast: only reads metadata JSON)
    genre_ids: Optional[Set[int]] = None
    if args.metadata_json and args.genres:
        genre_ids = load_genre_ids(Path(args.metadata_json), args.genres)
        print(f"Genre filter: {args.genres} → {len(genre_ids):,} matching IDs in metadata")

    # Collect MIDI files, applying genre filter inline to avoid double-scan
    all_files: List[Path] = []
    for mid in sorted(set(input_dir.rglob("*.mid")) | set(input_dir.rglob("*.midi"))):
        if genre_ids is not None:
            stem = mid.stem.split("_")[0]
            try:
                if int(stem) not in genre_ids:
                    continue
            except ValueError:
                continue
        all_files.append(mid)

    label = f" (genre={args.genres})" if genre_ids is not None else ""
    print(f"Found {len(all_files)} MIDI files{label}")

    midi_out: Optional[Path] = Path(args.midi_output_dir) if args.midi_output_dir else None
    if midi_out:
        midi_out.mkdir(parents=True, exist_ok=True)
        print(f"Chord MIDIs will be written to: {midi_out}")

    # Extract voicings in parallel
    global _ONSET_TOLERANCE, _MIN_NOTES
    _ONSET_TOLERANCE = args.onset_tolerance
    _MIN_NOTES = args.min_notes

    task_args = [(f, args.onset_tolerance, args.min_notes) for f in all_files]
    counter: Counter = Counter()
    skipped = 0
    total_events = 0
    midi_written = 0

    print(f"Extracting with {args.workers} worker(s) …")
    with mp.Pool(processes=args.workers) as pool:
        for src_path, timed_chords in tqdm(
            pool.imap_unordered(_extract_file_wrapper, task_args, chunksize=32),
            total=len(all_files),
            desc="Extracting",
        ):
            if not timed_chords:
                skipped += 1
                continue

            total_events += len(timed_chords)
            for _onset, pitches in timed_chords:
                counter[pitches] += 1

            if midi_out is not None:
                rel = Path(src_path).relative_to(input_dir)
                out_mid = midi_out / rel.with_suffix(".mid")
                write_chord_midi(timed_chords, out_mid, max_hold=args.max_hold)
                midi_written += 1

    print(f"\nDone.")
    print(f"  Files processed : {len(all_files) - skipped}")
    print(f"  Files skipped   : {skipped}")
    print(f"  Total events    : {total_events:,}")
    print(f"  Unique voicings : {len(counter):,}")
    if midi_out:
        print(f"  MIDI files written: {midi_written:,} → {midi_out}")

    # Save voicings JSON
    all_voicings = [
        {"pitches": list(pitches), "count": count}
        for pitches, count in counter.most_common()
    ]
    out_path = output_dir / "all_voicings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_voicings, f, indent=2)
    print(f"  Voicings JSON   : {out_path}")


if __name__ == "__main__":
    main()
