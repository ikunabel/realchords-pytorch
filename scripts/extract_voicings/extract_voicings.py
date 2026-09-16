#!/usr/bin/env python3
"""Extract chord voicings from a directory of MIDI files.

A "voicing" is a set of MIDI pitches whose note-on events fall within a short
time window (``--onset_tolerance``, default 50 ms). Only groups of at least
``--min_notes`` simultaneous notes are kept.

Every voicing count is tagged with a genre and a performer, so selection by
either can happen later (e.g. in ``match_voicings_to_chords.py``) without
re-running extraction:

- **Genre** comes from ``--metadata_json`` (e.g. aria-midi's per-file
  ``metadata.genre`` field). ``"unknown"`` if not given or not present.
- **Performer** comes from EITHER ``--metadata_json`` (aria-midi's
  ``metadata.performer`` field — sparsely populated, ~7% of aria-midi) OR
  ``--performer_path_index`` (a path-component index for datasets that encode
  performer in the folder structure, e.g. PIJAMA's
  ``<root>/<subdir>/midi/<live|studio>/<Performer Name>/...``, index 3
  relative to ``--input_dir``). ``"unknown"`` if neither is given/matches.

Two counts are kept per (genre, performer, pitches) voicing, NOT just one,
because a single song repeating a voicing many times (a vamp, an ostinato
comping figure) would otherwise make that voicing look far more common or
characteristic than it really is — the same problem text corpora solve with
document frequency instead of raw term frequency:

- ``count``     — total number of occurrences across all scanned files.
- ``num_songs`` — number of DISTINCT files it appeared in at least once
                  (deduplicated per file before counting).

Both survive merging (``merge_voicings.py``) and are available for
downstream filtering/weighting (``match_voicings_to_chords.py``).

Output
------
``<output_dir>/all_voicings.json``
    List of ``{"genre": ..., "performer": ..., "pitches": [...],
    "count": N, "num_songs": M}`` dicts, sorted by descending count.

``<midi_output_dir>/<relative_path>/<file>.mid``  (when --midi_output_dir is set)
    One MIDI file per input file containing only the detected chord voicings,
    placed at their original onset times.  The folder structure under
    ``--input_dir`` is mirrored under ``--midi_output_dir``.

Usage::

    # PIJAMA: no genre metadata, but performer is the folder right after
    # "live"/"studio" (depth differs between its two subtrees, so match on
    # the anchor name rather than a fixed index)
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/pijama \\
        --output_dir data/voicings/pijama \\
        --performer_path_anchor live studio

    # aria-midi, ALL genres + performers cached, 8 workers
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/aria-midi-v1-deduped-ext/data \\
        --metadata_json data/aria-midi-v1-deduped-ext/metadata.json \\
        --output_dir data/voicings/aria-midi-all \\
        --midi_output_dir data/voicings/aria-midi-all/chord_midi \\
        --workers 8

    # aria-midi, only scan jazz files (still tagged, just a smaller scan)
    python scripts/extract_voicings/extract_voicings.py \\
        --input_dir data/aria-midi-v1-deduped-ext/data \\
        --metadata_json data/aria-midi-v1-deduped-ext/metadata.json \\
        --genres jazz \\
        --output_dir data/voicings/aria-midi-jazz \\
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

def _load_metadata_field(metadata_json: Path, field: str) -> Dict[int, str]:
    """Return a mapping from integer file ID to a lowercased metadata field.

    aria-midi metadata keys are plain integers (e.g. "31357") while file stems
    are zero-padded (e.g. "031357"). We normalise both to int. Files with no
    value for this field (missing or empty string) are tagged ``"unknown"``.
    """
    with open(metadata_json, encoding="utf-8") as f:
        meta: Dict = json.load(f)

    values: Dict[int, str] = {}
    for file_id, entry in meta.items():
        v = str(entry.get("metadata", {}).get(field, "")).strip().lower() or "unknown"
        try:
            values[int(file_id)] = v
        except ValueError:
            pass
    return values


def load_file_genres(metadata_json: Path) -> Dict[int, str]:
    """Return a mapping from integer file ID to its genre label."""
    return _load_metadata_field(metadata_json, "genre")


def load_file_performers(metadata_json: Path) -> Dict[int, str]:
    """Return a mapping from integer file ID to its performer label."""
    return _load_metadata_field(metadata_json, "performer")


def file_id_from_stem(stem: str) -> Optional[int]:
    """Best-effort parse of an aria-midi-style numeric ID from a file stem."""
    try:
        return int(stem.split("_")[0])
    except ValueError:
        return None


def performer_from_path(
    midi_path: Path,
    input_dir: Path,
    path_index: int,
) -> Optional[str]:
    """Return the performer name from a FIXED path component of ``midi_path``.

    ``path_index`` is the 0-based index into the path relative to
    ``input_dir``. Only safe when every scanned subtree has the same folder
    depth up to the performer name — PIJAMA does NOT (see
    ``performer_from_path_after_anchor`` instead). Returns None if the
    relative path is too short.
    """
    try:
        parts = midi_path.relative_to(input_dir).parts
    except ValueError:
        return None
    if path_index >= len(parts):
        return None
    return parts[path_index].strip().lower() or None


def performer_from_path_after_anchor(
    midi_path: Path,
    input_dir: Path,
    anchors: List[str],
) -> Optional[str]:
    """Return the path component right after the first matching anchor.

    Robust to inconsistent folder depth across subtrees — e.g. PIJAMA has
    ``midi_hawthorne/midi/live/<performer>/...`` (anchor at index 2) but
    ``midi_kong/live/<performer>/...`` (anchor at index 1). Matching on the
    anchor name itself (case-insensitive) rather than a fixed index handles
    both. Returns None if no anchor is found or nothing follows it.
    """
    try:
        parts = midi_path.relative_to(input_dir).parts
    except ValueError:
        return None
    anchor_set = {a.lower() for a in anchors}
    for i, part in enumerate(parts):
        if part.lower() in anchor_set and i + 1 < len(parts):
            return parts[i + 1].strip().lower() or None
    return None


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
        "--performer_path_index",
        type=int,
        default=None,
        help=(
            "0-based index of the performer name within the path relative to "
            "--input_dir. Only safe if every scanned subtree has the same "
            "folder depth up to the performer name — prefer "
            "--performer_path_anchor otherwise. Ignored if --metadata_json "
            "already provides a 'performer' field for a file."
        ),
    )
    parser.add_argument(
        "--performer_path_anchor",
        nargs="+",
        default=None,
        help=(
            "Path component name(s) marking the performer folder as the NEXT "
            "component (case-insensitive), robust to inconsistent folder depth "
            "across subtrees — e.g. for PIJAMA: --performer_path_anchor live studio "
            "(midi_hawthorne/midi/live/<performer>/... and "
            "midi_kong/live/<performer>/... both resolve correctly). Takes "
            "precedence over --performer_path_index if both are given."
        ),
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

    # Load per-file genre/performer labels (used for output tagging
    # regardless of whether --genres is also used to restrict which files
    # are scanned).
    file_genres: Dict[int, str] = {}
    file_performers: Dict[int, str] = {}
    if args.metadata_json:
        file_genres = load_file_genres(Path(args.metadata_json))
        file_performers = load_file_performers(Path(args.metadata_json))

    # Optionally restrict which files are scanned to specific genres. This is
    # purely a scan-time speed optimization now — omit --genres to scan
    # everything and keep all genres in the cached output.
    genre_ids: Optional[Set[int]] = None
    if args.genres:
        genre_set = {g.lower() for g in args.genres}
        genre_ids = {fid for fid, g in file_genres.items() if g in genre_set}
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
    # Keyed by (genre, performer, pitches) so selection along either axis can
    # happen later without re-scanning. `counter` is raw occurrence count
    # (term-frequency-like); `song_counter` is deduplicated per file (one
    # increment per file that contains the voicing at all, regardless of how
    # many times) — document-frequency-like, so a single repetitive song
    # can't be mistaken for a widely-used voicing downstream.
    counter: Counter = Counter()
    song_counter: Counter = Counter()
    genre_file_counts: Counter = Counter()
    performer_file_counts: Counter = Counter()
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

            src_path = Path(src_path)
            fid = file_id_from_stem(src_path.stem) if file_genres or file_performers else None

            genre = "unknown"
            if file_genres and fid is not None:
                genre = file_genres.get(fid, "unknown")
            genre_file_counts[genre] += 1

            performer = "unknown"
            if file_performers and fid is not None:
                performer = file_performers.get(fid, "unknown")
            if performer == "unknown" and args.performer_path_anchor:
                performer = (
                    performer_from_path_after_anchor(src_path, input_dir, args.performer_path_anchor)
                    or "unknown"
                )
            elif performer == "unknown" and args.performer_path_index is not None:
                performer = performer_from_path(src_path, input_dir, args.performer_path_index) or "unknown"
            performer_file_counts[performer] += 1

            total_events += len(timed_chords)
            seen_in_this_file: Set[Tuple[int, ...]] = set()
            for _onset, pitches in timed_chords:
                key = (genre, performer, pitches)
                counter[key] += 1
                if pitches not in seen_in_this_file:
                    song_counter[key] += 1
                    seen_in_this_file.add(pitches)

            if midi_out is not None:
                rel = src_path.relative_to(input_dir)
                out_mid = midi_out / rel.with_suffix(".mid")
                write_chord_midi(timed_chords, out_mid, max_hold=args.max_hold)
                midi_written += 1

    print(f"\nDone.")
    print(f"  Files processed : {len(all_files) - skipped}")
    print(f"  Files skipped   : {skipped}")
    print(f"  Total events    : {total_events:,}")
    print(f"  Unique (genre, performer, pitches) rows: {len(counter):,}")
    if file_genres:
        print("  Files by genre  :")
        for genre, n in genre_file_counts.most_common():
            print(f"    {genre:12s}: {n:,}")
    if file_performers or args.performer_path_index is not None or args.performer_path_anchor:
        print("  Files by performer (top 15):")
        for performer, n in performer_file_counts.most_common(15):
            print(f"    {performer:20s}: {n:,}")
    if midi_out:
        print(f"  MIDI files written: {midi_written:,} → {midi_out}")

    # Save voicings JSON
    all_voicings = [
        {
            "genre": genre,
            "performer": performer,
            "pitches": list(pitches),
            "count": count,
            "num_songs": song_counter[(genre, performer, pitches)],
        }
        for (genre, performer, pitches), count in counter.most_common()
    ]
    out_path = output_dir / "all_voicings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_voicings, f, indent=2)
    print(f"  Voicings JSON   : {out_path}")


if __name__ == "__main__":
    main()
