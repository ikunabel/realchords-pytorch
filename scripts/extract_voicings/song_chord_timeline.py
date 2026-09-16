#!/usr/bin/env python3
"""Extract a time-aligned chord-symbol timeline for ONE song.

Unlike extract_voicings.py / match_voicings_to_chords.py, which aggregate
across an entire corpus, this operates on a single MIDI file and keeps every
detected chord cluster's own onset time — intended for building side-by-side
visualizations (e.g. "PiJAMA's own notes" vs. "chords extracted from it" vs.
"a human transcription"), where the first two are automatically time-aligned
with each other (same source clock) and the third needs separate alignment.

Output
------
``--output_json`` (optional)
    List of ``{"onset": t, "offset": t, "pitches": [...], "chord_name": name_or_null}``
    dicts, one per detected cluster, in time order. ``chord_name`` is null
    when the cluster's pitch-class set doesn't exactly match any chord in
    the vocabulary (see match_voicings_to_chords.py's matching rule).

``--output_midi`` (optional)
    A chord-only MIDI file (reuses extract_voicings.write_chord_midi),
    directly renderable to notation alongside the original song MIDI. Always
    built from the FULL detected timeline (matched + unmatched clusters)
    unless ``--only_matched`` is passed, in which case unmatched/noisy
    clusters are dropped first — the preceding real chord's held duration
    then simply extends across the gap to the next recognized chord, since
    write_chord_midi derives each chord's duration from the next entry in
    whatever list it's given.

The ``--output_json`` timeline always contains every detected cluster
(matched or not) with ``chord_name: null`` for unmatched ones, regardless of
``--only_matched`` — that flag only affects the MIDI rendering.

Usage::

    # Full detail: every detected cluster, matched or not
    python scripts/extract_voicings/song_chord_timeline.py \\
        --midi "data/pijama/midi_hawthorne/midi/studio/Brad Mehldau/Suite -  April 2020/New York State of Mind.midi" \\
        --output_json /tmp/nysom_hawthorne_chords.json \\
        --output_midi /tmp/nysom_hawthorne_chords_full.mid

    # Only the chords we could actually identify (drop unmatched noise)
    python scripts/extract_voicings/song_chord_timeline.py \\
        --midi "data/pijama/midi_hawthorne/midi/studio/Brad Mehldau/Suite -  April 2020/New York State of Mind.midi" \\
        --output_midi /tmp/nysom_hawthorne_chords_matched_only.mid \\
        --only_matched

    # Compare both PiJAMA transcriptions of the same song
    python scripts/extract_voicings/song_chord_timeline.py \\
        --midi "data/pijama/midi_kong/studio/Brad Mehldau/Suite -  April 2020/New York State of Mind.midi" \\
        --output_json /tmp/nysom_kong_chords.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_voicings import TimedChord, _extract_file, write_chord_midi
from match_voicings_to_chords import chord_name_to_pcs, pitches_to_pcs_mask, pcs_to_mask


def load_chord_vocab(chord_names_path: Path):
    with open(chord_names_path, encoding="utf-8") as f:
        chord_names: List[str] = json.load(f)
    masks, names = [], []
    for name in chord_names:
        pcs = chord_name_to_pcs(name)
        if pcs is not None:
            masks.append(pcs_to_mask(pcs))
            names.append(name)
    return np.array(masks, dtype=np.int32), names


def match_pitches(pitches, masks_np: np.ndarray, names: List[str]) -> Optional[str]:
    """Same exact-pitch-class-set matching rule as match_voicings_to_chords.py."""
    mask = pitches_to_pcs_mask(pitches)
    hits = np.where(masks_np == mask)[0]
    return names[int(hits[0])] if len(hits) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", required=True, help="Path to a single song's MIDI file.")
    parser.add_argument(
        "--chord_names",
        default="data/cache/chord_names_augmented.json",
        help="Path to chord_names_augmented.json (global vocabulary).",
    )
    parser.add_argument("--output_json", default=None, help="Optional path for the timed chord-symbol JSON.")
    parser.add_argument("--output_midi", default=None, help="Optional path for a chord-only MIDI rendering.")
    parser.add_argument(
        "--only_matched",
        action="store_true",
        help=(
            "For --output_midi only: drop clusters that didn't match a real chord "
            "symbol before writing, so the rendered MIDI shows only chords we could "
            "actually identify. The preceding chord's duration extends to cover the gap."
        ),
    )
    parser.add_argument("--onset_tolerance", type=float, default=0.05)
    parser.add_argument("--min_notes", type=int, default=3)
    parser.add_argument("--max_hold", type=float, default=2.0, help="Duration for the final chord (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import extract_voicings as ev
    ev._ONSET_TOLERANCE = args.onset_tolerance
    ev._MIN_NOTES = args.min_notes

    midi_path = Path(args.midi)
    timed_chords: List[TimedChord] = _extract_file(midi_path)
    print(f"{midi_path.name}: {len(timed_chords)} chord-like clusters detected")

    masks_np, names = load_chord_vocab(Path(args.chord_names))

    unmatched = 0
    timeline = []
    matched_only: List[TimedChord] = []
    for i, (onset, pitches) in enumerate(timed_chords):
        offset = timed_chords[i + 1][0] if i + 1 < len(timed_chords) else onset + args.max_hold
        chord_name = match_pitches(pitches, masks_np, names)
        if chord_name is None:
            unmatched += 1
        else:
            matched_only.append((onset, pitches))
        timeline.append({
            "onset": round(onset, 4),
            "offset": round(offset, 4),
            "pitches": list(pitches),
            "chord_name": chord_name,
        })

    print(f"  Matched: {len(timeline) - unmatched}  Unmatched: {unmatched}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)
        print(f"  Timeline JSON written to: {out_path}")

    if args.output_midi:
        midi_source = matched_only if args.only_matched else timed_chords
        if args.only_matched:
            print(f"  --only_matched: writing {len(midi_source)}/{len(timed_chords)} clusters "
                  f"(dropped {len(timed_chords) - len(midi_source)} unmatched)")
        write_chord_midi(midi_source, Path(args.output_midi), max_hold=args.max_hold)
        print(f"  Chord-only MIDI written to: {args.output_midi}")


if __name__ == "__main__":
    main()
