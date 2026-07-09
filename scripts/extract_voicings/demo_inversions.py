#!/usr/bin/env python3
"""Play real-world voicings for random chords and write to MIDI.

Voicings are looked up from ``chord_voicings.json`` (built by
``match_voicings_to_chords.py``), so every voicing you hear actually appeared
in real piano recordings.

The first voicing played for each chord is always the naive close-position
root-position voicing (all notes squeezed into one octave starting at
``--base_midi``).  After that, the remaining voicings from the lookup table are
played in descending order of occurrence count — one voicing per beat.

Chords are separated by a half-beat rest.

Usage::

    # 10 random chords, default options
    python scripts/extract_voicings/demo_inversions.py

    # specific chords
    python scripts/extract_voicings/demo_inversions.py --chords Cm7 F7 Bbmaj7 Eb Am7b5

    # triads only, 20 random picks, faster tempo
    python scripts/extract_voicings/demo_inversions.py --max_notes 3 --num_chords 20 --bpm 120

    # limit to N voicings per chord (default: 8)
    python scripts/extract_voicings/demo_inversions.py --max_voicings 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pretty_midi

SCRIPT_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from convert_wikifonia_to_cache import parse_chord_symbol_with_noteseq  # type: ignore


# ---------------------------------------------------------------------------
# Naive close-position root voicing (first voicing shown)
# ---------------------------------------------------------------------------

def close_position_root(root_pc: int, intervals: List[int], base_midi: int) -> List[int]:
    """Return MIDI pitches for a close-position root voicing starting at base_midi.

    The root lands at the first MIDI pitch >= base_midi that matches root_pc.
    Subsequent notes are stacked by the chord's stacked semitone intervals.
    """
    root_midi = base_midi + (root_pc - base_midi % 12) % 12
    pitches = [root_midi]
    for iv in intervals:
        pitches.append(pitches[-1] + iv)
    return pitches


# ---------------------------------------------------------------------------
# MIDI rendering
# ---------------------------------------------------------------------------

def render_to_midi(
    chord_voicings: List[Tuple[str, List[List[int]]]],
    bpm: float,
    note_fraction: float,
    rest_beats: float,
    output_path: Path,
) -> None:
    """Write chord voicings to a MIDI file — one voicing per beat."""
    beat_s = 60.0 / bpm
    rest_s = rest_beats * beat_s

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    t = 0.0
    for _chord_name, voicings in chord_voicings:
        for pitches in voicings:
            note_dur = beat_s * note_fraction
            for pitch in pitches:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=max(0, min(127, pitch)),
                    start=t,
                    end=t + note_dur,
                )
                piano.notes.append(note)
            t += beat_s
        t += rest_s

    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"Written: {output_path}  ({t:.1f} s total)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chords",
        nargs="+",
        default=None,
        help="Explicit chord names (must exist in chord_voicings.json).",
    )
    p.add_argument(
        "--num_chords",
        type=int,
        default=10,
        help="Number of random chords to pick (default: 10).",
    )
    p.add_argument(
        "--min_notes",
        type=int,
        default=3,
        help="Min chord tones for random filtering (default: 3).",
    )
    p.add_argument(
        "--max_notes",
        type=int,
        default=5,
        help="Max chord tones for random filtering (default: 5).",
    )
    p.add_argument(
        "--max_voicings",
        type=int,
        default=8,
        help="Max voicings to play per chord (default: 8, by descending count).",
    )
    p.add_argument(
        "--base_midi",
        type=int,
        default=48,
        help="Lowest MIDI pitch for the naive root voicing (default: 48 = C3).",
    )
    p.add_argument(
        "--bpm",
        type=float,
        default=80.0,
        help="Tempo in BPM (default: 80).",
    )
    p.add_argument(
        "--note_fraction",
        type=float,
        default=0.85,
        help="Fraction of a beat to hold each note (default: 0.85).",
    )
    p.add_argument(
        "--rest_beats",
        type=float,
        default=1.0,
        help="Silence (in beats) inserted between chords (default: 1.0).",
    )
    p.add_argument(
        "--chord_voicings",
        default="data/voicings/merged/chord_voicings.json",
        help="Path to chord_voicings.json lookup table.",
    )
    p.add_argument(
        "--output",
        default="data/voicings/demo_inversions.mid",
        help="Output MIDI path.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ------------------------------------------------------------------ #
    # 1. Load lookup table
    # ------------------------------------------------------------------ #
    with open(args.chord_voicings, encoding="utf-8") as f:
        lookup: Dict[str, List[dict]] = json.load(f)

    # ------------------------------------------------------------------ #
    # 2. Select chords
    # ------------------------------------------------------------------ #
    if args.chords:
        names = args.chords
        missing = [n for n in names if n not in lookup]
        if missing:
            print(f"Warning: not in chord_voicings.json — {missing}")
            names = [n for n in names if n in lookup]
    else:
        # Filter by note count using note_seq parsing
        candidates = []
        for name in lookup:
            try:
                _root_pc, intervals, _ = parse_chord_symbol_with_noteseq(name)
            except Exception:
                continue
            n_tones = 1 + len(intervals)
            if args.min_notes <= n_tones <= args.max_notes:
                candidates.append(name)

        if not candidates:
            print("No chords matched the filter — relax --min_notes/--max_notes.")
            sys.exit(1)

        names = random.sample(candidates, min(args.num_chords, len(candidates)))
        print(f"Selected {len(names)} random chords "
              f"({len(candidates)} candidates with {args.min_notes}–{args.max_notes} tones):\n")

    # ------------------------------------------------------------------ #
    # 3. Build voicing lists: naive root first, then lookup
    # ------------------------------------------------------------------ #
    PITCH_NAMES = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    chord_voicings: List[Tuple[str, List[List[int]]]] = []

    for name in names:
        # Naive close-position root voicing
        try:
            root_pc, intervals, _ = parse_chord_symbol_with_noteseq(name)
            naive = close_position_root(root_pc, intervals, args.base_midi)
        except Exception:
            naive = None

        # Real voicings from lookup (already sorted by count desc)
        real = [entry["pitches"] for entry in lookup.get(name, [])]
        real = real[: args.max_voicings]

        all_voicings: List[List[int]] = []
        if naive is not None:
            all_voicings.append(naive)
        all_voicings.extend(real)

        if not all_voicings:
            print(f"[skip] {name!r} — no voicings available")
            continue

        # Print summary
        print(f"{name}  ({len(all_voicings)} voicings played)")
        for i, pitches in enumerate(all_voicings):
            pcs = [PITCH_NAMES[p % 12] for p in pitches]
            tag = "naive" if i == 0 and naive is not None else f"#{i}"
            count_str = ""
            if i > 0 or naive is None:
                idx = i - (1 if naive is not None else 0)
                if 0 <= idx < len(lookup.get(name, [])):
                    count_str = f"  (count={lookup[name][idx]['count']})"
            print(f"  [{tag:5}]  pitches={pitches}  pc={pcs}{count_str}")

        chord_voicings.append((name, all_voicings))

    if not chord_voicings:
        print("Nothing to render.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. Render
    # ------------------------------------------------------------------ #
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_to_midi(chord_voicings, args.bpm, args.note_fraction, args.rest_beats, out)


if __name__ == "__main__":
    main()
