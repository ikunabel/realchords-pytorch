#!/usr/bin/env python3
"""CLI demo for the music-theory-aware voicing selector.

Usage::

    python scripts/extract_voicings/voicing_selector.py \\
        --voicings data/voicings/merged/chord_voicings.json \\
        --chords Cmaj7 Am7 Dm7 G7 Cmaj7

    # With a melody ceiling at A4 (MIDI 69)
    python scripts/extract_voicings/voicing_selector.py \\
        --chords Cmaj7 Am7 Dm7 G7 --melody_pitch 69
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from realchords.utils.voicing_selector import VoicingSelector, _vl_cost


def _demo(args: argparse.Namespace) -> None:
    sel = VoicingSelector(
        args.voicings,
        target_mid=args.target_mid,
        vl_weight=args.vl_weight,
        reg_weight=args.reg_weight,
        count_weight=args.count_weight,
    )

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    prev: Optional[List[int]] = None
    print(f"{'Chord':<12}  {'Pitches':<30}  {'Centroid':>8}  {'VL cost':>8}")
    print("-" * 65)
    for chord in args.chords:
        pitches = sel.select(
            chord,
            prev_voicing=prev,
            melody_pitch=args.melody_pitch,
            melody_role=args.melody_role,
        )
        if pitches is None:
            print(f"{chord:<12}  (no voicing found)")
        else:
            centroid = sum(pitches) / len(pitches)
            vl = _vl_cost(prev, pitches) if prev else 0.0
            note_names = [
                NOTE_NAMES[p % 12] + str(p // 12 - 1) for p in sorted(pitches)
            ]
            print(
                f"{chord:<12}  {str(sorted(pitches)):<30}  "
                f"{centroid:>8.1f}  {vl:>8.1f}  "
                f"({', '.join(note_names)})"
            )
            prev = pitches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voicings",
        default="data/voicings/merged/chord_voicings.json",
        help="Path to chord_voicings.json lookup table.",
    )
    parser.add_argument(
        "--chords",
        nargs="+",
        default=["Cmaj7", "Am7", "Dm7", "G7", "Cmaj7"],
        help="Chord sequence to demo.",
    )
    parser.add_argument("--target_mid", type=int, default=60)
    parser.add_argument("--melody_pitch", type=int, default=None)
    parser.add_argument("--melody_role", default="top", choices=["top", "bass"])
    parser.add_argument("--vl_weight", type=float, default=0.6)
    parser.add_argument("--reg_weight", type=float, default=0.3)
    parser.add_argument("--count_weight", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    _demo(_parse_args())
