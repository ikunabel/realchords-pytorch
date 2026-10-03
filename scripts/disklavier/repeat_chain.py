#!/usr/bin/env python3
"""Play bar 1 of the disklavier_hold-time demo straight to the piano.

Three legato quarter notes on one key (80 BPM by default), each released
--gap_ms before the next, repeated in --groups groups with a rest between.
Listen for whether notes 2 and 3 of each group sound. Bypasses the browser,
so it separates the piano's behaviour from ReaLJam's MIDI timing; used for
the velocity / key sweep in journal/DISKLAVIER.md.

Requires MIDI IN Delay OFF on the piano, and python-rtmidi (mido's backend).

Usage::

    python scripts/disklavier/repeat_chain.py --out "<output port>" \\
        [--velocity 55 --gap_ms 150 --pitch 60 --bpm 80 --groups 5]
"""

from __future__ import annotations

import argparse
import time

import mido

from mixed_velocity_restrike import wait_until

NOTES_PER_GROUP = 3
GROUP_REST_S = 1.5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output port name.")
    p.add_argument("--velocity", type=int, default=55)
    p.add_argument("--gap_ms", type=float, default=150, help="Release before each repeat.")
    p.add_argument("--pitch", type=int, default=60)
    p.add_argument("--bpm", type=float, default=80)
    p.add_argument("--groups", type=int, default=5)
    args = p.parse_args()

    spacing, gap = 60 / args.bpm, args.gap_ms / 1000
    with mido.open_output(args.out) as out:
        try:
            for _ in range(args.groups):
                t0 = time.perf_counter()
                for i in range(NOTES_PER_GROUP):
                    wait_until(t0 + i * spacing)
                    out.send(mido.Message("note_on", note=args.pitch, velocity=args.velocity))
                    last = i == NOTES_PER_GROUP - 1
                    wait_until(t0 + (i + 1) * spacing - (0 if last else gap))
                    out.send(mido.Message("note_off", note=args.pitch, velocity=0))
                wait_until(t0 + NOTES_PER_GROUP * spacing + GROUP_REST_S)
        finally:
            out.send(mido.Message("control_change", control=123, value=0))  # all notes off


if __name__ == "__main__":
    main()
