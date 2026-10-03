#!/usr/bin/env python3
"""First contact with a Disklavier: ports, playback, and echo.

1. Lists MIDI ports so you can find the interface.
2. Plays middle C at a few velocities so you can confirm the keys move.
3. Listens while doing so: if the piano's MIDI OUT reports the notes its own
   motors just played, that echo would reach ReaLJam as "your" melody during
   a simultaneous duet. Reports whether it happens and how late it arrives.
4. Optionally waits for you to play a few keys, to confirm input works.

Requires python-rtmidi (mido's backend).

Usage::

    python scripts/disklavier/check_connection.py --list
    python scripts/disklavier/check_connection.py --in "<input port>" --out "<output port>"
"""

from __future__ import annotations

import argparse
import time

import mido

TEST_PITCH = 60
VELOCITIES = (30, 64, 100)
LISTEN_S = 1.0


def list_ports() -> None:
    print("Inputs: ", mido.get_input_names() or "(none)")
    print("Outputs:", mido.get_output_names() or "(none)")


def drain(port) -> None:
    for _ in port.iter_pending():
        pass


def play_and_listen(inp, out) -> None:
    print(f"\nPlaying pitch {TEST_PITCH} at velocities {VELOCITIES}.")
    print("Watch/listen: the key should go down each time.\n")
    echoes = []
    for vel in VELOCITIES:
        drain(inp)
        sent = time.perf_counter()
        out.send(mido.Message("note_on", note=TEST_PITCH, velocity=vel))
        seen = None
        while time.perf_counter() - sent < LISTEN_S:
            for msg in inp.iter_pending():
                if msg.type == "note_on" and msg.velocity > 0 and msg.note == TEST_PITCH:
                    seen = (time.perf_counter() - sent) * 1000, msg.velocity
            if seen:
                break
            time.sleep(0.001)
        out.send(mido.Message("note_off", note=TEST_PITCH, velocity=0))
        if seen:
            print(f"  vel {vel:>3}: ECHO after {seen[0]:6.1f} ms (reported vel {seen[1]})")
            echoes.append(seen)
        else:
            print(f"  vel {vel:>3}: no echo")
        time.sleep(0.8)

    print()
    if echoes:
        print("The piano echoes notes it plays. During a duet ReaLJam would hear")
        print("its own chords as melody. Either find the Disklavier setting that")
        print("stops MIDI OUT from reporting playback, or filter echoes in software.")
    else:
        print("No echo: everything arriving on the input is the human. Good.")


def listen_to_human(inp, seconds: float) -> None:
    print(f"\nPlay a few keys on the piano ({seconds:.0f} s)...")
    drain(inp)
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        for msg in inp.iter_pending():
            if msg.type in ("note_on", "note_off", "control_change"):
                print("  ", msg)
                n += msg.type == "note_on" and msg.velocity > 0
        time.sleep(0.001)
    print(f"Received {n} note-ons." if n else "Nothing received: check the piano-OUT -> interface-IN cable.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List ports and exit.")
    p.add_argument("--in", dest="inp", help="Input port name (from --list).")
    p.add_argument("--out", help="Output port name (from --list).")
    p.add_argument("--human_s", type=float, default=8.0,
                   help="Seconds to listen for your playing (0 to skip).")
    args = p.parse_args()

    if args.list or not (args.inp and args.out):
        list_ports()
        return

    with mido.open_input(args.inp) as inp, mido.open_output(args.out) as out:
        try:
            play_and_listen(inp, out)
            if args.human_s > 0:
                listen_to_human(inp, args.human_s)
        finally:
            out.send(mido.Message("control_change", control=123, value=0))  # all notes off


if __name__ == "__main__":
    main()
