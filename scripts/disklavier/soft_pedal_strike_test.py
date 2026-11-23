#!/usr/bin/env python3
"""Listening test: does the soft pedal weaken the Disklavier's own strikes?

Background (journal/reference/REALJAM_CHORD_TIMING.md, 2026-11-17): chords
triggered with the left (soft) pedal in ReaLJam's triggered mode sounded much
quieter than the same chords triggered with space, although the same MIDI
velocity is sent. With the pedal already held down, space-triggered chords
were loud. Hypothesis: the strike is weakened when it falls while the pedal
(and with it the upright's hammer rail) is moving -- some form of lost motion.

The script drives everything itself: the Disklavier moves its soft pedal on
CC 67 (continuous values work), and the chord is struck at velocity 40 at
exactly defined moments. Each trial is a pair: a reference strike with the
pedal up, then the test strike; you rate the test strike against the
reference, without being told the condition:

    1 = much quieter   2 = quieter   3 = same   4 = louder   5 = much louder

Conditions (each repeated --reps times, in random order):
    up            pedal up (control: should be rated 3)
    held-32/64/96/127   pedal held still at that position, settled 1.5 s
    moving+0/25/50/100  pedal pressed from 0 to 127 over RAMP_S, the chord
                        struck that many ms after the movement starts

Safety: velocity is capped at 64, notes are released after NOTE_S, every
trial ends with the pedal up and a pause, and on exit (also Ctrl+C or an
error) all notes are released and the soft and damper pedals lifted.

Results go to logs/disklavier/soft_pedal_test_<time>.json with a summary
(mean rating per condition) printed at the end.

Usage (from the repo root; needs mido + python-rtmidi as check_connection.py)::

    python scripts/disklavier/soft_pedal_strike_test.py --list
    python scripts/disklavier/soft_pedal_strike_test.py [--out "Disklavier"] [--reps 4]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import mido

SOFT_PEDAL = 67
DAMPER_PEDAL = 64
MAX_VELOCITY = 64        # refuse anything louder
NOTE_S = 0.8             # each strike sounds this long
GAP_S = 1.2              # reference strike -> test strike
REST_S = 1.5             # after each trial, pedal up, before the next
SETTLE_S = 1.5           # a held pedal position settles this long before the strike
RAMP_S = 0.15            # pedal travel 0 -> 127 in "moving" trials (about a human press)
RAMP_STEP_S = 0.01       # one CC 67 message every 10 ms during the travel
HELD = (32, 64, 96, 127)
MOVING_MS = (0, 25, 50, 100)


def wait_until(t: float) -> None:
    """Sleep until perf_counter() reaches t (busy-wait for the last ms)."""
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        time.sleep(left - 0.002 if left > 0.003 else 0)


class Piano:
    def __init__(self, port, pitches, velocity):
        self.port, self.pitches, self.velocity = port, pitches, velocity
        self.log = []  # (perf_counter, message) for this trial

    def send(self, msg) -> float:
        self.port.send(msg)
        t = time.perf_counter()
        self.log.append((t, str(msg)))
        return t

    def pedal(self, value: int) -> float:
        return self.send(mido.Message("control_change", control=SOFT_PEDAL, value=int(value)))

    def strike(self) -> float:
        t = None
        for p in self.pitches:
            t = self.send(mido.Message("note_on", note=p, velocity=self.velocity))
        return t

    def release(self) -> None:
        for p in self.pitches:
            self.send(mido.Message("note_off", note=p, velocity=0))

    def all_off(self) -> None:
        for p in self.pitches:
            self.port.send(mido.Message("note_off", note=p, velocity=0))
        self.port.send(mido.Message("control_change", control=123, value=0))  # all notes off
        self.port.send(mido.Message("control_change", control=SOFT_PEDAL, value=0))
        self.port.send(mido.Message("control_change", control=DAMPER_PEDAL, value=0))


def run_trial(piano: Piano, condition: str) -> dict:
    """Reference strike (pedal up), then the test strike under `condition`."""
    piano.log = []
    piano.pedal(0)
    time.sleep(SETTLE_S)
    piano.strike()
    time.sleep(NOTE_S)
    piano.release()
    start = time.perf_counter() + GAP_S - NOTE_S
    info = {"condition": condition}
    if condition == "up":
        wait_until(start)
        info["strike_t"] = piano.strike()
    elif condition.startswith("held-"):
        value = int(condition.split("-")[1])
        piano.pedal(value)
        wait_until(time.perf_counter() + SETTLE_S)
        info["strike_t"] = piano.strike()
    else:  # moving+<ms>: the pedal travels 0 -> 127, the chord struck <ms> after it starts
        offset = int(condition.split("+")[1]) / 1000
        wait_until(start)
        t0 = time.perf_counter()
        steps = int(RAMP_S / RAMP_STEP_S)
        events = [(t0 + i * RAMP_STEP_S, "pedal", round(127 * i / steps)) for i in range(steps + 1)]
        events.append((t0 + offset, "strike", None))
        for when, kind, value in sorted(events, key=lambda e: (e[0], e[1] != "strike")):
            wait_until(when)
            if kind == "pedal":
                piano.pedal(value)
            else:
                info["strike_t"] = piano.strike()
        info["ramp_start_t"] = t0
    time.sleep(NOTE_S)
    piano.release()
    piano.pedal(0)
    info["messages"] = [(round(t - piano.log[0][0], 4), m) for t, m in piano.log]
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list MIDI outputs and exit")
    parser.add_argument("--out", default="Disklavier", help="output port (substring of its name)")
    parser.add_argument("--pitches", default="48,55,64", help="the chord, MIDI pitches (default C3 G3 E4)")
    parser.add_argument("--velocity", type=int, default=40)
    parser.add_argument("--reps", type=int, default=4, help="repetitions per condition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unattended", action="store_true",
                        help="play all trials without asking (3 s between trials); write the "
                             "ratings down per trial and add them to the saved file afterwards")
    parser.add_argument("--lead_in", type=float, default=0.0, help="seconds to wait before the first trial")
    args = parser.parse_args()

    if args.list:
        print("Outputs:", mido.get_output_names() or "(none)")
        return
    if not 1 <= args.velocity <= MAX_VELOCITY:
        raise SystemExit(f"velocity must be 1-{MAX_VELOCITY} (safety cap)")
    pitches = [int(p) for p in args.pitches.split(",")]
    if not all(21 <= p <= 108 for p in pitches) or len(pitches) > 6:
        raise SystemExit("pitches must be piano keys (21-108), at most 6")
    names = [n for n in mido.get_output_names() if args.out.lower() in n.lower()]
    if not names:
        raise SystemExit(f"no output matching {args.out!r}; available: {mido.get_output_names()}")

    conditions = ["up"] + [f"held-{v}" for v in HELD] + [f"moving+{ms}" for ms in MOVING_MS]
    order = conditions * args.reps
    rng = random.Random(args.seed)
    rng.shuffle(order)
    out_dir = Path("logs/disklavier"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"soft_pedal_test_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    results = []
    print(f"Port {names[0]}; chord {pitches} at velocity {args.velocity}; {len(order)} trials "
          f"(~{len(order) * 7 // 60 + 1} min plus your answers).")
    print("Each trial: a reference strike, then the test strike. Rate the test strike:")
    print("  1 = much quieter  2 = quieter  3 = same  4 = louder  5 = much louder"
          "   (r = repeat the trial, q = stop and save)\n")
    with mido.open_output(names[0]) as port:
        piano = Piano(port, pitches, args.velocity)
        try:
            time.sleep(args.lead_in)
            i = 0
            while i < len(order):
                info = run_trial(piano, order[i])
                if args.unattended:
                    info["rating"] = None
                    results.append(info)
                    print(f"trial {i + 1}/{len(order)} played", flush=True)
                    i += 1
                    time.sleep(3.0)
                    continue
                answer = input(f"trial {i + 1}/{len(order)}: rating 1-5? ").strip().lower()
                if answer == "q":
                    break
                if answer == "r" or answer not in {"1", "2", "3", "4", "5"}:
                    continue  # repeat the same trial
                info["rating"] = int(answer)
                results.append(info)
                i += 1
                time.sleep(REST_S)
        finally:
            piano.all_off()
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"pitches": pitches, "velocity": args.velocity, "seed": args.seed,
                           "ramp_s": RAMP_S, "settle_s": SETTLE_S, "trials": results}, f, indent=1)
            print(f"\nSaved {len(results)} rated trials to {out_path}")

    print("\nMean rating per condition (3 = as loud as the pedal-up reference):")
    for c in conditions:
        r = [t["rating"] for t in results if t["condition"] == c and t["rating"] is not None]
        if r:
            sd = f" ± {statistics.stdev(r):.2f}" if len(r) > 1 else ""
            print(f"  {c:12s} {statistics.mean(r):.2f}{sd}  (n={len(r)})")


if __name__ == "__main__":
    main()
