#!/usr/bin/env python3
"""Which note's velocity sets the release a re-strike needs after a long hold?

On the ENSPIRE U1, after a long held note, a release that re-strikes reliably
at velocity 40 sometimes fails at 55 (see journal/DISKLAVIER.md). Two
explanations predict different things once the held note and its repeat
have *different* velocities:

* next press: a soft repeat drives the key down slowly, giving the action
  extra time to reset beyond the release, so the *repeat's* velocity matters
  -> soft-then-loud fails, loud-then-soft works.
* solenoid release: a loud note's stronger drive current dies away more
  slowly after note-off, so the *held note's* velocity matters
  -> loud-then-soft fails, soft-then-loud works.

Each trial holds middle C for --hold_ms at one velocity, releases it for
--gap_ms, then strikes it again at the other velocity. Four blocks: the two
mixed orders plus soft/soft and loud/loud controls. After each block, type
how many of the repeats you heard (or saw the hammer strike); a summary
table is printed at the end.

Pick --gap_ms so the controls differ: soft/soft should always repeat and
loud/loud only sometimes (150 ms did that for 40 vs 55 at volume 3/5). If
both controls always repeat, lower the gap; if neither does, raise it.

Requires MIDI IN Delay OFF on the piano, and python-rtmidi (mido's backend).

Usage::

    python scripts/disklavier/mixed_velocity_restrike.py --list
    python scripts/disklavier/mixed_velocity_restrike.py --out "<output port>" \\
        [--soft 40 --loud 70 --hold_ms 600 --gap_ms 150 --trials 8 --blind]
"""

from __future__ import annotations

import argparse
import random
import time

import mido

PITCH = 60
REPEAT_HOLD_MS = 400   # how long the repeat is held, long enough to sound at any velocity
TRIAL_PAUSE_MS = 1200  # rest between trials, so each starts from a settled action


def wait_until(t: float) -> None:
    """Sleep until perf_counter() reaches t, finishing with a short spin for ms accuracy."""
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        time.sleep(left - 0.002 if left > 0.003 else 0)


def run_trial(out, first_vel: int, second_vel: int, hold_ms: float, gap_ms: float) -> None:
    t0 = time.perf_counter()
    out.send(mido.Message("note_on", note=PITCH, velocity=first_vel))
    wait_until(t0 + hold_ms / 1000)
    out.send(mido.Message("note_off", note=PITCH, velocity=0))
    wait_until(t0 + (hold_ms + gap_ms) / 1000)
    out.send(mido.Message("note_on", note=PITCH, velocity=second_vel))
    wait_until(t0 + (hold_ms + gap_ms + REPEAT_HOLD_MS) / 1000)
    out.send(mido.Message("note_off", note=PITCH, velocity=0))
    wait_until(t0 + (hold_ms + gap_ms + REPEAT_HOLD_MS + TRIAL_PAUSE_MS) / 1000)


def ask_count(trials: int) -> int | None:
    while True:
        answer = input(f"  How many of the {trials} repeats sounded? (Enter to skip) ").strip()
        if not answer:
            return None
        if answer.isdigit() and int(answer) <= trials:
            return int(answer)
        print(f"  Please enter a number from 0 to {trials}.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="List output ports and exit.")
    p.add_argument("--out", help="Output port name (from --list).")
    p.add_argument("--soft", type=int, default=40, help="Soft velocity.")
    p.add_argument("--loud", type=int, default=70, help="Loud velocity.")
    p.add_argument("--hold_ms", type=float, default=600, help="Hold of the first note.")
    p.add_argument("--gap_ms", type=float, default=150, help="Release before the repeat.")
    p.add_argument("--trials", type=int, default=8, help="Trials per block.")
    p.add_argument("--blind", action="store_true",
                   help="Shuffle the blocks and reveal which was which only at the end.")
    args = p.parse_args()

    if args.list or not args.out:
        print("Outputs:", mido.get_output_names() or "(none)")
        return

    s, l = args.soft, args.loud
    blocks = [
        ("soft -> loud (mixed)", s, l),
        ("loud -> soft (mixed)", l, s),
        ("soft -> soft (control)", s, s),
        ("loud -> loud (control)", l, l),
    ]
    if args.blind:
        random.shuffle(blocks)

    print(f"Middle C: hold {args.hold_ms:.0f} ms, release {args.gap_ms:.0f} ms, repeat; "
          f"{args.trials} trials per block. Count the repeats (the second strike of each pair).")
    results = []
    with mido.open_output(args.out) as out:
        try:
            for i, (name, first, second) in enumerate(blocks, 1):
                label = f"Block {i}" if args.blind else f"Block {i}: {name}, vel {first} -> {second}"
                input(f"\n{label}. Press Enter to start...")
                for _ in range(args.trials):
                    run_trial(out, first, second, args.hold_ms, args.gap_ms)
                results.append((i, name, first, second, ask_count(args.trials)))
        finally:
            out.send(mido.Message("note_off", note=PITCH, velocity=0))
            out.send(mido.Message("control_change", control=123, value=0))  # all notes off

    print(f"\nhold {args.hold_ms:.0f} ms, release {args.gap_ms:.0f} ms, {args.trials} trials per block")
    print(f"{'block':>5}  {'condition':<24} {'vel':>9}  {'repeats heard':>13}")
    for i, name, first, second, heard in results:
        count = "-" if heard is None else f"{heard}/{args.trials}"
        print(f"{i:>5}  {name:<24} {first:>3} -> {second:<3}  {count:>13}")
    print("\nNext-press explanation predicts: soft -> loud fails more than loud -> soft.")
    print("Solenoid-release explanation predicts the opposite.")


if __name__ == "__main__":
    main()
