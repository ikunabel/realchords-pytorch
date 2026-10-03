#!/usr/bin/env python3
"""Measure how loud and how late the Disklavier plays each MIDI velocity.

Records the laptop microphone while sending a chord at a series of
velocities, then finds each chord's acoustic onset and peak level in the
recording. Gives, per velocity:

  * loudness (peak dBFS)  -- does velocity actually change the volume?
  * latency (ms)          -- time from sending note-on to the sound starting,
                             i.e. what "MIDI Send Early" has to cover.

Latency is relative to the first chord's measured send time, so it includes
microphone/recording latency as a constant offset; differences between
velocities are exact, the absolute number is an upper bound.

Requires MIDI IN Delay OFF on the piano (otherwise every note is +500 ms).

Usage::

    python scripts/disklavier/measure_velocity_response.py \\
        --port "Disklavier:Disklavier MIDI 1 28:0" [--velocities 15,35,55,75,95,115]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import mido
import numpy as np

CHORD = [48, 52, 55]
RATE = 48000
HOLD_S = 1.0
GAP_S = 1.2


def record(path: Path, seconds: float) -> subprocess.Popen:
    return subprocess.Popen(
        ["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def analyse(path: Path, sends: list[tuple[int, float]], rec_start: float) -> None:
    with wave.open(str(path)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        rate = w.getframerate()
    hop = int(rate * 0.002)                                   # 2 ms envelope
    env = np.sqrt(np.convolve(audio ** 2, np.ones(hop) / hop, mode="same"))
    noise = np.percentile(env[: int(rate * 0.4)], 90) if len(env) > rate else env.min()

    print(f"\nnoise floor {20*np.log10(noise + 1e-9):6.1f} dBFS\n")
    print(f"{'velocity':>9} {'peak dBFS':>10} {'onset after send':>17}")
    for vel, t_send in sends:
        a = int((t_send - rec_start) * rate)
        b = a + int(rate * (HOLD_S + 0.4))
        seg = env[a:b]
        if len(seg) == 0:
            print(f"{vel:>9}   (outside recording)")
            continue
        peak = seg.max()
        thresh = noise + 0.3 * (peak - noise)
        above = np.nonzero(seg > thresh)[0]
        onset_ms = above[0] / rate * 1000 if len(above) and peak > 3 * noise else float("nan")
        print(f"{vel:>9} {20*np.log10(peak + 1e-9):>10.1f} {onset_ms:>14.0f} ms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--velocities", default="15,35,55,75,95,115")
    p.add_argument("--keep", help="Keep the recording at this path.")
    args = p.parse_args()
    velocities = [int(v) for v in args.velocities.split(",")]

    path = Path(args.keep) if args.keep else Path(tempfile.mkstemp(suffix=".wav")[1])
    total = 1.0 + len(velocities) * (HOLD_S + GAP_S) + 0.5
    rec = record(path, total)
    rec_start = time.perf_counter()
    time.sleep(1.0)                                            # noise floor + recorder spin-up

    sends = []
    with mido.open_output(args.port) as out:
        for vel in velocities:
            t = time.perf_counter()
            for n in CHORD:
                out.send(mido.Message("note_on", note=n, velocity=vel))
            sends.append((vel, t))
            time.sleep(HOLD_S)
            for n in CHORD:
                out.send(mido.Message("note_off", note=n, velocity=0))
            time.sleep(GAP_S)
        out.send(mido.Message("control_change", control=123, value=0))
    time.sleep(0.5)
    rec.terminate()
    rec.wait()
    analyse(path, sends, rec_start)
    if args.keep:
        print(f"\nrecording kept at {path}")


if __name__ == "__main__":
    main()
