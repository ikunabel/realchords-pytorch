"""The path of the chords through the Tonnetz: how they move, not just where they are.

Input: free-running runs (plot_onset_hold_free_running.py JSON: melody from a Hooktheory test song, chords sampled)
plus each song's real chords. Every chord change is reduced to a major/minor triad (as tonnetz_tonic_distance.py;
sus/dim/aug/power chords and silence skipped, re-strikes of the same triad are not moves). Distances are
neo-Riemannian: fewest P/L/R flips between triads on the Tonnetz; "home" is the annotated key's tonic triad.

Per chords source (real / each model):
  step size       Tonnetz distance between consecutive triads (1 = one shared edge, two common notes);
  direction       each step toward home (distance from the tonic falls), away (rises) or sideways;
  excursions      each trip from leaving the tonic to the next return: steps taken, furthest point, and whether
                  the way back is a steady approach (no step away after the furthest point);
  phrase endings  from the melody: the end of a note followed by >= 1 beat of rest, and the section end. At each:
                  the triad sounding there at the tonic? the last change before it a step toward home? arrival
                  (the last change goes into the tonic from elsewhere)?

    python scripts/eval/tonnetz_paths.py logs/eval/free_running_20songs/free_running_*.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval")]
from tonnetz_helpers import TRIADS, plr_distance, triad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+")
args = ap.parse_args()

songs = {}
for line in open(REPO / "data/cache/hooktheory/test.jsonl"):
    d = json.loads(line)
    k = d["annotations"]["keys"]
    if len(k) != 1:
        continue
    third = sum(k[0]["scale_degree_intervals"][:2])
    mel = sorted((round(n["onset"] * 4), round(n["offset"] * 4)) for n in d["annotations"]["melody"])
    songs[d["hooktheory"]["urls"]["song"]] = {"tonic": (k[0]["tonic_pitch_class"], "M" if third == 4 else "m"),
                                              "melody": mel}


def phrase_ends(melody, n):
    ends = []
    for i, (on, off) in enumerate(melody):
        if off > n:
            break
        nxt = melody[i + 1][0] if i + 1 < len(melody) else None
        if nxt is None or nxt - off >= 4:
            ends.append(min(off, n) - 1)
    if n - 1 not in ends:
        ends.append(n - 1)
    return sorted(set(e for e in ends if e >= 16))  # skip the first bar (cold start)


def moves(onsets, n):
    """[(frame, triad)] of triad changes (re-strikes and unplaceable chords dropped)."""
    out = []
    for k, name in sorted(onsets):
        t = triad(name)
        if t is None or k >= n:
            continue
        if not out or out[-1][1] != t:
            out.append((k, t))
    return out


acc = collections.defaultdict(lambda: collections.defaultdict(list))
seen_real = set()
for path in args.runs:
    meta = json.load(open(path))
    for r in meta["runs"]:
        song = songs.get(r["song_url"])
        if song is None:
            continue
        n = len(r["p_onset"])
        home = song["tonic"]
        ends = phrase_ends(song["melody"], n)
        sources = [(r["model"], r["model_onsets"])]
        if r["song_url"] not in seen_real:
            seen_real.add(r["song_url"])
            sources.append(("real", r["real_onsets"]))
        for label, onsets in sources:
            A = acc[label]
            mv = moves(onsets, n)
            dh = [plr_distance(t, home) for _, t in mv]
            for (k0, a), (k1, b), d0, d1 in zip(mv, mv[1:], dh, dh[1:]):
                A["step"].append(plr_distance(a, b))
                A["dir"].append(np.sign(d1 - d0))
            # excursions: from the first move away from the tonic to the next move back into it
            i = 0
            while i < len(dh):
                if dh[i] == 0 and i + 1 < len(dh) and dh[i + 1] > 0:
                    j = i + 1
                    while j < len(dh) and dh[j] != 0:
                        j += 1
                    if j < len(dh):
                        trip = dh[i:j + 1]
                        top = int(np.argmax(trip))
                        steady = all(b <= a for a, b in zip(trip[top:], trip[top + 1:]))
                        A["exc_steps"].append(j - i)
                        A["exc_far"].append(max(trip))
                        A["exc_steady"].append(steady)
                    i = j
                else:
                    i += 1
            # phrase endings
            frames = [k for k, _ in mv]
            for e in ends:
                idx = max([i for i, k in enumerate(frames) if k <= e], default=None)
                if idx is None:
                    continue
                A["end_home"].append(dh[idx] == 0)
                if idx >= 1:
                    A["end_last_toward"].append(dh[idx] < dh[idx - 1])
                    A["end_arrival"].append(dh[idx] == 0 and dh[idx - 1] > 0)
            A["runs"].append(1)
            A["moves_per_bar"].append(len(mv) / (n / 16))

labels = ["real"] + sorted(l for l in acc if l != "real")
pc = lambda xs: f"{100 * np.mean(xs):.0f}%" if xs else "-"
print("| | real | " + " | ".join(labels[1:]) + " |")
print("|---|" + "---|" * len(labels))
def row(name, f):
    print(f"| {name} | " + " | ".join(f(acc[l]) for l in labels) + " |")
row("runs (songs x seeds)", lambda A: str(len(A["runs"])))
row("triad changes per bar", lambda A: f"{np.mean(A['moves_per_bar']):.2f}")
for s in range(1, 5):
    row(f"step size {s}{'+' if s == 4 else ''}", lambda A, s=s: pc([(x >= 4 if s == 4 else x == s) for x in A["step"]]))
row("step toward home / away / sideways", lambda A: "/".join(pc([d == v for d in A["dir"]]) for v in (-1, 1, 0)))
row("excursions per run", lambda A: f"{len(A['exc_steps']) / max(len(A['runs']), 1):.1f}")
row("excursion: steps (median)", lambda A: f"{np.median(A['exc_steps']):.0f}" if A["exc_steps"] else "-")
row("excursion: furthest (mean)", lambda A: f"{np.mean(A['exc_far']):.2f}" if A["exc_far"] else "-")
row("excursion: steady way back", lambda A: pc(A["exc_steady"]))
row("phrase end on the tonic", lambda A: pc(A["end_home"]))
row("phrase end: last change toward home", lambda A: pc(A["end_last_toward"]))
row("phrase end: arrival into the tonic", lambda A: pc(A["end_arrival"]))
print("phrase endings counted:", {l: len(acc[l]["end_home"]) for l in labels})
