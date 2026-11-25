"""When the melody repeats, do the chords repeat? Real songs vs free-running models.

Input: free-running runs (plot_onset_hold_free_running.py JSON: melody from a Hooktheory test song, chords sampled)
plus each song's real chords and melody (data/cache/hooktheory/test.jsonl). Frames are 16ths, bars every 16 frames.

For window lengths W (bars), every pair of bar-aligned, non-overlapping windows is classed by melody:
  same melody   identical frame by frame (pitch and onset at every 16th), with >= 2 note onsets in the window;
  other melody  everything else (the baseline: how repetitive the chords are anyway).
Per pair, chord agreement = share of frames where the sounding chord name is the same in both windows;
"exact" = agreement 1. A model that repeats its progression when the melody repeats, as songs do, scores close
to the real chords on same-melody pairs; the gap between same and other melody is the melody-driven repetition.

    python scripts/eval/melody_repeat_chord_repeat.py logs/eval/free_running_20songs/free_running_*.json
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+")
ap.add_argument("--bars", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--skip_first_bar", action="store_true", help="drop windows starting at bar 0 (cold start)")
args = ap.parse_args()

songs = {}
for line in open(REPO / "data/cache/hooktheory/test.jsonl"):
    d = json.loads(line)
    songs[d["hooktheory"]["urls"]["song"]] = d["annotations"]["melody"]


def melody_grid(notes, n):
    g = [None] * n  # per frame: (pitch, is_onset) or None (rest)
    for x in notes:
        on, off, p = round(x["onset"] * 4), round(x["offset"] * 4), 12 * x["octave"] + x["pitch_class"]
        for k in range(max(on, 0), min(off, n)):
            g[k] = (p, k == on)
    return g


def sounding(onsets, n):
    out, cur, d = [None] * n, None, dict(onsets)
    for k in range(n):
        cur = d.get(k, cur)
        out[k] = cur
    return out


def pairs(mel, n, W):
    L = 16 * W
    starts = [s for s in range(0, n - L + 1, 16) if not (args.skip_first_bar and s == 0)]
    for i, a in enumerate(starts):
        for b in starts[i + 1:]:
            if b < a + L:
                continue
            wa, wb = mel[a:a + L], mel[b:b + L]
            same = wa == wb and sum(1 for x in wa if x and x[1]) >= 2
            yield a, b, L, same


acc = collections.defaultdict(list)  # (W, label, same) -> agreements
seen = set()
for path in args.runs:
    for r in json.load(open(path))["runs"]:
        n = len(r["p_onset"])
        mel = melody_grid(songs[r["song_url"]], n)
        srcs = [(r["model"], sounding(r["model_onsets"], n))]
        if r["song_url"] not in seen:
            seen.add(r["song_url"])
            srcs.append(("real", sounding(r["real_onsets"], n)))
        for W in args.bars:
            for a, b, L, same in pairs(mel, n, W):
                for label, ch in srcs:
                    acc[W, label, same].append(np.mean([x == y for x, y in zip(ch[a:a + L], ch[b:b + L])]))

labels = ["real"] + sorted({k[1] for k in acc} - {"real"})
print("| window | melody | pairs (real) | " + " | ".join(f"{l}: agree / exact" for l in labels) + " |")
print("|---|---|---|" + "---|" * len(labels))
for W in args.bars:
    for same in (True, False):
        cells = []
        for l in labels:
            xs = acc[W, l, same]
            cells.append(f"{np.mean(xs):.2f} / {100 * np.mean(np.array(xs) == 1):.0f}%" if xs else "-")
        print(f"| {W} bar{'s' if W > 1 else ''} | {'same' if same else 'other'} | {len(acc[W, 'real', same])} | "
              + " | ".join(cells) + " |")
