#!/usr/bin/env python3
"""Song-length table of journal/reference/DATASET_SUMMARY.md ("Song length"), from the current caches.

Per corpus, over train + valid + test of the plain (not augmented) cache: number of songs and the length
per lane in frames (16th notes, 4 per beat: frames = 4 x num_beats), mean, median, 10th and 90th
percentile, mean bars (16 frames per bar), and the share of songs shorter than 128 and 256 frames (the 8-
and 16-bar windows). For Hooktheory also the number of distinct (artist, song) pairs, since its entries
are song sections.

Usage:
    python scripts/data/song_length_stats.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORPORA = ["hooktheory", "emopia_plus", "nottingham", "pop909", "wikifonia", "chord_melody_dataset",
           "wjd", "jazzmus", "filobass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--out", type=Path, default=Path("data/cache/song_length_stats.json"))
    args = ap.parse_args()
    res = {}
    print(f"{'corpus':<22}{'songs':>7}{'mean':>7}{'median':>8}{'p10':>6}{'p90':>6}{'bars':>7}{'<128f':>7}{'<256f':>7}")
    for c in CORPORA:
        frames, pieces = [], set()
        for split in ("train", "valid", "test"):
            for line in (args.cache / c / f"{split}.jsonl").open():
                item = json.loads(line)
                frames.append(4 * item["annotations"]["num_beats"])
                if c == "hooktheory":
                    pieces.add((item["hooktheory"]["artist"], item["hooktheory"]["song"]))
        f = np.array(frames, float)
        r = {"songs": len(f), "mean": float(f.mean()), "median": float(np.median(f)),
             "p10": float(np.percentile(f, 10)), "p90": float(np.percentile(f, 90)),
             "mean_bars": float(f.mean() / 16), "share_below_128": float((f < 128).mean()),
             "share_below_256": float((f < 256).mean())}
        if c == "hooktheory":
            r["distinct_artist_song"] = len(pieces)
        res[c] = r
        print(f"{c:<22}{r['songs']:>7}{r['mean']:>7.0f}{r['median']:>8.0f}{r['p10']:>6.0f}{r['p90']:>6.0f}"
              f"{r['mean_bars']:>7.1f}{r['share_below_128']:>7.0%}{r['share_below_256']:>7.0%}")
    print("Hooktheory distinct (artist, song):", res["hooktheory"]["distinct_artist_song"])
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
