#!/usr/bin/env python3
"""Chord-distribution distances of the four-way Hooktheory run (journal/reference/METRICS.md,
"Key-relative and per-song chord distribution distances").

That table was computed on 2026-09-21 before these metrics were part of the evaluation, so the run's
means.json does not contain them. Recomputed here from the saved chords_per_frame.pt files with the
evaluation's functions (eval_utils.py), onset weighting:

  absolute pooled      all sequences' chord pitch-class sets pooled, model vs GT (no transposition)
  key-relative pooled  each sequence transposed so its annotated tonic is pitch class 0, then pooled
  per song             model vs GT chords of the same sequence, averaged

The run's metadata has song URLs but no section ids, so the tonic comes from the URL where every
Hooktheory test section of that URL has the same single key (as in the original computation).

Usage:
    python scripts/eval/custom_eval/four_way_chord_distances.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from realchords.utils.eval_utils import (key_relative_chord_distribution_wasserstein,
                                         per_song_chord_distribution_wasserstein)


def url_tonics(test_jsonl: Path) -> dict:
    keys_by_url = {}
    for line in test_jsonl.open():
        item = json.loads(line)
        keys = item["annotations"]["keys"]
        tonic = int(keys[0]["tonic_pitch_class"]) if len(keys) == 1 else None
        keys_by_url.setdefault(item["hooktheory"]["urls"]["song"], set()).add(tonic)
    return {u: next(iter(t)) for u, t in keys_by_url.items() if len(t) == 1 and None not in t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("logs/custom_eval/four_way_hooktheory_vs_7sets"))
    ap.add_argument("--test_jsonl", type=Path, default=Path("data/cache/hooktheory/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/four_way_hooktheory_vs_7sets/chord_distances.json"))
    args = ap.parse_args()
    meta = [json.loads(l) for l in (args.run / "metadata.jsonl").read_text().splitlines() if l.strip()]
    by_url = url_tonics(args.test_jsonl)
    tonics = [by_url.get(m["song_url"]) for m in meta]
    gt = torch.load(args.run / "gt" / "chords_per_frame.pt", weights_only=False)
    res = {"sequences": len(meta), "sequences_with_key": sum(t is not None for t in tonics), "models": {}}
    print(f"sequences {res['sequences']}, with an unambiguous key {res['sequences_with_key']}")
    print(f"{'model':<16}{'absolute':>10}{'key-relative':>14}{'per song':>10}")
    for d in sorted((args.run / "models").iterdir()):
        ch = torch.load(d / "chords_per_frame.pt", weights_only=False)
        absolute = key_relative_chord_distribution_wasserstein(ch, gt, [0] * len(meta), "onset")[0]
        rel = key_relative_chord_distribution_wasserstein(ch, gt, tonics, "onset")[0]
        song = per_song_chord_distribution_wasserstein(ch, gt, "onset")[0]
        res["models"][d.name] = {"absolute_pooled": absolute, "key_relative_pooled": rel, "per_song": song}
        print(f"{d.name:<16}{absolute:>10.4f}{rel:>14.4f}{song:>10.4f}")
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
