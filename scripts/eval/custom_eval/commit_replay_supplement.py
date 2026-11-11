#!/usr/bin/env python3
"""Numbers in journal/REALJAM_CHORD_TIMING.md, Part 5, that no other script saves.

No generation; reads the saved replay outputs of commit_replay.py and writes one JSON.

  anticipation         "How well can the model anticipate the melody?": per steady-state call,
                       share of imagined frames whose note equals the real one (same pitch, or
                       both silent), against the baseline "the last real note keeps holding".
                       Seed 42, the 32-window --keep_imagined runs.
  offbeat_first_bar    finding 3: share of chord changes on 8th/16th offbeats, with the first
                       bar of each window excluded (cold start) and included; means over seeds.
  rate_monotonicity    finding 1: chord changes per bar per seed and k, and in how many
                       seed-corpus series the rate falls strictly at every step k = 1 -> 16.
  fix_cells            "The fix": per seed x corpus x k, change rate and offbeat share for GAPT
                       with self- vs MLE-imagined melody, and the note-in-chord difference.

Tail windows are dropped as in chord_placement.py. Rate and offbeat share are pooled over
windows (changes / bars, offbeat changes / changes), as in the Part 5 tables.

Usage:
    python scripts/eval/custom_eval/commit_replay_supplement.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from chord_placement import FRAMES_PER_BAR, _vocab_maps, change_frames, keep_rows  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

CORPORA = ["hooktheory", "pop909"]
SEEDS = ["s42", "s43", "s44"]
STAGES = ["mle", "gapt"]
K_ALL = [1, 4, 8, 16]
K_IMAG = [8, 16]


def _load(p: Path) -> torch.Tensor:
    return torch.load(p, map_location="cpu", weights_only=False)


def _tok(corpus_dir: Path) -> HooktheoryTokenizer:
    return HooktheoryTokenizer(chord_names=json.loads((corpus_dir / "chord_names_augmented.json").read_text()))


def anticipation(root: Path) -> dict:
    out = {}
    for k in K_IMAG:
        w = k - 1
        for corpus in CORPORA:
            d = root / f"k{k}" / corpus
            tok = _tok(d)
            hold_lo, hold_hi = tok.note_hold_token_range
            on_lo, on_hi = tok.note_on_token_range

            def pitch(t: int):
                """Sounding pitch, -1 for silence, None for any other token."""
                if t == tok.silence_token:
                    return -1
                if hold_lo <= t <= hold_hi:
                    return t - hold_lo
                if on_lo <= t <= on_hi:
                    return t - on_lo
                return None

            gt = _load(d / "gt" / "gt.pt")
            real = gt[:, 2::2]
            rows, _ = keep_rows(d / "metadata.jsonl", gt.shape[0])
            for stage in STAGES:
                imag = _load(d / "models" / f"{stage}_s42" / "imagined.pt")
                hit, base, n = [0] * w, [0] * w, [0] * w
                for i in rows:
                    valid = int((real[i] != tok.pad_token).sum())
                    for u in range(w, valid):
                        f = u - w
                        last = pitch(int(real[i, f - 1])) if f >= 1 else -1   # last real note heard
                        for j in range(w):
                            r, g = pitch(int(real[i, f + j])), pitch(int(imag[i, u, j]))
                            if r is None or g is None:
                                continue
                            n[j] += 1
                            hit[j] += g == r
                            base[j] += last == r

                def cell(js):
                    m = sum(n[j] for j in js)
                    return {"model": sum(hit[j] for j in js) / m, "hold_baseline": sum(base[j] for j in js) / m}

                out[f"k{k}/{corpus}/{stage}"] = {
                    "1_frame_ahead": cell([0]), "middle": cell([w // 2]),
                    "last_frame": cell([w - 1]), "whole_window": cell(range(w)),
                    "frames_per_offset": n,
                }
    return out


def _changes(path: Path, rows, events, pads, skip: int = 0):
    """Pooled over rows: (changes per bar, offbeat share), changes before frame `skip` ignored."""
    x = _load(path)
    n = off = frames = 0
    for i in rows:
        ch, valid = change_frames(pm.chord_lane(x[i].tolist()), events, pads)
        ch = [t for t in ch if t >= skip]
        n += len(ch)
        off += sum(1 for t in ch if t % 4)
        frames += valid
    return n / (frames / FRAMES_PER_BAR), (off / n if n else float("nan"))


def placement(root: Path, fix_root: Path) -> tuple[dict, dict, dict]:
    offbeat, rates, fix = {}, {}, {"cells": {}}
    falling = {s: [] for s in STAGES}
    for corpus in CORPORA:
        d1 = root / "k1" / corpus
        events, pads = _vocab_maps(_tok(d1).id_to_name)
        rows, _ = keep_rows(d1 / "metadata.jsonl", _load(d1 / "gt" / "gt.pt").shape[0])
        pred = lambda r, k, stage, s: r / f"k{k}" / corpus / "models" / f"{stage}_{s}" / "preds.pt"

        offbeat[f"{corpus}/gt"] = {"excluded": _changes(d1 / "gt" / "gt.pt", rows, events, pads, 16)[1],
                                   "included": _changes(d1 / "gt" / "gt.pt", rows, events, pads)[1]}
        for stage in STAGES:
            for k in K_ALL:
                offbeat[f"{corpus}/{stage}/k{k}"] = {
                    "excluded": st.mean(_changes(pred(root, k, stage, s), rows, events, pads, 16)[1] for s in SEEDS),
                    "included": st.mean(_changes(pred(root, k, stage, s), rows, events, pads)[1] for s in SEEDS),
                }
            for s in SEEDS:
                r = [_changes(pred(root, k, stage, s), rows, events, pads)[0] for k in K_ALL]
                strict = all(r[i] > r[i + 1] for i in range(len(r) - 1))
                rates[f"{corpus}/{stage}_{s}"] = {"rate_by_k": dict(zip(map(str, K_ALL), r)), "strictly_falling": strict}
                falling[stage].append(strict)

        for k in K_IMAG:
            for s in SEEDS:
                r0, o0 = _changes(pred(root, k, "gapt", s), rows, events, pads)
                r1, o1 = _changes(pred(fix_root, k, "gapt", s), rows, events, pads)
                fix["cells"][f"{corpus}/k{k}/{s}"] = {"rate_self": r0, "rate_mle_imagined": r1,
                                                      "offbeat_self": o0, "offbeat_mle_imagined": o1}

    rates["strictly_falling_count"] = {s: f"{sum(v)}/{len(v)}" for s, v in falling.items()}
    cells = fix["cells"].values()
    fix["rate_higher"] = f"{sum(c['rate_mle_imagined'] > c['rate_self'] for c in cells)}/{len(cells)}"
    fix["offbeat_lower"] = f"{sum(c['offbeat_mle_imagined'] < c['offbeat_self'] for c in cells)}/{len(cells)}"
    summ = json.loads((fix_root / "summary.json").read_text())
    nd = [a - b for c in CORPORA for k in K_IMAG
          for a, b in zip(summ[f"{c}/mle{k}"]["nicr_seeds"], summ[f"{c}/self{k}"]["nicr_seeds"])]
    fix["nicr_diff"] = {"per_cell": nd, "min": min(nd), "max": max(nd), "lower": f"{sum(x < 0 for x in nd)}/{len(nd)}"}
    return offbeat, rates, fix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--imagined_root", type=Path, default=Path("logs/custom_eval/commit_replay_imagined"))
    ap.add_argument("--fix_root", type=Path, default=Path("logs/custom_eval/commit_replay_mleimag"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/commit_replay/supplement.json"))
    args = ap.parse_args()

    res = {"anticipation": anticipation(args.imagined_root)}
    res["offbeat_first_bar"], res["rate_monotonicity"], res["fix_cells"] = placement(args.replay_root, args.fix_root)

    print("anticipation, model (hold-last-note baseline)")
    for key, v in res["anticipation"].items():
        print(f"  {key:<22}" + "".join(f"{v[c]['model']:>7.2f} ({v[c]['hold_baseline']:.2f})"
                                     for c in ("1_frame_ahead", "middle", "last_frame", "whole_window")))
    print("\noffbeat share, first bar excluded (included)")
    for key, v in res["offbeat_first_bar"].items():
        print(f"  {key:<22}{v['excluded']:>7.1%} ({v['included']:.1%})")
    print(f"\nstrictly falling change rate k=1..16: {res['rate_monotonicity']['strictly_falling_count']}")
    f = res["fix_cells"]
    print(f"fix: rate higher {f['rate_higher']}, offbeat lower {f['offbeat_lower']}, NiCR lower "
          f"{f['nicr_diff']['lower']} (min {f['nicr_diff']['min']:+.3f}, max {f['nicr_diff']['max']:+.3f})")
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
