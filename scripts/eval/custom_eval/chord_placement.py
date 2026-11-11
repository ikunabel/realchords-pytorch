#!/usr/bin/env python3
"""Where in the bar do generated chord changes land, compared with real music?

GAPT plays far more chords per window than real music (19.6 vs 12.8, journal/METRICS.md),
with more uniform durations. Two readings predict different placements:

    H1  off-grid       -- the excess changes land on 8th/16th offbeats
    H2  harmonic rhythm too fast -- the excess is on-grid (beat 3, beats 2/4) where real
                          music holds the chord through the bar

Teacher-forced, the model puts ~0.81 of its mass on a chord onset at downbeats and ~0.01 at
16th offbeats (scripts/eval/analyze_chord_onset_timing.py), which favours H2. This measures
the free-running generations instead, which nothing else in the repo does.

A change is counted exactly as chord_events() counts one -- a frame whose parsed (root,
quality) differs from the last chord sounding before it, carried across silences -- but
the frame index is kept. The first chord of a window is never a change (a window boundary
is not a musical event), so a window yields len(chord_events) - 1 changes.

Two corrections, both decided before the first run:

* Tail windows dropped. Window starts are multiples of 256 frames from the song's start
  except a song's final window, which is aligned to the song's END
  (sequence_utils.window_starts) and so has phase song_frames mod 16 -- up to a third of
  EMOPIA+'s windows. The last window of every song with more than one window is dropped
  (via metadata.jsonl song_id; rows verified to be in preds.pt order and contiguous per
  song). Conservative: also drops some windows that were aligned.
* Nottingham excluded from bar position. At least 36% of it is compound meter (6/8, 9/8,
  12-frame bars), so position mod 16 is not bar position there. Beat-level position
  (mod 4) is meter-robust at the quarter/8th/16th level and is reported for all corpora.
  Hooktheory's ~9% non-4/4 rows are left in as noise; the other corpora hardcode 4/4 in
  their caches, so their true meter cannot be checked.

Reported per run: changes per 16 frames; share of changes per bar bucket and per beat
bucket; total-variation distance to GT's distribution; and the EXCESS RATE per bucket
(model changes per 16 frames in that bucket minus GT's), which is the quantity that
decides between H1 and H2.

--loo replaces the grand mean with the leave-one-out contrast: for each corpus C, models
trained on all seven (7sets_s*) against models with C held out (7sets_no_C_s*), both
scored on C.

Usage:
    python scripts/eval/custom_eval/chord_placement.py \
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new GAPT=logs/custom_eval/gapt_loo \
        --out logs/custom_eval/chord_placement_loo.json
    # leave-one-out contrast
    python scripts/eval/custom_eval/chord_placement.py --loo \
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new GAPT=logs/custom_eval/gapt_loo \
        --out logs/custom_eval/chord_placement_loo_contrast.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm
from realchords.utils.eval_utils import _parse_chord_symbol

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}

FRAMES_PER_BAR = 16
# Same position names as scripts/eval/analyze_completion_timing.py.
BAR_BUCKETS = ["downbeat", "beat 3", "beats 2/4", "8th off", "16th off"]
BEAT_BUCKETS = ["on beat", "8th off", "16th off"]
# Compound meter makes position mod 16 meaningless; decided before the first run.
BAR_EXCLUDED = {"nottingham"}
SEEDS = ["s42", "s43", "s44"]


def bar_bucket(pos: int) -> int:
    if pos == 0:
        return 0
    if pos == 8:
        return 1
    if pos in (4, 12):
        return 2
    return 3 if pos % 2 == 0 else 4


def beat_bucket(pos: int) -> int:
    p = pos % 4
    return 0 if p == 0 else (1 if p == 2 else 2)


def _vocab_maps(id_to_name):
    """token id -> (root, quality) event or None, and the set of PAD ids. Built once."""
    events, pads = {}, set()
    for tok, name in id_to_name.items():
        if name == "PAD":
            pads.add(int(tok))
        symbol = _parse_chord_symbol(name)
        events[int(tok)] = (pm.chord_symbol_to_event(symbol, quality="full")
                            if symbol is not None else None)
    return events, pads


def change_frames(lane, events, pads):
    """(frames of chord changes, number of valid frames) for one chord lane."""
    out, previous, valid = [], None, 0
    for t, tok in enumerate(lane):
        tok = int(tok)
        if tok in pads:
            continue
        valid += 1
        event = events.get(tok)
        if event is None:
            continue
        if previous is not None and event != previous:
            out.append(t)
        previous = event
    return out, valid


def keep_rows(meta_path: Path, limit: int):
    """Row indices to keep: drop the last window of every song with more than one."""
    ids = [json.loads(l)["song_id"] for l in meta_path.read_text().splitlines() if l.strip()]
    ids = ids[:limit]
    counts = Counter(ids)
    last = {s: i for i, s in enumerate(ids)}
    drop = {last[s] for s, n in counts.items() if n > 1}
    return [i for i in range(len(ids)) if i not in drop], len(drop)


def profile(path: Path, rows, events, pads, bar_ok: bool):
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    bar, beat, n_changes, n_frames = [0] * 5, [0] * 3, 0, 0
    for i in rows:
        frames, valid = change_frames(pm.chord_lane(seqs[i].tolist()), events, pads)
        n_changes += len(frames)
        n_frames += valid
        for t in frames:
            beat[beat_bucket(t)] += 1
            if bar_ok:
                bar[bar_bucket(t % FRAMES_PER_BAR)] += 1
    rate = n_changes / (n_frames / FRAMES_PER_BAR) if n_frames else 0.0
    share = lambda h: [v / sum(h) for v in h] if sum(h) else None
    return {"rate": rate,
            "bar_share": share(bar) if bar_ok else None,
            "beat_share": share(beat)}


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q)) if p and q else None


def _mean_profiles(profs, gt, bar_ok):
    out = {"rate": st.mean(p["rate"] for p in profs),
           "beat_share": [st.mean(p["beat_share"][k] for p in profs) for k in range(3)],
           "tv_beat": st.mean(tv(p["beat_share"], gt["beat_share"]) for p in profs),
           "n_models": len(profs)}
    if bar_ok:
        out["bar_share"] = [st.mean(p["bar_share"][k] for p in profs) for k in range(5)]
        out["tv_bar"] = st.mean(tv(p["bar_share"], gt["bar_share"]) for p in profs)
        # changes per 16 frames in each bucket, minus GT's: where the excess goes
        out["excess_rate"] = [out["rate"] * out["bar_share"][k] - gt["rate"] * gt["bar_share"][k]
                              for k in range(5)]
    else:
        out["bar_share"] = out["tv_bar"] = out["excess_rate"] = None
    return out


def _print_row(name, label, r, bar_ok):
    bars = ("".join(f"{v:>8.1%}" for v in r["bar_share"]) if bar_ok and r["bar_share"]
            else "".join(f"{'-':>8}" for _ in range(5)))
    beats = "".join(f"{v:>8.1%}" for v in r["beat_share"])
    tvb = f"{r['tv_bar']:>8.3f}" if r.get("tv_bar") is not None else f"{'-':>8}"
    tvt = f"{r['tv_beat']:>8.3f}" if r.get("tv_beat") is not None else f"{'-':>8}"
    print(f"{name:<12}{label:<10}{r['rate']:>7.2f}{bars} |{beats}{tvb}{tvt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR",
                    help="GT, metadata and the chord vocabulary are read from the first one.")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="only models whose directory name starts with one of these")
    ap.add_argument("--loo", action="store_true",
                    help="per corpus C: 7sets_s* (all seven) vs 7sets_no_C_s* (C held out)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = [(i.split("=", 1)[0], Path(i.split("=", 1)[1])) for i in args.eval_dirs]
    results = {}
    head = "".join(f"{b:>8}" for b in ["down", "beat3", "b2/4", "8th", "16th"])
    print(f"\nchord-change placement   rate = changes per 16 frames; "
          f"TV = total-variation distance to GT")
    print(f"{'corpus':<12}{'run':<10}{'rate':>7}{head} |"
          f"{'on':>8}{'8th':>8}{'16th':>8}{'TV bar':>8}{'TV beat':>8}")

    for corpus in CORPORA:
        base = runs[0][1] / corpus
        vocab = base / "chord_names_augmented.json"
        if not vocab.exists():
            continue
        names = json.loads(vocab.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        events, pads = _vocab_maps(HooktheoryTokenizer(chord_names=names).id_to_name)
        bar_ok = corpus not in BAR_EXCLUDED

        # same truncation rule as fifths_motion.py
        def _rows(p):
            return torch.load(p, map_location="cpu", weights_only=False).shape[0]
        counts = [_rows(base / "gt" / "gt.pt")]
        for _, d in runs:
            first = next((s / "preds.pt" for s in sorted((d / corpus / "models").iterdir())
                          if (s / "preds.pt").exists()), None)
            if first is not None:
                counts.append(_rows(first))
        limit = min(counts)
        rows, n_drop = keep_rows(base / "metadata.jsonl", limit)
        note = f"  [{SHORT[corpus]}] {len(rows):,} windows ({n_drop} tail windows dropped"
        note += f", truncated to {limit:,})" if len(set(counts)) > 1 else ")"
        print(note + ("" if bar_ok else "  -- bar position excluded: compound meter"))

        gt = profile(base / "gt" / "gt.pt", rows, events, pads, bar_ok)
        row = {"gt": gt, "n_windows": len(rows), "n_dropped": n_drop}
        _print_row(SHORT[corpus], "GT", {**gt, "tv_bar": None, "tv_beat": None}, bar_ok)

        groups = []
        for label, d in runs:
            if args.loo:
                groups.append((f"{label} full", d, ["7sets_s"]))
                groups.append((f"{label} -C", d, [f"7sets_no_{corpus}_"]))
            else:
                groups.append((label, d, args.cells))
        for label, d, prefixes in groups:
            subs = [s for s in sorted((d / corpus / "models").iterdir())
                    if (s / "preds.pt").exists()
                    and (prefixes is None or any(s.name.startswith(p) for p in prefixes))]
            if not subs:
                continue
            profs = [profile(s / "preds.pt", rows, events, pads, bar_ok) for s in subs]
            row[label] = _mean_profiles(profs, gt, bar_ok)
            _print_row("", label, row[label], bar_ok)
        results[corpus] = row
        print()

    # where does each run's excess change rate go?
    labels = [k for k in next(iter(results.values())) if k not in ("gt", "n_windows",
                                                                    "n_dropped")]
    print("excess change rate by bar position (model minus GT, changes per 16 frames)")
    print(f"{'corpus':<12}{'run':<10}" + "".join(f"{b:>9}" for b in BAR_BUCKETS))
    for corpus, row in results.items():
        for label in labels:
            ex = row.get(label, {}).get("excess_rate") if row.get(label) else None
            if ex is None:
                continue
            print(f"{SHORT[corpus]:<12}{label:<10}" + "".join(f"{v:>+9.3f}" for v in ex))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
