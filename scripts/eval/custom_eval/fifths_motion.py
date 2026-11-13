#!/usr/bin/env python3
"""How far does the model move the chord root, measured on the line of fifths?

Every chord change is one root interval. Converting that interval to a number of
steps on the line of fifths turns it into a graded harmonic distance: a fifth or a
fourth is one step, a whole tone is two, a semitone is five, a tritone is six. Steps
of four or more are chromatic motion -- the moves that sound harmonically remote.

    k = 7 * root_interval (mod 12), folded into [-6, 6]

7 is its own inverse mod 12, which is why the same expression maps pitch classes to
line-of-fifths positions and intervals to step counts.

Two numbers are reported per model: the share of its root motions at four or more
steps, and the Wasserstein-1 distance between its full step distribution and GT's.
Both are read against the GT row, which is held-out real music through the same
computation, and GT differs greatly between corpora -- 74.1% of Nottingham's root
motions are single fifth steps against 29.6% of EMOPIA+'s -- so the GT row is the
reference and neither number is comparable across corpora without it.

The distribution carries the signal and the mean does not: a model can hold GT's mean
step size while having both more chromatic motion and more static-root motion than GT,
the two errors cancelling. The mean step size was measured and separates nothing.

Why the line of fifths rather than a published pitch-space model: see
journal/reference/METRICS.md. Chew's Spiral Array and Lerdahl's Tonal Pitch Space have no Python
implementation; music21's neo-Riemannian operations are defined only for consonant
triads, whose share differs between GT and the models by enough to bias the comparison;
and librosa's Harte tonal centroid was measured and does not separate the models,
because weighting the circles of fifths, major thirds and minor thirds equally makes a
two-step motion and a five-step motion almost equidistant.

Duration is invisible here: a repeated chord is not an
event, so only changes in (root, quality) are counted. Quality is invisible too -- a
C-to-Cm change is zero steps -- so report the zero-step column, which is where quality
change without root motion appears.

Usage:
    python scripts/eval/custom_eval/fifths_motion.py \
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new \
                    GAPT=logs/custom_eval/gapt_loo \
        --out logs/custom_eval/fifths_motion_loo.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import torch
from scipy.stats import wasserstein_distance

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}
MAX_STEPS = 6


def fifths_step_signed(root_interval: int) -> int:
    """Root interval in semitones -> signed position on the line of fifths, [-6, 6].

    Sign is direction along the line of fifths, which tonal harmony is not symmetric
    about: C->G (up a fifth, I-V) is +1, C->F (up a fourth, the V-I direction) is -1.
    Descending-fifths motion carries authentic cadences and ii-V-I, so the signed
    distribution is expected to lean negative in real music. Folding to |k| hides that,
    which is why both are reported.
    """
    k = (7 * root_interval) % 12
    return k - 12 if k > MAX_STEPS else k


def fifths_steps(root_interval: int) -> int:
    """Unsigned distance in steps on the line of fifths."""
    return abs(fifths_step_signed(root_interval))


def _rows(path: Path) -> int:
    return torch.load(path, map_location="cpu", weights_only=False).shape[0]


def _first_preds(models_dir: Path):
    return next((s / "preds.pt" for s in sorted(models_dir.iterdir())
                 if (s / "preds.pt").exists()), None)


def step_sequence(path: Path, id_to_name, limit: int, signed: bool = False):
    """Every root motion in one .pt file, as line-of-fifths steps (signed optional)."""
    seqs = torch.load(path, map_location="cpu", weights_only=False)[:limit]
    step = fifths_step_signed if signed else fifths_steps
    out = []
    for row in seqs.tolist():
        events = pm.chord_events(pm.chord_lane(row), id_to_name, quality="full")
        out += [step((b[0] - a[0]) % 12) for a, b in zip(events, events[1:])]
    return out


def descending_share(signed_steps):
    """Share of non-zero root motions that go DOWN the line of fifths (the V-I direction)."""
    moved = [s for s in signed_steps if s != 0]
    return sum(1 for s in moved if s < 0) / len(moved) if moved else None


def _histogram(steps):
    counts = Counter(steps)
    return [counts[k] / len(steps) for k in range(MAX_STEPS + 1)]


def _remote_share(steps, threshold: int = 4):
    return sum(1 for s in steps if s >= threshold) / len(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR",
                    help="GT and the chord vocabulary are read from the first one.")
    ap.add_argument("--threshold", type=int, default=4,
                    help="steps at or above this count as chromatic motion")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="only models whose directory name starts with one of these")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = [(i.split("=", 1)[0], Path(i.split("=", 1)[1])) for i in args.eval_dirs]
    results = {}

    header = "".join(f"{k:>7}" for k in range(MAX_STEPS + 1))
    print(f"\nroot motion on the line of fifths   (>={args.threshold} steps = chromatic)")
    print(f"{'corpus':<12}{'run':<6}{header}{'>=' + str(args.threshold):>8}{'desc':>8}{'EMD vs GT':>11}")

    for corpus in CORPORA:
        vocab = runs[0][1] / corpus / "chord_names_augmented.json"
        if not vocab.exists():
            continue
        names = json.loads(vocab.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        id_to_name = HooktheoryTokenizer(chord_names=names).id_to_name

        # Runs can differ in window count (e.g. hooktheory GAPT ran num_batches 16 =
        # 1,024 of 2,785 windows). The eval loader is shuffle=False and every run uses
        # the same dataset/split/cropped_max_len/all_windows, so row i is the same
        # window everywhere. Truncate all runs and GT to the shortest; no-op when equal.
        counts = [_rows(runs[0][1] / corpus / "gt" / "gt.pt")]
        for _, directory in runs:
            first = _first_preds(directory / corpus / "models")
            if first is not None:
                counts.append(_rows(first))
        limit = min(counts)
        if len(set(counts)) > 1:
            print(f"  [{SHORT[corpus]}] window counts {sorted(set(counts))} "
                  f"-> truncating every run and GT to {limit:,}")

        gt_signed = step_sequence(runs[0][1] / corpus / "gt" / "gt.pt", id_to_name,
                                  limit, signed=True)
        gt_steps = [abs(v) for v in gt_signed]
        row = {"gt": {"hist": _histogram(gt_steps),
                      "remote": _remote_share(gt_steps, args.threshold),
                      "descending": descending_share(gt_signed),
                      "emd": 0.0}}
        print(f"{SHORT[corpus]:<12}{'GT':<6}"
              + "".join(f"{v:>6.1%}" for v in row["gt"]["hist"])
              + f"{row['gt']['remote']:>8.1%}{row['gt']['descending']:>8.1%}{'-':>11}")

        for label, directory in runs:
            subs = [s for s in sorted((directory / corpus / "models").iterdir())
                    if (s / "preds.pt").exists()
                    and (args.cells is None
                         or any(s.name.startswith(c) for c in args.cells))]
            if not subs:
                continue
            signed = [step_sequence(s / "preds.pt", id_to_name, limit, signed=True)
                      for s in subs]
            per_model = [[abs(v) for v in sg] for sg in signed]
            hists = [_histogram(p) for p in per_model]
            row[label] = {
                "hist": [st.mean(h[k] for h in hists) for k in range(MAX_STEPS + 1)],
                "remote": st.mean(_remote_share(p, args.threshold) for p in per_model),
                "descending": st.mean(descending_share(v) for v in signed
                                      if descending_share(v) is not None),
                "emd": st.mean(wasserstein_distance(gt_steps, p) for p in per_model),
                "n_models": len(subs),
            }
            print(f"{'':<12}{label:<6}"
                  + "".join(f"{v:>6.1%}" for v in row[label]["hist"])
                  + f"{row[label]['remote']:>8.1%}{row[label]['descending']:>8.1%}{row[label]['emd']:>11.4f}")

        results[corpus] = row
        print()

    # Across-corpus spread of the single-fifth share: GT varies far more than the
    # models do, which is the style-adaptation result rather than a per-corpus one.
    print("single-fifth share, spread across corpora")
    for label in (["gt"] + [r[0] for r in runs]):
        shares = [results[c][label]["hist"][1] for c in results if label in results[c]]
        if len(shares) > 1:
            print(f"  {label.upper():<6} range {min(shares):.1%}-{max(shares):.1%}"
                  f"   spread {max(shares) - min(shares):.1%}"
                  f"   sd {st.stdev(shares):.1%}")

    print(f"\nis each model closer to GT than {runs[0][0]}?")
    for label, _ in runs[1:]:
        closer_remote = sum(
            1 for c in results
            if abs(results[c][label]["remote"] - results[c]["gt"]["remote"])
            < abs(results[c][runs[0][0]]["remote"] - results[c]["gt"]["remote"]))
        closer_emd = sum(1 for c in results
                         if results[c][label]["emd"] < results[c][runs[0][0]]["emd"])
        print(f"  {label}: closer on >={args.threshold} share {closer_remote}/{len(results)}"
              f"   closer on EMD {closer_emd}/{len(results)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
