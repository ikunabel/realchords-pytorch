#!/usr/bin/env python3
"""Repetition and closure in generated chord progressions.

Motivation: the divergence metrics say a model's chord *distribution* is off but not
why it sounds off. A very common cause is that generated harmony wanders -- real
progressions loop and come back, models drift through fresh chords. That is measured
directly here rather than inferred from surprisal.

Four statistics per window, all comparing a model against the same corpus's ground
truth on identical windows:

  n_chords      chord changes per window
  distinct      distinct chords used
  repeat_rate   1 - distinct/n_chords: share of changes that re-use a chord
  returns       does the opening chord recur later in the window (tonal closure)
  loop4         does any 4-chord block recur verbatim immediately after itself

`returns` is deliberately conservative as a criticism of a model: a model that emits
MORE chords per window has more opportunities to return, so a lower rate despite a
higher chord count cannot be explained by length.

These are descriptive, not distances: read each against the GT row, in both directions
-- too little repetition is wandering, too much is a model stuck in a loop.

Usage:
    python scripts/eval/custom_eval/progression_structure.py \
        --eval_dirs MLE=logs/custom_eval/alpha_sweep_allwin \
                    GAPT=logs/custom_eval/gapt_alpha_allwin --cells a0p5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}
MIN_CHORDS = 4


def stats(sequences) -> dict:
    seqs = [s for s in sequences if len(s) >= MIN_CHORDS]
    if not seqs:
        return {}

    def loop4(s):
        return any(s[i:i + 4] == s[i + 4:i + 8] for i in range(len(s) - 7))

    return {
        "n_chords": st.mean(len(s) for s in seqs),
        "distinct": st.mean(len(set(s)) for s in seqs),
        "repeat_rate": st.mean(1 - len(set(s)) / len(s) for s in seqs),
        "returns": st.mean(1.0 if s[0] in s[1:] else 0.0 for s in seqs),
        "loop4": st.mean(1.0 if loop4(s) else 0.0 for s in seqs),
        "n_windows": len(seqs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="model-label prefixes to average over, e.g. a0p5 (all seeds)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = [(i.split("=", 1)[0], Path(i.split("=", 1)[1])) for i in args.eval_dirs]
    results = {}
    hdr = f"{'corpus':<13}{'run':<6}{'n_ch':>7}{'distinct':>9}{'repeat':>8}{'returns':>9}{'loop4':>7}"
    print(hdr)
    print("-" * len(hdr))
    for c in CORPORA:
        vocab = runs[0][1] / c / "chord_names_augmented.json"
        if not vocab.exists():
            continue
        names = json.loads(vocab.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        id_to_name = HooktheoryTokenizer(chord_names=names).id_to_name

        def decode(path):
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            return [pm.chord_events(pm.chord_lane(r), id_to_name) for r in seqs.tolist()]

        row = {"gt": stats(decode(runs[0][1] / c / "gt" / "gt.pt"))}
        print(f"{SHORT[c]:<13}{'GT':<6}" + "".join(
            f"{row['gt'][k]:>{w}.{p}f}" if k not in ("returns", "repeat_rate", "loop4")
            else f"{row['gt'][k]:>{w}.1%}"
            for k, w, p in (("n_chords", 7, 1), ("distinct", 9, 1),
                            ("repeat_rate", 8, 0), ("returns", 9, 0), ("loop4", 7, 0))))
        for label, d in runs:
            per = [sub for sub in sorted((d / c / "models").iterdir())
                   if (sub / "preds.pt").exists()
                   and (args.cells is None or any(sub.name.startswith(x) for x in args.cells))]
            if not per:
                continue
            agg = [stats(decode(sub / "preds.pt")) for sub in per]
            m = {k: st.mean(a[k] for a in agg) for k in agg[0]}
            row[label] = m
            print(f"{'':<13}{label:<6}" + "".join(
                f"{m[k]:>{w}.{p}f}" if k not in ("returns", "repeat_rate", "loop4")
                else f"{m[k]:>{w}.1%}"
                for k, w, p in (("n_chords", 7, 1), ("distinct", 9, 1),
                                ("repeat_rate", 8, 0), ("returns", 9, 0), ("loop4", 7, 0))))
        results[c] = row
        print()
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
