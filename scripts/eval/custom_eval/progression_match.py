#!/usr/bin/env python3
"""How many of a model's chord progressions actually occur in real music?

For every window, take each span of n consecutive chords, and ask whether that exact
progression appears anywhere in the corpus's training split. The reported number is the
share that do.

Read it against the GT row, which is the held-out real music put through the same
lookup. That row is the reference and it is NOT 100%: real music contains progressions
the training split happens not to have, so e.g. 45% at six chords is what "idiomatic"
looks like for Hooktheory. It also makes the measure two-sided -- below the reference
means unidiomatic, above it means the model is leaning on the commonest patterns harder
than real music does.

Representation. A progression is stored as the opening chord's quality followed by one
(root interval, landing quality) pair per move:

    C  G  Am F   ->  ('MP', (7,'MP'), (2,'mP'), (8,'MP'))

so it is transposition-invariant by construction -- the same shape in D matches -- while
the quality of every chord including the first is kept. An earlier version omitted the
opening quality, which made Cm-G-Am-F match C-G-Am-F.

--quality full keeps extensions apart (C7 and C13 are different tokens); --quality
function collapses them to one, since they share a harmonic function. Full is the
stricter test and lowers every row including GT, which is why GT must be reported.

The absolute rate rises with training-set size, so numbers are not comparable across
corpora; the GT row normalises that within each corpus.

Usage:
    python scripts/eval/custom_eval/progression_match.py \
        --eval_dirs MLE=logs/custom_eval/alpha_sweep_allwin \
                    GAPT=logs/custom_eval/gapt_alpha_allwin --quality full
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm
from realchords.utils.experiment_utils import create_dataset_dataloaders

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}


def patterns(seq, n_chords: int):
    """Every n-chord progression in one window, each as a transposition-free tuple."""
    out = []
    for i in range(len(seq) - n_chords + 1):
        block = seq[i:i + n_chords]
        moves = tuple(((block[j][0] - block[j - 1][0]) % 12, block[j][1])
                      for j in range(1, n_chords))
        out.append((block[0][1],) + moves)      # opening quality + the moves
    return out


def train_events(dataset_name: str, quality: str, max_len: int = 512):
    meta = {"max_len": max_len, "split": "train", "dataset_name": dataset_name,
            "quality": quality}
    path = pm.events_cache_path(dataset_name, "train", quality=quality)
    cached = pm.load_events(path, meta)
    if cached is not None:
        return cached
    built = create_dataset_dataloaders(
        dataset_name=dataset_name, dataset_split="train", model_part="chord",
        batch_size=64, max_len=max_len, num_workers=0, all_windows=True)
    loader = next(x for x in built if x is not None)
    ds = loader.dataset
    i2n = ds.tokenizer.id_to_name
    events = [pm.chord_events(pm.chord_lane(ds[i]["targets"].tolist()), i2n, quality=quality)
              for i in range(len(ds))]
    pm.save_events(path, events, meta)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR")
    ap.add_argument("--quality", choices=["function", "full"], default="full")
    ap.add_argument("--spans", nargs="+", type=int, default=[3, 4, 6])
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    runs = [(i.split("=", 1)[0], Path(i.split("=", 1)[1])) for i in args.eval_dirs]
    results = {}
    cols = "".join(f"{str(n) + ' chords':>12}" for n in args.spans)
    print(f"\nverbatim match against training  (quality={args.quality})")
    print(f"{'corpus':<13}{'run':<6}{cols}{'ref patterns':>26}")
    for c in CORPORA:
        vocab = runs[0][1] / c / "chord_names_augmented.json"
        if not vocab.exists():
            continue
        names = json.loads(vocab.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        i2n = HooktheoryTokenizer(chord_names=names).id_to_name

        ref = {n: set() for n in args.spans}
        for seq in train_events(c, args.quality):
            for n in args.spans:
                ref[n].update(patterns(seq, n))

        def decode(path):
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            return [pm.chord_events(pm.chord_lane(r), i2n, quality=args.quality)
                    for r in seqs.tolist()]

        def rate(seqs, n):
            per = []
            for s in seqs:
                ps = patterns(s, n)
                if ps:
                    per.append(sum(1 for p in ps if p in ref[n]) / len(ps))
            return st.mean(per) if per else None

        row = {}
        gt = decode(runs[0][1] / c / "gt" / "gt.pt")
        row["gt"] = {n: rate(gt, n) for n in args.spans}
        sizes = "  ".join(f"{len(ref[n]):,}" for n in args.spans)
        print(f"{SHORT[c]:<13}{'GT':<6}"
              + "".join(f"{row['gt'][n]:>11.1%}" if row['gt'][n] is not None else f"{'--':>12}"
                        for n in args.spans) + f"{sizes:>26}")
        for label, d in runs:
            subs = [s for s in sorted((d / c / "models").iterdir())
                    if (s / "preds.pt").exists()
                    and (args.cells is None or any(s.name.startswith(x) for x in args.cells))]
            if not subs:
                continue
            per_cell = [{n: rate(decode(s / "preds.pt"), n) for n in args.spans} for s in subs]
            m = {n: st.mean(p[n] for p in per_cell if p[n] is not None) for n in args.spans}
            row[label] = m
            print(f"{'':<13}{label:<6}" + "".join(f"{m[n]:>11.1%}" for n in args.spans))
        results[c] = row
        print()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
