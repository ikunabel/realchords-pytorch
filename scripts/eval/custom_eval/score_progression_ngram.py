#!/usr/bin/env python3
"""Fit a chord-progression n-gram per corpus and score generated progressions with it.

Complements score_progressions.py: those are bigram histograms, which cannot see past
adjacent chord pairs. This fits a model with longer context, so it responds to phrase
structure -- whether a progression cadences and returns, or merely strings locally
plausible steps together.

The reported quantity is NOT perplexity to be minimised. A model looping one cliche
forever is maximally predictable and would win. Instead each progression gets a mean
log2 probability per chord move, and the *distribution* of those values is compared
against the same distribution for held-out real music:

    mean_shift > 0   model is more predictable than real music  (cliche collapse)
    mean_shift < 0   model is less predictable                  (scattered)
    sd_ratio  < 1    model's progressions are more uniform than real ones
    ks               largest gap between the two distributions

Held-out ground truth is scored too, as the floor: the n-gram is fit on train, so real
test progressions do not score perfectly either, and that gap is the baseline.

Usage:
    python scripts/eval/custom_eval/score_progression_ngram.py \
        --eval_dirs MLE=logs/custom_eval/alpha_sweep_allwin \
                    GAPT=logs/custom_eval/gapt_alpha_allwin
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm
from realchords.utils import progression_ngram as png
from realchords.utils.experiment_utils import create_dataset_dataloaders

DATASETS = ["hooktheory", "pop909", "emopia_plus", "nottingham",
            "wikifonia", "chord_melody_dataset", "jazzmus"]


def train_events(dataset_name: str, max_len: int = 512):
    meta = {"max_len": max_len, "split": "train", "dataset_name": dataset_name}
    cached = pm.load_events(pm.events_cache_path(dataset_name, "train"), meta)
    if cached is not None:
        print(f"  loaded {len(cached):,} train sequences from cache", flush=True)
        return cached
    built = create_dataset_dataloaders(
        dataset_name=dataset_name, dataset_split="train", model_part="chord",
        batch_size=64, max_len=max_len, num_workers=0, all_windows=True,
    )
    loader = next(x for x in built if x is not None)
    dataset = loader.dataset
    id_to_name = dataset.tokenizer.id_to_name
    events = [
        pm.chord_events(pm.chord_lane(dataset[i]["targets"].tolist()), id_to_name)
        for i in range(len(dataset))
    ]
    pm.save_events(pm.events_cache_path(dataset_name, "train"), events, meta)
    return events


def dump_events(path: Path, id_to_name):
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    return [pm.chord_events(pm.chord_lane(row), id_to_name) for row in seqs.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--orders", nargs="+", type=int, default=[3],
                    help="n-gram orders to fit. A symbol is a chord *move*, so order k "
                         "spans k+1 chords. The training pass is shared across orders, "
                         "which is the expensive part.")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out_dir", type=Path, default=Path("logs/custom_eval"),
                    help="one file per order: progression_ngram_order<k>.json")
    args = ap.parse_args()

    runs = []
    for item in args.eval_dirs:
        label, path = item.split("=", 1)
        runs.append((label, Path(path)))

    results = {order: {} for order in args.orders}
    for name in args.datasets:
        corpus_dirs = [(lab, d / name) for lab, d in runs if (d / name / "gt" / "gt.pt").exists()]
        if not corpus_dirs:
            continue
        print(f"\n=== {name} ===", flush=True)
        train = train_events(name)          # shared across orders
        # Decode the token dumps ONCE and reuse across orders. Decoding dominates the
        # runtime (189 dumps for the LOO grid), and re-doing it per order made a 6-order
        # sweep 6x more expensive than it needs to be.
        vocab_path = corpus_dirs[0][1] / "chord_names_augmented.json"
        names = json.loads(vocab_path.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        id_to_name = HooktheoryTokenizer(chord_names=names).id_to_name
        decoded = {"gt": dump_events(corpus_dirs[0][1] / "gt" / "gt.pt", id_to_name)}
        for label, cdir in corpus_dirs:
            for sub in sorted((cdir / "models").iterdir()):
                preds = sub / "preds.pt"
                if preds.exists():
                    decoded[(label, sub.name)] = dump_events(preds, id_to_name)
        print(f"  decoded {len(decoded)} dumps once, reused across {len(args.orders)} orders",
              flush=True)
        for order in args.orders:
            results[order][name] = _score_one(name, order, args, train, corpus_dirs, decoded)
    for order, payload in results.items():
        out = args.out_dir / f"progression_ngram_order{order}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")


def _score_one(name, order, args, train, corpus_dirs, decoded):
        model = png.ProgressionNGram(order=order, alpha=args.alpha).fit(train)
        print(f"  order {order} (spans {order+1} chords): {model.num_symbols_seen:,} moves, "
              f"{len(model.qualities)} qualities, {len(model.ngrams[order-1]):,} contexts")

        gt_ll = model.per_sequence_logprob(decoded["gt"])
        gt_sum = png.distribution_summary(gt_ll)
        print(f"  held-out GT (the floor): mean={gt_sum['mean']:.3f} sd={gt_sum['sd']:.3f} "
              f"n={gt_sum['n']}")
        entry = {"gt": gt_sum, "models": {}}

        print(f"  {'run':<6}{'cell':<12}{'mean':>9}{'shift':>9}{'sd_ratio':>10}{'ks':>7}")
        for label, cdir in corpus_dirs:
            per_cell = {}
            for sub in sorted((cdir / "models").iterdir()):
                if (label, sub.name) not in decoded:
                    continue
                ll = model.per_sequence_logprob(decoded[(label, sub.name)])
                per_cell[sub.name] = {
                    "summary": png.distribution_summary(ll),
                    "vs_gt": png.two_sample(ll, gt_ll),
                }
            entry["models"][label] = per_cell
            if per_cell:
                shifts = [v["vs_gt"]["mean_shift"] for v in per_cell.values()]
                sds = [v["vs_gt"]["sd_ratio"] for v in per_cell.values() if v["vs_gt"]["sd_ratio"]]
                kss = [v["vs_gt"]["ks"] for v in per_cell.values()]
                means = [v["summary"]["mean"] for v in per_cell.values()]
                print(f"  {label:<6}{'(mean of ' + str(len(per_cell)) + ')':<12}"
                      f"{st.mean(means):>9.3f}{st.mean(shifts):>+9.3f}"
                      f"{st.mean(sds):>10.3f}{st.mean(kss):>7.3f}")
        return entry


if __name__ == "__main__":
    main()
