#!/usr/bin/env python3
"""Score generated chord progressions against a corpus's progression reference.

Reads the token dumps an evaluation already wrote (gt/gt.pt and
models/<slug>/preds.pt), so it needs no regeneration -- any completed custom_eval
run can be scored retrospectively.

Every number is reported against the floor from the reference file: the same
divergence between held-out ground truth and the training reference, which carries
the same finite-sample bias. A model's `excess` (distance - floor) is the quantity to
read; the raw distance is not comparable across corpora, whose sample sizes differ by
two orders of magnitude.

All models within a corpus are subsampled to a common transition count, because they
emit different numbers of chord changes (on Hooktheory the RL policy produced ~49k
against ground truth's ~32k) and an estimated divergence grows with bins/n.

Usage:
    python scripts/eval/custom_eval/score_progressions.py \
        --eval_dir logs/custom_eval/alpha_sweep_allwin \
        --label MLE --out logs/custom_eval/progression_scores.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

REPO_ROOT = Path(__file__).resolve().parents[3]
REF_DIR = REPO_ROOT / "data" / "cache" / "progression_reference"


def load_reference(dataset_name: str) -> dict:
    path = REF_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise SystemExit(
            f"no progression reference for {dataset_name}: {path}\n"
            f"  build it: python scripts/eval/custom_eval/build_progression_reference.py "
            f"--datasets {dataset_name}"
        )
    ref = json.loads(path.read_text(encoding="utf-8"))
    ref["_hist"] = {
        name: Counter({tuple(json.loads(k)) if json.loads(k).__class__ is list
                       else json.loads(k): v for k, v in hist.items()})
        for name, hist in ref["train"]["histograms"].items()
    }
    return ref


def events_from_dump(path: Path, id_to_name, classes) -> list:
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    return [
        pm.chord_events(pm.chord_lane(row), id_to_name, classes=classes)
        for row in seqs.tolist()
    ]


def score_corpus(corpus_dir: Path, dataset_name: str, tokenizer_names) -> dict | None:
    gt_path = corpus_dir / "gt" / "gt.pt"
    models_dir = corpus_dir / "models"
    if not gt_path.exists() or not models_dir.exists():
        return None
    ref = load_reference(dataset_name)
    classes = ref["classes"]
    tok = HooktheoryTokenizer(chord_names=tokenizer_names)
    id_to_name = tok.id_to_name

    series = {"gt": events_from_dump(gt_path, id_to_name, classes)}
    for sub in sorted(models_dir.iterdir()):
        preds = sub / "preds.pt"
        if preds.exists():
            series[sub.name] = events_from_dump(preds, id_to_name, classes)

    trans = {k: pm.transitions(v) for k, v in series.items()}
    match_n = min(len(t) for t in trans.values() if t)

    floor = ref["test_floor"]["js_vs_train"]
    out = {
        "dataset_name": dataset_name,
        "classes": classes,
        "match_n": match_n,
        "floor": floor,
        "reference_transitions": ref["train"]["num_transitions"],
        "models": {},
    }
    for label, tr in trans.items():
        js = pm.compare(tr, ref["_hist"], classes, match_n=match_n)
        out["models"][label] = {
            "num_transitions": len(tr),
            "js": js,
            "excess": {
                k: (None if js.get(k) is None or floor.get(k) is None
                    else js[k] - floor[k])
                for k in js
            },
            "conditionals": pm.conditionals(tr),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, type=Path)
    ap.add_argument("--label", default=None, help="name for this run in the output")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    for sub in sorted(args.eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        vocab = sub / "chord_names_augmented.json"
        if not vocab.exists():
            continue
        names = json.loads(vocab.read_text(encoding="utf-8"))
        names = names if isinstance(names, list) else list(names)
        scored = score_corpus(sub, sub.name, names)
        if scored is None:
            continue
        results[sub.name] = scored
        f = scored["floor"]
        print(f"\n=== {sub.name} === (match_n={scored['match_n']:,})")
        print(f"  floor  droot={f['droot']:.4f}" if f.get("droot") is not None else "  floor  --")
        print(f"  {'model':<14}{'n':>8}{'droot':>9}{'quality':>9}{'droot_qto':>11}"
              f"{'fifth%':>9}{'tritone%':>10}")
        for label, m in scored["models"].items():
            e, c = m["excess"], m["conditionals"]
            def fmt(x, w=9, p=4):
                return f"{x:>{w}.{p}f}" if x is not None else f"{'--':>{w}}"
            print(f"  {label:<14}{m['num_transitions']:>8,}{fmt(e.get('droot'))}"
                  f"{fmt(e.get('quality'))}{fmt(e.get('droot_qto'),11)}"
                  f"{fmt(c.get('fifth_motion_rate'),9,3)}{fmt(c.get('tritone_rate'),10,4)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
