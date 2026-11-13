#!/usr/bin/env python3
"""Leave-one-out significance for every reported metric, with one family-wise correction.

The LOO question: does training WITHOUT corpus C make generation on C worse? For each corpus C
the full-mixture models (7sets_s42/43/44) are compared with the models trained with C held out
(7sets_no_C_s42/43/44), both scored on C's test windows, within one stage (MLE or GAPT).

Every metric is turned into a distance from ground truth, so "worse" means the same thing
everywhere -- GT is the target, not the maximum (journal/reference/METRICS.md):

    note-in-chord, chord complexity, chord duration entropy   |model - GT|
    sync EMD, root-motion EMD, chord-placement TV    already distances to GT

damage = distance(held-out) - distance(full); positive means holding C out hurt on C.

Two tests per (stage, metric):

  per corpus  Welch t-test, 3 held-out vs 3 full seeds, Holm-corrected across corpora.
              Says WHICH corpora show the effect; low power at 3 seeds.
  pooled      stratified permutation test. Under "no LOO effect" the six models of a corpus are
              exchangeable, so full/held-out labels are permuted within each corpus (20
              labelings per corpus, Monte Carlo draws). Statistic = sum over corpora of
              (mean held-out - mean full distance) / sd of that corpus's six values, which is
              invariant under the permutation. Says whether there is an effect ON AVERAGE.

The pooled p-values of all (stage, metric) pairs form one family and are Holm-corrected
together. Pre-registered exclusion: Nottingham for chord placement (compound meter, 2026-10-14).

Within one stage every model was scored on the same windows, so no truncation is needed (the
GAPT Hooktheory run has 1,024 windows, the MLE one 2,785; each stage uses its own GT).

Usage:
    python scripts/eval/custom_eval/loo_significance.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_ind, wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))
from chord_placement import BAR_EXCLUDED, _vocab_maps, keep_rows, profile, tv  # noqa: E402
from fifths_motion import step_sequence  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}
STAGES = {"MLE": Path("logs/custom_eval/dataset_loo_new"),
          "GAPT": Path("logs/custom_eval/gapt_loo")}
SEEDS = ["s42", "s43", "s44"]
# name, source, means.json key (or None), how to turn it into a distance, excluded corpora
METRICS = [
    ("note-in-chord", "means", "note_in_chord_ratio_mean", "abs", set()),
    ("chord complexity", "means", "chord_complexity_mean", "abs", set()),
    ("chord duration entropy", "means", "chord_duration_entropy", "abs", set()),
    ("sync EMD", "means", "sync_emd_vs_gt", "direct", set()),
    ("root-motion EMD", "computed", None, "direct", set()),
    ("chord placement TV", "computed", None, "direct", set(BAR_EXCLUDED)),
]


def _cells(corpus):
    return {"full": [f"7sets_{s}" for s in SEEDS],
            "held": [f"7sets_no_{corpus}_{s}" for s in SEEDS]}


def _computed_values(stage_dir: Path, corpus: str, labels=None) -> dict:
    """Per-model values (and GT) for the three metrics computed from saved sequences."""
    base = stage_dir / corpus
    names = json.loads((base / "chord_names_augmented.json").read_text())
    tok = HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names))
    i2n = tok.id_to_name
    events, pads = _vocab_maps(i2n)
    n_rows = torch.load(base / "gt" / "gt.pt", map_location="cpu", weights_only=False).shape[0]
    rows, _ = keep_rows(base / "metadata.jsonl", n_rows)
    bar_ok = corpus not in BAR_EXCLUDED
    gt_path = base / "gt" / "gt.pt"
    gt_steps = step_sequence(gt_path, i2n, n_rows)
    gt_prof = profile(gt_path, rows, events, pads, bar_ok)
    out = {"gt": {}, "models": {}}
    labels = labels or _cells(corpus)["full"] + _cells(corpus)["held"]
    for label in labels:
        p = base / "models" / label / "preds.pt"
        prof = profile(p, rows, events, pads, bar_ok)
        out["models"][label] = {
            "root-motion EMD": float(wasserstein_distance(gt_steps,
                                                          step_sequence(p, i2n, n_rows))),
            "chord placement TV": (tv(prof["bar_share"], gt_prof["bar_share"])
                                   if bar_ok else None),
        }
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def pooled_permutation(blocks, draws: int, rng) -> tuple:
    """blocks: list of (values[6] with full first, sd). Returns (statistic, two-sided p)."""
    # Each corpus has only C(6,3) = 20 labelings, so precompute its 20 possible contributions
    # and sum independent random picks -- the same null distribution, vectorised.
    perms = [np.array(list(c) + [i for i in range(6) if i not in c])
             for c in itertools.combinations(range(6), 3)]       # perms[0] = observed labeling
    table = np.array([[(v[l[3:]].mean() - v[l[:3]].mean()) / sd for l in perms]
                      for v, sd in blocks])                     # [n_blocks, 20]
    obs = table[:, 0].sum()
    picks = rng.integers(0, len(perms), size=(draws, len(blocks)))
    null = table[np.arange(len(blocks)), picks].sum(axis=1)
    return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/loo_significance.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    results = {}
    for stage, stage_dir in STAGES.items():
        # distances per corpus: {metric: {corpus: (full[3], held[3])}}
        dist = {m[0]: {} for m in METRICS}
        for corpus in CORPORA:
            means = json.loads((stage_dir / corpus / "means.json").read_text())
            comp = _computed_values(stage_dir, corpus)
            cells = _cells(corpus)
            for name, source, key, kind, excluded in METRICS:
                if corpus in excluded:
                    continue
                if source == "means":
                    gt = means["gt"].get(key)
                    vals = {g: [means["models"][l][key] for l in ls] for g, ls in cells.items()}
                else:
                    gt = comp["gt"].get(name)
                    vals = {g: [comp["models"][l][name] for l in ls] for g, ls in cells.items()}
                if kind == "abs":
                    vals = {g: [abs(v - gt) for v in vs] for g, vs in vals.items()}
                dist[name][corpus] = (vals["full"], vals["held"])
            print(f"  [{stage}] {SHORT[corpus]} done", flush=True)

        for name, *_ in METRICS:
            per = dist[name]
            corp = list(per)
            dmg = [float(np.mean(h) - np.mean(f)) for f, h in per.values()]
            p_corp = [float(ttest_ind(h, f, equal_var=False).pvalue)
                      if np.std(f) + np.std(h) > 0 else 1.0 for f, h in per.values()]
            p_holm = holm(p_corp)
            blocks = []
            for f, h in per.values():
                v = np.array(list(f) + list(h), dtype=float)
                blocks.append((v, v.std(ddof=1) if v.std(ddof=1) > 0 else 1.0))
            obs, p_pool = pooled_permutation(blocks, args.draws, rng)
            k = sum(d > 0 for d in dmg)
            p_sign = min(1.0, 2 * sum(math.comb(len(dmg), i) for i in range(max(k, len(dmg) - k),
                                                                              len(dmg) + 1))
                         / 2 ** len(dmg))
            results[f"{stage}/{name}"] = {
                "corpora": corp, "damage": dmg, "p_corpus": p_corp, "p_corpus_holm": p_holm,
                "held_out_worse": k, "n": len(dmg), "p_sign": p_sign,
                "pooled_statistic": obs, "p_pooled": p_pool,
                "distances": {c: {"full": f, "held": h} for c, (f, h) in per.items()},
            }

    keys = list(results)
    family = holm([results[k]["p_pooled"] for k in keys])
    for k, adj in zip(keys, family):
        results[k]["p_pooled_holm_family"] = adj

    print(f"\nLOO significance -- {len(keys)} pooled tests, Holm across all of them\n")
    print(f"{'stage':<6}{'metric':<24}{'worse':>7}{'sign p':>8}{'pooled p':>10}{'Holm p':>9}"
          f"   per-corpus significant, Welch + Holm (+ held-out worse, - held-out better)")
    for k in keys:
        r = results[k]
        # "+" = held-out worse (the expected direction), "-" = held-out BETTER on its own corpus
        sig = [f"{SHORT[c]} {'+' if d > 0 else '-'}{p:.3f}" for c, p, d in
               zip(r["corpora"], r["p_corpus_holm"], r["damage"]) if p < 0.05]
        stage, name = k.split("/", 1)
        print(f"{stage:<6}{name:<24}{r['held_out_worse']:>4}/{r['n']:<2}{r['p_sign']:>8.3f}"
              f"{r['p_pooled']:>10.4f}{r['p_pooled_holm_family']:>9.4f}   "
              f"{', '.join(sig) if sig else '-'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
