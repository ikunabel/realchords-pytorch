#!/usr/bin/env python3
"""Does RL (GAPT) move generation closer to real music than its MLE initialisation?

Paired, stratified sign-flip permutation test, one per metric, Holm across metrics.

Pairs. Every GAPT model in logs/custom_eval/gapt_loo was RL-tuned from the MLE checkpoint with
the same label in logs/custom_eval/dataset_loo_new (configs/rl/gapt_loo_*.yml,
pretrain_model_path), so each corpus has 27 pairs (9 training mixes x 3 seeds), scored on the
same test windows. The GAPT Hooktheory run has the first 1,024 of MLE's 2,785 windows, so both
stages are scored on the windows they share (GT checked identical).

Every metric is a distance from ground truth, as in loo_significance.py:

    note-in-chord, chord complexity, chord duration entropy,
    chord changes per bar                                     |model - GT|
    sync EMD, root-motion EMD                                 already distances to GT

The four metrics from custom_evaluation are recomputed here with its own functions (so they can
be restricted to the shared windows); on full windows they must reproduce means.json, which is
checked for one model per stage and corpus.

Test, per metric:
  d_i = distance(GAPT_i) - distance(MLE_i) for each pair; negative = RL moved closer to GT.
  Corpus statistic z_C = mean(d) / rms(d); rms is unchanged by sign flips, so it is a fixed
  per-corpus scale. T = sum over corpora.
  Null: RL does not change the metric, so within a pair the GAPT/MLE labels are exchangeable,
  i.e. each d_i is equally likely to have either sign. 200,000 random sign flips (independent
  per pair, within corpus); two-sided p.
Holm across the metrics.

Usage:
    python scripts/eval/custom_eval/gapt_vs_mle_significance.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))
from chord_placement import _vocab_maps, keep_rows, profile  # noqa: E402
from fifths_motion import step_sequence  # noqa: E402
from loo_significance import holm  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.eval_utils import (duration_entropy, evaluate_chord_complexity,
                                         evaluate_chord_durations,
                                         evaluate_chord_to_note_onset_intervals,
                                         evaluate_note_in_chord_ratio, synchronization_emd)
from realchords.utils.sequence_penalty_analysis import strip_bos

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
MLE_DIR, GAPT_DIR = Path("logs/custom_eval/dataset_loo_new"), Path("logs/custom_eval/gapt_loo")
# name, kind ("abs" = |model - GT|, "direct" = already a distance), excluded corpora
METRICS = [
    ("root-motion EMD", "direct", set()),
    ("note-in-chord", "abs", set()),
    ("chord complexity", "abs", set()),
    ("chord changes per bar", "abs", set()),
    ("chord duration entropy", "abs", set()),
    ("sync EMD", "direct", set()),
]
MEANS_KEYS = {"note-in-chord": "note_in_chord_ratio_mean",
              "chord complexity": "chord_complexity_mean",
              "chord duration entropy": "chord_duration_entropy"}


def basic_metrics(seqs: torch.Tensor, tok) -> dict:
    """note-in-chord (pooled over frames), complexity, duration entropy and sync intervals,
    exactly as custom_evaluation computes them for means.json."""
    stripped = strip_bos(seqs, tok)
    _, valid, correct = evaluate_note_in_chord_ratio(stripped, tok, model_part="chord",
                                                     return_count=True)
    dur = evaluate_chord_durations(stripped, tok)["durations_flat"]
    comp = evaluate_chord_complexity(stripped, tok)["mean"]
    return {
        "note-in-chord": float(correct.sum()) / float(valid.sum()),
        "chord complexity": float(comp.float().nanmean()),
        "chord duration entropy": duration_entropy(dur) if dur.numel() else float("nan"),
        "_sync": evaluate_chord_to_note_onset_intervals(stripped, tok)["intervals_flat"],
    }


def corpus_values(corpus: str, check: bool) -> dict:
    """{"gt": {...}, "MLE": {label: {...}}, "GAPT": {label: {...}}} on the shared windows."""
    names = json.loads((GAPT_DIR / corpus / "chord_names_augmented.json").read_text())
    names_mle = json.loads((MLE_DIR / corpus / "chord_names_augmented.json").read_text())
    assert names == names_mle, f"{corpus}: chord vocabularies differ between stages"
    tok = HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names))
    i2n = tok.id_to_name
    events, pads = _vocab_maps(i2n)

    gt_g = torch.load(GAPT_DIR / corpus / "gt" / "gt.pt", map_location="cpu", weights_only=False)
    gt_m = torch.load(MLE_DIR / corpus / "gt" / "gt.pt", map_location="cpu", weights_only=False)
    n = min(len(gt_g), len(gt_m))
    assert torch.equal(gt_g[:n], gt_m[:n]), f"{corpus}: ground truth differs on shared windows"
    gt = gt_g[:n]
    rows, _ = keep_rows(GAPT_DIR / corpus / "metadata.jsonl", n)

    gt_path = GAPT_DIR / corpus / "gt" / "gt.pt"
    gt_steps = step_sequence(gt_path, i2n, n)
    out = {"gt": basic_metrics(gt, tok), "n_windows": n}
    out["gt"]["chord changes per bar"] = profile(gt_path, rows, events, pads, False)["rate"]

    labels = sorted(p.name for p in (GAPT_DIR / corpus / "models").iterdir())
    for stage, base in (("MLE", MLE_DIR), ("GAPT", GAPT_DIR)):
        means = json.loads((base / corpus / "means.json").read_text())["models"]
        out[stage] = {}
        for i, label in enumerate(labels):
            path = base / corpus / "models" / label / "preds.pt"
            full = torch.load(path, map_location="cpu", weights_only=False)
            seqs = full[:n]
            v = basic_metrics(seqs, tok)
            if check and i == 0 and len(full) == n:      # full windows: must equal means.json
                for name, key in MEANS_KEYS.items():
                    assert abs(v[name] - means[label][key]) < 1e-3, (corpus, stage, name,
                                                                     v[name], means[label][key])
            v["sync EMD"] = synchronization_emd(v.pop("_sync"), out["gt"]["_sync"])
            v["root-motion EMD"] = float(wasserstein_distance(gt_steps,
                                                              step_sequence(path, i2n, n)))
            v["chord changes per bar"] = profile(path, rows, events, pads, False)["rate"]
            out[stage][label] = v
    out["gt"].pop("_sync")
    return out


def paired_test(blocks, draws: int, rng) -> tuple:
    """blocks: one array of pair differences per corpus. Returns (T, two-sided p)."""
    T, null = 0.0, np.zeros(draws)
    for d in blocks:
        scale = np.sqrt(np.mean(d ** 2)) or 1.0
        T += d.mean() / scale
        signs = rng.choice([-1.0, 1.0], size=(draws, len(d)))
        null += (signs @ d) / len(d) / scale
    return float(T), float((np.abs(null) >= abs(T) - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/gapt_vs_mle_significance.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    values = {}
    for corpus in CORPORA:
        values[corpus] = corpus_values(corpus, check=True)
        print(f"  {corpus}: {values[corpus]['n_windows']} windows, "
              f"{len(values[corpus]['GAPT'])} pairs", flush=True)

    results = {}
    for name, kind, excluded in METRICS:
        blocks, per_corpus = [], {}
        for corpus in CORPORA:
            if corpus in excluded:
                continue
            v = values[corpus]
            gt = v["gt"].get(name)
            dist = lambda x: abs(x - gt) if kind == "abs" else x
            d = np.array([dist(v["GAPT"][l][name]) - dist(v["MLE"][l][name]) for l in v["GAPT"]])
            blocks.append(d)
            per_corpus[corpus] = {"mean_d": float(d.mean()), "gapt_closer_pairs": int((d < 0).sum()),
                                  "pairs": len(d)}
        T, p = paired_test(blocks, args.draws, rng)
        results[name] = {"T": T, "p": p, "per_corpus": per_corpus,
                         "corpora_gapt_closer": sum(c["mean_d"] < 0 for c in per_corpus.values()),
                         "n_corpora": len(per_corpus),
                         "pairs_gapt_closer": sum(c["gapt_closer_pairs"] for c in per_corpus.values()),
                         "n_pairs": sum(c["pairs"] for c in per_corpus.values())}
    adj = holm([results[m]["p"] for m in results])
    for m, a in zip(results, adj):
        results[m]["p_holm"] = a

    print(f"\nGAPT vs MLE, paired sign-flip test, Holm across {len(results)} metrics"
          f"  (d = GAPT - MLE distance to GT; negative = GAPT closer)\n")
    print(f"{'metric':<24}{'corpora':>9}{'pairs':>10}{'T':>8}{'p':>9}{'Holm':>8}  closer")
    for m, r in results.items():
        closer = "GAPT" if r["T"] < 0 else "MLE"
        print(f"{m:<24}{r['corpora_gapt_closer']:>5}/{r['n_corpora']:<3}"
              f"{r['pairs_gapt_closer']:>6}/{r['n_pairs']:<4}{r['T']:>7.2f}{r['p']:>9.4f}"
              f"{r['p_holm']:>8.4f}  {closer}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results, "values": {
        c: {k: v for k, v in val.items()} for c, val in values.items()}}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
