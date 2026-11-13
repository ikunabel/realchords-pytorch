#!/usr/bin/env python3
"""Significance tests for the commit-ahead study (journal/experiments/COMMIT_AHEAD.md).

Commit-ahead is an inference setting applied to a fixed trained model, so the replicates are test
windows: each window is replayed under both conditions by the same model, and the test is paired
by window. Strata = model seed x corpus (3 x 2). Training-run generality is reported separately as
the number of strata whose mean difference has the expected sign.

Per window: d = value(condition B) - value(condition A). Per stratum z = mean(d) / rms(d) (rms
is unchanged by sign flips); T = sum over strata. Null: the condition does not change the value,
so each d is equally likely to have either sign. Random sign flips within strata; two-sided p.

Tests, fixed before running, one family, Holm:
  1  GAPT chord changes per bar: 4 beats (k=16) vs commit 0 (k=1)
  2  that fall, GAPT minus MLE (same window, same seed): d = dGAPT - dMLE
  3  fix: GAPT chord changes per bar at 4 beats, MLE-imagined vs self-imagined melody
  4  fix: share of chord changes on 8th/16th offbeats, same comparison (windows with at least
     one change under both conditions)

"No difference" claims are reported as estimates with a 95% CI (stratified bootstrap over
windows), not as tests: MLE's change-rate difference 4 beats vs commit 0, and the note-in-chord
drop GAPT minus MLE.

Tail windows (not phase-aligned to the bar) are dropped as in chord_placement.py.

Usage:
    # exploratory: the windows the study was built on (rows 0-127)
    python scripts/eval/custom_eval/commit_ahead_significance.py
    # confirmatory: fresh windows (rows 128-255)
    python scripts/eval/custom_eval/commit_ahead_significance.py \\
        --mle_root logs/custom_eval/commit_replay_confirm_mle \\
        --gapt_root logs/custom_eval/commit_replay_confirm_gapt \\
        --mleimag_root logs/custom_eval/commit_replay_confirm_mleimag \\
        --out logs/custom_eval/commit_ahead_significance_confirm.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from chord_placement import FRAMES_PER_BAR, _vocab_maps, bar_bucket, change_frames, keep_rows  # noqa: E402
from loo_significance import holm  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

CORPORA = ["hooktheory", "pop909"]
SEEDS = ["s42", "s43", "s44"]
OFFBEAT = (3, 4)          # bar_bucket indices: 8th and 16th offbeats


def window_values(root: Path, k: int, corpus: str, model: str):
    """Per kept window: chord changes per bar, offbeat share (None without changes), note-in-chord."""
    base = root / f"k{k}" / corpus
    names = json.loads((base / "chord_names_augmented.json").read_text())
    tok = HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names))
    events, pads = _vocab_maps(tok.id_to_name)
    preds = torch.load(base / "models" / model / "preds.pt", map_location="cpu", weights_only=False)
    nicr = torch.load(base / "models" / model / "nicr_per_frame.pt", map_location="cpu",
                      weights_only=False)["mean"]
    rows, _ = keep_rows(base / "metadata.jsonl", len(preds))
    gt = torch.load(base / "gt" / "gt.pt", map_location="cpu", weights_only=False)
    out = {"rate": [], "off": [], "nicr": []}
    for i in rows:
        frames, valid = change_frames(pm.chord_lane(preds[i].tolist()), events, pads)
        out["rate"].append(len(frames) / (valid / FRAMES_PER_BAR) if valid else float("nan"))
        off = sum(1 for t in frames if bar_bucket(t % FRAMES_PER_BAR) in OFFBEAT)
        out["off"].append(off / len(frames) if frames else float("nan"))
        out["nicr"].append(float(nicr[i]))
    return {key: np.array(v, dtype=float) for key, v in out.items()}, gt[rows]


def sign_flip(blocks, draws: int, rng, chunk: int = 20_000) -> tuple:
    """blocks: per stratum, the paired differences. Returns (T, two-sided p)."""
    blocks = [b[~np.isnan(b)] for b in blocks]
    scales = [np.sqrt(np.mean(b ** 2)) or 1.0 for b in blocks]
    T = float(sum(b.mean() / s for b, s in zip(blocks, scales)))
    hits = 0
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        null = np.zeros(n)
        for b, s in zip(blocks, scales):
            signs = rng.integers(0, 2, size=(n, len(b)), dtype=np.int8) * 2 - 1
            null += (signs @ b) / len(b) / s
        hits += int((np.abs(null) >= abs(T) - 1e-12).sum())
    return T, hits / draws


def bootstrap_ci(blocks, reps: int, rng) -> tuple:
    """Mean of the per-stratum mean differences, with a 95% stratified-bootstrap CI."""
    blocks = [b[~np.isnan(b)] for b in blocks]
    est = float(np.mean([b.mean() for b in blocks]))
    boot = np.mean([[rng.choice(b, size=len(b)).mean() for b in blocks] for _ in range(reps)], axis=1)
    return est, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mle_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--gapt_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--mleimag_root", type=Path, default=Path("logs/custom_eval/commit_replay_mleimag"))
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/commit_ahead_significance.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    V, gts = {}, {}
    for corpus in CORPORA:
        for s in SEEDS:
            for key, root, k, model in (("mle1", args.mle_root, 1, f"mle_{s}"),
                                        ("mle16", args.mle_root, 16, f"mle_{s}"),
                                        ("gapt1", args.gapt_root, 1, f"gapt_{s}"),
                                        ("gapt16", args.gapt_root, 16, f"gapt_{s}"),
                                        ("fix16", args.mleimag_root, 16, f"gapt_{s}")):
                V[(corpus, s, key)], gt = window_values(root, k, corpus, model)
                gts.setdefault(corpus, gt)
                assert torch.equal(gts[corpus], gt), f"{corpus}: windows differ between runs"
    strata = [(c, s) for c in CORPORA for s in SEEDS]
    v = lambda c, s, key, m: V[(c, s, key)][m]

    diffs = {
        "1 GAPT change rate, 4 beats - commit 0":
            [v(c, s, "gapt16", "rate") - v(c, s, "gapt1", "rate") for c, s in strata],
        "2 change-rate fall, GAPT - MLE":
            [(v(c, s, "gapt16", "rate") - v(c, s, "gapt1", "rate"))
             - (v(c, s, "mle16", "rate") - v(c, s, "mle1", "rate")) for c, s in strata],
        "3 fix: change rate, MLE-imagined - self":
            [v(c, s, "fix16", "rate") - v(c, s, "gapt16", "rate") for c, s in strata],
        "4 fix: offbeat share, MLE-imagined - self":
            [v(c, s, "fix16", "off") - v(c, s, "gapt16", "off") for c, s in strata],
    }
    results = {"tests": {}, "estimates": {}, "n_windows": {c: int(len(gts[c])) for c in CORPORA}}
    for name, blocks in diffs.items():
        T, p = sign_flip(blocks, args.draws, rng)
        means = [float(np.nanmean(b)) for b in blocks]
        same = sum(np.sign(m) == np.sign(T) for m in means)
        results["tests"][name] = {
            "T": T, "p": p, "mean_diff": float(np.mean(means)),
            "strata_same_sign": int(same), "n_strata": len(means),
            "p_strata_sign": min(1.0, 2 * sum(math.comb(6, i) for i in range(same, 7)) / 64)
            if same >= 3 else 1.0,
            "per_stratum_mean": {f"{c}/{s}": m for (c, s), m in zip(strata, means)},
        }
    for name, a in zip(results["tests"], holm([t["p"] for t in results["tests"].values()])):
        results["tests"][name]["p_holm"] = a

    estimates = {
        "MLE change rate, 4 beats - commit 0":
            [v(c, s, "mle16", "rate") - v(c, s, "mle1", "rate") for c, s in strata],
        "note-in-chord drop, GAPT - MLE":
            [(v(c, s, "gapt16", "nicr") - v(c, s, "gapt1", "nicr"))
             - (v(c, s, "mle16", "nicr") - v(c, s, "mle1", "nicr")) for c, s in strata],
    }
    for name, blocks in estimates.items():
        est, lo, hi = bootstrap_ci(blocks, args.boot, rng)
        results["estimates"][name] = {"mean": est, "ci95": [lo, hi]}

    print(f"windows per corpus (tail windows dropped): {results['n_windows']}\n")
    print(f"{'test':<46}{'mean d':>9}{'strata':>8}{'p':>10}{'Holm':>9}")
    for name, t in results["tests"].items():
        print(f"{name:<46}{t['mean_diff']:>+9.3f}{t['strata_same_sign']:>5}/{t['n_strata']}"
              f"{t['p']:>10.5f}{t['p_holm']:>9.5f}")
    print(f"\n{'estimate (no-difference claims)':<46}{'mean':>9}   95% CI")
    for name, e in results["estimates"].items():
        print(f"{name:<46}{e['mean']:>+9.3f}   [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
