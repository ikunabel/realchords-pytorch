#!/usr/bin/env python3
"""95% confidence intervals for the commit-ahead effects (journal/experiments/COMMIT_AHEAD.md).

No significance tests. For every reported metric and each comparison, the effect is the difference of
the metric between two conditions, averaged over the 3 model seeds; its 95% interval comes from a
bootstrap over test windows: windows are resampled with replacement within each corpus, and the same
resampled windows are used for every condition, seed and the ground truth, so the comparison stays
paired. The models are fixed, so the interval describes how precisely these trained models' effect is
measured on this test material, not variation over retraining.

Comparisons: MLE 4 beats - commit 0; GAPT 4 beats - commit 0; GAPT's change - MLE's change; GAPT with
MLE-imagined melody - GAPT self-imagined, at 2 and at 4 beats. Reported per corpus and as the mean of the
two corpora (Hooktheory, POP909).

Per window, from the saved replay outputs (`commit_replay_suite.py` parts, `commit_replay.py` preds):
changes per bar and offbeat share (as `commit_ahead_significance.window_values`, last window of each song
excluded), note-in-chord (frames pooled), note-in-mode (chord segments pooled, as the evaluation), chord duration entropy and sync EMD (pooled
over the resampled windows, as in the evaluation), chord complexity, chord distance to the song, tonal
agreement, root-motion EMD, and the n-gram log-probability shift against ground truth (orders 2 and 7).

Usage:
    python scripts/eval/custom_eval/commit_ahead_ci.py
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
from chord_placement import keep_rows  # noqa: E402
from commit_ahead_significance import window_values  # noqa: E402
from fifths_motion import fifths_steps  # noqa: E402
from score_progression_ngram import train_events  # noqa: E402
from tonal_agreement import profiles, window_agreement  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm
from realchords.utils import progression_ngram as png
from realchords.utils.eval_utils import (_row_pitch_class_set_weights, duration_entropy,
                                         evaluate_melody_mode_fit_ratio, pitch_class_set_wasserstein,
                                         synchronization_emd)
from realchords.utils.sequence_penalty_analysis import strip_bos

CORPORA = ["hooktheory", "pop909"]
SEEDS = ["s42", "s43", "s44"]
CONDS = {"mle1": ("all", 1, "mle"), "mle16": ("all", 16, "mle"), "gapt1": ("all", 1, "gapt"),
         "gapt8": ("all", 8, "gapt"), "gapt16": ("all", 16, "gapt"),
         "fix8": ("fix", 8, "gapt"), "fix16": ("fix", 16, "gapt")}
COMPARISONS = {"MLE: 4 beats - commit 0": [("mle16", 1), ("mle1", -1)],
               "GAPT: 4 beats - commit 0": [("gapt16", 1), ("gapt1", -1)],
               "GAPT change - MLE change": [("gapt16", 1), ("gapt1", -1), ("mle16", -1), ("mle1", 1)],
               "fix - self, 2 beats": [("fix8", 1), ("gapt8", -1)],
               "fix - self, 4 beats": [("fix16", 1), ("gapt16", -1)]}
METRICS = ["changes_per_bar", "offbeat_share", "note_in_chord", "note_in_mode", "duration_entropy",
           "sync_emd", "complexity", "chord_distance_to_song", "tonal_agreement", "root_motion_emd",
           "ngram_shift_order2", "ngram_shift_order7"]


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def per_window(root: Path, parts: Path, k: int, corpus: str, label: str, tok, ngrams, gt_chords) -> dict:
    """Per-window ingredients of every metric for one model (or GT if label == 'gt')."""
    base = root / f"k{k}" / corpus
    part = parts / root.name / f"k{k}" / corpus / label
    preds = _load(base / ("gt/gt.pt" if label == "gt" else f"models/{label}/preds.pt"))
    n = preds.shape[0]
    i2n = tok.id_to_name
    out = {}
    if label != "gt":
        v, _ = window_values(root, k, corpus, label)
        rows, _ = keep_rows(base / "metadata.jsonl", n)
        rate, off = np.full(n, np.nan), np.full(n, np.nan)
        rate[rows], off[rows] = v["rate"], v["off"]
        out["rate"], out["off"] = rate, off
    d = _load(part / "nicr_per_frame.pt")
    out["nicr_m"] = np.nansum(d["matches"].float().numpy(), axis=1)
    out["nicr_v"] = d["valid"].float().sum(1).numpy()
    # note-in-mode as the evaluation pools it: per-chord-segment fit, weighted by segment count
    ratio, _, counts = evaluate_melody_mode_fit_ratio(strip_bos(preds, tok), tok, model_part="chord", return_count=True)
    ok = torch.isfinite(ratio) & (counts > 0)
    out["mode_m"] = torch.where(ok, ratio * counts.float(), torch.zeros_like(ratio)).numpy()
    out["mode_v"] = torch.where(ok, counts.float(), torch.zeros_like(ratio)).numpy()
    out["durations"] = [np.asarray(x, float) for x in _load(part / "chord_durations.pt")["durations"]]
    out["intervals"] = [np.asarray(x, float) for x in _load(part / "sync_intervals.pt")["intervals"]]
    out["complexity"] = _load(part / "chord_complexity.pt")["mean"].float().numpy()
    chords = _load(part / "chords_per_frame.pt")
    if gt_chords is not None:
        dist = np.full(n, np.nan)
        for r in range(n):
            wm, wg = _row_pitch_class_set_weights(chords, r, "onset"), _row_pitch_class_set_weights(gt_chords, r, "onset")
            if wm and wg:
                dist[r] = pitch_class_set_wasserstein(wm, wg)
        out["dist"] = dist
    rows_list = preds.tolist()
    out["tonal"] = np.array([np.nan if (a := window_agreement(p)) is None else a
                             for p in profiles(rows_list, i2n, local=False)], float)
    steps, lp = [], {o: np.full(n, np.nan) for o in ngrams}
    for r, row in enumerate(rows_list):
        ev_full = pm.chord_events(pm.chord_lane(row), i2n, quality="full")
        steps.append(np.array([fifths_steps((b[0] - a[0]) % 12) for a, b in zip(ev_full, ev_full[1:])], float))
        ev = pm.chord_events(pm.chord_lane(row), i2n)
        for o, model in ngrams.items():
            total, moves = model.logprob(ev)
            if moves >= 3:
                lp[o][r] = total / moves
    out["steps"] = steps
    for o in ngrams:
        out[f"lp{o}"] = lp[o]
    return out, chords


def metrics(w: dict, g: dict, idx: np.ndarray) -> dict:
    cat = lambda xs: np.concatenate([xs[i] for i in idx]) if len(idx) else np.array([])
    return {
        "changes_per_bar": float(np.nanmean(w["rate"][idx])),
        "offbeat_share": float(np.nanmean(w["off"][idx])),
        "note_in_chord": float(w["nicr_m"][idx].sum() / w["nicr_v"][idx].sum()),
        "note_in_mode": float(w["mode_m"][idx].sum() / w["mode_v"][idx].sum()),
        "duration_entropy": float(duration_entropy(torch.tensor(cat(w["durations"])))),
        "sync_emd": float(synchronization_emd(torch.tensor(cat(w["intervals"])), torch.tensor(cat(g["intervals"])))),
        "complexity": float(np.nanmean(w["complexity"][idx])),
        "chord_distance_to_song": float(np.nanmean(w["dist"][idx])),
        "tonal_agreement": float(np.nanmean(w["tonal"][idx])),
        "root_motion_emd": float(wasserstein_distance(cat(w["steps"]), cat(g["steps"]))),
        "ngram_shift_order2": float(np.nanmean(w["lp2"][idx]) - np.nanmean(g["lp2"][idx])),
        "ngram_shift_order7": float(np.nanmean(w["lp7"][idx]) - np.nanmean(g["lp7"][idx])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_root", type=Path, default=Path("logs/custom_eval/commit_replay_all"))
    ap.add_argument("--fix_root", type=Path, default=Path("logs/custom_eval/commit_replay_all_mleimag"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/commit_replay_all/ci.json"))
    args = ap.parse_args()
    roots = {"all": args.all_root, "fix": args.fix_root}
    parts = args.all_root / "suite.parts"
    rng = np.random.default_rng(args.seed)

    data, n_win = {}, {}
    for c in CORPORA:
        tok = HooktheoryTokenizer(chord_names=json.loads((args.all_root / "k1" / c / "chord_names_augmented.json").read_text()))
        train = train_events(c)
        ngrams = {o: png.ProgressionNGram(order=o, alpha=1.0).fit(train) for o in (2, 7)}
        g, gt_chords = per_window(args.all_root, parts, 1, c, "gt", tok, ngrams, None)
        data[(c, "gt")] = g
        for cond, (rk, k, stage) in CONDS.items():
            for s in SEEDS:
                data[(c, cond, s)], _ = per_window(roots[rk], parts, k, c, f"{stage}_{s}", tok, ngrams, gt_chords)
        n_win[c] = len(g["complexity"])
        print(f"loaded {c}: {n_win[c]} windows", flush=True)

    def effects(idx_by_corpus):
        per_c = {}
        for c in CORPORA:
            idx, g = idx_by_corpus[c], data[(c, "gt")]
            val = {cond: {m: np.mean([metrics(data[(c, cond, s)], g, idx)[m] for s in SEEDS]) for m in METRICS}
                   for cond in CONDS}
            per_c[c] = {comp: {m: float(sum(sign * val[cond][m] for cond, sign in terms)) for m in METRICS}
                        for comp, terms in COMPARISONS.items()}
        both = {comp: {m: float(np.mean([per_c[c][comp][m] for c in CORPORA])) for m in METRICS} for comp in COMPARISONS}
        return {**per_c, "both": both}

    est = effects({c: np.arange(n_win[c]) for c in CORPORA})
    draws = [effects({c: rng.integers(0, n_win[c], n_win[c]) for c in CORPORA}) for _ in range(args.boot)]
    res = {"n_windows": n_win, "boot": args.boot, "effects": {}}
    for scope in CORPORA + ["both"]:
        res["effects"][scope] = {comp: {m: {"estimate": est[scope][comp][m],
                                            "ci95": [float(np.percentile([d[scope][comp][m] for d in draws], 2.5)),
                                                     float(np.percentile([d[scope][comp][m] for d in draws], 97.5))]}
                                        for m in METRICS} for comp in COMPARISONS}
    args.out.write_text(json.dumps(res, indent=2))
    for comp in COMPARISONS:
        print(f"\n{comp} (mean of both corpora)")
        for m in METRICS:
            e = res["effects"]["both"][comp][m]
            print(f"  {m:<24}{e['estimate']:+9.3f}  [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
