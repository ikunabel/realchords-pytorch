#!/usr/bin/env python3
"""GAPT 7sets_no_hooktheory trained with its own reward models vs with the shared 7-set ones
(journal/experiments/SHARED_REWARD_MODELS.md).

No significance tests: point estimates with 95% bootstrap intervals over test windows. Windows are
resampled with replacement within each corpus, and the same resampled windows are used for every model,
seed and the ground truth, so every comparison is paired. The models are fixed, so an interval describes
how precisely these trained models' difference is measured on this test material, not retraining noise
(the per-seed values are saved too).

Models (3 seeds each):
  shared  gapt_loo_7sets_no_hooktheory              (rewards: 7-set reward models)
  own     gapt_loo_ownrewards_7sets_no_hooktheory   (rewards: 7sets_no_hooktheory reward models)
  7sets   gapt_loo_7sets                            (the full-mixture reference of the leave-one-out test)
`shared` and `own` come from one evaluation run (`configs/custom_eval/gapt_loo_ownrewards/`); `7sets`
from the leave-one-out run (`configs/custom_eval/gapt_loo/`), same windows (checked: identical GT).
`shared` was also scored in that run; its values there are reported as `shared_loo_run`, which shows how
much the evaluation's own sampling moves a model (the random stream depends on the other models in a run).

The six leave-one-out metrics, each as a distance from ground truth, as in `loo_significance.py`:
|model - GT| for note-in-chord (frames pooled), chord complexity (mean over windows) and chord duration
entropy (durations pooled); sync EMD, root-motion EMD (all windows) and chord-placement TV (last window
of each song dropped; Nottingham excluded) are distances already. Every pooled quantity is computed from
per-window histograms, so a resample sums histograms.

Reported per corpus and metric: the mean over seeds of each group's distance, and with intervals
  own - shared        did giving the policy its own reward models change it?
  damage shared       shared - 7sets  (the leave-one-out damage as measured so far)
  damage own          own - 7sets     (the damage with the reward models also held out)

Usage:
    python scripts/eval/custom_eval/gapt_ownrewards_ci.py
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
from chord_placement import (BAR_EXCLUDED, FRAMES_PER_BAR, _vocab_maps, bar_bucket,  # noqa: E402
                             change_frames, keep_rows)
from fifths_motion import fifths_steps  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm
from realchords.utils.eval_utils import duration_entropy

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham", "wikifonia", "chord_melody_dataset", "jazzmus"]
SEEDS = ["s42", "s43", "s44"]
GROUPS = {"shared": ("new", "7sets_no_hooktheory_{}"),
          "own": ("new", "ownrewards_7sets_no_hooktheory_{}"),
          "7sets": ("loo", "7sets_{}"),
          "shared_loo_run": ("loo", "7sets_no_hooktheory_{}")}
COMPARISONS = {"own - shared": ("own", "shared"), "damage shared": ("shared", "7sets"),
               "damage own": ("own", "7sets")}
METRICS = ["note-in-chord", "chord complexity", "chord duration entropy", "sync EMD", "root-motion EMD",
           "chord placement TV"]
MAX_INTERVAL = 18          # synchronization_emd's overflow bin
MAX_DURATION = 600


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def per_window(d: Path, preds_path: Path, i2n, events, pads, kept: np.ndarray, bar_ok: bool) -> dict:
    """Per-window ingredients of the six metrics for one model or the GT."""
    nicr = _load(d / "nicr_per_frame.pt")
    out = {"nicr_m": np.nansum(nicr["matches"].float().numpy(), axis=1),
           "nicr_v": nicr["valid"].float().sum(1).numpy(),
           "complexity": _load(d / "chord_complexity.pt")["mean"].float().numpy()}
    durs = _load(d / "chord_durations.pt")["durations"]
    ints = _load(d / "sync_intervals.pt")["intervals"]
    n = len(out["nicr_m"])
    out["dur_h"] = np.zeros((n, MAX_DURATION + 1))
    out["int_h"] = np.zeros((n, MAX_INTERVAL + 1))
    for r in range(n):
        x = np.asarray(durs[r], int)
        if len(x):
            assert x.max() <= MAX_DURATION
            out["dur_h"][r] = np.bincount(x, minlength=MAX_DURATION + 1)
        y = np.clip(np.asarray(ints[r], int), 0, MAX_INTERVAL)
        if len(y):
            out["int_h"][r] = np.bincount(y, minlength=MAX_INTERVAL + 1)
    seqs = _load(preds_path).tolist()
    out["step_h"] = np.zeros((n, 7))
    out["bar_h"] = np.zeros((n, 5))
    for r, row in enumerate(seqs[:n]):
        ev = pm.chord_events(pm.chord_lane(row), i2n, quality="full")
        for a, b in zip(ev, ev[1:]):
            out["step_h"][r, fifths_steps((b[0] - a[0]) % 12)] += 1
        if bar_ok and kept[r]:
            frames, _ = change_frames(pm.chord_lane(row), events, pads)
            for t in frames:
                out["bar_h"][r, bar_bucket(t % FRAMES_PER_BAR)] += 1
    return out


def _entropy(h):
    vals = np.repeat(np.arange(len(h)), h.astype(int))
    return float(duration_entropy(torch.tensor(vals)))


def _emd(h, g):
    if h.sum() == 0 or g.sum() == 0:
        return float("nan")
    x = np.arange(len(h))
    return float(wasserstein_distance(x, x, h, g))


def raw(w: dict, idx: np.ndarray) -> dict:
    """Metric values of one source on the windows idx (with repetition)."""
    return {"note-in-chord": float(w["nicr_m"][idx].sum() / w["nicr_v"][idx].sum()),
            "chord complexity": float(np.nanmean(w["complexity"][idx])),
            "chord duration entropy": _entropy(w["dur_h"][idx].sum(0)),
            "_int": w["int_h"][idx].sum(0), "_step": w["step_h"][idx].sum(0), "_bar": w["bar_h"][idx].sum(0)}


def distances(m: dict, g: dict, bar_ok: bool) -> dict:
    bar = m["_bar"] / m["_bar"].sum() if m["_bar"].sum() else None
    gbar = g["_bar"] / g["_bar"].sum() if g["_bar"].sum() else None
    return {"note-in-chord": float(abs(m["note-in-chord"] - g["note-in-chord"])),
            "chord complexity": abs(m["chord complexity"] - g["chord complexity"]),
            "chord duration entropy": abs(m["chord duration entropy"] - g["chord duration entropy"]),
            "sync EMD": _emd(m["_int"], g["_int"]),
            "root-motion EMD": _emd(m["_step"], g["_step"]),
            "chord placement TV": (0.5 * float(np.abs(bar - gbar).sum()) if bar_ok and bar is not None
                                   else float("nan"))}


def group_distances(win: dict, gt: dict, idx: np.ndarray, bar_ok: bool) -> dict:
    """{group: {metric: [per-seed distance]}} on the windows idx."""
    g = raw(gt, idx)
    return {grp: {k: [distances(raw(win[grp][s], idx), g, bar_ok)[k] for s in SEEDS] for k in METRICS}
            for grp in win}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_root", type=Path, default=Path("logs/custom_eval/gapt_loo_ownrewards"))
    ap.add_argument("--loo_root", type=Path, default=Path("logs/custom_eval/gapt_loo"))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/gapt_ownrewards_ci.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    roots = {"new": args.new_root, "loo": args.loo_root}
    comparisons = {k: v for k, v in COMPARISONS.items() if set(v) <= set(args.groups)}
    res = {"boot": args.boot, "corpora": {}}
    for c in CORPORA:
        bar_ok = c not in BAR_EXCLUDED
        used = {GROUPS[g][0] for g in args.groups}
        gts = {r: _load(roots[r] / c / "gt" / "gt.pt") for r in used}
        ref = next(iter(gts.values()))
        assert all(torch.equal(t, ref) for t in gts.values()), f"{c}: runs scored different windows"
        base = roots[next(iter(used))] / c
        names = json.loads((base / "chord_names_augmented.json").read_text())
        tok = HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names))
        events, pads = _vocab_maps(tok.id_to_name)
        n = ref.shape[0]
        rows, _ = keep_rows(base / "metadata.jsonl", n)
        kept = np.zeros(n, bool)
        kept[rows] = True
        gt = per_window(base / "gt", base / "gt" / "gt.pt", tok.id_to_name, events, pads, kept, bar_ok)
        win = {}
        for grp in args.groups:
            root, pat = GROUPS[grp]
            win[grp] = {s: per_window(roots[root] / c / "models" / pat.format(s),
                                      roots[root] / c / "models" / pat.format(s) / "preds.pt",
                                      tok.id_to_name, events, pads, kept, bar_ok) for s in SEEDS}
        full = group_distances(win, gt, np.arange(n), bar_ok)
        draws = []
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            draws.append(group_distances(win, gt, idx, bar_ok))
        out = {"windows": n, "per_seed": full, "mean": {}, "comparisons": {}}
        for grp in args.groups:
            out["mean"][grp] = {k: float(np.mean(full[grp][k])) for k in METRICS}
        for name, (a, b) in comparisons.items():
            out["comparisons"][name] = {}
            for k in METRICS:
                est = float(np.mean(full[a][k]) - np.mean(full[b][k]))
                bs = np.array([np.mean(d[a][k]) - np.mean(d[b][k]) for d in draws])
                lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(est) else (np.nan, np.nan))
                out["comparisons"][name][k] = {"estimate": est, "ci95": [float(lo), float(hi)]}
        res["corpora"][c] = out
        print(f"\n{c} ({n} windows)")
        print(f"  {'metric':<24}" + "".join(f"{g:>16}" for g in args.groups)
              + "".join(f"{nm:>28}" for nm in comparisons))
        for k in METRICS:
            cells = "".join(f"{out['mean'][g][k]:>16.4f}" for g in args.groups)
            comp = "".join(f"{v[k]['estimate']:>+10.4f} [{v[k]['ci95'][0]:+.4f},{v[k]['ci95'][1]:+.4f}]"
                           for v in out["comparisons"].values())
            print(f"  {k:<24}{cells}{comp}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
