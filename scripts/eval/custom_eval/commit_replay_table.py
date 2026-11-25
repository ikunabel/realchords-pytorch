#!/usr/bin/env python3
"""All metrics of the commit-ahead replays in one table (journal/experiments/COMMIT_AHEAD.md).

Reads the replay folders merged by merge_commit_replays.py (256 windows per corpus) and the metric
outputs written for them, and collects, per corpus and condition, every metric per model seed:

  from window_values (as commit_ahead_significance.py, tail windows dropped):
      chord changes per bar, share of changes on 8th/16th offbeats
  computed here (as fifths_motion.py): root-motion EMD to GT on the line of fifths
  suite.json (commit_replay_suite.py): note-in-chord, chord duration entropy, sync EMD,
      chord complexity, chord silence, chord-distribution distance per song and key-relative

Writes mean, min and max over the 3 seeds per condition, the GT value where the metric has one, and,
for commit 0 -> 4 beats, in how many of the 6 seed x corpus series the metric moves in the direction
of the mean change. Descriptive only.

Usage:
    python scripts/eval/custom_eval/commit_replay_table.py
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
from chord_placement import FRAMES_PER_BAR, _vocab_maps, bar_bucket, change_frames, keep_rows  # noqa: E402
from commit_ahead_significance import OFFBEAT, window_values  # noqa: E402
from fifths_motion import step_sequence  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils import progression_metrics as pm

CORPORA = ["hooktheory", "pop909"]
SEEDS = ["s42", "s43", "s44"]
CONDITIONS = [  # (name, root key, k, stage, label in the metric outputs)
    ("MLE, commit 0", "all", 1, "mle", "k1"), ("MLE, 1 beat", "all", 4, "mle", "k4"),
    ("MLE, 2 beats", "all", 8, "mle", "k8"), ("MLE, 4 beats", "all", 16, "mle", "k16"),
    ("GAPT, commit 0", "all", 1, "gapt", "k1"), ("GAPT, 1 beat", "all", 4, "gapt", "k4"),
    ("GAPT, 2 beats", "all", 8, "gapt", "k8"), ("GAPT, 4 beats", "all", 16, "gapt", "k16"),
    ("GAPT + MLE melody, 2 beats", "fix", 8, "gapt", "fix8"),
    ("GAPT + MLE melody, 4 beats", "fix", 16, "gapt", "fix16"),
]
SUITE = {"note_in_chord": "note_in_chord_ratio_mean",
         "duration_entropy": "chord_duration_entropy", "sync_emd": "sync_emd_vs_gt",
         "complexity": "chord_complexity_mean", "chord_silence": "chord_silence_ratio_mean",
         "chord_dist_song": "chord_dist_wasserstein_per_song_onset_vs_gt",
         "chord_dist_key": "chord_dist_wasserstein_key_relative_onset_vs_gt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_root", type=Path, default=Path("logs/custom_eval/commit_replay_all"))
    ap.add_argument("--fix_root", type=Path, default=Path("logs/custom_eval/commit_replay_all_mleimag"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/commit_replay_all/table.json"))
    args = ap.parse_args()
    roots = {"all": args.all_root, "fix": args.fix_root}
    A = args.all_root
    suite = json.loads((A / "suite.json").read_text())

    out = {}
    for corpus in CORPORA:
        base = A / "k1" / corpus
        i2n = HooktheoryTokenizer(chord_names=json.loads((base / "chord_names_augmented.json").read_text())).id_to_name
        gt_path = base / "gt" / "gt.pt"
        gt = torch.load(gt_path, map_location="cpu", weights_only=False)
        n = gt.shape[0]
        gt_steps = step_sequence(gt_path, i2n, n)
        gt_suite = suite["mean_over_seeds"][f"{corpus}/gt"]
        events, pads = _vocab_maps(i2n)                          # GT rate/offbeat, as window_values
        gt_rate, gt_off = [], []
        for i in keep_rows(base / "metadata.jsonl", n)[0]:
            frames, valid = change_frames(pm.chord_lane(gt[i].tolist()), events, pads)
            gt_rate.append(len(frames) / (valid / FRAMES_PER_BAR) if valid else np.nan)
            gt_off.append(sum(bar_bucket(t % FRAMES_PER_BAR) in OFFBEAT for t in frames) / len(frames)
                          if frames else np.nan)
        rows = {}
        for name, rk, k, stage, lab in CONDITIONS:
            per_seed = []
            for s in SEEDS:
                model = f"{stage}_{s}"
                v, _ = window_values(roots[rk], k, corpus, model)
                pred = roots[rk] / f"k{k}" / corpus / "models" / model / "preds.pt"
                sk = f"{roots[rk].name}/k{k}/{corpus}/{model}"
                r = {"changes_per_bar": float(np.nanmean(v["rate"])),
                     "offbeat_share": float(np.nanmean(v["off"])),
                     "root_motion_emd": float(wasserstein_distance(gt_steps, step_sequence(pred, i2n, n))),
                     **{m: suite["per_seed"][sk][key] for m, key in SUITE.items()}}
                per_seed.append(r)
            rows[name] = {m: {"mean": float(np.mean([r[m] for r in per_seed if r[m] is not None]))
                              if all(r[m] is not None for r in per_seed) else None,
                              "min": min((r[m] for r in per_seed if r[m] is not None), default=None),
                              "max": max((r[m] for r in per_seed if r[m] is not None), default=None),
                              "per_seed": [r[m] for r in per_seed]} for m in per_seed[0]}
        out[corpus] = {"n_windows": int(n), "conditions": rows,
                       "gt": {"changes_per_bar": float(np.nanmean(gt_rate)),
                              "offbeat_share": float(np.nanmean(gt_off)),
                              **{m: gt_suite.get(key) for m, key in SUITE.items() if key in gt_suite}}}

    # direction counts, commit 0 -> 4 beats, per stage, over 6 seed x corpus series
    counts = {}
    for stage in ("MLE", "GAPT"):
        for m in out["hooktheory"]["conditions"][f"{stage}, commit 0"]:
            d = [b - a for c in CORPORA
                 for a, b in zip(out[c]["conditions"][f"{stage}, commit 0"][m]["per_seed"],
                                 out[c]["conditions"][f"{stage}, 4 beats"][m]["per_seed"])
                 if a is not None and b is not None]
            if d:
                up = sum(x > 0 for x in d)
                counts[f"{stage}/{m}"] = {"series": len(d), "up": up, "down": sum(x < 0 for x in d)}
    out["commit0_to_4beats_direction"] = counts
    args.out.write_text(json.dumps(out, indent=2))

    for corpus in CORPORA:
        print(f"\n{corpus} ({out[corpus]['n_windows']} windows)")
        metrics = list(out[corpus]["conditions"]["MLE, commit 0"])
        print(f"{'condition':<28}" + "".join(f"{m[:11]:>12}" for m in metrics))
        print(f"{'GT':<28}" + "".join(f"{out[corpus]['gt'][m]:>12.3f}" if out[corpus]['gt'].get(m) is not None
                                      else f"{'':>12}" for m in metrics))
        for name, r in out[corpus]["conditions"].items():
            print(f"{name:<28}" + "".join(f"{r[m]['mean']:>12.3f}" if r[m]["mean"] is not None else f"{'-':>12}"
                                          for m in metrics))
    print("\ncommit 0 -> 4 beats, series up / down:")
    for k, v in counts.items():
        print(f"  {k:<32} up {v['up']}/{v['series']}  down {v['down']}/{v['series']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
