#!/usr/bin/env python3
"""Descriptive metrics of commit-ahead replays (journal/experiments/COMMIT_AHEAD.md).

Per corpus and condition, the mean over seeds of: chord changes per bar and offbeat share (as in
commit_ahead_significance.py, tail windows dropped), note-in-chord (pooled over frames), chord duration
entropy and sync EMD against the windows' ground truth (as in custom_evaluation.py), and root-motion
EMD (as in fifths_motion.py). No tests; the planned tests are in commit_ahead_significance.py.

Usage:
    python scripts/eval/custom_eval/commit_replay_metrics.py      # exploratory windows (rows 0-127)
    python scripts/eval/custom_eval/commit_replay_metrics.py \\
        --mle_root logs/custom_eval/commit_replay_confirm_mle \\
        --gapt_root logs/custom_eval/commit_replay_confirm_gapt \\
        --mleimag_root logs/custom_eval/commit_replay_confirm_mleimag \\
        --out logs/custom_eval/commit_replay_metrics_confirm.json
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
from commit_ahead_significance import window_values  # noqa: E402
from fifths_motion import step_sequence  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.eval_utils import (duration_entropy, evaluate_chord_durations,
                                         evaluate_chord_to_note_onset_intervals,
                                         evaluate_note_in_chord_ratio, synchronization_emd)
from realchords.utils.sequence_penalty_analysis import strip_bos

CORPORA = ["hooktheory", "pop909"]
SEEDS = ["s42", "s43", "s44"]


def frame_metrics(path: Path, gt: torch.Tensor, tok, i2n) -> dict:
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    st, sg = strip_bos(seqs, tok), strip_bos(gt, tok)
    _, valid, correct = evaluate_note_in_chord_ratio(st, tok, model_part="chord", return_count=True)
    dur = evaluate_chord_durations(st, tok)["durations_flat"]
    sync = synchronization_emd(evaluate_chord_to_note_onset_intervals(st, tok)["intervals_flat"],
                               evaluate_chord_to_note_onset_intervals(sg, tok)["intervals_flat"])
    return {"nicr": float(correct.sum()) / float(valid.sum()),
            "dur_entropy": duration_entropy(dur) if dur.numel() else float("nan"),
            "sync_emd": float(sync)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mle_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--gapt_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--mleimag_root", type=Path, default=Path("logs/custom_eval/commit_replay_mleimag"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/commit_replay_metrics.json"))
    args = ap.parse_args()
    conditions = [("MLE, commit 0", args.mle_root, 1, "mle"), ("MLE, 4 beats", args.mle_root, 16, "mle"),
                  ("GAPT, commit 0", args.gapt_root, 1, "gapt"), ("GAPT, 2 beats", args.gapt_root, 8, "gapt"),
                  ("GAPT, 4 beats", args.gapt_root, 16, "gapt"),
                  ("GAPT + MLE melody, 2 beats", args.mleimag_root, 8, "gapt"),
                  ("GAPT + MLE melody, 4 beats", args.mleimag_root, 16, "gapt")]
    out = {}
    for corpus in CORPORA:
        base = args.gapt_root / "k1" / corpus
        tok = HooktheoryTokenizer(chord_names=json.load(open(base / "chord_names_augmented.json")))
        i2n = tok.id_to_name
        gt = torch.load(base / "gt" / "gt.pt", map_location="cpu", weights_only=False)
        gt_steps = step_sequence(base / "gt" / "gt.pt", i2n, len(gt))
        out[corpus] = {}
        for name, root, k, stage in conditions:
            rows = []
            for s in SEEDS:
                path = root / f"k{k}" / corpus / "models" / f"{stage}_{s}" / "preds.pt"
                g = torch.load(root / f"k{k}" / corpus / "gt" / "gt.pt", map_location="cpu", weights_only=False)
                assert torch.equal(g, gt), f"{name}: windows differ"
                v, _ = window_values(root, k, corpus, f"{stage}_{s}")
                m = frame_metrics(path, gt, tok, i2n)
                m["rate"] = float(np.nanmean(v["rate"]))
                m["offbeat"] = float(np.nanmean(v["off"]))
                m["root_emd"] = float(wasserstein_distance(gt_steps, step_sequence(path, i2n, len(gt))))
                rows.append(m)
            out[corpus][name] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        print(f"\n{corpus} ({len(gt)} windows)")
        print(f"{'condition':<30}{'changes/bar':>12}{'offbeat':>9}{'NiCR':>8}{'dur. H':>8}{'sync EMD':>10}{'root EMD':>10}")
        for name, m in out[corpus].items():
            print(f"{name:<30}{m['rate']:>12.2f}{m['offbeat']*100:>8.1f}%{m['nicr']:>8.3f}"
                  f"{m['dur_entropy']:>8.3f}{m['sync_emd']:>10.3f}{m['root_emd']:>10.3f}")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
