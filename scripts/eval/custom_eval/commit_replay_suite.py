#!/usr/bin/env python3
"""The evaluation's means.json metrics for commit-ahead replays (journal/experiments/COMMIT_AHEAD.md).

No generation: scores the preds.pt that commit_replay.py saved, with the same functions
custom_evaluation.py uses, so every value is defined exactly as in the evaluation's means.json
(windows, no tail-window drop). Melody-dependent metrics are computed against the REAL melody
of each window -- the melody the user played -- not the melody the model imagined.

Per <root>/k<k>/<corpus>/models/<label>: note-in-chord (pooled, per song), note-in-mode (pooled,
per song), chord duration entropy, chord silence ratio, chord complexity, sync EMD vs GT, chord
distribution distances vs GT (per song; key-relative on Hooktheory). Melody-only metrics (note
duration entropy, melody silence ratio) are left out: the melody is the same real melody in every
condition. Vendi needs a contrastive checkpoint and is null in the LOO evaluations too.

Per-sequence side files the evaluation writes go to <out>.parts/ instead of the replay folders.

Usage:
    python scripts/eval/custom_eval/commit_replay_suite.py --roots logs/custom_eval/commit_replay \\
        logs/custom_eval/commit_replay_mleimag --out logs/custom_eval/commit_replay/suite.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.custom_evaluation import (_compare_to_gt, _hooktheory_tonics, _nanmean,
                                                _paired_chord_metrics, _save_chords_per_frame,
                                                _save_mean_metrics, _save_mode_per_frame,
                                                _save_nicr_per_frame)

KEEP = ["note_in_chord_ratio_mean", "note_in_chord_ratio_per_song_mean", "note_in_mode_ratio_mean",
        "note_in_mode_ratio_per_song_mean", "chord_duration_entropy", "chord_silence_ratio_mean",
        "chord_complexity_mean", "sync_emd_vs_gt", "chord_dist_wasserstein_per_song_onset_vs_gt",
        "chord_dist_wasserstein_per_song_frame_vs_gt",
        "chord_dist_wasserstein_key_relative_onset_vs_gt",
        "chord_dist_wasserstein_key_relative_frame_vs_gt"]


def _load(p: Path) -> torch.Tensor:
    return torch.load(p, map_location="cpu", weights_only=False)


def score(tensor, tok, parts: Path, gt=None):
    """means.json entries for one source; gt = (gt_means, gt_chords, tonics) for a model."""
    parts.mkdir(parents=True, exist_ok=True)
    nicr_means, nicr_pooled = _save_nicr_per_frame(tensor, tok, parts / "nicr_per_frame.pt")
    _, mode_seg, mode_pooled = _save_mode_per_frame(tensor, tok, parts / "mode_per_frame.pt")
    chords = _save_chords_per_frame(tensor, tok, parts / "chords_per_frame.pt")
    means = _save_mean_metrics(tensor, tok, parts, "", chords, reward_wrapper=None,
                               reward_model=None, device=torch.device("cpu"))
    row = {"note_in_chord_ratio_mean": nicr_pooled,
           "note_in_chord_ratio_per_song_mean": _nanmean(nicr_means),
           "note_in_mode_ratio_mean": mode_pooled,
           "note_in_mode_ratio_per_song_mean": _nanmean(mode_seg),
           **{k: means[k] for k in ("chord_duration_entropy", "chord_silence_ratio_mean",
                                    "chord_complexity_mean")}}
    if gt is None:
        return row, means, chords
    gt_means, gt_chords, tonics = gt
    row.update(_compare_to_gt(means, gt_means))
    row.update(_paired_chord_metrics(chords, gt_chords, tonics))
    return {k: row.get(k) for k in KEEP}, means, chords


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path, required=True,
                    help="commit_replay.py output roots (each with k<k>/<corpus>/...)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parts_root = args.out.with_suffix(".parts")

    res = {}
    for root in args.roots:
        for kdir in sorted(root.glob("k*"), key=lambda p: int(p.name[1:])):
            for cdir in sorted(p for p in kdir.iterdir() if p.is_dir()):
                corpus = cdir.name
                tok = HooktheoryTokenizer(chord_names=json.loads((cdir / "chord_names_augmented.json").read_text()))
                gt_t = _load(cdir / "gt" / "gt.pt")
                meta = [json.loads(l) for l in (cdir / "metadata.jsonl").read_text().splitlines() if l.strip()]
                assert len(meta) == gt_t.shape[0], f"{cdir}: metadata rows != gt rows"
                tonics = _hooktheory_tonics(meta, corpus, "test")
                parts = parts_root / root.name / kdir.name / corpus
                gt_row, gt_means, gt_chords = score(gt_t, tok, parts / "gt")
                res.setdefault(f"{corpus}/gt", {k: gt_row.get(k) for k in KEEP if k in gt_row})
                for mdir in sorted((cdir / "models").iterdir()):
                    p = mdir / "preds.pt"
                    if not p.exists():
                        continue
                    row, _, _ = score(_load(p), tok, parts / mdir.name, (gt_means, gt_chords, tonics))
                    res[f"{root.name}/{kdir.name}/{corpus}/{mdir.name}"] = row
                    print(f"{root.name}/{kdir.name}/{corpus}/{mdir.name}: "
                          + "  ".join(f"{k.replace('_mean', '')[:22]}={v:.3f}" for k, v in row.items()
                                      if isinstance(v, float)), flush=True)

    # mean over seeds per condition
    groups = {}
    for key, row in res.items():
        if key.endswith("/gt"):
            continue
        cond = key.rsplit("_s", 1)[0]
        groups.setdefault(cond, []).append(row)
    summary = {cond: {k: st.mean(r[k] for r in rows) if all(r.get(k) is not None for r in rows) else None
                      for k in KEEP} for cond, rows in groups.items()}
    summary.update({k: v for k, v in res.items() if k.endswith("/gt")})
    args.out.write_text(json.dumps({"per_seed": res, "mean_over_seeds": summary}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
