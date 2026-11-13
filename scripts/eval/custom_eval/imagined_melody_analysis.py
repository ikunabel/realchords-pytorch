#!/usr/bin/env python3
"""The melody the model imagines under commit-ahead, against the real melody, with 95% intervals
(journal/experiments/COMMIT_AHEAD.md).

Reads replays run with `commit_replay.py --keep_imagined`: for the call that decides the chord at frame u
with commit-ahead k, `imagined.pt[i, u]` holds the melody the model sampled for frames u-k+1 .. u-1 under
the chords already committed. Only calls with a complete imagined span (u >= k-1) are used, inside the
window's valid length; the last window of each song is dropped (as in chord_placement.py).

Per call, over the same frames and the same committed chords, imagined vs real melody:
  fit         note-in-chord of the committed chords against the melody (pooled over calls)
  onsets      melody note onsets per bar, overall and per distance ahead (1 .. k-1 frames)
  offbeat     share of onsets off the quarter-note grid
  anticipation  share of imagined frames with the real note (same pitch, or both silent), against the
              baseline "the last real note keeps sounding"

Effects are averaged over the model seeds; 95% intervals from a bootstrap over windows (resampled within
corpus, the same windows for every seed). No significance tests.

Usage:
    python scripts/eval/custom_eval/imagined_melody_analysis.py \\
        --root logs/custom_eval/commit_replay_imagined_all --out logs/custom_eval/commit_replay_imagined_all/analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from chord_placement import keep_rows  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.eval_utils import evaluate_note_in_chord_ratio

CORPORA = ["hooktheory", "pop909"]
STAGES = ["mle", "gapt"]


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def per_window(cdir: Path, label: str, k: int, tok) -> dict:
    """Per kept window: counts for every statistic, imagined and real."""
    gt = _load(cdir / "gt" / "gt.pt")
    preds = _load(cdir / "models" / label / "preds.pt")
    imag = _load(cdir / "models" / label / "imagined.pt")
    rows, _ = keep_rows(cdir / "metadata.jsonl", gt.shape[0])
    chords, real = preds[:, 1::2], gt[:, 2::2]
    w = k - 1
    on_lo, on_hi = tok.note_on_token_range
    hold_lo, hold_hi = tok.note_hold_token_range

    def pitch(t: torch.Tensor) -> torch.Tensor:
        """Sounding pitch, -1 for silence, -2 for any other token."""
        p = torch.full_like(t, -2)
        p[t == tok.silence_token] = -1
        h = (t >= hold_lo) & (t <= hold_hi)
        o = (t >= on_lo) & (t <= on_hi)
        p[h], p[o] = t[h] - hold_lo, t[o] - on_lo
        return p

    keys = ["fit_c_i", "fit_v_i", "fit_c_r", "fit_v_r", "frames", "off_i", "off_r"]
    out = {key: [] for key in keys} | {"on_i": [], "on_r": [], "hit": [], "base": [], "n": []}
    seq_i, seq_r, owner = [], [], []
    for wi, i in enumerate(rows):
        valid = int((real[i] != tok.pad_token).sum())
        U = torch.arange(w, valid)
        if len(U) == 0:
            for key in keys: out[key].append(0.0)
            for key in ("on_i", "on_r", "hit", "base", "n"): out[key].append(np.zeros(w))
            continue
        F = (U - w)[:, None] + torch.arange(w)[None, :]                # frames of each call's span
        mi, mr, c = imag[i, U], real[i, F], chords[i, F]
        is_on_i, is_on_r = (mi >= on_lo) & (mi <= on_hi), (mr >= on_lo) & (mr <= on_hi)
        offgrid = (F % 4 != 0)
        out["on_i"].append(is_on_i.sum(0).numpy().astype(float))
        out["on_r"].append(is_on_r.sum(0).numpy().astype(float))
        out["off_i"].append(float((is_on_i & offgrid).sum()))
        out["off_r"].append(float((is_on_r & offgrid).sum()))
        out["frames"].append(float(F.numel()))
        pr, pg = pitch(mr), pitch(mi)
        last = torch.where(U - w >= 1, pitch(real[i, (U - w - 1).clamp(min=0)]), torch.full_like(U, -1))
        ok = (pr != -2) & (pg != -2)
        out["hit"].append(((pg == pr) & ok).sum(0).numpy().astype(float))
        out["base"].append(((last[:, None] == pr) & ok).sum(0).numpy().astype(float))
        out["n"].append(ok.sum(0).numpy().astype(float))
        seq_i.append(torch.stack([c, mi], -1).reshape(len(U), -1))
        seq_r.append(torch.stack([c, mr], -1).reshape(len(U), -1))
        owner += [wi] * len(U)
    owner = np.array(owner)
    for name, seqs in (("i", seq_i), ("r", seq_r)):
        _, valid_c, correct_c = evaluate_note_in_chord_ratio(torch.cat(seqs), tok, model_part="chord", return_count=True)
        out[f"fit_c_{name}"] = np.bincount(owner, weights=correct_c.float().numpy(), minlength=len(rows))
        out[f"fit_v_{name}"] = np.bincount(owner, weights=valid_c.float().numpy(), minlength=len(rows))
    return {key: np.asarray(v, float) for key, v in out.items()}


def stats(d: dict, idx: np.ndarray, w: int) -> dict:
    s = {key: d[key][idx].sum(0) for key in d}
    per_offset_frames = s["frames"] / w
    return {
        "fit_imagined": s["fit_c_i"] / s["fit_v_i"], "fit_real": s["fit_c_r"] / s["fit_v_r"],
        "fit_diff": s["fit_c_i"] / s["fit_v_i"] - s["fit_c_r"] / s["fit_v_r"],
        "onsets_per_bar_imagined": 16 * s["on_i"].sum() / s["frames"],
        "onsets_per_bar_real": 16 * s["on_r"].sum() / s["frames"],
        "onset_deficit": s["on_i"].sum() / s["on_r"].sum() - 1,
        "onset_deficit_by_offset": (s["on_i"] / s["on_r"] - 1).tolist(),
        "offbeat_imagined": s["off_i"] / s["on_i"].sum(), "offbeat_real": s["off_r"] / s["on_r"].sum(),
        "anticipation_model": s["hit"].sum() / s["n"].sum(), "anticipation_baseline": s["base"].sum() / s["n"].sum(),
        "anticipation_gain": (s["hit"].sum() - s["base"].sum()) / s["n"].sum(),
        "anticipation_gain_by_offset": ((s["hit"] - s["base"]) / s["n"]).tolist(),
        "_frames_per_offset": float(per_offset_frames),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--k", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {}
    for k in args.k:
        w = k - 1
        for corpus in CORPORA:
            cdir = args.root / f"k{k}" / corpus
            tok = HooktheoryTokenizer(chord_names=json.loads((cdir / "chord_names_augmented.json").read_text()))
            for stage in STAGES:
                labels = sorted(m.name for m in (cdir / "models").iterdir()
                                if m.name.startswith(stage + "_") and (m / "imagined.pt").exists())
                if not labels:
                    continue
                data = [per_window(cdir, lab, k, tok) for lab in labels]
                n = len(data[0]["frames"])

                def avg(idx):
                    per_seed = [stats(d, idx, w) for d in data]
                    return {key: (np.mean([p[key] for p in per_seed], axis=0).tolist()
                                  if isinstance(per_seed[0][key], list) else float(np.mean([p[key] for p in per_seed])))
                            for key in per_seed[0]}

                est = avg(np.arange(n))
                draws = [avg(rng.integers(0, n, n)) for _ in range(args.boot)]
                cell = {"seeds": labels, "windows": n}
                for key, v in est.items():
                    if key.startswith("_"):
                        cell[key] = v
                        continue
                    arr = np.array([d[key] for d in draws])
                    cell[key] = {"estimate": v, "ci95": [np.percentile(arr, 2.5, axis=0).tolist(),
                                                         np.percentile(arr, 97.5, axis=0).tolist()]}
                res[f"k{k}/{corpus}/{stage}"] = cell
                e = lambda key: f"{cell[key]['estimate']:+.3f} [{cell[key]['ci95'][0]:+.3f}, {cell[key]['ci95'][1]:+.3f}]"
                print(f"k{k:<3}{corpus:<11}{stage:<5} n={n:<4} fit diff {e('fit_diff')}  onset deficit {e('onset_deficit')}"
                      f"  anticipation gain {e('anticipation_gain')}", flush=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
