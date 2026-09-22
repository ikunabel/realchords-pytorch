"""Summarize the dataset-mix leave-one-out benchmark (configs/custom_eval/dataset_loo/*.yml).

For every test set D, compares the 7sets baseline (D in training) with no_D (D held out) and
with the other leave-one-out models (D in training, another set held out):

  - means from means.json: note-in-chord, note-in-mode, chord duration entropy (|diff| to GT),
    sync EMD, per-song chord distance, key-relative distance (hooktheory only), CLaMP 2 FMD
    (only meaningful for large test sets; N is reported)
  - paired bootstrap (over test pieces, 10k resamples) of the per-piece note-in-chord and
    note-in-mode difference no_D - 7sets, from metadata.jsonl (per-piece means, so values differ
    slightly from the pooled means.json ratios)

Writes logs/custom_eval/dataset_loo/summary.json and prints markdown tables.

Usage:
    python scripts/eval/custom_eval/summarize_dataset_loo.py [--root logs/custom_eval/dataset_loo]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATASETS = ["hooktheory", "pop909", "nottingham", "wikifonia", "chord_melody_dataset", "jazzmus", "emopia_plus"]
BASE = "7sets"
N_BOOT = 10_000


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    """Mean of a-b with a 95% percentile interval, resampling pieces."""
    diff = a - b
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boots = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_piece(root: Path, ds: str, field: str, models: list[str]):
    rows = [json.loads(l) for l in open(root / ds / "metadata.jsonl")]
    rows = [r for r in rows if all(r[field].get(m) is not None for m in models)]
    return {m: np.array([r[field][m] for r in rows], dtype=float) for m in models}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs/custom_eval/dataset_loo")
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(0)

    summary = {}
    for ds in DATASETS:
        means = json.load(open(root / ds / "means.json"))
        gt, models = means["gt"], means["models"]
        held = f"no_{ds}"
        others = [m for m in models if m not in (BASE, held)]
        entry = {"num_sequences": int(models[BASE].get("clamp2_num_sequences_chords") or 0)}
        for key, name in [
            ("note_in_chord_ratio_mean", "nicr"),
            ("note_in_mode_ratio_mean", "mode"),
            ("sync_emd_vs_gt", "sync"),
            ("chord_dist_wasserstein_per_song_onset_vs_gt", "per_song"),
            ("chord_dist_wasserstein_key_relative_onset_vs_gt", "key_rel"),
            ("clamp2_fmd_chords_vs_gt", "fmd_chords"),
        ]:
            if models[BASE].get(key) is None:
                continue
            entry[name] = {BASE: models[BASE][key], held: models[held][key],
                           "others_mean": float(np.mean([models[m][key] for m in others])),
                           "others_min": float(np.min([models[m][key] for m in others])),
                           "others_max": float(np.max([models[m][key] for m in others]))}
        ent = {m: abs(models[m]["chord_duration_entropy"] - gt["chord_duration_entropy"]) for m in models}
        entry["entropy_gap"] = {BASE: ent[BASE], held: ent[held],
                                "others_mean": float(np.mean([ent[m] for m in others]))}
        for field, name in [("nicr", "nicr_per_piece"), ("mode_fit", "mode_per_piece")]:
            pp = per_piece(root, ds, field, [BASE, held])
            d, lo, hi = paired_bootstrap(pp[held], pp[BASE], rng)
            entry[name + "_diff_held_minus_base"] = {"mean": d, "ci95": [lo, hi], "n": len(pp[BASE])}
        summary[ds] = entry

    json.dump(summary, open(root / "summary.json", "w"), indent=2)

    print("| Test set | N | NiCR 7sets | NiCR held-out | Δ held-out − 7sets (95% CI) | NiCR others (min–max) "
          "| Mode Δ (95% CI) | Per-song dist 7sets / held-out | |entropy−GT| 7sets / held-out |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ds, e in summary.items():
        held = f"no_{ds}"
        n, m = e["nicr"], e["nicr_per_piece_diff_held_minus_base"]
        md = e["mode_per_piece_diff_held_minus_base"]
        print(f"| {ds} | {m['n']} | {n[BASE]:.3f} | {n[held]:.3f} | {m['mean']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}] "
              f"| {n['others_mean']:.3f} ({n['others_min']:.3f}–{n['others_max']:.3f}) "
              f"| {md['mean']:+.3f} [{md['ci95'][0]:+.3f}, {md['ci95'][1]:+.3f}] "
              f"| {e['per_song'][BASE]:.3f} / {e['per_song'][held]:.3f} "
              f"| {e['entropy_gap'][BASE]:.2f} / {e['entropy_gap'][held]:.2f} |")
    print(f"\nwrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
