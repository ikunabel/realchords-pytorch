#!/usr/bin/env python3
"""How well does profile-correlation key estimation recover Hooktheory's annotated keys?

Validation before using a key estimator in a metric. Test windows are the evaluation's own
Hooktheory windows (logs/custom_eval/dataset_loo_new/hooktheory/gt/gt.pt, current vocabulary),
matched to their section's key annotation by song_id, as custom_evaluation._hooktheory_tonics
does. Only single-key sections are used: a multi-key section's windows cannot be assigned one key.

Hooktheory annotates a tonic plus a full scale (Ionian, Aeolian, Dorian, Mixolydian, Phrygian,
Lydian, Locrian, harmonic minor, ...). Profile methods estimate major or minor, so each scale is
mapped by its third: major third -> major (Ionian, Lydian, Mixolydian), minor third -> minor
(Aeolian, Dorian, Phrygian, Locrian, harmonic minor).

Estimation is music21's (realchords/utils/key_estimation.py; Krumhansl-Schmuckler by default).
For every analyser x input (melody, chords, both), reports the share of windows whose estimate is
exact, a fifth above, a fifth below, the relative key, the parallel key, or other -- and the MIREX
score from mir_eval.key.weighted_score, which credits only the fifth ABOVE (0.5). Only
Hooktheory's key labels are usable; every other corpus's are placeholders
(journal/reference/DATASET_SUMMARY.md).

Usage:
    python scripts/eval/custom_eval/key_estimator_validation.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import mir_eval

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.key_estimation import (ANALYSERS, chord_stream, combined_stream,
                                             estimate_key, key_relation, melody_stream,
                                             mir_eval_key)

RELATIONS = ["exact", "fifth above", "fifth below", "relative", "parallel", "other"]


def section_keys(jsonl: Path) -> dict:
    """Hooktheory section id -> key index (tonic + 12 * minor) for single-key sections."""
    out = {}
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            keys = item["annotations"]["keys"]
            if len(keys) != 1:
                continue
            k = keys[0]
            third = sum(k["scale_degree_intervals"][:2])
            if third not in (3, 4):
                continue
            out[item["hooktheory"]["id"]] = int(k["tonic_pitch_class"]) + (12 if third == 3 else 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, default=Path("logs/custom_eval/dataset_loo_new/hooktheory"))
    ap.add_argument("--labels", type=Path, default=Path("data/cache/hooktheory/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/key_estimator_validation.json"))
    args = ap.parse_args()

    names = json.loads((args.eval_dir / "chord_names_augmented.json").read_text())
    i2n = HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names)).id_to_name
    gt = torch.load(args.eval_dir / "gt" / "gt.pt", map_location="cpu", weights_only=False)
    meta = [json.loads(l) for l in (args.eval_dir / "metadata.jsonl").read_text().splitlines() if l.strip()]
    labels = section_keys(args.labels)

    windows = []
    for row, m in zip(gt.tolist(), meta):
        key = labels.get(m.get("song_id"))
        if key is None:
            continue
        mel = melody_stream(row[2::2])
        cho = chord_stream(row[1::2], i2n)
        windows.append((key, {"melody": mel, "chords": cho, "both": combined_stream(mel, cho)}))
    n_major = sum(1 for k, _ in windows if k < 12)
    print(f"{len(windows)} of {len(meta)} windows have a single-key label "
          f"({n_major} major, {len(windows) - n_major} minor)\n")

    results = {}
    print(f"{'analyser':<11}{'input':<8}" + "".join(f"{r:>13}" for r in RELATIONS)
          + f"{'MIREX':>8}{'major ok':>10}{'minor ok':>10}")
    for profile in ANALYSERS:
        for inp in ("melody", "chords", "both"):
            rel, per_mode, scores = Counter(), {False: [0, 0], True: [0, 0]}, []
            for key, h in windows:
                est = estimate_key(h[inp], profile)
                r = "other" if est is None else key_relation(est, key)
                rel[r] += 1
                scores.append(0.0 if est is None else
                              mir_eval.key.weighted_score(mir_eval_key(key), mir_eval_key(est)))
                per_mode[key >= 12][0] += r == "exact"
                per_mode[key >= 12][1] += 1
            n = sum(rel.values())
            share = {r: rel[r] / n for r in RELATIONS}
            score = float(np.mean(scores))
            maj, mnr = (per_mode[False][0] / per_mode[False][1], per_mode[True][0] / per_mode[True][1])
            results[f"{profile}/{inp}"] = {**share, "score": score, "exact_major": maj,
                                           "exact_minor": mnr, "n": n}
            print(f"{profile:<11}{inp:<8}" + "".join(f"{share[r]:>13.1%}" for r in RELATIONS)
                  + f"{score:>8.3f}{maj:>10.1%}{mnr:>10.1%}")
        print()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
