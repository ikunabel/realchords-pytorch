#!/usr/bin/env python3
"""Does tonal agreement carry information beyond note-in-chord? (journal/experiments/GAPT_VS_MLE.md)

Per corpus, correlates the two metrics across the models of the leave-one-out grid (27 MLE, 27 GAPT):
over all 54 models, and within each stage. Tonal agreement per model from tonal_agreement.py's output
(global, per 16-bar window, averaged); note-in-chord from each model's means.json. Each stage is scored
on its own windows (GAPT Hooktheory: the first 1,024 of MLE's 2,785), as in those outputs.

Within a stage the models differ only by training mixture and seed, so a high within-stage correlation
means the two metrics rank models alike; the all-models correlation is dominated by the stage difference.

Usage:
    python scripts/eval/custom_eval/tonal_vs_nicr.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

STAGES = {"MLE": Path("logs/custom_eval/dataset_loo_new"), "GAPT": Path("logs/custom_eval/gapt_loo")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tonal", type=Path, default=Path("logs/custom_eval/tonal_agreement_loo.json"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/tonal_vs_nicr.json"))
    args = ap.parse_args()
    tonal = json.loads(args.tonal.read_text())
    res = {}
    for corpus, row in tonal.items():
        pts = {}
        for stage, d in STAGES.items():
            means = json.loads((d / corpus / "means.json").read_text())["models"]
            t = dict(zip(row[stage]["models"], row[stage]["per_model_global"]))
            pts[stage] = [(t[m], means[m]["note_in_chord_ratio_mean"]) for m in row[stage]["models"]]
        out = {}
        for scope, p in (("all", pts["MLE"] + pts["GAPT"]), ("MLE", pts["MLE"]), ("GAPT", pts["GAPT"])):
            x, y = np.array(p).T
            out[scope] = {"n": len(p), "pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]),
                          "tonal_range": [float(x.min()), float(x.max())], "nicr_range": [float(y.min()), float(y.max())]}
        res[corpus] = out
        print(f"{corpus:<22}" + "  ".join(f"{s}: r {v['pearson']:+.2f} rho {v['spearman']:+.2f}" for s, v in out.items()))
    args.out.write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
