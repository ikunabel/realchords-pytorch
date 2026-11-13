#!/usr/bin/env python3
"""Leave-one-out, raw metric values. No normalisation, no test statistics.

One panel per test corpus. Each panel shows, on that corpus's own test set, the
full-mixture model and the model trained without that corpus, as the three individual
training seeds plus their mean. Ground truth is a dashed line where the metric has one.

This is deliberately the unprocessed view: it lets the reader see the seed spread, the
distance to ground truth and the size of the 7sets-vs-no_X gap directly, rather than
taking a ratio on trust.

Reads means.json only, so --metric is one of its keys: chord_complexity_mean,
note_in_chord_ratio_mean, chord_duration_entropy, sync_emd_vs_gt. Progression matching,
root motion, chord placement and the n-gram are scored by their own scripts into their own
JSON and are not plotted here.

Usage:
    python scripts/eval/custom_eval/plot_loo_raw.py --metric chord_complexity_mean
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

SEEDS = ["s42", "s43", "s44"]
CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}
NICE = {"chord_complexity_mean": "chord complexity (pitch classes per chord)",
        "note_in_chord_ratio_mean": "note-in-chord ratio",
        "chord_duration_entropy": "chord duration entropy",
        "sync_emd_vs_gt": "sync EMD vs GT (lower = better)"}


def series(corpus: str, metric: str, eval_dir: Path):
    """-> (gt value or None, {'7sets': [per-seed], 'no_X': [per-seed]}) from means.json."""
    m = json.loads((eval_dir / corpus / "means.json").read_text(encoding="utf-8"))
    gt, models = m["gt"].get(metric), m["models"]

    def get(pre):
        return [models[f"{pre}_{s}"].get(metric) for s in SEEDS if f"{pre}_{s}" in models]

    out = {}
    for label, pre in (("7sets", "7sets"), ("no_X", f"7sets_no_{corpus}")):
        out[label] = [v for v in get(pre) if v is not None]
    return gt, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, help="a key of means.json")
    ap.add_argument("--eval_dir", type=Path, default=Path("logs/custom_eval/dataset_loo_new"))
    ap.add_argument("--out_dir", type=Path, default=Path("logs/custom_eval/loo_raw"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 7, figsize=(16.5, 3.9), squeeze=False)
    for i, c in enumerate(CORPORA):
        ax = axes[0][i]
        gt, s = series(c, args.metric, args.eval_dir)
        xs = [0, 1]
        for x, label in zip(xs, ("7sets", "no_X")):
            vals = s[label]
            if not vals:
                continue
            ax.scatter([x] * len(vals), vals, s=34, color="#1f77b4", alpha=0.45,
                       zorder=3, linewidths=0)
            ax.plot([x - 0.17, x + 0.17], [st.mean(vals)] * 2, lw=2.4,
                    color="#1f77b4", zorder=4)
        if s["7sets"] and s["no_X"]:
            ax.plot(xs, [st.mean(s["7sets"]), st.mean(s["no_X"])], lw=1.2,
                    color="#1f77b4", alpha=0.6, zorder=2)
        if gt is not None:
            ax.axhline(gt, ls="--", lw=1.2, color="#d62728", zorder=1)
            ax.annotate("GT", xy=(1.38, gt), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color="#d62728", va="center")
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["7sets", f"no\n{SHORT[c]}"], fontsize=8)
        ax.set_title(SHORT[c], fontsize=9.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0][0].set_ylabel(NICE.get(args.metric, args.metric), fontsize=9)
    fig.suptitle(f"Leave-one-out, raw values: {NICE.get(args.metric, args.metric)}"
                 f"   (3 seeds per cell, bar = mean)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"loo_raw_{args.metric}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
