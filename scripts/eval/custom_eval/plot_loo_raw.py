#!/usr/bin/env python3
"""Leave-one-out, raw metric values. No normalisation, no test statistics.

One panel per test corpus. Each panel shows, on that corpus's own test set, the
full-mixture model and the model trained without that corpus, as the three individual
training seeds plus their mean. Ground truth is a dashed line where the metric has one.

This is deliberately the unprocessed view: it is what the damage-per-seed-sd summary is
computed from, and it lets the reader see the seed spread, the distance to ground truth
and the size of the 7sets-vs-no_X gap directly, rather than taking a ratio on trust.

Usage:
    python scripts/eval/custom_eval/plot_loo_raw.py --metric chord_complexity_mean
    python scripts/eval/custom_eval/plot_loo_raw.py --metric droot --source progression
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
        "sync_emd_vs_gt": "sync EMD vs GT (lower = better)",
        "droot": "root-motion divergence from real music (lower = better)",
        "quality": "chord-vocabulary divergence from real music (lower = better)",
        "ks": "n-gram KS vs real progressions (lower = better)",
        "mean_shift": "n-gram log-prob shift vs real progressions",
        "sd_ratio": "n-gram log-prob spread, model / real"}


def series(corpus: str, metric: str, source: str, eval_dir: Path, prog: dict):
    """-> (gt value or None, {'7sets': [per-seed], 'no_X': [per-seed]})"""
    if source == "means":
        m = json.loads((eval_dir / corpus / "means.json").read_text(encoding="utf-8"))
        gt, models = m["gt"].get(metric), m["models"]
        get = lambda pre: [models[f"{pre}_{s}"].get(metric) for s in SEEDS
                           if f"{pre}_{s}" in models]
    elif source == "ngram":
        models = prog[corpus]["models"]["LOO"]
        # KS and mean_shift are already differences from held-out real progressions, so
        # a perfect model sits at 0; sd_ratio sits at 1.
        gt = 1.0 if metric == "sd_ratio" else 0.0
        get = lambda pre: [models[f"{pre}_{s}"]["vs_gt"].get(metric) for s in SEEDS
                           if f"{pre}_{s}" in models]
    else:
        models = prog[corpus]["models"]
        gt = 0.0  # progression scores are excess over the floor, so GT sits at 0
        get = lambda pre: [models[f"{pre}_{s}"]["excess"].get(metric) for s in SEEDS
                           if f"{pre}_{s}" in models]
    out = {}
    for label, pre in (("7sets", "7sets"), ("no_X", f"7sets_no_{corpus}")):
        out[label] = [v for v in get(pre) if v is not None]
    return gt, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--source", choices=["means", "progression", "ngram"], default="means")
    ap.add_argument("--order", type=int, default=3,
                    help="n-gram order when --source ngram (a symbol is a chord move, so "
                         "order k spans k+1 chords)")
    ap.add_argument("--eval_dir", type=Path, default=Path("logs/custom_eval/dataset_loo_new"))
    ap.add_argument("--prog", type=Path,
                    default=Path("logs/custom_eval/progression_scores_loo_new.json"))
    ap.add_argument("--out_dir", type=Path, default=Path("logs/custom_eval/loo_raw"))
    args = ap.parse_args()

    if args.source == "ngram":
        args.prog = Path(f"logs/custom_eval/ngram_loo/progression_ngram_order{args.order}.json")
    prog = json.loads(args.prog.read_text(encoding="utf-8")) if args.source != "means" else {}
    fig, axes = plt.subplots(1, 7, figsize=(16.5, 3.9), squeeze=False)
    for i, c in enumerate(CORPORA):
        ax = axes[0][i]
        gt, s = series(c, args.metric, args.source, args.eval_dir, prog)
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
            ax.annotate("GT" if args.source == "means" else
                        ("real" if args.source == "ngram" else "floor"),
                        xy=(1.38, gt), xycoords=("axes fraction", "data"),
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
    suffix = f"_order{args.order}" if args.source == "ngram" else ""
    stem = args.out_dir / f"loo_raw_{args.metric}{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
