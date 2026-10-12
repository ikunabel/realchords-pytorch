#!/usr/bin/env python3
"""Leave-one-out, raw values, compact version for slides.

Same data and layout as plot_loo_raw.py --source progression (one panel per test corpus:
full mixture vs. the model trained without that corpus, three seeds as dots, mean as a
bar, the floor at 0), but sized for a slide: no suptitle (the slide has one), short
tick labels, short y label.

With --plot delta it draws the leave-one-out effect as differences: one panel per metric
(root-motion divergence, chord-quality divergence, note-in-chord ratio), one horizontal bar per test
corpus = mean(without X) - mean(all 7) on X's test set, error bar = standard error of that
difference (unpaired, 3 + 3 seeds), dagger where the seed ranges do not overlap. Corpora are
ordered by size and labelled with their alpha = 0.5 sampling weight.

With --plot pairs it draws the same three metrics as raw values instead (no dagger marks): per
test corpus one marker for the full mixture and one for the model without X, the mean of the three
seeds with a min-max whisker (no sd: three seeds are too few for a meaningful one) (dot-and-whisker, so the note-in-chord axis need not start at 0). Corpora in the same
order (by size), labelled by name only.

With --plot bars it instead draws one of the named conditionals (e.g. fifth_motion_rate) as
grouped bars per test corpus: held-out real music vs. the full-mixture model (mean of the three
seeds, seeds as dots), corpora sorted by the real value.

Usage:
    python scripts/eval/custom_eval/plot_loo_slide.py --metric droot \
        --prog presentation/data/progression_scores_loo_new.json \
        --out journal/Docs/pitch_presentation/figures/loo_droot.pdf
    python scripts/eval/custom_eval/plot_loo_slide.py --plot bars --metric fifth_motion_rate \
        --ylabel "fifth motion" --prog presentation/data/progression_scores_loo_new.json \
        --out journal/Docs/pitch_presentation/figures/loo_fifth_motion.pdf
    python scripts/eval/custom_eval/plot_loo_slide.py --plot delta \
        --prog presentation/data/progression_scores_loo_new.json \
        --means_dir presentation/data/logs/custom_eval/dataset_loo_new \
        --out journal/Docs/pitch_presentation/figures/loo_delta.pdf
    python scripts/eval/custom_eval/plot_loo_slide.py --plot pairs \
        --prog presentation/data/progression_scores_loo_new.json \
        --means_dir presentation/data/logs/custom_eval/dataset_loo_new \
        --out journal/Docs/pitch_presentation/figures/loo_pairs.pdf
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

SEEDS = ["s42", "s43", "s44"]
CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}


def conditional_bars(prog: dict, metric: str, ylabel: str, out: Path) -> None:
    """Real music vs. full-mixture model for one named conditional, one bar pair per corpus."""
    rows = []
    for corpus in CORPORA:
        models = prog[corpus]["models"]
        seeds = [models[f"7sets_{s}"]["conditionals"][metric] for s in SEEDS
                 if f"7sets_{s}" in models]
        rows.append((corpus, models["gt"]["conditionals"][metric], seeds))
    rows.sort(key=lambda r: -r[1])
    xs = range(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.bar([x - w / 2 for x in xs], [r[1] for r in rows], w, color="#7f7f7f",
           label="real music (test set)")
    ax.bar([x + w / 2 for x in xs], [st.mean(r[2]) for r in rows], w, color="#1f77b4",
           label="model trained on all 7 corpora")
    for x, r in zip(xs, rows):
        ax.scatter([x + w / 2] * len(r[2]), r[2], s=10, color="black", zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SHORT[r[0]] for r in rows], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


# slide orange = the frame-grid figure's cells: orange!82 / orange3 / orange4 of the
# pitch deck's rgb.tex (which redefines orange as rgb 1.00, 0.65, 0.00)
# alpha = 0.5 sampling weights of the 7-corpus mixture (frames^0.5, normalised)
WEIGHT = {"hooktheory": 38, "pop909": 19, "emopia_plus": 10, "nottingham": 10,
          "wikifonia": 9, "chord_melody_dataset": 9, "jazzmus": 5}


def _loo_getters(prog: dict, means_dir: Path):
    def nicr(corpus, pre):
        m = json.loads((means_dir / corpus / "means.json").read_text(encoding="utf-8"))["models"]
        return [m[f"{pre}_{s}"]["note_in_chord_ratio_mean"] for s in SEEDS if f"{pre}_{s}" in m]

    def excess(key):
        def get(corpus, pre):
            m = prog[corpus]["models"]
            return [100 * m[f"{pre}_{s}"]["excess"][key] for s in SEEDS if f"{pre}_{s}" in m]
        return get

    return [
        ("Note-in-chord ratio", "ratio\n(real music: 0.65–0.79)", nicr),
        ("Root motion", "divergence ×100  (lower = better)", excess("droot")),
        ("Chord quality", "divergence ×100  (lower = better)", excess("quality")),
    ]


def pair_panels(prog: dict, means_dir: Path, out: Path) -> None:
    """Raw values, full mixture vs. without X: mean of the 3 seeds, whisker = min-max."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    ys = list(range(len(CORPORA)))[::-1]          # largest corpus on top
    off = 0.11          # pair gap 0.22 vs. 0.78 between corpora
    for ax, (title, xlabel, get) in zip(axes, _loo_getters(prog, means_dir)):
        for corpus, y in zip(CORPORA, ys):
            a, b = get(corpus, "7sets"), get(corpus, f"7sets_no_{corpus}")
            # 3 seeds are too few for a meaningful sd: whisker = min-max range of the seeds
            for vals, yy, face, edge, whisk, fmt, ms, label in (
                (a, y + off, "#1f77b4", "#1f77b4", "#1f77b4", "o", 4, "all 7 corpora"),
                (b, y - off, "#FFB62E", "#8C5900", "#CC8500", "s", 4.5, "without X"),
            ):
                m = st.mean(vals)
                ax.errorbar(m, yy, xerr=[[m - min(vals)], [max(vals) - m]], fmt=fmt, ms=ms,
                            color=face, markeredgecolor=edge, markeredgewidth=0.7,
                            ecolor=whisk, elinewidth=1, capsize=2,
                            label=label if corpus == CORPORA[0] else None)
        if "divergence" in xlabel:
            # Nottingham sits an order of magnitude above the rest; log keeps all readable
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            xlabel = "divergence ×100, log scale\n(lower = better)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color="#dddddd", lw=0.6)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([SHORT[c] for c in CORPORA], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(CORPORA) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.55, -0.04))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def delta_panels(prog: dict, means_dir: Path, out: Path) -> None:
    """Leave-one-out differences (without X - all 7) for three metrics, one panel each."""
    def nicr(corpus, pre):
        m = json.loads((means_dir / corpus / "means.json").read_text(encoding="utf-8"))["models"]
        return [m[f"{pre}_{s}"]["note_in_chord_ratio_mean"] for s in SEEDS if f"{pre}_{s}" in m]

    def excess(key):
        def get(corpus, pre):
            m = prog[corpus]["models"]
            return [100 * m[f"{pre}_{s}"]["excess"][key] for s in SEEDS if f"{pre}_{s}" in m]
        return get

    panels = [
        ("Root motion", "Δ divergence ×100\n(> 0: worse w/o X)", excess("droot"), None),
        ("Chord quality", "Δ divergence ×100\n(> 0: worse w/o X)", excess("quality"), None),
        ("Note-in-chord ratio", "Δ ratio\n(w/o X − all 7)", nicr, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    ys = list(range(len(CORPORA)))[::-1]          # largest corpus on top
    for ax, (title, xlabel, get, note) in zip(axes, panels):
        diffs, errs, seps = [], [], []
        for corpus in CORPORA:
            a, b = get(corpus, "7sets"), get(corpus, f"7sets_no_{corpus}")
            diffs.append(st.mean(b) - st.mean(a))
            errs.append((st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5)
            seps.append(min(b) > max(a) or max(b) < min(a))
        ax.barh(ys, diffs, 0.6, xerr=errs, capsize=2.5, color="#1f77b4",
                error_kw={"lw": 0.9, "ecolor": "black"})
        lo = min(min(d - e for d, e in zip(diffs, errs)), 0)
        hi = max(max(d + e for d, e in zip(diffs, errs)), 0)
        pad = 0.08 * (hi - lo)
        for y, d, e, sep in zip(ys, diffs, errs, seps):
            if sep:
                x = d + e + pad / 3 if d >= 0 else d - e - pad / 3
                ax.text(x, y, "†", ha="left" if d >= 0 else "right", va="center", fontsize=10)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(lo - pad, hi + 2 * pad)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{SHORT[c]} ({WEIGHT[c]}%)" for c in CORPORA], fontsize=8.5)
    fig.tight_layout(w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", choices=["loo", "bars", "delta", "pairs"], default="loo")
    ap.add_argument("--metric", default="droot",
                    help="loo: a key of `excess` (droot, quality, ...); "
                         "bars: a key of `conditionals` (fifth_motion_rate, ...)")
    ap.add_argument("--prog", type=Path,
                    default=Path("logs/custom_eval/progression_scores_loo_new.json"))
    ap.add_argument("--means_dir", type=Path,
                    default=Path("logs/custom_eval/dataset_loo_new"),
                    help="delta: per-corpus means.json (for note-in-chord ratio)")
    ap.add_argument("--ylabel", default="divergence")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prog = json.loads(args.prog.read_text(encoding="utf-8"))
    if args.plot == "bars":
        conditional_bars(prog, args.metric, args.ylabel, args.out)
        return
    if args.plot == "pairs":
        pair_panels(prog, args.means_dir, args.out)
        return
    if args.plot == "delta":
        delta_panels(prog, args.means_dir, args.out)
        return
    fig, axes = plt.subplots(1, len(CORPORA), figsize=(9.5, 2.6))
    for ax, corpus in zip(axes, CORPORA):
        models = prog[corpus]["models"]
        vals = {
            label: [models[f"{pre}_{s}"]["excess"][args.metric] for s in SEEDS
                    if f"{pre}_{s}" in models]
            for label, pre in (("7sets", "7sets"), ("no_X", f"7sets_no_{corpus}"))
        }
        for x, label in enumerate(("7sets", "no_X")):
            ax.scatter([x] * len(vals[label]), vals[label], s=22, color="#1f77b4",
                       alpha=0.5, zorder=3)
            ax.hlines(st.mean(vals[label]), x - 0.18, x + 0.18, color="#1f77b4", lw=2.5,
                      zorder=4)
        ax.plot([0, 1], [st.mean(vals["7sets"]), st.mean(vals["no_X"])], color="#1f77b4",
                lw=1.1, alpha=0.7)
        ax.axhline(0, color="#d62728", ls="--", lw=1.1)   # floor: held-out real music
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["all 7", "w/o"], fontsize=9)
        ax.set_title(SHORT[corpus], fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(args.ylabel, fontsize=9)
    fig.tight_layout(w_pad=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
