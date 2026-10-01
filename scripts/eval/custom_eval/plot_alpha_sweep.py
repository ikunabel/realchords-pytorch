#!/usr/bin/env python3
"""Plot the alpha sweep: metric vs alpha, one panel per test corpus.

Reads the per-window records in <eval_dir>/<corpus>/metadata.jsonl, which carry one
entry per (window, model) with model labels "a{alpha}_s{seed}". Points are the mean
over windows then over seeds; error bars are +/- 1 sd over the three training seeds,
which is the dominant noise term (see journal/DATASET_MIX_LOO.md). The ground-truth
level for each corpus is drawn as a dashed line, because every model cell sits far
below it and the alpha differences are small next to that gap.

Panels are ordered by corpus size (frames in the training cache) so a size-dependent
trend, if any, reads left to right.

Usage:
    python scripts/eval/custom_eval/plot_alpha_sweep.py \
        --eval_dir logs/custom_eval/alpha_sweep_allwin \
        --metric nicr [--zoom]

Writes both a .pdf (vector, for the thesis) and a .png next to the eval dir unless
--out names one file explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# embed TrueType rather than Type 3 so the vector output is editable and portable
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ALPHAS = [("a0", 0.0), ("a0p25", 0.25), ("a0p5", 0.5), ("a1", 1.0)]
SEEDS = ["s42", "s43", "s44"]

# gt key in means.json for each metadata.jsonl metric
GT_KEY = {
    "nicr": "note_in_chord_ratio_mean",
    "mode_fit": "note_in_mode_ratio_mean",
}
NICE = {
    "nicr": "note-in-chord ratio",
    "mode_fit": "note-in-mode ratio",
}
DISPLAY = {
    "chord_melody_dataset": "CMD",
    "emopia_plus": "EMOPIA+",
    "jazzmus": "JAZZMUS",
    "nottingham": "Nottingham",
    "pop909": "POP909",
    "wikifonia": "Wikifonia",
    "hooktheory": "Hooktheory",
}


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def corpus_sizes(stats_path: Path) -> Dict[str, float]:
    """Frame counts per corpus, used only to order the panels."""
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    out = {}
    for name, entry in stats.items():
        train = entry.get("train") if isinstance(entry, dict) else None
        if isinstance(train, dict) and "total_frames" in train:
            out[name] = float(train["total_frames"])
    return out


def load_corpus(path: Path, metric: str):
    """-> {alpha: [per-seed mean]}, n_windows, n_null. Nulls dropped per model."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    cells: Dict[float, List[float]] = {}
    n_null = 0
    for label, alpha in ALPHAS:
        per_seed = []
        for seed in SEEDS:
            key = f"{label}_{seed}"
            vals = [r[metric].get(key) for r in rows if metric in r]
            n_null += sum(1 for v in vals if v is None)
            vals = [v for v in vals if v is not None]
            if vals:
                per_seed.append(_mean(vals))
        if per_seed:
            cells[alpha] = per_seed
    return cells, len(rows), n_null


def gt_level(corpus_dir: Path, metric: str) -> Optional[float]:
    try:
        means = json.loads((corpus_dir / "means.json").read_text(encoding="utf-8"))
    except OSError:
        return None
    return means.get("gt", {}).get(GT_KEY.get(metric, ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, type=Path)
    ap.add_argument(
        "--label",
        default=None,
        help="series name for --eval_dir; only shown when overlays are present",
    )
    ap.add_argument(
        "--overlay",
        nargs="*",
        default=[],
        metavar="LABEL=DIR",
        help="additional eval dirs drawn in the same panels, e.g. GAPT=logs/custom_eval/"
        "gapt_alpha_allwin. Valid only when the runs share the eval seed and all_windows "
        "setting, which makes their windows byte-identical.",
    )
    ap.add_argument("--metric", default="nicr")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="explicit output path; its extension picks the format and --formats is ignored",
    )
    ap.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="extensions to write when --out is not given (default: pdf png)",
    )
    ap.add_argument("--skip", nargs="*", default=[], help="corpus names to leave out")
    ap.add_argument("--stats", type=Path, default=Path("data/cache/dataset_stats.json"))
    ap.add_argument("--share_y", action="store_true", help="one y-scale for all panels")
    ap.add_argument(
        "--errorbar",
        choices=["seeds", "pooled", "percell"],
        default="seeds",
        help="pooled (default): one bar height per corpus, sd pooled over the alpha cells "
        "(8 df). percell: each cell's own sd over 3 seeds (2 df) -- individually unreliable, "
        "its length varies ~50%% by chance, so bars are not comparable between cells. "
        "seeds: no summary statistic at all -- draw the three seed values themselves, with a "
        "thin min-max line. Nothing is estimated, so nothing can be mis-estimated; the cost is "
        "that the figure then carries no quantity an interval could be built from.",
    )
    ap.add_argument(
        "--zoom",
        action="store_true",
        help="scale y to the model curves and report the GT gap in the panel title "
        "instead of drawing GT in range (the GT line otherwise flattens every curve)",
    )
    args = ap.parse_args()

    found = []
    for sub in sorted(args.eval_dir.iterdir()):
        meta = sub / "metadata.jsonl"
        if not sub.is_dir() or sub.name in args.skip or not meta.exists():
            continue
        if meta.stat().st_size == 0:
            continue
        cells, n_win, n_null = load_corpus(meta, args.metric)
        if len(cells) < len(ALPHAS):
            print(f"  skipping {sub.name}: only {len(cells)}/{len(ALPHAS)} alpha cells")
            continue
        found.append((sub.name, cells, n_win, n_null, gt_level(sub, args.metric)))

    overlays = []
    for item in args.overlay:
        if "=" not in item:
            raise SystemExit(f"--overlay expects LABEL=DIR, got {item!r}")
        lab, d = item.split("=", 1)
        per_corpus = {}
        for sub in sorted(Path(d).iterdir()):
            meta = sub / "metadata.jsonl"
            if not sub.is_dir() or not meta.exists() or meta.stat().st_size == 0:
                continue
            cells, n_win, _ = load_corpus(meta, args.metric)
            if len(cells) == len(ALPHAS):
                per_corpus[sub.name] = (cells, n_win)
        overlays.append((lab, per_corpus))

    if not found:
        raise SystemExit(f"no usable corpora under {args.eval_dir}")

    sizes = corpus_sizes(args.stats)
    found.sort(key=lambda t: -sizes.get(t[0], 0.0))

    n = len(found)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow), squeeze=False, sharey=args.share_y
    )

    xs = [a for _, a in ALPHAS]
    OVERLAY_COLORS = ["#e8710a", "#2ca02c", "#9467bd"]
    for i, (name, cells, n_win, n_null, gt) in enumerate(found):
        ax = axes[i // ncol][i % ncol]
        extra = []
        for j, (lab, per_corpus) in enumerate(overlays):
            if name not in per_corpus:
                continue
            ocells, on = per_corpus[name]
            if on != n_win:
                print(f"  WARNING {name}: {lab} has {on} windows vs {n_win} -- "
                      f"not the same windows, overlay is not comparable")
            col = OVERLAY_COLORS[j % len(OVERLAY_COLORS)]
            oys = [_mean(ocells[a]) for a in xs]
            for a in xs:
                vals = ocells[a]
                ax.plot([a, a], [min(vals), max(vals)], lw=1.0, color=col,
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.scatter([a] * len(vals), vals, s=16, color=col, alpha=0.55,
                           linewidths=0, zorder=3)
            ax.plot(xs, oys, marker="o", lw=1.6, color=col, zorder=4,
                    label=lab if i == 0 else None)
            extra.extend(v for a in xs for v in ocells[a])
        ys = [_mean(cells[a]) for a in xs]
        if args.errorbar == "seeds":
            es = None
            for a in xs:
                vals = cells[a]
                ax.plot([a, a], [min(vals), max(vals)], lw=1.0, color="#1f77b4",
                        alpha=0.55, zorder=2, solid_capstyle="butt")
                ax.scatter([a] * len(vals), vals, s=16, color="#1f77b4", alpha=0.55,
                           linewidths=0, zorder=3)
        elif args.errorbar == "pooled":
            # Pool the per-cell variances across alpha: 4 cells x 2 df = 8 df, so the bar
            # height is estimated ~2x more precisely than any single cell's sd. Assumes the
            # seed spread does not depend on alpha, which the data are consistent with.
            pooled = math.sqrt(sum(_sd(cells[a]) ** 2 for a in xs) / len(xs))
            es = [pooled] * len(xs)
        else:
            es = [_sd(cells[a]) for a in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.6, color="#1f77b4", zorder=5,
                    label=(args.label or "MLE") if (overlays and i == 0) else None)
        if gt is not None and not args.zoom:
            ax.axhline(gt, ls="--", lw=1.2, color="#d62728", zorder=2)
            ax.annotate(
                f"GT {gt:.3f}",
                xy=(1.0, gt),
                xytext=(-2, 3),
                textcoords="offset points",
                ha="right",
                fontsize=7.5,
                color="#d62728",
            )
        if args.zoom:
            if es is None:
                lo = min(min(cells[a]) for a in xs)
                hi = max(max(cells[a]) for a in xs)
            else:
                lo = min(y - e for y, e in zip(ys, es))
                hi = max(y + e for y, e in zip(ys, es))
            if extra:
                lo = min(lo, min(extra))
                hi = max(hi, max(extra))
            pad = max((hi - lo) * 0.35, 0.004)
            ax.set_ylim(lo - pad, hi + pad)
        size = sizes.get(name)
        sub = f"n={n_win} windows" + (f", {size/1e6:.2f}M frames" if size else "")
        if args.zoom and gt is not None:
            sub += f"  |  GT {gt:.3f}"
        ax.set_title(f"{DISPLAY.get(name, name)}\n{sub}", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        if i % ncol == 0:
            ax.set_ylabel(NICE.get(args.metric, args.metric), fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        if n_null:
            print(f"  {name}: {n_null} null values dropped")

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(
        f"{NICE.get(args.metric, args.metric)} vs sampling temperature "
        f"(mean over 3 training seeds, "
        + ("individual seed values shown" if args.errorbar == "seeds" else "error bars $\\pm$1 sd")
        + {"pooled": " pooled over $\\alpha$", "percell": " per cell"}.get(args.errorbar, "")
        + "; panels ordered by corpus size)"
        + ("  [y-scale zoomed to models; GT is far above range]" if args.zoom else ""),
        fontsize=10,
    )
    if overlays:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", ncol=1, frameon=False,
                   bbox_to_anchor=(0.98, 0.06), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    suffix = ("_zoom" if args.zoom else "") + ("" if args.errorbar == "seeds" else f"_{args.errorbar}")
    if args.out is not None:
        outs = [args.out]
    else:
        stem = args.eval_dir / f"alpha_vs_{args.metric}{suffix}"
        outs = [stem.with_suffix(f".{fmt.lstrip('.')}") for fmt in args.formats]
    for out in outs:
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"wrote {out}  ({n} corpora)")


if __name__ == "__main__":
    main()
