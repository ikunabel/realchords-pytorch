#!/usr/bin/env python3
"""Compare several chord models' metrical profiles on one figure.

Same two panels as plot_chord_onset_timing.py -- probability of a chord change
by position in the bar (with the ground-truth rate), and the entropy of the
chord choice if forced to change there -- but with one bar per model at each
position, so the effect of RL fine-tuning on calibration and on the tail of
the onset distribution can be read side by side.

Usage::

    python scripts/plotting/plot_chord_onset_timing_comparison.py \\
        --inputs logs/eval/chord_onset_timing/chord_onset_timing_20260921_105402.json \\
                 logs/eval/chord_onset_timing/chord_onset_timing_20260921_103914.json \\
        [--models "Online MLE" GAPT ReaLchords] \\
        [--output scripts/plotting/chord_onset_timing_comparison.pdf]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_chord_onset_timing import (  # noqa: E402
    GRID, INK, INK_MUTED, INK_SECONDARY, SURFACE, style_axis, tick_label,
)

# Categorical slots 1-3, validated all-pairs (dataviz palette), fixed order
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


LOG_DIR = Path("logs/eval/chord_onset_timing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Result JSONs; models are gathered from all of them.")
    parser.add_argument("--models", nargs="+",
                        default=["Online MLE", "GAPT", "ReaLchords"])
    parser.add_argument("--output",
                        default="scripts/plotting/chord_onset_timing_comparison.pdf")
    args = parser.parse_args()

    by_model = {}
    for path in args.inputs:
        for result in json.loads(Path(path).read_text())["results"]:
            by_model[result["model"]] = result
    missing = [m for m in args.models if m not in by_model]
    if missing:
        raise SystemExit(f"Not found in inputs: {missing}. Have: {list(by_model)}")
    if len(args.models) > len(MODEL_COLORS):
        raise SystemExit(f"At most {len(MODEL_COLORS)} models per figure.")

    rows = {m: sorted(by_model[m]["bar_position"], key=lambda r: r["bucket"])
            for m in args.models}
    frames = [r["bucket"] for r in rows[args.models[0]]]
    gt_rate = [r["gt_onset_rate"] for r in rows[args.models[0]]]
    max_entropy = math.log(2821)

    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax_p, ax_h) = plt.subplots(
        2, 1, figsize=(7.6, 5.6), dpi=200, sharex=True,
        gridspec_kw={"hspace": 0.3},
    )
    fig.patch.set_facecolor(SURFACE)

    n = len(args.models)
    width = 0.82 / n
    offsets = [(i - (n - 1) / 2) * width for i in range(n)]

    style_axis(ax_p)
    style_axis(ax_h)
    for model, color, off in zip(args.models, MODEL_COLORS, offsets):
        xs = [f + off for f in frames]
        ax_p.bar(xs, [r["mean_p_onset"] for r in rows[model]], width=width,
                 color=color, edgecolor=SURFACE, linewidth=0.8)
        ax_h.bar(xs, [r["mean_entropy"] for r in rows[model]], width=width,
                 color=color, edgecolor=SURFACE, linewidth=0.8)

    ax_p.plot(frames, gt_rate, linestyle="none", marker="D", markersize=5,
              markerfacecolor=INK, markeredgecolor=SURFACE, markeredgewidth=1.2,
              zorder=3)
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("P(chord change)", color=INK_SECONDARY, fontsize=9)
    ax_p.set_title("Probability of a chord change, by position in the bar",
                   loc="left", color=INK, fontsize=10, pad=8)

    ax_h.axhline(max_entropy, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_h.text(15.55, max_entropy + 0.12, "uniform over all 2,821 chords",
              ha="right", va="bottom", fontsize=7.5, color=INK_MUTED)
    ax_h.set_ylim(0, 8.6)
    ax_h.set_ylabel("Entropy of chord choice\n(nats)", color=INK_SECONDARY, fontsize=9)
    ax_h.set_title("Uncertainty about which chord, if forced to change here",
                   loc="left", color=INK, fontsize=10, pad=8)
    ax_h.set_xticks(frames)
    ax_h.set_xticklabels([tick_label(f) for f in frames])
    ax_h.set_xlabel("Position in 4/4 bar (sixteenth notes)", color=INK_SECONDARY,
                    fontsize=9)
    ax_h.set_xlim(-0.6, 15.6)

    handles = [Patch(facecolor=c, edgecolor=SURFACE, label=m)
               for m, c in zip(args.models, MODEL_COLORS)]
    handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5,
                          markerfacecolor=INK, markeredgecolor=SURFACE,
                          label="Ground-truth change rate"))
    ax_p.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5,
                labelcolor=INK_SECONDARY, handlelength=1.2)

    songs = by_model[args.models[0]]["songs"]
    n_frames = sum(r["n_frames"] for r in rows[args.models[0]])
    fig.text(0.99, 0.0, f"{songs} Hooktheory test songs · {n_frames:,} frames",
             ha="right", fontsize=7, color=INK_MUTED)

    # Also keep a copy next to the result JSONs it was drawn from
    for out in (Path(args.output), LOG_DIR / Path(args.output).name):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor=SURFACE)
        print(f"Written {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
