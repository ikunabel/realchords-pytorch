#!/usr/bin/env python3
"""Plot where in the bar the chord model expects a chord change.

Reads the JSON written by scripts/eval/analyze_chord_onset_timing.py and draws
two panels over the 16 sixteenth-note positions of a 4/4 bar:

  * top    -- probability mass the model puts on *any* chord onset, with the
              ground-truth rate of chord changes at that position overlaid;
  * bottom -- entropy of the model's choice among onsets once renormalised,
              i.e. how arbitrary a chord forced at that position would be.

Bar colour follows bar height (light = low, dark = high) on each panel's own
axis range, reinforcing the value rather than adding a second encoding.

Usage::

    python scripts/plotting/plot_chord_onset_timing.py \\
        [--input logs/eval/chord_onset_timing/chord_onset_timing_20260921_103914.json] \\
        [--model ReaLchords] [--output scripts/plotting/chord_onset_timing.pdf]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FRAMES_PER_BEAT = 4

# Sequential single-hue blue ramp, light -> dark. Starts at step 250 rather
# than the palest step so even near-zero bars stay visible on the surface.
RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def shade(values, vmax):
    """Colour each bar by its height on a 0..vmax scale."""
    return [RAMP(min(v / vmax, 1.0)) for v in values]


def tick_label(frame: int) -> str:
    beat, sub = divmod(frame, FRAMES_PER_BEAT)
    return str(beat + 1) if sub == 0 else ["", "e", "&", "a"][sub]


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0, pad=4)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


LOG_DIR = Path("logs/eval/chord_onset_timing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="logs/eval/chord_onset_timing/chord_onset_timing_20260921_103914.json",
    )
    parser.add_argument("--model", default=None,
                        help="Model label inside the JSON (default: first).")
    parser.add_argument("--output", default="scripts/plotting/chord_onset_timing.pdf")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    results = data["results"]
    result = (next(r for r in results if r["model"] == args.model)
              if args.model else results[0])
    rows = sorted(result["bar_position"], key=lambda r: r["bucket"])
    vocab = data["args"].get("chord_names_path", "")

    frames = [r["bucket"] for r in rows]
    p_onset = [r["mean_p_onset"] for r in rows]
    gt_rate = [r["gt_onset_rate"] for r in rows]
    entropy = [r["mean_entropy"] for r in rows]
    # Uniform over the onset vocabulary; 2,821 chords for the shipped checkpoints
    max_entropy = math.log(2821)

    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax_p, ax_h) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), dpi=200, sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    fig.patch.set_facecolor(SURFACE)

    bar_kw = dict(width=0.78, edgecolor=SURFACE, linewidth=1.2)

    # -- Top: how much the model wants a chord change here ------------------
    style_axis(ax_p)
    ax_p.bar(frames, p_onset, color=shade(p_onset, 1.0), **bar_kw)
    ax_p.plot(frames, gt_rate, linestyle="none", marker="D", markersize=5,
              markerfacecolor=INK, markeredgecolor=SURFACE, markeredgewidth=1.2,
              zorder=3)
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel("P(chord change)", color=INK_SECONDARY, fontsize=9)
    ax_p.set_title("Model probability of a chord change, by position in the bar",
                   loc="left", color=INK, fontsize=10, pad=8)
    for f, p in zip(frames, p_onset):
        if p >= 0.3:
            ax_p.annotate(f"{p:.2f}", (f, p), textcoords="offset points",
                          xytext=(0, 4), ha="center", fontsize=7.5,
                          color=INK_SECONDARY)

    # -- Bottom: how arbitrary a forced chord would be ------------------------
    style_axis(ax_h)
    ax_h.bar(frames, entropy, color=shade(entropy, max_entropy), **bar_kw)
    ax_h.axhline(max_entropy, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_h.text(15.45, max_entropy + 0.12, "uniform over all 2,821 chords",
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

    handles = [
        Patch(facecolor=RAMP(0.6), edgecolor=SURFACE, label="Model"),
        Line2D([], [], linestyle="none", marker="D", markersize=5,
               markerfacecolor=INK, markeredgecolor=SURFACE,
               label="Ground-truth change rate"),
    ]
    ax_p.legend(handles=handles, loc="upper right", frameon=False,
                fontsize=7.5, labelcolor=INK_SECONDARY, handlelength=1.2)

    n_frames = sum(r["n_frames"] for r in rows)
    fig.text(0.99, 0.0,
             f"{result['model']} · {result['songs']} Hooktheory test songs · "
             f"{n_frames:,} frames",
             ha="right", fontsize=7, color=INK_MUTED)

    # Also keep a copy next to the result JSONs it was drawn from
    for out in (Path(args.output), LOG_DIR / Path(args.output).name):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor=SURFACE)
        print(f"Written {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
