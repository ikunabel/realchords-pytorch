#!/usr/bin/env python3
"""Chord-onset timing of two models across corpora, by metrical position class.

Reads the result JSONs of scripts/eval/analyze_chord_onset_timing.py (one per
corpus, each with both models) and draws one figure: per corpus, the probability
of a chord change (with the real change rate) and the effective number of chords
if forced to change there, at five position classes of the 4/4 bar -- downbeat,
beat 3, beats 2 and 4, the eighth offbeats ("and"), the sixteenth offsets.
Classes average their frames (weighted by frame count; effective choices =
exp(mean entropy)). Also prints the table.

Usage::

    python scripts/plotting/plot_chord_onset_timing_corpora.py \\
        --inputs hooktheory=logs/eval/chord_onset_timing/chord_onset_timing_20261119_170537.json,logs/eval/chord_onset_timing/chord_onset_timing_20260921_105402.json \\
                 pop909=logs/eval/chord_onset_timing/pop909 ... \\
        --models GAPT "GAPT 7 datasets" \\
        --output scripts/plotting/chord_onset_timing_corpora.pdf
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
from plot_chord_onset_timing import GRID, INK, INK_MUTED, SURFACE, style_axis  # noqa: E402

MODEL_COLORS = ["#2a78d6", "#eb6834"]
CLASSES = [("downbeat", [0]), ("beat 3", [8]), ("beats 2, 4", [4, 12]),
           ("and", [2, 6, 10, 14]), ("16ths", [1, 3, 5, 7, 9, 11, 13, 15])]


def load(spec: str) -> tuple[str, dict]:
    """'name=path[,path]' (files or directories of JSONs) -> (name, {model: bar_position rows})."""
    name, paths = spec.split("=", 1)
    files = []
    for p in paths.split(","):
        p = Path(p)
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    by_model, songs, dropped = {}, {}, None
    for f in files:
        d = json.loads(f.read_text())
        dropped = d["args"].get("songs_dropped", dropped)
        for r in d["results"]:
            by_model[r["model"]] = r["bar_position"]
            songs[r["model"]] = r["songs"]
    return name, {"rows": by_model, "songs": songs, "dropped": dropped}


def classes(rows: list) -> list:
    by_bucket = {r["bucket"]: r for r in rows}
    out = []
    for label, buckets in CLASSES:
        rs = [by_bucket[b] for b in buckets if b in by_bucket]
        n = sum(r["n_frames"] for r in rs)
        out.append({
            "label": label,
            "p": sum(r["mean_p_onset"] * r["n_frames"] for r in rs) / n,
            "eff": math.exp(sum(r["mean_entropy"] * r["n_frames"] for r in rs) / n),
            "gt": sum(r["gt_onset_rate"] * r["n_frames"] for r in rs) / n,
            "frames": n,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="name=path[,path] per corpus")
    ap.add_argument("--models", nargs=2, default=["GAPT", "GAPT 7 datasets"])
    ap.add_argument("--output", default="scripts/plotting/chord_onset_timing_corpora.pdf")
    args = ap.parse_args()

    corpora = [load(s) for s in args.inputs]
    table = {}
    print(f"{'corpus':22s} {'class':11s} {'real':>6s} | " +
          " | ".join(f"P {m[:14]:>14s}  eff {m[:14]:>14s}" for m in args.models))
    for name, c in corpora:
        missing = [m for m in args.models if m not in c["rows"]]
        if missing:
            raise SystemExit(f"{name}: models missing {missing}")
        table[name] = {m: classes(c["rows"][m]) for m in args.models}
        for i, (label, _) in enumerate(CLASSES):
            a = [table[name][m][i] for m in args.models]
            print(f"{name:22s} {label:11s} {100 * a[0]['gt']:5.1f}% | " +
                  " | ".join(f"P {x['p']:14.3f}  eff {x['eff']:14.1f}" for x in a))
        print(f"{'':22s} songs {c['songs']}, dropped {c['dropped']}")

    plt.rcParams["font.family"] = "sans-serif"
    n = len(corpora)
    fig, axes = plt.subplots(2, n, figsize=(2.15 * n + 0.6, 5.4), dpi=200, sharey="row",
                             gridspec_kw={"hspace": 0.32, "wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    xs = list(range(len(CLASSES)))
    w = 0.38
    for j, (name, c) in enumerate(corpora):
        ax_p, ax_e = axes[0][j], axes[1][j]
        for ax in (ax_p, ax_e):
            style_axis(ax)
        for k, m in enumerate(args.models):
            vals = table[name][m]
            off = (k - 0.5) * w
            ax_p.bar([x + off for x in xs], [v["p"] for v in vals], width=w, color=MODEL_COLORS[k],
                     edgecolor="white", linewidth=0.6, zorder=2)
            ax_e.bar([x + off for x in xs], [v["eff"] for v in vals], width=w, color=MODEL_COLORS[k],
                     edgecolor="white", linewidth=0.6, zorder=2)
        gt = [v["gt"] for v in table[name][args.models[0]]]
        ax_p.scatter(xs, gt, marker="D", s=22, color=INK, edgecolor="white", linewidth=0.8, zorder=3)
        songs = c["songs"][args.models[0]]
        ax_p.set_title(f"{name}\n{songs} songs", fontsize=9, color=INK, pad=6)
        ax_e.set_yscale("log")
        ax_p.set_ylim(0, 1.0)
        for ax in (ax_p, ax_e):
            ax.set_xticks(xs)
            ax.set_xticklabels([l for l, _ in CLASSES], rotation=45, ha="right", fontsize=7.5, color=INK_MUTED)
            ax.tick_params(axis="y", labelsize=7.5, colors=INK_MUTED)
    axes[0][0].set_ylabel("P(chord change)", fontsize=9, color=INK)
    axes[1][0].set_ylabel("effective chords\nif forced (log)", fontsize=9, color=INK)
    handles = [Patch(color=MODEL_COLORS[k], label=m) for k, m in enumerate(args.models)]
    handles.append(Line2D([], [], marker="D", linestyle="", color=INK, markeredgecolor="white",
                          markersize=5, label="real change rate"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor=SURFACE)
    print(f"Written {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
