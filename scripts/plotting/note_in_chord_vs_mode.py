#!/usr/bin/env python3
"""Scatter plot: note-in-chord vs strict note-in-mode from eval summary.json."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SYSTEM_GROUPS: Dict[str, List[str]] = {
    "gt": [
        "hooktheory_gt",
        "wikifonia_gt",
        "nottingham_gt",
        "pop909_gt",
    ],
    "melody_vs_mle": [
        "hooktheory_melody_vs_mle_chord",
        "wikifonia_melody_vs_mle_chord",
        "pop909_melody_vs_mle_chord",
        "nottingham_melody_vs_mle_chord",
    ],
    "melody_vs_realchords": [
        "hooktheory_melody_vs_realchords_chord",
        "wikifonia_melody_vs_realchords_chord",
        "pop909_melody_vs_realchords_chord",
        "nottingham_melody_vs_realchords_chord",
    ],
    "melody_vs_gapt": [
        "hooktheory_melody_vs_gapt_chord",
        "wikifonia_melody_vs_gapt_chord",
        "pop909_melody_vs_gapt_chord",
        "nottingham_melody_vs_gapt_chord",
    ],
    "free_generation": [
        "mle_melody_vs_mle_chord_free_generation",
        "mle_melody_vs_realchords_chord_free_generation",
        "mle_melody_vs_gapt_chord_free_generation",
        "realchords_melody_vs_mle_chord_free_generation",
        "realchords_melody_vs_realchords_chord_free_generation",
        "realchords_melody_vs_gapt_chord_free_generation",
        "gapt_melody_vs_mle_chord_free_generation",
        "gapt_melody_vs_realchords_chord_free_generation",
        "gapt_melody_vs_gapt_chord_free_generation",
    ],
}

DATASET_GROUP_STYLES = {
    "gt": {"label": "GT", "color": "#333333", "marker": "o"},
    "melody_vs_mle": {"label": "GT melody / MLE chord", "color": "#1f77b4", "marker": "^"},
    "melody_vs_realchords": {
        "label": "GT melody / ReaLchords chord",
        "color": "#d62728",
        "marker": "s",
    },
    "melody_vs_gapt": {"label": "GT melody / GAPT chord", "color": "#2ca02c", "marker": "*"},
}

MELODY_MODEL_STYLES = {
    "mle": {"marker": "^", "short": "M", "label": "MLE melody"},
    "realchords": {"marker": "s", "short": "R", "label": "ReaLchords melody"},
    "gapt": {"marker": "*", "short": "G", "label": "GAPT melody"},
}

CHORD_MODEL_COLORS = {
    "mle": {"color": "#1f77b4", "short": "M", "label": "MLE chord"},
    "realchords": {"color": "#d62728", "short": "R", "label": "ReaLchords chord"},
    "gapt": {"color": "#2ca02c", "short": "G", "label": "GAPT chord"},
}

GROUP_STYLES = {
    **DATASET_GROUP_STYLES,
    "free_generation": {"label": "Free generation", "color": "#666666", "marker": "D"},
}

DEFAULT_DATASET_MELODY_GROUPS = (
    "gt",
    "melody_vs_mle",
    "melody_vs_realchords",
    "melody_vs_gapt",
)


@dataclass(frozen=True)
class MetricPoint:
    system_name: str
    group: str
    dataset: str
    note_in_chord: float
    mode_fit: float

    @property
    def gap(self) -> float:
        return self.mode_fit - self.note_in_chord


def _load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_free_generation_models(system_name: str) -> Optional[Tuple[str, str]]:
    if not system_name.endswith("_free_generation"):
        return None
    core = system_name[: -len("_free_generation")]
    if "_melody_vs_" not in core:
        return None
    melody, chord = core.split("_melody_vs_", 1)
    if chord.endswith("_chord"):
        chord = chord[: -len("_chord")]
    return melody, chord


def _free_generation_label(melody: str, chord: str) -> str:
    melody_short = MELODY_MODEL_STYLES[melody]["short"]
    chord_short = CHORD_MODEL_COLORS[chord]["short"]
    return f"{melody_short}/{chord_short}"


def _parse_dataset(system_name: str) -> str:
    if system_name.endswith("_gt"):
        return system_name[:-3]
    if "_melody_vs_" in system_name:
        return system_name.split("_melody_vs_", 1)[0]
    return system_name


def _dataset_label(dataset: str) -> str:
    return {
        "hooktheory": "Hooktheory",
        "wikifonia": "Wikifonia",
        "nottingham": "Nottingham",
        "pop909": "POP909",
    }.get(dataset, dataset)


def collect_points(
    summary: dict,
    groups: Sequence[str],
) -> List[MetricPoint]:
    systems = summary.get("systems", {})
    if not isinstance(systems, dict):
        raise ValueError("Expected summary['systems'] to be a dict.")

    points: List[MetricPoint] = []
    missing: List[str] = []
    for group in groups:
        if group not in SYSTEM_GROUPS:
            raise ValueError(
                f"Unknown group '{group}'. Expected one of: {', '.join(SYSTEM_GROUPS)}"
            )
        for system_name in SYSTEM_GROUPS[group]:
            payload = systems.get(system_name)
            if not isinstance(payload, dict):
                missing.append(system_name)
                continue
            note_in_chord = payload.get("overall_note_in_chord_ratio")
            mode_fit = payload.get("overall_mode_fit_ratio")
            if note_in_chord is None or mode_fit is None:
                missing.append(system_name)
                continue
            points.append(
                MetricPoint(
                    system_name=system_name,
                    group=group,
                    dataset=_parse_dataset(system_name),
                    note_in_chord=float(note_in_chord),
                    mode_fit=float(mode_fit),
                )
            )

    if missing:
        import warnings

        warnings.warn(
            "Missing note-in-chord or mode-fit metrics for: " + ", ".join(missing),
            stacklevel=2,
        )
    if not points:
        raise ValueError("No plottable systems found for the requested groups.")
    return points


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


def _group_stats(points: Iterable[MetricPoint]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[MetricPoint]] = {}
    for point in points:
        buckets.setdefault(point.group, []).append(point)

    stats: Dict[str, Dict[str, float]] = {}
    for group, group_points in buckets.items():
        gaps = [p.gap for p in group_points]
        stats[group] = {
            "count": float(len(group_points)),
            "mean_gap": sum(gaps) / len(gaps),
            "mean_note_in_chord": sum(p.note_in_chord for p in group_points)
            / len(group_points),
            "mean_mode_fit": sum(p.mode_fit for p in group_points) / len(group_points),
        }
    return stats


def _axis_limits(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _dataset_melody_axis_limits(summary: dict) -> Tuple[float, float]:
    points = collect_points(summary, DEFAULT_DATASET_MELODY_GROUPS)
    xs = [p.note_in_chord for p in points]
    ys = [p.mode_fit for p in points]
    return _axis_limits(xs, ys)


def _format_scatter_axes(
    ax,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    title: Optional[str],
    limits: Optional[Tuple[float, float]] = None,
) -> None:
    from matplotlib.ticker import FormatStrFormatter

    lo, hi = limits if limits is not None else _axis_limits(xs, ys)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#999999", linewidth=1.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Note-in-chord ratio")
    ax.set_ylabel("Note-in-mode ratio (strict)")
    ax.set_title(title or "Note-in-chord vs note-in-mode")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle=":", alpha=0.4)


def _free_generation_legend(ax) -> None:
    from matplotlib.lines import Line2D

    melody_handles = [
        Line2D(
            [0],
            [0],
            marker=MELODY_MODEL_STYLES[key]["marker"],
            color="w",
            markerfacecolor="#666666",
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            markersize=8,
            label=MELODY_MODEL_STYLES[key]["label"],
        )
        for key in ("mle", "realchords", "gapt")
    ]
    chord_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CHORD_MODEL_COLORS[key]["color"],
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            markersize=8,
            label=CHORD_MODEL_COLORS[key]["label"],
        )
        for key in ("mle", "realchords", "gapt")
    ]
    ax.legend(
        handles=melody_handles + chord_handles,
        loc="lower right",
        frameon=True,
        fontsize=8,
        title="Marker = melody, color = chord",
        title_fontsize=8,
    )


def _plot_free_generation_points(ax, points: Sequence[MetricPoint]) -> None:
    for point in points:
        models = _parse_free_generation_models(point.system_name)
        if models is None:
            continue
        melody, chord = models
        melody_style = MELODY_MODEL_STYLES[melody]
        chord_style = CHORD_MODEL_COLORS[chord]
        ax.scatter(
            [point.note_in_chord],
            [point.mode_fit],
            marker=melody_style["marker"],
            c=chord_style["color"],
            s=110 if melody_style["marker"] != "*" else 160,
            edgecolors="#222222",
            linewidths=0.7,
            zorder=4,
        )
        ax.annotate(
            _free_generation_label(melody, chord),
            (point.note_in_chord, point.mode_fit),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            color=chord_style["color"],
        )


def _print_plot_stats(
    *,
    points: Sequence[MetricPoint],
    groups: Sequence[str],
    scoring: str,
    pearson: float,
    spearman: float,
    gen_pearson: float,
) -> None:
    stats = _group_stats(points)
    print(f"Pearson r (all): {pearson:.3f}")
    print(f"Spearman ρ (all): {spearman:.3f}")
    if not math.isnan(gen_pearson):
        print(f"Pearson r (generated): {gen_pearson:.3f}")
    print(f"scoring={scoring}")
    for group in groups:
        if group not in stats:
            continue
        row = stats[group]
        print(
            f"{GROUP_STYLES[group]['label']}: "
            f"Δ={row['mean_gap']:.3f} "
            f"(mode {row['mean_mode_fit']:.3f}, chord {row['mean_note_in_chord']:.3f})"
        )


def plot_note_in_chord_vs_mode_free_generation(
    *,
    summary_path: Path,
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.2, 6.0),
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt

    summary = _load_summary(summary_path)
    groups = ("free_generation",)
    points = collect_points(summary, groups)
    scoring = summary.get("scoring", "unknown")
    axis_limits = _dataset_melody_axis_limits(summary)

    xs = [p.note_in_chord for p in points]
    ys = [p.mode_fit for p in points]
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_free_generation_points(ax, points)
    _format_scatter_axes(
        ax,
        xs,
        ys,
        title=title or "Free generation: note-in-chord vs note-in-mode",
        limits=axis_limits,
    )
    _free_generation_legend(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path.resolve()}")
    _print_plot_stats(
        points=points,
        groups=groups,
        scoring=scoring,
        pearson=pearson,
        spearman=spearman,
        gen_pearson=float("nan"),
    )


def plot_note_in_chord_vs_mode(
    *,
    summary_path: Path,
    groups: Sequence[str],
    output_path: Path,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (7.2, 6.0),
    dpi: int = 200,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if "free_generation" in groups:
        raise ValueError(
            "Use plot_note_in_chord_vs_mode_free_generation() or --preset free_generation "
            "for free-generation systems."
        )

    summary = _load_summary(summary_path)
    points = collect_points(summary, groups)
    scoring = summary.get("scoring", "unknown")

    xs = [p.note_in_chord for p in points]
    ys = [p.mode_fit for p in points]
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    generated = [p for p in points if p.group != "gt"]
    gen_xs = [p.note_in_chord for p in generated]
    gen_ys = [p.mode_fit for p in generated]
    gen_pearson = _pearson(gen_xs, gen_ys) if len(generated) >= 2 else float("nan")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for group in groups:
        style = DATASET_GROUP_STYLES[group]
        group_points = [p for p in points if p.group == group]
        ax.scatter(
            [p.note_in_chord for p in group_points],
            [p.mode_fit for p in group_points],
            label=style["label"],
            c=style["color"],
            marker=style["marker"],
            s=90 if style["marker"] != "*" else 140,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        for point in group_points:
            ax.annotate(
                _dataset_label(point.dataset),
                (point.note_in_chord, point.mode_fit),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color=style["color"],
            )

    _format_scatter_axes(
        ax,
        xs,
        ys,
        title=title or "GT melody + generated chord: note-in-chord vs note-in-mode",
    )
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=DATASET_GROUP_STYLES[group]["marker"],
            color="w",
            markerfacecolor=DATASET_GROUP_STYLES[group]["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8 if DATASET_GROUP_STYLES[group]["marker"] != "*" else 10,
            label=DATASET_GROUP_STYLES[group]["label"],
        )
        for group in groups
    ]
    ax.legend(
        handles=dataset_handles,
        loc="lower right",
        frameon=True,
        fontsize=8,
        title="GT melody + generated chord",
        title_fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path.resolve()}")
    _print_plot_stats(
        points=points,
        groups=groups,
        scoring=scoring,
        pearson=pearson,
        spearman=spearman,
        gen_pearson=gen_pearson,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("logs/eval/summary.json"),
        help="Path to evaluation summary JSON.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="System group to include (repeatable). Defaults to melody-vs-* + gt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/plotting/note_in_chord_vs_mode.pdf"),
        help="Output figure path (.pdf, .png, ...).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--preset",
        choices=("dataset_melody", "free_generation"),
        default="dataset_melody",
        help="Figure preset (default: dataset_melody).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "free_generation":
        plot_note_in_chord_vs_mode_free_generation(
            summary_path=args.summary,
            output_path=args.out,
            title=args.title,
        )
        return

    groups = args.group or list(DEFAULT_DATASET_MELODY_GROUPS)
    plot_note_in_chord_vs_mode(
        summary_path=args.summary,
        groups=groups,
        output_path=args.out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
