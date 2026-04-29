from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PointStyle:
    label: str
    marker: str
    color: str


DEFAULT_STYLES: Dict[str, PointStyle] = {
    "decoder_only_online_chord": PointStyle(
        label="Online MLE", marker="^", color="#1f77b4"
    ),
    "decoder_only_online_chord_3_datasets": PointStyle(
        label="Online MLE (3 datasets)", marker="^", color="#4aa3df"
    ),
    "realchords": PointStyle(label="ReaLchords", marker="s", color="#d62728"),
    "gapt": PointStyle(label="GAPT", marker="*", color="#2ca02c"),
}


def _load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_points(summary: dict) -> Iterable[Tuple[str, dict]]:
    systems = summary.get("systems", {})
    if not isinstance(systems, dict):
        raise ValueError("Expected summary['systems'] to be a dict.")
    for name, payload in systems.items():
        if isinstance(payload, dict):
            yield name, payload


def _parse_dataset_and_variant(system_name: str) -> Tuple[str, str]:
    # Expected name pattern from your scripts:
    #   "<dataset>_melody_vs_<variant>"
    if "_melody_vs_" not in system_name:
        return "unknown", system_name
    dataset, variant = system_name.split("_melody_vs_", 1)
    return dataset, variant


def collect_system_metrics(
    summary: dict,
    *,
    dataset: Optional[str] = None,
    variants: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str, str, float, float]]:
    """Return (system_name, dataset, variant, vendi, harmony) rows from a summary.json dict."""
    wanted = set(variants) if variants is not None else None
    rows: List[Tuple[str, str, str, float, float]] = []
    for sys_name, payload in _iter_points(summary):
        vendi = payload.get("overall_vendi_score")
        harmony = payload.get("overall_note_in_chord_ratio")
        if vendi is None or harmony is None:
            continue
        sys_dataset, variant = _parse_dataset_and_variant(sys_name)
        if dataset is not None and sys_dataset != dataset:
            continue
        if wanted is not None:
            # `--variant` can be either:
            # - the suffix after "<dataset>_melody_vs_" (e.g. "realchords"), OR
            # - the full system key (e.g. "hooktheory_melody_vs_realchords").
            if (variant not in wanted) and (sys_name not in wanted):
                continue
        rows.append((sys_name, sys_dataset, variant, float(vendi), float(harmony)))
    return rows


def plot_harmony_vs_diversity_one(
    *,
    summary_path: str | Path,
    dataset: Optional[str] = None,
    variants: Sequence[str],
    output_path: Optional[str | Path] = None,
    styles: Optional[Dict[str, PointStyle]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (5.2, 4.2),
    dpi: int = 200,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    x_frac_pad: float = 0.1,
    y_frac_pad: float = 0.1,
) -> "object":
    """
    One figure (one dataset) harmony-vs-diversity scatter:
      x = overall_vendi_score (diversity)
      y = overall_note_in_chord_ratio (harmony)
    """
    summary = _load_summary(Path(summary_path))
    styles = styles or DEFAULT_STYLES

    # Lazy import so the repo can be used without matplotlib in non-plot contexts.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    points = collect_system_metrics(summary, dataset=dataset, variants=variants)
    if not points:
        raise ValueError(
            f"No matching systems found for dataset={dataset!r} and variants={list(variants)}."
        )

    datasets_present = {ds for _, ds, _, _, _ in points}
    if dataset is None:
        title_default = " / ".join(sorted(datasets_present))
    else:
        title_default = dataset

    ax.set_title(title or title_default)
    ax.set_xlabel("Diversity (Vendi score)")
    ax.set_ylabel("Harmony (note-in-chord ratio)")
    ax.set_facecolor("#e3e3e3")
    ax.grid(True, color="white", alpha=0.9, linewidth=1.0)

    xs = [x for *_, x, __ in points]
    ys = [y for *_, __, y in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if xlim is None:
        ax.set_xlim(
            max(0.0, min_x * (1.0 - x_frac_pad)),
            max_x * (1.0 + x_frac_pad),
        )
    else:
        ax.set_xlim(*xlim)
    if ylim is None:
        ax.set_ylim(
            max(0.0, min_y * (1.0 - y_frac_pad)),
            min(1.0, max_y * (1.0 + y_frac_pad)),
        )
    else:
        ax.set_ylim(*ylim)

    # Harmony axis formatting: 0.38, 0.40, ... (avoid 0.375-style ticks)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    include_dataset_in_label = len(datasets_present) > 1
    for _, ds, variant, x, y in points:
        style = styles.get(
            variant,
            PointStyle(label=variant, marker="o", color="#7f7f7f"),
        )
        size = 180 if style.marker == "*" else 90
        ax.scatter(
            [x],
            [y],
            marker=style.marker,
            s=size,
            c=style.color,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f" {style.label} ({ds})"
            if include_dataset_in_label
            else f" {style.label}",
            fontsize=9,
            va="center",
            ha="left",
        )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot harmony vs diversity from evaluation summary.json."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("logs/eval/summary.json"),
        help="Path to summary.json produced by evaluate_generated_sequences.py",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help=(
            "Variant suffix after '<dataset>_melody_vs_'. Repeat for multiple. "
            "Also accepts full system keys, e.g. 'hooktheory_melody_vs_gapt'."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path (prefer .pdf for vector).",
    )
    args = parser.parse_args(argv)

    plot_harmony_vs_diversity_one(
        summary_path=args.summary,
        dataset=None,
        variants=args.variant,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()

