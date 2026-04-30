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


@dataclass(frozen=True)
class VariantSpec:
    """Parsed from a ``--variant`` CLI argument.

    Plain form   ``gapt``                  → style_key="gapt",  system_key="gapt"
    Mapped form  ``gapt=wikifonia_..._gapt_paper_authentic``
                                           → style_key="gapt",  system_key="wikifonia_..."
    """

    style_key: str   # key into DEFAULT_STYLES (and label source)
    system_key: str  # key/suffix used to look up data in summary.json


def _parse_variant_arg(arg: str) -> VariantSpec:
    """Parse a single ``--variant`` argument into a :class:`VariantSpec`."""
    if "=" in arg:
        style_key, system_key = arg.split("=", 1)
        return VariantSpec(style_key=style_key.strip(), system_key=system_key.strip())
    return VariantSpec(style_key=arg, system_key=arg)


DEFAULT_STYLES: Dict[str, PointStyle] = {
    # Human-readable keys (used when style_key= is a label in --variant args)
    "Online MLE": PointStyle(label="Online MLE", marker="^", color="#1f77b4"),
    "Online MLE (3 datasets)": PointStyle(
        label="Online MLE (3 datasets)", marker="^", color="#4aa3df"
    ),
    "ReaLchords": PointStyle(label="ReaLchords", marker="s", color="#d62728"),
    "GAPT": PointStyle(label="GAPT", marker="*", color="#2ca02c"),
    # Technical-suffix aliases (backward compatibility / plain --variant form)
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
    variant_specs: Optional[Sequence[VariantSpec]] = None,
) -> List[Tuple[str, str, str, str, float, float]]:
    """Return (system_name, dataset, variant_suffix, style_key, vendi, harmony) rows.

    ``variant_specs`` is a list of :class:`VariantSpec`.  Each spec's ``system_key``
    is matched against either the variant suffix *or* the full system name in the
    summary; its ``style_key`` is carried through so the caller can look up the
    right :class:`PointStyle`.
    """
    # Build lookup: system_key (suffix or full name) → style_key
    key_to_style: Optional[Dict[str, str]] = None
    if variant_specs is not None:
        key_to_style = {}
        for spec in variant_specs:
            key_to_style[spec.system_key] = spec.style_key

    rows: List[Tuple[str, str, str, str, float, float]] = []
    for sys_name, payload in _iter_points(summary):
        vendi = payload.get("overall_vendi_score")
        harmony = payload.get("overall_note_in_chord_ratio")
        if vendi is None or harmony is None:
            continue
        sys_dataset, variant = _parse_dataset_and_variant(sys_name)
        if dataset is not None and sys_dataset != dataset:
            continue
        if key_to_style is not None:
            # Match by variant suffix first, then full system name.
            if variant in key_to_style:
                style_key = key_to_style[variant]
            elif sys_name in key_to_style:
                mapped = key_to_style[sys_name]
                # Plain spec (no '='): style_key == system_key == full name.
                # Prefer the short variant suffix so DEFAULT_STYLES is found.
                style_key = variant if mapped == sys_name else mapped
            else:
                continue
        else:
            style_key = variant
        rows.append((sys_name, sys_dataset, variant, style_key, float(vendi), float(harmony)))
    return rows


def plot_harmony_vs_diversity_one(
    *,
    summary_path: str | Path,
    dataset: Optional[str] = None,
    variant_specs: Sequence[VariantSpec],
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

    points = collect_system_metrics(summary, dataset=dataset, variant_specs=variant_specs)
    if not points:
        raise ValueError(
            f"No matching systems found for dataset={dataset!r} "
            f"and variants={[s.system_key for s in variant_specs]}."
        )

    datasets_present = {ds for _, ds, _, _, _, _ in points}
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
    for _, ds, _variant, style_key, x, y in points:
        style = styles.get(
            style_key,
            PointStyle(label=style_key, marker="o", color="#7f7f7f"),
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
            "Variant to plot. Two forms are accepted:\n"
            "  Plain:  '<system_key>'  — style is looked up by the same key.\n"
            "  Mapped: '<style_key>=<system_key>'  — use the style of <style_key>\n"
            "          for the data of <system_key>.  Example:\n"
            "          --variant gapt=wikifonia_melody_vs_gapt_paper_authentic\n"
            "Repeat for multiple variants."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path (prefer .pdf for vector).",
    )
    args = parser.parse_args(argv)

    variant_specs = [_parse_variant_arg(v) for v in args.variant]

    plot_harmony_vs_diversity_one(
        summary_path=args.summary,
        dataset=None,
        variant_specs=variant_specs,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()

