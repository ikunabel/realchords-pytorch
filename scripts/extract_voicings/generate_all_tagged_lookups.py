#!/usr/bin/env python3
"""Generate a chord_voicings.json for EVERY genre and EVERY performer tag,
plus one combined coverage-statistics report file, all in one pass.

Loads the (potentially huge) merged all_voicings.json and the chord
vocabulary exactly once, then reuses match_voicings_to_chords.build_chord_lookup
in-memory for every tag — avoids re-reading/re-parsing the source file once
per tag, which would be the dominant cost otherwise (the merged cache is
close to 1 GB).

Genre and performer tags are used EXACTLY as recorded by extract_voicings.py
(one file per distinct tag string) — no identity-merging across differently
spelled tags for the same real person (e.g. aria-midi's "jarrett" vs
PIJAMA's "keith jarrett" are two distinct tags here, each getting its own
file). Merge specific spellings yourself via match_voicings_to_chords.py's
--performers flag (which does accept several spellings at once) if you want
a unified view for one person.

Output
------
``<output_dir>/by_genre/chord_voicings_<slug>.json``       one per genre
``<output_dir>/by_performer/chord_voicings_<slug>.json``   one per performer
``<report_json>``  machine-readable stats for every generated file (+ the
                    overall "all genres, all performers" file, if present)
``<report_md>``     the same, as a human-readable table

Usage::

    python scripts/extract_voicings/generate_all_tagged_lookups.py \\
        [--voicings data/voicings/merged/all_voicings.json] \\
        [--chord_names data/cache/chord_names_augmented.json] \\
        [--output_dir data/voicings/merged] \\
        [--report_json data/voicings/coverage_report.json] \\
        [--report_md data/voicings/COVERAGE_REPORT.md] \\
        [--min_count_genre 3] [--min_count_performer 1]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_voicings_to_chords import build_chord_lookup, load_chord_vocab


def slugify(tag: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
    return slug or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voicings", default="data/voicings/merged/all_voicings.json")
    parser.add_argument("--chord_names", default="data/cache/chord_names_augmented.json")
    parser.add_argument("--output_dir", default="data/voicings/merged")
    parser.add_argument("--report_json", default="data/voicings/coverage_report.json")
    parser.add_argument("--report_md", default="data/voicings/COVERAGE_REPORT.md")
    parser.add_argument(
        "--min_count_genre", type=int, default=3,
        help="--min_count used for genre-level lookups (large samples; default 3).",
    )
    parser.add_argument(
        "--min_count_performer", type=int, default=1,
        help=(
            "--min_count used for performer-level lookups (default 1 — a single "
            "performer's sample is much smaller than a whole genre, so the default "
            "genre threshold of 3 would discard legitimate but rare voicings)."
        ),
    )
    parser.add_argument("--min_songs", type=int, default=1)
    parser.add_argument("--min_notes", type=int, default=3)
    parser.add_argument("--max_notes", type=int, default=8)
    parser.add_argument("--max_voicings_per_chord", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_path = Path(args.chord_names)
    output_dir = Path(args.output_dir)
    by_genre_dir = output_dir / "by_genre"
    by_performer_dir = output_dir / "by_performer"
    by_genre_dir.mkdir(parents=True, exist_ok=True)
    by_performer_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing chord vocabulary …")
    chord_masks_np, valid_names, n_total, n_failed = load_chord_vocab(vocab_path)
    vocab_size = len(valid_names)
    print(f"  Parsed {vocab_size}/{n_total} chord names ({n_failed} failed)")

    print(f"Loading voicings from {args.voicings} …")
    with open(args.voicings, encoding="utf-8") as f:
        all_voicings: List[dict] = json.load(f)
    print(f"  {len(all_voicings):,} raw (genre, performer, pitches) rows")

    genres = sorted({v.get("genre", "unknown") for v in all_voicings})
    performers = sorted({v.get("performer", "unknown") for v in all_voicings})
    print(f"  {len(genres)} distinct genres, {len(performers)} distinct performers")

    report_rows: List[dict] = []

    # -- Overall (no filter), if it already exists, include it in the report --
    overall_path = output_dir / "chord_voicings.json"
    if overall_path.exists():
        with open(overall_path, encoding="utf-8") as f:
            overall = json.load(f)
        total_entries = sum(len(v) for v in overall.values())
        total_occ = sum(e["count"] for v in overall.values() for e in v)
        report_rows.append({
            "tag_type": "all", "tag": "ALL", "path": str(overall_path),
            "distinct_chords": len(overall), "vocab_coverage_pct": round(100 * len(overall) / vocab_size, 1),
            "total_voicing_entries": total_entries, "total_occurrences": total_occ,
        })

    def run_tag(tag_type: str, tag: str, min_count: int, out_dir: Path) -> None:
        lookup, stats = build_chord_lookup(
            all_voicings, chord_masks_np, valid_names,
            genres=[tag] if tag_type == "genre" else None,
            performers=[tag] if tag_type == "performer" else None,
            min_count=min_count, min_songs=args.min_songs,
            min_notes=args.min_notes, max_notes=args.max_notes,
            max_voicings_per_chord=args.max_voicings_per_chord,
            show_progress=False,
        )
        out_path = out_dir / f"chord_voicings_{slugify(tag)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lookup, f, indent=2)
        report_rows.append({
            "tag_type": tag_type, "tag": tag, "path": str(out_path),
            "distinct_chords": stats["distinct_chords"],
            "vocab_coverage_pct": round(100 * stats["distinct_chords"] / vocab_size, 1),
            "total_voicing_entries": stats["total_voicing_entries"],
            "total_occurrences": stats["total_occurrences"],
        })

    print(f"\nGenerating {len(genres)} genre lookups …")
    for i, genre in enumerate(genres, 1):
        run_tag("genre", genre, args.min_count_genre, by_genre_dir)
        print(f"  [{i}/{len(genres)}] {genre}")

    print(f"\nGenerating {len(performers)} performer lookups …")
    for i, performer in enumerate(performers, 1):
        run_tag("performer", performer, args.min_count_performer, by_performer_dir)
        if i % 20 == 0 or i == len(performers):
            print(f"  [{i}/{len(performers)}] ...")

    # -- Write the combined report --
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_voicings": args.voicings,
        "chord_vocab_size": vocab_size,
        "params": {
            "min_count_genre": args.min_count_genre,
            "min_count_performer": args.min_count_performer,
            "min_songs": args.min_songs,
            "min_notes": args.min_notes,
            "max_notes": args.max_notes,
            "max_voicings_per_chord": args.max_voicings_per_chord,
        },
        "rows": sorted(report_rows, key=lambda r: (r["tag_type"], -r["distinct_chords"])),
    }
    report_json_path = Path(args.report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport JSON written to: {report_json_path}")

    # -- Human-readable Markdown version --
    lines = [
        "# Chord voicing coverage report",
        "",
        f"Generated: {report['generated_at']}",
        f"Source: `{args.voicings}`",
        f"Chord vocabulary size: {vocab_size}",
        "",
        "| Tag type | Tag | Chords covered | Vocab % | Voicing entries | Total occurrences | File |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['tag_type']} | {row['tag']} | {row['distinct_chords']} | "
            f"{row['vocab_coverage_pct']}% | {row['total_voicing_entries']:,} | "
            f"{row['total_occurrences']:,} | `{row['path']}` |"
        )
    report_md_path = Path(args.report_md)
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report Markdown written to: {report_md_path}")


if __name__ == "__main__":
    main()
