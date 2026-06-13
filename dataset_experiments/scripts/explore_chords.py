#!/usr/bin/env python3
"""Summarize harmony chords across Hooktheory cache-eligible songs.

Uses Hooktheory's native chord representation only:
  root_pitch_class + root_position_intervals + inversion

No chord-symbol conversion (to_chord_name). Only songs that would appear in
data/cache/hooktheory are included (MELODY + HARMONY, no TEMPO_CHANGES).
See hooktheory_cache_filter.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from _common import PROJECT_ROOT, write_json, write_text
from hooktheory_cache_filter import passes_hooktheory_cache_filter

PITCH_CLASS_NAMES = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


def open_hooktheory_json(path: Path):
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def resolve_hooktheory_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        PROJECT_ROOT / "data/hooktheory/Hooktheory.json.gz",
        PROJECT_ROOT / "data/hooktheory/Hooktheory.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Hooktheory.json(.gz) not found under data/hooktheory/")


def chord_key(event: dict) -> tuple[int, tuple[int, ...], int]:
    return (
        event["root_pitch_class"],
        tuple(event["root_position_intervals"]),
        event.get("inversion", 0),
    )


def pitch_count(intervals: tuple[int, ...]) -> int:
    """Number of pitch classes: root + one per interval."""
    return 1 + len(intervals)


def format_chord(root_pc: int, intervals: tuple[int, ...], inversion: int) -> str:
    root_name = PITCH_CLASS_NAMES.get(root_pc, str(root_pc))
    inv = f", inversion={inversion}" if inversion != 0 else ""
    return f"root={root_name} ({root_pc}), intervals={list(intervals)}{inv}"


def build_overview(dataset: dict) -> dict:
    kept_songs = [v for v in dataset.values() if passes_hooktheory_cache_filter(v)]

    chord_counts: Counter = Counter()
    interval_pattern_counts: Counter = Counter()
    root_pc_counts: Counter = Counter()
    inversion_counts: Counter = Counter()
    pitch_count_event_counts: Counter = Counter()
    songs_per_chord: dict[tuple[int, tuple[int, ...], int], set[str]] = defaultdict(set)
    songs_with_inversion = 0

    for song in kept_songs:
        song_id = song["hooktheory"]["id"]
        harmony = (song.get("annotations") or {}).get("harmony") or []
        song_has_inversion = False

        for event in harmony:
            key = chord_key(event)
            root_pc, intervals, inversion = key

            chord_counts[key] += 1
            interval_pattern_counts[intervals] += 1
            root_pc_counts[PITCH_CLASS_NAMES.get(root_pc, str(root_pc))] += 1
            inversion_counts[inversion] += 1
            pitch_count_event_counts[pitch_count(intervals)] += 1
            songs_per_chord[key].add(song_id)

            if inversion != 0:
                song_has_inversion = True

        if song_has_inversion:
            songs_with_inversion += 1

    total_events = sum(chord_counts.values())
    unique_pitch_counts = {pitch_count(intervals) for (_, intervals, _) in chord_counts}
    chords = []
    for (root_pc, intervals, inversion), count in chord_counts.most_common():
        n_pitches = pitch_count(intervals)
        chords.append(
            {
                "root_pitch_class": root_pc,
                "root_name": PITCH_CLASS_NAMES.get(root_pc, str(root_pc)),
                "root_position_intervals": list(intervals),
                "inversion": inversion,
                "is_inverted": inversion != 0,
                "pitch_count": n_pitches,
                "count": count,
                "songs_using_chord": len(songs_per_chord[(root_pc, intervals, inversion)]),
                "label": format_chord(root_pc, intervals, inversion),
            }
        )

    inverted_chords = [c for c in chords if c["is_inverted"]]

    return {
        "source": {
            "description": "Hooktheory annotations.harmony (structural representation)",
            "cache_filter": "MELODY + HARMONY tags, no TEMPO_CHANGES",
            "songs_included": len(kept_songs),
            "representation": "root_pitch_class + root_position_intervals + inversion",
        },
        "summary": {
            "total_harmony_events": total_events,
            "unique_chords": len(chord_counts),
            "unique_interval_patterns": len(interval_pattern_counts),
            "unique_roots": len(root_pc_counts),
            "inverted_events": sum(
                count for (_, _, inv), count in chord_counts.items() if inv != 0
            ),
            "root_position_events": sum(
                count for (_, _, inv), count in chord_counts.items() if inv == 0
            ),
            "songs_with_any_inverted_chord": songs_with_inversion,
            "min_pitches_per_chord": min(unique_pitch_counts),
            "max_pitches_per_chord": max(unique_pitch_counts),
        },
        "counts_by_pitch_count": {
            str(n): {
                "harmony_events": pitch_count_event_counts[n],
                "unique_chords": sum(
                    1
                    for (_, intervals, _), _ in chord_counts.items()
                    if pitch_count(intervals) == n
                ),
            }
            for n in sorted(pitch_count_event_counts)
        },
        "counts_by_inversion": {
            str(k): v for k, v in sorted(inversion_counts.items())
        },
        "counts_by_root": dict(root_pc_counts.most_common()),
        "counts_by_interval_pattern": {
            str(list(pattern)): count
            for pattern, count in interval_pattern_counts.most_common()
        },
        "chords": chords,
        "inverted_chords_only": inverted_chords,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to Hooktheory.json or Hooktheory.json.gz",
    )
    args = parser.parse_args()

    data_path = resolve_hooktheory_path(args.data_path)
    with open_hooktheory_json(data_path) as f:
        dataset = json.load(f)

    overview = build_overview(dataset)
    overview["source"]["data_path"] = str(data_path)

    out_path = write_json("chords", "chords_overview.json", overview)
    summary = write_text(
        "chords",
        "run_summary.txt",
        f"Wrote {out_path}\n"
        f"Songs: {overview['source']['songs_included']}\n"
        f"Harmony events: {overview['summary']['total_harmony_events']}\n"
        f"Unique chords (structural): {overview['summary']['unique_chords']}\n"
        f"Pitches per chord: {overview['summary']['min_pitches_per_chord']}"
        f"-{overview['summary']['max_pitches_per_chord']}\n"
        f"Inverted events: {overview['summary']['inverted_events']}\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
