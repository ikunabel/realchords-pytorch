#!/usr/bin/env python3
"""Summarize annotations.keys across Hooktheory cache-eligible songs.

Only songs that would appear in data/cache/hooktheory are included
(MELODY + HARMONY, no TEMPO_CHANGES). See hooktheory_cache_filter.py.

Mode names follow the usual diatonic step patterns (W = whole = 2 semitones,
H = half = 1 semitone), e.g. major/ionian = W W H W W W W.
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

# First six semitone steps between scale degrees 1..7 (degree 7 -> octave implied).
# Full octave patterns from Wikipedia / standard mode definitions:
#   ionian (major):     W W H W W W W  -> [2,2,1,2,2,2,2]
#   dorian:             W H W W W H W  -> [2,1,2,2,2,1,2]
#   phrygian:           H W W W H W W  -> [1,2,2,2,1,2,2]
#   lydian:             W W W H W W W  -> [2,2,2,1,2,2,2]
#   mixolydian:         W W H W W H W  -> [2,2,1,2,2,1,2]
#   aeolian (nat minor): W H W W H W W -> [2,1,2,2,1,2,2]
#   locrian:            H W W H W W W  -> [1,2,2,1,2,2,2]
MODE_BY_INTERVALS = {
    (2, 2, 1, 2, 2, 2): {
        "mode": "ionian",
        "common_name": "major",
        "step_pattern": "W W H W W W W",
    },
    (2, 1, 2, 2, 1, 2): {
        "mode": "aeolian",
        "common_name": "natural minor",
        "step_pattern": "W H W W H W W",
    },
    (2, 2, 1, 2, 2, 1): {
        "mode": "mixolydian",
        "common_name": "mixolydian",
        "step_pattern": "W W H W W H W",
    },
    (2, 1, 2, 2, 2, 1): {
        "mode": "dorian",
        "common_name": "dorian",
        "step_pattern": "W H W W W H W",
    },
    (2, 2, 2, 1, 2, 2): {
        "mode": "lydian",
        "common_name": "lydian",
        "step_pattern": "W W W H W W W",
    },
    (1, 2, 2, 2, 1, 2): {
        "mode": "phrygian",
        "common_name": "phrygian",
        "step_pattern": "H W W W H W W",
    },
    (1, 2, 2, 1, 2, 2): {
        "mode": "locrian",
        "common_name": "locrian",
        "step_pattern": "H W W H W W W",
    },
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


def classify_key(key_entry: dict) -> dict:
    intervals = tuple(key_entry["scale_degree_intervals"])
    tonic_pc = key_entry["tonic_pitch_class"]
    tonic_name = PITCH_CLASS_NAMES.get(tonic_pc, str(tonic_pc))
    mode_info = MODE_BY_INTERVALS.get(intervals)

    if mode_info is None:
        label = f"{tonic_name} (unknown mode)"
        return {
            "beat": key_entry.get("beat"),
            "tonic_pitch_class": tonic_pc,
            "tonic_name": tonic_name,
            "scale_degree_intervals": list(intervals),
            "mode": None,
            "common_name": "unknown",
            "step_pattern": None,
            "key_label": label,
        }

    label = f"{tonic_name} {mode_info['common_name']}"
    return {
        "beat": key_entry.get("beat"),
        "tonic_pitch_class": tonic_pc,
        "tonic_name": tonic_name,
        "scale_degree_intervals": list(intervals),
        **mode_info,
        "key_label": label,
    }


def build_overview(dataset: dict) -> dict:
    kept_songs = [v for v in dataset.values() if passes_hooktheory_cache_filter(v)]

    key_region_counts: Counter = Counter()
    mode_counts: Counter = Counter()
    tonic_counts: Counter = Counter()
    interval_pattern_counts: Counter = Counter()
    key_label_counts: Counter = Counter()
    keys_per_song: Counter = Counter()
    songs_with_key_changes = 0

    by_mode: dict[str, Counter] = defaultdict(Counter)
    unknown_patterns: Counter = Counter()
    examples_key_change: list[dict] = []

    for song in kept_songs:
        keys = (song.get("annotations") or {}).get("keys") or []
        keys_per_song[len(keys)] += 1
        if len(keys) > 1:
            songs_with_key_changes += 1

        classified = [classify_key(k) for k in keys]
        if len(classified) > 1 and len(examples_key_change) < 5:
            ht = song["hooktheory"]
            examples_key_change.append(
                {
                    "id": ht["id"],
                    "artist": ht["artist"],
                    "song": ht["song"],
                    "urls": ht.get("urls", {}),
                    "keys": classified,
                }
            )

        for entry in classified:
            key_region_counts["total_key_regions"] += 1
            intervals = tuple(entry["scale_degree_intervals"])
            interval_pattern_counts[intervals] += 1
            tonic_counts[entry["tonic_name"]] += 1
            key_label_counts[entry["key_label"]] += 1

            mode_name = entry["common_name"]
            mode_counts[mode_name] += 1
            by_mode[mode_name][entry["tonic_name"]] += 1
            if entry["mode"] is None:
                unknown_patterns[intervals] += 1

    mode_reference = {
        info["mode"]: {
            "scale_degree_intervals": list(intervals),
            **info,
        }
        for intervals, info in sorted(
            MODE_BY_INTERVALS.items(),
            key=lambda item: item[1]["mode"],
        )
    }

    return {
        "source": {
            "description": "Hooktheory annotations.keys for cache-eligible songs",
            "cache_filter": "MELODY + HARMONY tags, no TEMPO_CHANGES",
            "songs_included": len(kept_songs),
        },
        "mode_reference": mode_reference,
        "summary": {
            "total_key_regions": key_region_counts["total_key_regions"],
            "songs_with_single_key": keys_per_song.get(1, 0),
            "songs_with_key_changes": songs_with_key_changes,
            "unique_key_labels": len(key_label_counts),
            "unique_interval_patterns": len(interval_pattern_counts),
        },
        "counts_by_mode": dict(mode_counts.most_common()),
        "counts_by_tonic": dict(tonic_counts.most_common()),
        "counts_by_interval_pattern": {
            str(list(pattern)): {
                "count": count,
                **(
                    MODE_BY_INTERVALS[pattern]
                    if pattern in MODE_BY_INTERVALS
                    else {"mode": None, "common_name": "unknown", "step_pattern": None}
                ),
            }
            for pattern, count in interval_pattern_counts.most_common()
        },
        "counts_by_key_label": dict(key_label_counts.most_common(40)),
        "tonics_by_mode": {
            mode: dict(counter.most_common()) for mode, counter in sorted(by_mode.items())
        },
        "keys_per_song": {str(k): v for k, v in sorted(keys_per_song.items())},
        "unknown_interval_patterns": {
            str(list(pattern)): count for pattern, count in unknown_patterns.most_common()
        },
        "examples_songs_with_key_changes": examples_key_change,
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

    out_path = write_json("keys", "keys_overview.json", overview)
    summary = write_text(
        "keys",
        "run_summary.txt",
        f"Wrote {out_path}\n"
        f"Songs: {overview['source']['songs_included']}\n"
        f"Key regions: {overview['summary']['total_key_regions']}\n"
        f"Songs with key changes: {overview['summary']['songs_with_key_changes']}\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
