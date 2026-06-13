#!/usr/bin/env python3
"""Summarize Hookpad-native chords in Hooktheory_Raw.json.

Mirrors explore_chords.py but uses the raw Hookpad export (json.chords) instead of
processed annotations.harmony (root_pitch_class + root_position_intervals).

Only songs that would appear in data/cache/hooktheory are included (MELODY + HARMONY,
no TEMPO_CHANGES), matched by hooktheory song id / raw hash.

The raw→processed conversion (Hooktheory.json.gz) was done externally (Sheet Sage /
Hooktheory release); this repo does not ship that script. This overview documents
what the raw format contains for voicing reconstruction.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from _common import PROJECT_ROOT, write_json, write_text
from hooktheory_cache_filter import passes_hooktheory_cache_filter

RAW_CHORD_FIELDS = (
    "root",
    "beat",
    "duration",
    "type",
    "inversion",
    "applied",
    "adds",
    "omits",
    "alterations",
    "suspensions",
    "pedal",
    "alternate",
    "borrowed",
    "isRest",
    "recordingEndBeat",
)


def open_hooktheory_json(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def resolve_processed_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for candidate in (
        PROJECT_ROOT / "data/hooktheory/Hooktheory.json.gz",
        PROJECT_ROOT / "data/hooktheory/Hooktheory.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Hooktheory.json(.gz) not found under data/hooktheory/")


def resolve_raw_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return PROJECT_ROOT / "data/hooktheory/Hooktheory_Raw.json"


def freeze(value):
    """Convert nested lists to tuples for use as dict keys."""
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def raw_chord_key(chord: dict) -> tuple:
    """Hashable key for a Hookpad chord event (symbolic, not voiced MIDI)."""
    return (
        chord.get("root"),
        chord.get("type"),
        chord.get("inversion", 0),
        chord.get("applied", 0),
        freeze(chord.get("adds") or []),
        freeze(chord.get("omits") or []),
        freeze(chord.get("alterations") or []),
        freeze(chord.get("suspensions") or []),
        freeze(chord.get("pedal")),
        freeze(chord.get("borrowed")),
        chord.get("alternate") or "",
    )


def format_raw_chord(key: tuple) -> str:
    (
        root,
        ctype,
        inversion,
        applied,
        adds,
        omits,
        alterations,
        suspensions,
        pedal,
        borrowed,
        alternate,
    ) = key
    parts = [f"root={root}", f"type={ctype}", f"inversion={inversion}"]
    if applied:
        parts.append(f"applied={applied}")
    if adds:
        parts.append(f"adds={list(adds)}")
    if omits:
        parts.append(f"omits={list(omits)}")
    if alterations:
        parts.append(f"alterations={list(alterations)}")
    if suspensions:
        parts.append(f"suspensions={list(suspensions)}")
    if pedal is not None:
        parts.append(f"pedal={pedal}")
    if borrowed is not None:
        parts.append(f"borrowed={borrowed}")
    if alternate:
        parts.append(f"alternate={alternate!r}")
    return ", ".join(parts)


def extract_raw_chords(raw_song: dict) -> list[dict] | None:
    """Return active chord events, or None if raw has no chord list."""
    json_block = raw_song.get("json") or {}
    chords = json_block.get("chords")
    if chords is None:
        return None
    return [c for c in chords if not c.get("isRest")]


def band_settings_summary(raw_song: dict) -> dict | None:
    json_block = raw_song.get("json") or {}
    bands = json_block.get("bands") or json_block.get("bandGui")
    if not bands:
        return None
    harmony_specs = []
    bass_specs = []
    for band in bands if isinstance(bands, list) else []:
        for entry in band.get("harmony") or []:
            harmony_specs.append(entry.get("specification"))
        for entry in band.get("bass") or []:
            bass_specs.append(entry.get("specification"))
    if not harmony_specs and not bass_specs:
        return None
    return {"harmony_instruments": harmony_specs, "bass_instruments": bass_specs}


def build_overview(processed: dict, raw_data: dict) -> dict:
    eligible = {
        song_id: song
        for song_id, song in processed.items()
        if passes_hooktheory_cache_filter(song)
    }

    chord_counts: Counter = Counter()
    type_counts: Counter = Counter()
    inversion_counts: Counter = Counter()
    extension_flags: Counter = Counter()
    songs_per_chord: dict[tuple, set[str]] = defaultdict(set)

    songs_in_raw = 0
    songs_missing_raw = 0
    songs_null_chords = 0
    songs_empty_chords = 0
    songs_with_band = 0
    raw_events = 0
    processed_harmony_events = 0
    event_count_pairs: list[tuple[int, int]] = []

    for song_id, proc_song in eligible.items():
        harmony = (proc_song.get("annotations") or {}).get("harmony") or []
        processed_harmony_events += len(harmony)

        raw_song = raw_data.get(song_id)
        if raw_song is None:
            songs_missing_raw += 1
            continue

        songs_in_raw += 1
        raw_chords = extract_raw_chords(raw_song)
        if raw_chords is None:
            songs_null_chords += 1
            continue
        if not raw_chords:
            songs_empty_chords += 1
            continue

        if band_settings_summary(raw_song):
            songs_with_band += 1

        event_count_pairs.append((len(raw_chords), len(harmony)))

        for chord in raw_chords:
            key = raw_chord_key(chord)
            chord_counts[key] += 1
            type_counts[chord.get("type")] += 1
            inversion_counts[chord.get("inversion", 0)] += 1
            songs_per_chord[key].add(song_id)
            raw_events += 1

            if chord.get("applied", 0):
                extension_flags["applied"] += 1
            if chord.get("adds"):
                extension_flags["adds"] += 1
            if chord.get("omits"):
                extension_flags["omits"] += 1
            if chord.get("alterations"):
                extension_flags["alterations"] += 1
            if chord.get("suspensions"):
                extension_flags["suspensions"] += 1
            if chord.get("pedal") is not None:
                extension_flags["pedal"] += 1
            if chord.get("borrowed") is not None:
                extension_flags["borrowed"] += 1
            if chord.get("alternate"):
                extension_flags["alternate"] += 1

    chords = []
    for key, count in chord_counts.most_common():
        chords.append(
            {
                "root": key[0],
                "type": key[1],
                "inversion": key[2],
                "applied": key[3],
                "adds": list(key[4]) if key[4] else [],
                "omits": list(key[5]) if key[5] else [],
                "alterations": list(key[6]) if key[6] else [],
                "suspensions": list(key[7]) if key[7] else [],
                "pedal": key[8],
                "borrowed": key[9],
                "alternate": key[10],
                "count": count,
                "songs_using_chord": len(songs_per_chord[key]),
                "label": format_raw_chord(key),
            }
        )

    inverted = [c for c in chords if c["inversion"] != 0]

    mismatch_songs = sum(1 for raw_n, proc_n in event_count_pairs if raw_n != proc_n)

    return {
        "source": {
            "description": "Hookpad json.chords from Hooktheory_Raw.json",
            "cache_filter": "MELODY + HARMONY tags, no TEMPO_CHANGES",
            "processed_dataset": "Hooktheory.json.gz annotations.harmony is a separate conversion",
            "conversion_script_in_repo": False,
            "songs_eligible_in_processed": len(eligible),
            "representation": (
                "root (scale degree 1-7 in key) + type (Hookpad enum) + inversion + extensions"
            ),
        },
        "voicing_reconstruction": {
            "in_raw_chord_objects": [
                "root (diatonic scale degree, needs key to map to pitch class)",
                "type (Hookpad chord quality enum)",
                "inversion (0=root position, 1/2/3=figured-bass inversion index)",
                "applied, adds, omits, alterations, suspensions, borrowed, pedal, alternate",
                "beat, duration (timing only)",
            ],
            "not_in_raw_chord_objects": [
                "Per-chord MIDI pitches or octave placement",
                "Open vs closed voicing / note spacing across octaves",
                "Hookpad Band octave-centering or voicing style per chord",
                "Explicit bass pitch separate from inversion (inversion implies bass tone)",
            ],
            "related_elsewhere_in_raw_json": [
                "json.keys: tonic + scale for resolving scale degree → pitch class",
                "json.notes: melody only (sd + octave), not chord voice leading",
                "json.bands[].harmony/bass: instrument style names (playback preset), not note lists",
            ],
            "processed_harmony_adds": [
                "root_pitch_class + root_position_intervals: stacked-interval compression",
                "Same inversion field preserved from raw",
                "Still no register/voicing — only symbolic harmony",
            ],
            "theorytab_playback": (
                "Hookpad voicing on the website is rendered by the playback engine from "
                "symbol + inversion + band settings; that rendered note list is not stored in "
                "Hooktheory_Raw.json or Hooktheory.json.gz."
            ),
        },
        "summary": {
            "songs_with_raw_entry": songs_in_raw,
            "songs_missing_from_raw": songs_missing_raw,
            "songs_with_null_chords_field": songs_null_chords,
            "songs_with_empty_chord_list": songs_empty_chords,
            "songs_with_band_harmony_specs": songs_with_band,
            "total_raw_chord_events": raw_events,
            "total_processed_harmony_events": processed_harmony_events,
            "songs_raw_vs_processed_event_mismatch": mismatch_songs,
            "unique_raw_chord_shapes": len(chord_counts),
            "unique_type_values": len(type_counts),
            "inverted_events": sum(
                count for key, count in chord_counts.items() if key[2] != 0
            ),
            "min_inversion": min(inversion_counts) if inversion_counts else None,
            "max_inversion": max(inversion_counts) if inversion_counts else None,
        },
        "counts_by_type": {
            str(k): v for k, v in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "counts_by_inversion": {
            str(k): v for k, v in sorted(inversion_counts.items())
        },
        "extension_field_usage": dict(extension_flags.most_common()),
        "chords": chords,
        "inverted_chords_only": inverted,
        "examples_high_inversion": [
            c for c in chords if c["inversion"] >= 2
        ][:20],
        "examples_with_extensions": [
            c
            for c in chords
            if c["adds"] or c["omits"] or c["alterations"] or c["suspensions"]
            or c["applied"] or c["borrowed"] is not None or c["pedal"] is not None
        ][:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-path", type=Path, default=None)
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=None,
        help="Path to Hooktheory_Raw.json",
    )
    args = parser.parse_args()

    processed_path = resolve_processed_path(args.processed_path)
    raw_path = resolve_raw_path(args.raw_json)

    if not raw_path.exists():
        msg = f"Raw JSON not found: {raw_path}\n"
        write_text("raw_chords", "run_summary.txt", msg)
        print(msg)
        return

    with open_hooktheory_json(processed_path) as f:
        processed = json.load(f)

    with raw_path.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    overview = build_overview(processed, raw_data)
    overview["source"]["processed_path"] = str(processed_path)
    overview["source"]["raw_path"] = str(raw_path)
    overview["source"]["raw_chord_field_names"] = list(RAW_CHORD_FIELDS)

    out_path = write_json("raw_chords", "raw_chords_overview.json", overview)
    summary = write_text(
        "raw_chords",
        "run_summary.txt",
        f"Wrote {out_path}\n"
        f"Eligible songs: {overview['source']['songs_eligible_in_processed']}\n"
        f"In raw: {overview['summary']['songs_with_raw_entry']}\n"
        f"Raw chord events: {overview['summary']['total_raw_chord_events']}\n"
        f"Processed harmony events: {overview['summary']['total_processed_harmony_events']}\n"
        f"Unique raw shapes: {overview['summary']['unique_raw_chord_shapes']}\n"
        f"Inverted raw events: {overview['summary']['inverted_events']}\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
