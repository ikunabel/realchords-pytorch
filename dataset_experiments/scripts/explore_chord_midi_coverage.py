#!/usr/bin/env python3
"""Map every cache-eligible chord through the stack's conversion pipeline.

Each unique Hooktheory harmony shape (root + intervals + inversion) is written
to a single JSON file with:
  hooktheory representation -> chord name -> token id -> chord pitches -> MIDI

Pipeline matches training tokenization + RealJam decode_chord_token.
Only cache-eligible songs (MELODY + HARMONY, no TEMPO_CHANGES).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from _common import PROJECT_ROOT, setup_imports, write_json, write_text
from hooktheory_cache_filter import passes_hooktheory_cache_filter

setup_imports()

import note_seq
import note_seq.chord_symbols_lib as chord_symbols_lib
from realchords.constants import BASS_OCTAVE, CHORD_OCTAVE
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.data_utils import to_chord_name

PIANO_MIDI_MIN = 21
PIANO_MIDI_MAX = 108
MIDDLE_C_MIDI = 60

PITCH_CLASS_NAMES = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
}


def open_hooktheory_json(path: Path):
    if str(path).endswith(".gz"):
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


def load_chord_names() -> list[str]:
    path = PROJECT_ROOT / "data/cache/hooktheory/chord_names.json"
    if not path.exists():
        raise FileNotFoundError(f"Chord names not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def structural_key(event: dict) -> tuple[int, tuple[int, ...], int]:
    return (
        event["root_pitch_class"],
        tuple(event["root_position_intervals"]),
        event.get("inversion", 0),
    )


def hooktheory_pitch_count(intervals: tuple[int, ...]) -> int:
    """Pitch classes in Hooktheory harmony: root + one per interval."""
    return 1 + len(intervals)


@lru_cache(maxsize=None)
def chord_name_from_structure(root_pc: int, intervals: tuple[int, ...]) -> str:
    return to_chord_name(root_pc, list(intervals))


@lru_cache(maxsize=None)
def symbol_to_pitches(chord_symbol: str) -> tuple[tuple[int, ...], int | None, tuple[int, ...]]:
    """Return (chord_pitch_classes, bass_pitch_class, midi_pitches)."""
    if not chord_symbol:
        return (), None, ()
    chord_pitch_classes = tuple(chord_symbols_lib.chord_symbol_pitches(chord_symbol))
    bass_pitch_class = chord_symbols_lib.chord_symbol_bass(chord_symbol)
    midi_pitches = tuple(
        sorted(
            {CHORD_OCTAVE * 12 + p for p in chord_pitch_classes}
            | {BASS_OCTAVE * 12 + bass_pitch_class}
        )
    )
    return chord_pitch_classes, bass_pitch_class, midi_pitches


def convert_chord(
    root_pc: int,
    intervals: tuple[int, ...],
    inversion: int,
    tokenizer: HooktheoryTokenizer,
) -> dict:
    hooktheory = {
        "root_pitch_class": root_pc,
        "root_name": PITCH_CLASS_NAMES.get(root_pc, str(root_pc)),
        "root_position_intervals": list(intervals),
        "inversion": inversion,
    }

    chord_name = chord_name_from_structure(root_pc, intervals)
    token_name_on = f"CHORD_ON_{chord_name}"
    token_name_hold = f"CHORD_{chord_name}"
    chord_token_id = tokenizer.name_to_id.get(token_name_on)
    chord_token_id_hold = tokenizer.name_to_id.get(token_name_hold)

    record = {
        "hooktheory": hooktheory,
        "hooktheory_pitch_count": hooktheory_pitch_count(intervals),
        "chord_name": chord_name,
        "chord_token_id": chord_token_id,
        "chord_token_id_hold": chord_token_id_hold,
        "chord_pitches": [],
        "bass_pitch_class": None,
        "midi_pitches": [],
        "in_token_vocab": chord_token_id is not None,
    }

    if chord_token_id is None:
        record["error"] = "chord_name not in token vocabulary"
        return record

    try:
        pitch_classes, bass_pc, midi_pitches = symbol_to_pitches(chord_name)
        record["chord_pitches"] = list(pitch_classes)
        record["bass_pitch_class"] = bass_pc
        record["midi_pitches"] = list(midi_pitches)
    except (note_seq.ChordSymbolError, TypeError) as exc:
        record["error"] = str(exc)

    return record


def collect_structural_chords(dataset: dict) -> dict[tuple[int, tuple[int, ...], int], int]:
    counts: Counter = Counter()
    kept = [v for v in dataset.values() if passes_hooktheory_cache_filter(v)]
    for song in tqdm(kept, desc="Scanning songs"):
        for event in (song.get("annotations") or {}).get("harmony") or []:
            counts[structural_key(event)] += 1
    return dict(counts)


def build_output(
    structural_counts: dict[tuple[int, tuple[int, ...], int], int],
    tokenizer: HooktheoryTokenizer,
) -> dict:
    chords = []
    midi_pitch_unique: Counter = Counter()
    midi_pitch_weighted: Counter = Counter()
    unique_midi_patterns: set[tuple[int, ...]] = set()
    missing_vocab = 0
    decode_errors = 0
    hooktheory_pitch_counts: set[int] = set()

    items = sorted(structural_counts.items(), key=lambda kv: -kv[1])
    for key, count in tqdm(items, desc="Converting chords"):
        root_pc, intervals, inversion = key
        record = convert_chord(root_pc, intervals, inversion, tokenizer)
        record["occurrence_count"] = count
        chords.append(record)
        hooktheory_pitch_counts.add(record["hooktheory_pitch_count"])

        if not record.get("in_token_vocab"):
            missing_vocab += 1
            continue
        if record.get("error"):
            decode_errors += 1
            continue
        unique_midi_patterns.add(tuple(record["midi_pitches"]))
        for pitch in record["midi_pitches"]:
            midi_pitch_unique[pitch] += 1
            midi_pitch_weighted[pitch] += count

    piano_keys = list(range(PIANO_MIDI_MIN, PIANO_MIDI_MAX + 1))
    covered_on_piano = sorted(
        p for p in midi_pitch_unique if PIANO_MIDI_MIN <= p <= PIANO_MIDI_MAX
    )
    uncovered_on_piano = [p for p in piano_keys if p not in midi_pitch_unique]

    return {
        "source": {
            "cache_filter": "MELODY + HARMONY tags, no TEMPO_CHANGES",
            "chord_names_path": str(
                PROJECT_ROOT / "data/cache/hooktheory/chord_names.json"
            ),
            "pipeline": [
                "hooktheory: root_pitch_class + root_position_intervals + inversion",
                "hooktheory_pitch_count: 1 + len(root_position_intervals)",
                "chord_name: to_chord_name (no inversion, as chord_to_frames)",
                "chord_token_id: HooktheoryTokenizer CHORD_ON_{name}",
                "chord_pitches: chord_symbols_lib.chord_symbol_pitches (note_seq expansion)",
                "midi_pitches: CHORD_OCTAVE=4, BASS_OCTAVE=3 (RealJam)",
            ],
        },
        "summary": {
            "unique_structural_chords": len(chords),
            "unique_midi_patterns": len(unique_midi_patterns),
            "total_harmony_events": sum(structural_counts.values()),
            "missing_from_token_vocab": missing_vocab,
            "decode_errors": decode_errors,
            "min_hooktheory_pitch_count": min(hooktheory_pitch_counts),
            "max_hooktheory_pitch_count": max(hooktheory_pitch_counts),
            "piano_keys_covered": len(covered_on_piano),
            "piano_keys_total": len(piano_keys),
            "piano_coverage_fraction": len(covered_on_piano) / len(piano_keys),
            "midi_pitch_min": min(midi_pitch_unique) if midi_pitch_unique else None,
            "midi_pitch_max": max(midi_pitch_unique) if midi_pitch_unique else None,
            "all_midi_below_middle_c": all(p < MIDDLE_C_MIDI for p in midi_pitch_unique)
            if midi_pitch_unique
            else None,
        },
        "piano_key_coverage": {
            "covered_midi_pitches": covered_on_piano,
            "uncovered_midi_pitches": uncovered_on_piano,
        },
        "chords": chords,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()

    data_path = resolve_hooktheory_path(args.data_path)
    with open_hooktheory_json(data_path) as f:
        dataset = json.load(f)

    tokenizer = HooktheoryTokenizer(chord_names=load_chord_names())
    structural_counts = collect_structural_chords(dataset)
    output = build_output(structural_counts, tokenizer)
    output["source"]["data_path"] = str(data_path)

    out_path = write_json("chord_midi_coverage", "chord_midi_coverage.json", output)
    summary = write_text(
        "chord_midi_coverage",
        "run_summary.txt",
        f"Wrote {out_path}\n"
        f"Structural chords: {output['summary']['unique_structural_chords']}\n"
        f"Unique MIDI patterns: {output['summary']['unique_midi_patterns']}\n"
        f"Piano coverage: {output['summary']['piano_keys_covered']}/"
        f"{output['summary']['piano_keys_total']}\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
