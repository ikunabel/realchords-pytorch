#!/usr/bin/env python3
"""Convert the Weimar Jazz Database (WJD) to the Hooktheory-compatible cache format.

Data source: ``data/wjazzd/wjazzd.db`` (SQLite).

Mapping to the 2-channel model:
  - harmony  ← ``beats.chord``      (chord changes at beat boundaries)
  - melody   ← ``beats.bass_pitch`` (walking bass, one MIDI note per beat)

The jazz solo (``melody`` table) is intentionally omitted: its sub-beat
swing/triplet timing cannot be losslessly represented on the 4-frames-per-beat
grid used by ``HooktheoryTokenizer``.

Only 4/4 solos are included; other time signatures are filtered out.
Pickup bars (``bar < 0``) are skipped.

Usage::

    python scripts/convert_data_to_cache/convert_wjd_to_cache.py [--wjd_db data/wjazzd/wjazzd.db]
                                            [--output_dir data/cache/wjd]
                                            [--augmentation]
                                            [--min_beats 16]
                                            [--report_only]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from realchords.constants import CACHE_DIR, CHORD_NAMES_AUG_PATH, ZERO_OCTAVE
from realchords.utils.data_utils import (
    transpose_chord,
    transpose_melody,
    update_global_chord_names,
)
from realchords.utils.io_utils import save_jsonl

# Reuse the note_seq chord parser and simplifier from the wikifonia converter
import sys

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from convert_wikifonia_to_cache import (
    parse_chord_symbol_with_noteseq,
    simplify_complex_chord,
    _is_no_chord_symbol,
)

BEATS_PER_BAR = 4  # 4/4 only
SEMITONES_PER_OCTAVE = 12
GLOBAL_CACHE_DIR = str(Path(CACHE_DIR))


# ---------------------------------------------------------------------------
# WJD chord-name normalization
# ---------------------------------------------------------------------------

def transform_wjd_chord_symbol(chord_symbol: str) -> str:
    """Normalize WJD jazz chord notation to a format note_seq can parse.

    WJD conventions that differ from note_seq / Wikifonia:
    - Root + ``-`` → minor  (``C-7`` = Cm7, NOT C♭7)
    - ``j7``        → major 7th (``Abj7`` = Ab maj7)
    - Numeric extensions concatenated (``A-79`` = Am9, ``A7911#`` = A7#11)
    - ``alt``       → altered dominant (``C7alt`` → ``C7alt``, let note_seq
                       handle or fall back to simplify)
    - ``NC``        → no chord (return empty string)
    """
    if not chord_symbol or _is_no_chord_symbol(chord_symbol) or chord_symbol.strip() == "NC":
        return ""

    chord = chord_symbol.strip()

    # Handle slash chords: normalize base and preserve bass note
    slash_part = ""
    if "/" in chord:
        base, bass = chord.split("/", 1)
        chord = base
        slash_part = "/" + bass.strip()

    # Extract root note (with optional accidental)
    root_match = re.match(r"^([A-G][#b]?)", chord)
    if not root_match:
        return ""
    root = root_match.group(1)
    rest = chord[len(root):]

    # WJD: '-' immediately after root = minor
    if rest.startswith("-"):
        rest = "m" + rest[1:]

    # 'j7' → 'maj7'
    rest = rest.replace("j7", "maj7")

    # Collapse concatenated numeric extensions to just the leading token.
    # Examples: 79 → 9, 79b → 7b9, 79# → 7#9, 7911# → 7#11, 7913b → 7b13,
    #           79b13 → 7b13, 6911 → 6/9, m79 → m9
    rest = _normalize_wjd_extensions(rest)

    return root + rest + slash_part


# Simple extension remappings (applied in order)
_EXT_RULES = [
    # Keep these as-is (note_seq handles them)
    (r"^(m?)(maj7)(#?11)?$", None),
    (r"^(m?)(7)(alt)$", None),
    # Compound numeric extensions → simplify to highest useful extension
    (r"79#", "7#9"),
    (r"79b", "7b9"),
    (r"7911#", "7#11"),
    (r"7913b", "7b13"),
    (r"79b13", "7b13"),
    (r"7913$", "13"),
    (r"7911$", "11"),
    (r"79$", "9"),
    (r"m79", "m9"),
    (r"m711", "m11"),
    (r"m713", "m13"),
    (r"6911", "9"),          # 6/9 chord → treat as 9
    (r"69$", "9"),
]


def _normalize_wjd_extensions(rest: str) -> str:
    for pattern, replacement in _EXT_RULES:
        if replacement is None:
            # keep-as-is rule — just return if it matches
            if re.fullmatch(pattern, rest):
                return rest
        else:
            rest = re.sub(pattern, replacement, rest)
    return rest


def parse_wjd_chord(chord_symbol: str) -> Optional[Tuple[int, List[int], int]]:
    """Parse a WJD chord string to (root_pitch_class, intervals, inversion).

    Returns None for no-chord symbols.
    """
    transformed = transform_wjd_chord_symbol(chord_symbol)
    if not transformed:
        return None
    root_pc, intervals, inversion = parse_chord_symbol_with_noteseq(
        transformed, chord_symbol_transform=lambda s: s
    )
    if not intervals:
        # note_seq couldn't parse even the simplified form → try again simplified
        simplified = simplify_complex_chord(transformed)
        root_pc, intervals, inversion = parse_chord_symbol_with_noteseq(
            simplified, chord_symbol_transform=lambda s: s
        )
    if not intervals:
        return None
    return root_pc, intervals, inversion


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _iter_solos_4_4(conn: sqlite3.Connection):
    """Yield (melid, performer, title, key) for all 4/4 solos."""
    rows = conn.execute(
        "SELECT melid, performer, title, key FROM solo_info WHERE signature = '4/4'"
    ).fetchall()
    for melid, performer, title, key in rows:
        yield melid, performer, title, key


def _get_beats(conn: sqlite3.Connection, melid: int) -> List[tuple]:
    """Return beat rows for *melid*, skipping pickup bars (bar < 0)."""
    return conn.execute(
        "SELECT bar, beat, chord, bass_pitch "
        "FROM beats WHERE melid = ? AND bar >= 0 "
        "ORDER BY bar, beat",
        (melid,),
    ).fetchall()


def _forward_fill_chords(beats: List[tuple]) -> List[tuple]:
    """Replace empty chord fields with the last seen chord name."""
    result = []
    current = None
    for bar, beat, chord, bass in beats:
        if chord and chord != "NC":
            current = chord
        result.append((bar, beat, current, bass))
    return result


def _beat_number(bar: int, beat: int) -> int:
    """Convert (bar, beat) to a 0-indexed beat position."""
    return bar * BEATS_PER_BAR + (beat - 1)


def _bass_midi_to_annotation(beat_pos: int, midi_pitch: int) -> Optional[Dict]:
    """Convert a bass MIDI pitch to a melody annotation dict."""
    if not midi_pitch:
        return None
    pitch_class = midi_pitch % SEMITONES_PER_OCTAVE
    octave = midi_pitch // SEMITONES_PER_OCTAVE - ZERO_OCTAVE // SEMITONES_PER_OCTAVE
    return {
        "onset": float(beat_pos),
        "offset": float(beat_pos + 1),
        "pitch_class": pitch_class,
        "octave": octave,
    }


# ---------------------------------------------------------------------------
# Solo → annotation dict
# ---------------------------------------------------------------------------

def solo_to_annotation(
    conn: sqlite3.Connection,
    melid: int,
    performer: str,
    title: str,
    key: str,
    min_beats: int = 16,
) -> Optional[Dict]:
    """Convert one WJD solo to an annotation dict.

    Returns None if the solo is too short or has no parseable chords.
    """
    beats = _get_beats(conn, melid)
    if not beats:
        return None

    beats = _forward_fill_chords(beats)
    total_beats = _beat_number(beats[-1][0], beats[-1][1]) + 1

    if total_beats < min_beats:
        return None

    # --- Harmony: chord segments ---
    harmony = []
    prev_chord_name = None
    prev_chord_parsed = None
    seg_start = None

    def _close_segment(end_beat):
        if prev_chord_parsed is not None and seg_start is not None:
            root_pc, intervals, inversion = prev_chord_parsed
            harmony.append({
                "onset": float(seg_start),
                "offset": float(end_beat),
                "root_pitch_class": root_pc,
                "root_position_intervals": intervals,
                "inversion": inversion,
            })

    for bar, beat, chord_name, _ in beats:
        bp = _beat_number(bar, beat)
        if chord_name != prev_chord_name:
            _close_segment(bp)
            if chord_name is not None:
                parsed = parse_wjd_chord(chord_name)
            else:
                parsed = None
            prev_chord_name = chord_name
            prev_chord_parsed = parsed
            seg_start = bp

    _close_segment(total_beats)

    if not harmony:
        return None

    # --- Melody: walking bass ---
    melody = []
    for bar, beat, _, bass_pitch in beats:
        bp = _beat_number(bar, beat)
        ann = _bass_midi_to_annotation(bp, bass_pitch)
        if ann is not None:
            melody.append(ann)

    if not melody:
        return None

    return {
        "wjazzd": {
            "melid": melid,
            "performer": performer,
            "title": title,
            "key": key,
        },
        "annotations": {
            "num_beats": total_beats,
            "melody": melody,
            "harmony": harmony,
        },
    }


# ---------------------------------------------------------------------------
# Augmentation (transposition)
# ---------------------------------------------------------------------------

def augment_item(item: Dict) -> List[Dict]:
    """Return 12 transpositions of *item* (semitones 0..11)."""
    results = []
    for semitone in range(12):
        if semitone == 0:
            results.append(item)
            continue
        new_item = {
            "wjazzd": item["wjazzd"],
            "annotations": {
                "num_beats": item["annotations"]["num_beats"],
                "melody": [transpose_melody(n, semitone) for n in item["annotations"]["melody"]],
                "harmony": [transpose_chord(c, semitone) for c in item["annotations"]["harmony"]],
            },
        }
        results.append(new_item)
    return results


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wjd_db",
        type=str,
        default="data/wjazzd/wjazzd.db",
        help="Path to the WJD SQLite database.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/wjd",
        help="Output directory for cache files.",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Include all 12 transpositions.",
    )
    parser.add_argument(
        "--min_beats",
        type=int,
        default=16,
        help="Minimum number of beats to include a solo (default: 16 = 4 bars).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Parse and print stats without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    conn = sqlite3.connect(args.wjd_db)
    solos = list(_iter_solos_4_4(conn))
    print(f"Found {len(solos)} 4/4 solos in WJD")

    from realchords.utils.data_utils import to_chord_name

    items: List[Dict] = []
    skipped = 0

    for melid, performer, title, key in tqdm(solos, desc="Converting solos"):
        item = solo_to_annotation(conn, melid, performer, title, key, args.min_beats)
        if item is None:
            skipped += 1
            continue
        items.append(item)

    print(f"Converted: {len(items)}, Skipped: {skipped}")

    if args.report_only:
        return

    def _collect_chord_names(item_list: List[Dict]) -> set:
        names: set = set()
        for it in item_list:
            for h in it["annotations"]["harmony"]:
                name = to_chord_name(
                    h["root_pitch_class"],
                    h["root_position_intervals"],
                    h.get("inversion"),
                )
                if name:
                    names.add(name)
        return names

    # Augmentation
    if args.augmentation:
        augmented: List[Dict] = []
        for item in tqdm(items, desc="Augmenting"):
            augmented.extend(augment_item(item))
        all_items = augmented
    else:
        all_items = items

    chord_names_set = _collect_chord_names(all_items)

    # Train / valid / test split (by original solo index so augmentations stay together)
    random.seed(args.seed)
    indices = list(range(len(items)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    split_map: Dict[int, str] = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            split_map[idx] = "train"
        elif i < n_train + n_valid:
            split_map[idx] = "valid"
        else:
            split_map[idx] = "test"

    splits: Dict[str, List] = {"train": [], "valid": [], "test": []}
    for orig_idx, item in enumerate(items):
        split = split_map[orig_idx]
        if args.augmentation:
            splits[split].extend(augment_item(item))
        else:
            splits[split].append(item)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = "_augmented" if args.augmentation else ""
    for split_name, split_items in splits.items():
        path = output_dir / f"{split_name}{postfix}.jsonl"
        save_jsonl(split_items, str(path))
        print(f"Wrote {len(split_items)} items to {path}")

    # Write dataset-local chord names
    local_path = output_dir / f"chord_names{postfix}.json"
    with open(local_path, "w") as f:
        json.dump(sorted(chord_names_set), f, indent=2)
    print(f"Wrote {len(chord_names_set)} chord names to {local_path}")

    # Update global chord names (both base and augmented files if not augmenting)
    update_global_chord_names(list(chord_names_set), GLOBAL_CACHE_DIR, augmented=args.augmentation)
    if not args.augmentation:
        update_global_chord_names(list(chord_names_set), GLOBAL_CACHE_DIR, augmented=True)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
