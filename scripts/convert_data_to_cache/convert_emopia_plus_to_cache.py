#!/usr/bin/env python3
"""Convert EMOPIA+ MIDI files to Hooktheory-compatible cache format.

Source: EMOPIA+ (`data/EMOPIA+/EMOPIA+/midis/*.mid`, 1071 clips). Each MIDI
file has separate ``Melody``, ``Bass``, ``Chord`` (and usually ``Texture``)
tracks -- a symbolic, already beat-quantized multi-track format, not audio-
aligned like POP909's MIDI. ``Texture`` (an accompaniment/arpeggiation layer)
is intentionally unused, the same way ``convert_wjd_to_cache.py`` deliberately
omits WJD's jazz solo track: it doesn't fit the melody+harmony cache schema.

Timing: reads each note's position via ``PrettyMIDI.time_to_tick()`` /
``resolution`` rather than reconstructing beats from seconds+tempo (POP909's
approach) -- this file's tempo map is internally consistent, so tick/
resolution round-trips losslessly back to the exact quarter-note grid
position the symbolic data was authored on.

Harmony: the ``Chord`` track alone is not a reliable root/bass source -- its
own lowest note matches the concurrently-sounding ``Bass`` note only ~50% of
the time (verified empirically), so chord tones are combined with the active
Bass note before resolving root/intervals/inversion via
``note_seq.chord_symbols_lib.pitches_to_chord_symbol`` (which already knows
how to test the bass pitch as a candidate root and fall back to a slash-chord
when the bass isn't the root) and the existing
``parse_chord_symbol_with_noteseq`` (from ``convert_wikifonia_to_cache.py``).

Splits: uses EMOPIA+'s own clip-level split CSVs
(`data/EMOPIA+/EMOPIA+/split/{train,val,test}_clip.csv`) rather than deriving
a new random split.

Usage::

    python scripts/convert_data_to_cache/convert_emopia_plus_to_cache.py
    python scripts/convert_data_to_cache/convert_emopia_plus_to_cache.py --report_only
    python scripts/convert_data_to_cache/convert_emopia_plus_to_cache.py --max_files 20
    python scripts/convert_data_to_cache/convert_emopia_plus_to_cache.py --augmentation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import note_seq.chord_symbols_lib as chord_symbols_lib
import pretty_midi
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_cocopops_to_cache import _midi_to_hooktheory
from convert_wikifonia_to_cache import (
    collect_chord_names,
    create_augmented_dataset,
    filter_zero_duration_chords,
    parse_chord_symbol_with_noteseq,
    quantize_timing_to_beat_grid,
    resolve_melody_overlaps,
    set_chord_symbol_parse_verbose,
)
from realchords.utils.data_utils import update_global_chord_names
from realchords.utils.io_utils import save_jsonl

_MELODY_TRACK = "Melody"
_CHORD_TRACK = "Chord"
_BASS_TRACK = "Bass"

_CLIP_NAME_RE = re.compile(r"^(Q[1-4])_([^_]+)_(\d+)$")


def _default_data_path() -> Path:
    return Path("data/EMOPIA+/EMOPIA+")


def discover_midi_files(data_path: Path) -> List[Path]:
    return sorted((data_path / "midis").glob("*.mid"))


def _load_split_map(data_path: Path) -> Dict[str, str]:
    """Map clip filename (e.g. "Q3_xxx_1.mid") -> split ("TRAIN"/"VALID"/"TEST")."""
    split_files = {
        "TRAIN": "train_clip.csv",
        "VALID": "val_clip.csv",
        "TEST": "test_clip.csv",
    }
    mapping: Dict[str, str] = {}
    for split_name, filename in split_files.items():
        csv_path = data_path / "split" / filename
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mapping[row["clip_name"]] = split_name
    return mapping


def _identity_chord_symbol(symbol: str) -> str:
    """No-op transform: symbols already come from note_seq's own vocabulary."""
    return symbol


def _find_instrument(
    midi: pretty_midi.PrettyMIDI, name: str
) -> Optional[pretty_midi.Instrument]:
    for instrument in midi.instruments:
        if instrument.name == name:
            return instrument
    return None


def _note_beat_bounds(
    midi: pretty_midi.PrettyMIDI, note: pretty_midi.Note
) -> Tuple[float, float]:
    resolution = midi.resolution
    onset = midi.time_to_tick(note.start) / resolution
    offset = midi.time_to_tick(note.end) / resolution
    return onset, offset


def extract_melody(midi: pretty_midi.PrettyMIDI) -> List[Dict]:
    instrument = _find_instrument(midi, _MELODY_TRACK)
    if instrument is None:
        return []

    melody: List[Dict] = []
    for note in instrument.notes:
        onset, offset = _note_beat_bounds(midi, note)
        if offset <= onset:
            continue
        pitch_class, octave = _midi_to_hooktheory(note.pitch)
        melody.append(
            {"onset": onset, "offset": offset, "pitch_class": pitch_class, "octave": octave}
        )
    return melody


def extract_harmony(midi: pretty_midi.PrettyMIDI) -> List[Dict]:
    """Reconstruct chord segments from the Chord + Bass tracks."""
    chord_instr = _find_instrument(midi, _CHORD_TRACK)
    if chord_instr is None or not chord_instr.notes:
        return []
    bass_instr = _find_instrument(midi, _BASS_TRACK)
    bass_notes = sorted(bass_instr.notes, key=lambda n: n.start) if bass_instr else []

    resolution = midi.resolution
    groups: Dict[int, List[int]] = defaultdict(list)
    for note in chord_instr.notes:
        groups[midi.time_to_tick(note.start)].append(note.pitch)

    def _active_bass_pitch(onset_time: float) -> Optional[int]:
        active = None
        for bass_note in bass_notes:
            if bass_note.start <= onset_time + 1e-6:
                active = bass_note
            else:
                break
        return active.pitch if active is not None else None

    harmony: List[Dict] = []
    sorted_onset_ticks = sorted(groups)
    for onset_tick in sorted_onset_ticks:
        pitches = groups[onset_tick]
        onset_time = midi.tick_to_time(onset_tick)
        bass_pitch = _active_bass_pitch(onset_time)
        combined = ([bass_pitch] if bass_pitch is not None else []) + pitches

        try:
            symbol = chord_symbols_lib.pitches_to_chord_symbol(combined)
        except chord_symbols_lib.ChordSymbolError:
            continue

        root_pc, intervals, inversion = parse_chord_symbol_with_noteseq(
            symbol, chord_symbol_transform=_identity_chord_symbol,
        )
        if not intervals:
            continue

        harmony.append(
            {
                "onset": onset_tick / resolution,
                "offset": onset_tick / resolution,  # filled in below
                "root_pitch_class": root_pc,
                "root_position_intervals": intervals,
                "inversion": inversion,
            }
        )

    # Block-chord convention: each segment runs until the next chord's onset.
    for i in range(len(harmony) - 1):
        harmony[i]["offset"] = harmony[i + 1]["onset"]
    if harmony:
        last_onset_tick = sorted_onset_ticks[-1]
        last_end_tick = max(
            midi.time_to_tick(note.end)
            for note in chord_instr.notes
            if midi.time_to_tick(note.start) == last_onset_tick
        )
        harmony[-1]["offset"] = max(
            harmony[-1]["onset"] + 0.25, last_end_tick / resolution
        )

    return harmony


def process_emopia_file(
    midi_path: Path,
    *,
    dataset_key: str = "emopia_plus",
    source_label: str = "EMOPIA+",
) -> Optional[Dict]:
    """Convert one EMOPIA+ MIDI clip into Hooktheory cache format."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        print(f"Error loading {midi_path.name}: {exc}")
        return None

    melody_notes = extract_melody(midi)
    raw_harmony = extract_harmony(midi)
    if not melody_notes or not raw_harmony:
        return None

    melody = resolve_melody_overlaps(
        quantize_timing_to_beat_grid(melody_notes, resolution=0.25)
    )
    harmony = filter_zero_duration_chords(
        quantize_timing_to_beat_grid(raw_harmony, resolution=0.25)
    )
    if not melody or not harmony:
        return None

    max_offset = max(
        max(note["offset"] for note in melody),
        max(chord["offset"] for chord in harmony),
    )

    match = _CLIP_NAME_RE.match(midi_path.stem)
    emotion_quadrant = match.group(1) if match else None
    youtube_id = match.group(2) if match else None

    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",  # reassigned by the caller
        dataset_key: {
            "id": midi_path.stem,
            "title": midi_path.stem,
            "composer": None,
            "source": source_label,
            "file": midi_path.name,
            "emotion_quadrant": emotion_quadrant,
            "youtube_id": youtube_id,
        },
        "annotations": {
            "num_beats": int(max_offset) if max_offset > 0 else 32,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "keys": [
                {
                    "beat": 0,
                    # Placeholder (no key annotation available), matching
                    # Wikifonia/Chord-Melody-Dataset's converters.
                    "tonic_pitch_class": 0,
                    "scale_degree_intervals": [2, 2, 1, 2, 2, 2],
                }
            ],
            "melody": melody,
            "harmony": harmony,
        },
    }


def convert_emopia_corpus(
    midi_files: List[Path],
    data_path: Path,
    output_dir: Path,
    *,
    augmentation: bool = False,
    max_files: Optional[int] = None,
) -> Dict[str, int]:
    """Convert EMOPIA+ MIDI files to cache JSONL splits."""
    if max_files is not None:
        midi_files = midi_files[:max_files]
    if not midi_files:
        print("No .mid files found for EMOPIA+")
        return {"total_files": 0, "processed": 0, "failed": 0}

    print(f"Found {len(midi_files)} EMOPIA+ MIDI files to process")
    split_map = _load_split_map(data_path)

    all_songs: List[Dict] = []
    failed = 0
    for midi_path in tqdm(midi_files, desc="Processing EMOPIA+"):
        try:
            song = process_emopia_file(midi_path)
        except Exception as exc:
            print(f"Error processing {midi_path.name}: {exc}")
            failed += 1
            continue
        if not song:
            failed += 1
            continue
        song["split"] = split_map.get(midi_path.name, "TRAIN")
        all_songs.append(song)

    print(f"Successfully processed {len(all_songs)} songs ({failed} failed/skipped)")
    if not all_songs:
        print("No songs were successfully processed!")
        return {"total_files": len(midi_files), "processed": 0, "failed": failed}

    splits: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}
    for song in all_songs:
        splits[song["split"].lower()].append(song)
    print("Dataset splits:")
    print(f"  Train: {len(splits['train'])} songs")
    print(f"  Valid: {len(splits['valid'])} songs")
    print(f"  Test:  {len(splits['test'])} songs")

    chord_names = collect_chord_names(all_songs)
    print(f"Found {len(chord_names)} unique chord names")

    cache_dir = str(output_dir.parent)

    if augmentation:
        print("\n=== Creating Augmented Dataset ===")
        augmented_train = create_augmented_dataset(splits["train"])
        augmented_chord_names = collect_chord_names(
            augmented_train + splits["valid"] + splits["test"]
        )
        augmented_splits = {
            "train": augmented_train,
            "valid": splits["valid"],
            "test": splits["test"],
        }
        for split_name, split_songs in augmented_splits.items():
            cache_path = output_dir / f"{split_name}_augmented.jsonl"
            save_jsonl(split_songs, cache_path)
            print(f"Saved {split_name} augmented split to {cache_path}")

        chord_names_aug_path = output_dir / "chord_names_augmented.json"
        with open(chord_names_aug_path, "w", encoding="utf-8") as handle:
            json.dump(augmented_chord_names, handle, indent=2)
        print(f"Saved augmented chord names to {chord_names_aug_path}")
        update_global_chord_names(augmented_chord_names, cache_dir, augmented=True)

    for split_name, split_songs in splits.items():
        cache_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(split_songs, cache_path)
        print(f"Saved {split_name} split to {cache_path}")

    chord_names_path = output_dir / "chord_names.json"
    with open(chord_names_path, "w", encoding="utf-8") as handle:
        json.dump(chord_names, handle, indent=2)
    print(f"Saved chord names to {chord_names_path}")
    update_global_chord_names(chord_names, cache_dir, augmented=False)
    update_global_chord_names(chord_names, cache_dir, augmented=True)

    return {
        "total_files": len(midi_files),
        "processed": len(all_songs),
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="EMOPIA+ root (default: data/EMOPIA+/EMOPIA+)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/emopia_plus",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of .mid files to process (for testing)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Create augmented dataset with transposition",
    )
    parser.add_argument(
        "--report_only",
        action="store_true",
        help="Parse files and print success stats without writing cache output",
    )
    parser.add_argument(
        "--verbose-chord-warnings",
        action="store_true",
        help="Print per-chord simplification warnings during parsing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_chord_symbol_parse_verbose(args.verbose_chord_warnings)

    data_path = Path(args.data_path) if args.data_path else _default_data_path()
    midi_files = discover_midi_files(data_path)
    if args.max_files is not None:
        midi_files = midi_files[: args.max_files]

    if args.report_only:
        ok = 0
        failed: List[str] = []
        for midi_path in tqdm(midi_files, desc="Scanning EMOPIA+"):
            try:
                song = process_emopia_file(midi_path)
                if song:
                    ok += 1
                else:
                    failed.append(midi_path.name)
            except Exception as exc:
                failed.append(f"{midi_path.name}: {exc}")

        print(f"Found {len(midi_files)} .mid files")
        print(f"Parsed successfully: {ok}")
        print(f"Failed/skipped: {len(failed)}")
        if failed:
            preview = failed[:20]
            print("Examples:")
            for name in preview:
                print(f"  - {name}")
            if len(failed) > len(preview):
                print(f"  ... and {len(failed) - len(preview)} more")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = convert_emopia_corpus(
        midi_files,
        data_path,
        output_dir,
        augmentation=args.augmentation,
        max_files=args.max_files,
    )
    if stats["processed"]:
        print("EMOPIA+ dataset conversion completed!")


if __name__ == "__main__":
    main()
