#!/usr/bin/env python3
"""Convert the Chord Melody Dataset to Hooktheory-compatible cache format.

Source: https://github.com/shiehn/chord-melody-dataset (``data/chord-melody-dataset``).
Reuses the Wikifonia MusicXML pipeline (``extract_melody_and_chords_from_musicxml``,
quantization, chord parsing) -- the format is standard MusicXML with ``<harmony>``
chord symbols, same as Wikifonia/JAZZMUS.

Layout differs from Wikifonia/JAZZMUS though: each *song* is a folder
(``data/chord-melody-dataset/<song>/``) containing one MusicXML file per
transposed key (``c.xml``, ``cs.xml``, ..., up to 12 -- some songs have fewer;
file stems and counts aren't perfectly uniform across the corpus). Rather than
re-deriving our own +-N semitone augmentation like the other converters, we
treat the corpus's own 12-key duplication as the augmentation: every available
key is used for TRAIN, but only one canonical key per song goes into
VALID/TEST, to avoid near-duplicate transposed copies of the same song
leaking across the eval split.

All songs were engraved for guitar ("Nylon Guitar"), and about half the files
declare ``<transpose><octave-change>-1</octave-change>`` (written a full
octave above sounding pitch). ``extract_melody_and_chords_from_musicxml`` now
calls ``score.toSoundingPitch()`` before reading pitches, so this is handled
automatically -- without it, roughly half the corpus's melodies would come out
a full octave too high.

Usage::

    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py
    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py --report_only
    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py --max_songs 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_wikifonia_to_cache import (
    collect_chord_names,
    extract_melody_and_chords_from_musicxml,
    filter_zero_duration_chords,
    quantize_timing_to_beat_grid,
    resolve_melody_overlaps,
    set_chord_symbol_parse_verbose,
    transform_wikifonia_chord_symbol,
)
from realchords.utils.data_utils import update_global_chord_names
from realchords.utils.io_utils import save_jsonl

_EVAL_KEY_PREFERENCE = "c"  # canonical, unaugmented key used for VALID/TEST


def _default_data_path() -> Path:
    return Path("data/chord-melody-dataset")


def discover_song_dirs(data_path: Path) -> List[Path]:
    """Find song folders, each containing one or more per-key .xml files."""
    return sorted(
        d
        for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and any(d.glob("*.xml"))
    )


def _pick_eval_key_file(xml_files: List[Path]) -> Path:
    """Choose one representative key file for VALID/TEST.

    Prefers the "c" (concert C) key if present; otherwise falls back to the
    alphabetically-first available file (``xml_files`` is pre-sorted), so the
    choice is deterministic across runs.
    """
    for f in xml_files:
        if f.stem.lower() == _EVAL_KEY_PREFERENCE:
            return f
    return xml_files[0]


def _raw_harmony_offsets(xml_file: Path) -> List[float]:
    """Raw MusicXML <offset> value (in quarter notes) for each <harmony>
    element in the file, in document order.

    SmartScore (the OCR tool behind this corpus -- see README.md) writes a
    per-<harmony> <offset> sub-element that looks like a fine sub-beat timing
    nudge but is actually graphical/OCR placement noise: subtracting it from
    music21's computed chord-symbol offset lands exactly on a melody note
    onset in every case checked (including "offsets" of 1-2+ quarters, not
    just sub-16th jitter). Trusting it as real timing means a meaningful
    fraction of chords sit close enough to a 16th-note-grid boundary that
    quantization rounds them to the wrong frame -- audible as a chord landing
    a bit early or late relative to the melody.
    """
    divisions = 1
    offsets: List[float] = []
    tree = ET.parse(xml_file)
    for measure in tree.getroot().iter("measure"):
        for element in measure:
            if element.tag == "attributes":
                divisions_el = element.find("divisions")
                if divisions_el is not None and divisions_el.text:
                    divisions = int(divisions_el.text)
            elif element.tag == "harmony":
                offset_el = element.find("offset")
                raw = int(offset_el.text) if offset_el is not None and offset_el.text else 0
                offsets.append(raw / divisions)
    return offsets


def _correct_chord_onsets(chords: List[Dict], xml_file: Path) -> List[Dict]:
    """Strip the OCR placement noise described in `_raw_harmony_offsets`.

    Falls back to the uncorrected chords if the raw <harmony> count doesn't
    match the parsed chord count (shouldn't happen -- verified 1:1 across the
    corpus -- but parsing failures elsewhere could in principle drop one).
    """
    corrections = _raw_harmony_offsets(xml_file)
    if len(corrections) != len(chords):
        return chords

    corrected = [deepcopy(chord) for chord in chords]
    prev_onset = -1.0
    for chord, correction, original in zip(corrected, corrections, chords):
        candidate = original["onset"] - correction
        # Occasionally two consecutive <harmony> elements share the same
        # underlying note's cursor position and are distinguished only by
        # their <offset> (e.g. two chords packed into one short note) -- in
        # that case the offset is real information, not noise. Stripping it
        # would collapse both onto the same instant, so fall back to the
        # original (noisy but still monotonic) onset instead.
        if candidate <= prev_onset:
            candidate = original["onset"]
        chord["onset"] = candidate
        prev_onset = candidate

    # Re-derive each chord's offset from the next (corrected) onset, since the
    # original offset/duration was computed relative to the noisy onsets.
    for i in range(len(corrected) - 1):
        corrected[i]["offset"] = corrected[i + 1]["onset"]
    if corrected:
        last_duration = chords[-1]["offset"] - chords[-1]["onset"]
        corrected[-1]["offset"] = corrected[-1]["onset"] + last_duration

    return corrected


def process_chord_melody_file(
    xml_file: Path,
    song_slug: str,
    *,
    dataset_key: str = "chord_melody_dataset",
    source_label: str = "Chord Melody Dataset",
    chord_symbol_transform=transform_wikifonia_chord_symbol,
) -> Optional[Dict]:
    """Convert one per-key MusicXML file into Hooktheory cache format."""
    parsed = extract_melody_and_chords_from_musicxml(
        xml_file, chord_symbol_transform=chord_symbol_transform,
    )
    if not parsed:
        return None

    melody_notes = parsed["melody"]
    chords = _correct_chord_onsets(parsed["chords"], xml_file)
    if not melody_notes or not chords:
        return None

    melody = resolve_melody_overlaps(
        quantize_timing_to_beat_grid(melody_notes, resolution=0.25)
    )
    harmony = filter_zero_duration_chords(
        quantize_timing_to_beat_grid(chords, resolution=0.25)
    )
    if not melody or not harmony:
        return None

    max_offset = max(
        max(note["offset"] for note in melody),
        max(chord["offset"] for chord in harmony),
    )

    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",  # reassigned by the caller
        dataset_key: {
            "id": f"{song_slug}__{xml_file.stem}",
            "title": song_slug.replace("_", " ").title(),
            "composer": None,
            "source": source_label,
            "file": xml_file.name,
            "key_label": xml_file.stem,
            "time_signature": parsed["metadata"].get("time_signature"),
            "key_signature": parsed["metadata"].get("key_signature"),
        },
        "annotations": {
            "num_beats": int(max_offset) if max_offset > 0 else 32,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "keys": [
                {
                    "beat": 0,
                    # Placeholder, matching Wikifonia's converter: actual key
                    # isn't tracked per transposition, only relative chord/
                    # melody shape matters for training.
                    "tonic_pitch_class": 0,
                    "scale_degree_intervals": [2, 2, 1, 2, 2, 2],
                }
            ],
            "melody": melody,
            "harmony": harmony,
        },
    }


def _split_song_names(song_names: List[str]) -> Dict[str, str]:
    """Assign each song to TRAIN/VALID/TEST (80/10/10, seed 42).

    Splitting by song (not by file) keeps every key-transposition of a song
    in the same split -- otherwise a transposed copy of a test song could
    leak into train.
    """
    shuffled = list(song_names)
    random.seed(42)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.8)
    valid_end = train_end + int(total * 0.1)

    split_by_song = {}
    for name in shuffled[:train_end]:
        split_by_song[name] = "TRAIN"
    for name in shuffled[train_end:valid_end]:
        split_by_song[name] = "VALID"
    for name in shuffled[valid_end:]:
        split_by_song[name] = "TEST"
    return split_by_song


def convert_chord_melody_corpus(
    song_dirs: List[Path],
    output_dir: Path,
    *,
    max_songs: Optional[int] = None,
) -> Dict[str, int]:
    """Convert Chord Melody Dataset song folders to cache JSONL splits."""
    if max_songs is not None:
        song_dirs = song_dirs[:max_songs]
    if not song_dirs:
        print("No song folders found for Chord Melody Dataset")
        return {"total_songs": 0, "processed": 0, "failed": 0}

    print(f"Found {len(song_dirs)} Chord Melody Dataset song folders")

    split_by_song = _split_song_names([d.name for d in song_dirs])

    all_songs: List[Dict] = []
    failed_songs = 0
    for song_dir in tqdm(song_dirs, desc="Processing Chord Melody Dataset"):
        split = split_by_song[song_dir.name]
        xml_files = sorted(song_dir.glob("*.xml"), key=lambda p: p.stem)
        if not xml_files:
            failed_songs += 1
            continue

        # TRAIN gets every available key (the corpus's built-in
        # augmentation); VALID/TEST get exactly one canonical key each.
        keys_to_process = (
            xml_files if split == "TRAIN" else [_pick_eval_key_file(xml_files)]
        )

        processed_any = False
        for xml_file in keys_to_process:
            try:
                song = process_chord_melody_file(xml_file, song_dir.name)
            except Exception as exc:
                print(f"Error processing {xml_file}: {exc}")
                continue
            if song:
                song["split"] = split
                all_songs.append(song)
                processed_any = True
        if not processed_any:
            failed_songs += 1

    print(
        f"Successfully processed {len(all_songs)} examples from "
        f"{len(song_dirs) - failed_songs} songs ({failed_songs} songs failed/skipped)"
    )
    if not all_songs:
        print("No songs were successfully processed!")
        return {"total_songs": len(song_dirs), "processed": 0, "failed": failed_songs}

    splits: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}
    for song in all_songs:
        splits[song["split"].lower()].append(song)

    print("Dataset splits:")
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Valid: {len(splits['valid'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")

    chord_names = collect_chord_names(all_songs)
    print(f"Found {len(chord_names)} unique chord names")

    cache_dir = str(output_dir.parent)

    for split_name, split_songs in splits.items():
        cache_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(split_songs, cache_path)
        print(f"Saved {split_name} split to {cache_path}")

    # No separate transposition augmentation here -- TRAIN already contains
    # every available key. Still write "_augmented" files (identical to the
    # plain ones) so HooktheoryDataset(data_augmentation=True) can load this
    # dataset the same way it loads any other.
    for split_name, split_songs in splits.items():
        cache_path = output_dir / f"{split_name}_augmented.jsonl"
        save_jsonl(split_songs, cache_path)
        print(f"Saved {split_name} augmented split (identical) to {cache_path}")

    chord_names_path = output_dir / "chord_names.json"
    with open(chord_names_path, "w", encoding="utf-8") as handle:
        json.dump(chord_names, handle, indent=2)
    print(f"Saved chord names to {chord_names_path}")

    chord_names_aug_path = output_dir / "chord_names_augmented.json"
    with open(chord_names_aug_path, "w", encoding="utf-8") as handle:
        json.dump(chord_names, handle, indent=2)
    print(f"Saved augmented chord names to {chord_names_aug_path}")

    update_global_chord_names(chord_names, cache_dir, augmented=False)
    update_global_chord_names(chord_names, cache_dir, augmented=True)

    return {
        "total_songs": len(song_dirs),
        "processed": len(all_songs),
        "failed": failed_songs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Chord Melody Dataset root (default: data/chord-melody-dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/chord_melody_dataset",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=None,
        help="Maximum number of song folders to process (for testing)",
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
    song_dirs = discover_song_dirs(data_path)
    if args.max_songs is not None:
        song_dirs = song_dirs[: args.max_songs]

    if args.report_only:
        split_by_song = _split_song_names([d.name for d in song_dirs])
        ok = 0
        failed: List[str] = []
        for song_dir in tqdm(song_dirs, desc="Scanning Chord Melody Dataset"):
            xml_files = sorted(song_dir.glob("*.xml"), key=lambda p: p.stem)
            if not xml_files:
                failed.append(song_dir.name)
                continue
            split = split_by_song[song_dir.name]
            probe_file = xml_files if split == "TRAIN" else [_pick_eval_key_file(xml_files)]
            try:
                song = process_chord_melody_file(probe_file[0], song_dir.name)
                if song:
                    ok += 1
                else:
                    failed.append(song_dir.name)
            except Exception as exc:
                failed.append(f"{song_dir.name}: {exc}")

        print(f"Found {len(song_dirs)} song folders")
        print(f"Parsed successfully (first key checked): {ok}")
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

    stats = convert_chord_melody_corpus(
        song_dirs,
        output_dir,
        max_songs=args.max_songs,
    )
    if stats["processed"]:
        print("Chord Melody Dataset conversion completed!")


if __name__ == "__main__":
    main()
