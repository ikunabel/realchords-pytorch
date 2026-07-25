#!/usr/bin/env python3
"""Convert the Chord Melody Dataset to Hooktheory-compatible cache format.

Source: https://github.com/shiehn/chord-melody-dataset (``data/chord-melody-dataset``).
Reuses the Wikifonia MusicXML pipeline (``extract_melody_and_chords_from_musicxml``,
quantization, chord parsing, splitting, and transposition augmentation) --
the format is standard MusicXML with ``<harmony>`` chord symbols, same as
Wikifonia/JAZZMUS.

Layout differs from Wikifonia/JAZZMUS though: each *song* is a folder
(``data/chord-melody-dataset/<song>/``) containing one MusicXML file per
transposed key (``c.xml``, ``cs.xml``, ..., up to 12 -- some songs have fewer;
file stems and counts aren't perfectly uniform across the corpus). We used to
treat this built-in 12-key duplication as the augmentation (every key for
TRAIN, one canonical key for VALID/TEST), but that made TRAIN's *row* count
~12x its *song* count while VALID/TEST stayed 1:1 -- a split that looked like
80/10/10 by song was ~98/1/1 by row, inconsistent with every other dataset.

Instead we now pick exactly **one** representative file per song folder --
prefer ``d.xml``, falling back to the alphabetically-first available file for
the ~10% of songs (49/474) that don't have one -- and feed that flat list of
one-file-per-song into the same generic ``convert_musicxml_corpus`` pipeline
Wikifonia uses: a plain 80/10/10 split by song, and (if ``--augmentation`` is
passed) the standard [-6, +6] semitone transposition augmentation applied to
TRAIN only. This keeps row counts equal to song counts in every split, same
as every other dataset.

All songs were engraved for guitar ("Nylon Guitar"), and about half the files
declare ``<transpose><octave-change>-1</octave-change>`` (written a full
octave above sounding pitch). ``extract_melody_and_chords_from_musicxml`` now
calls ``score.toSoundingPitch()`` before reading pitches, so this is handled
automatically -- without it, roughly half the corpus's melodies would come out
a full octave too high.

Usage::

    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py --augmentation
    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py --report_only
    python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py --max_songs 20
"""

from __future__ import annotations

import argparse
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
    convert_musicxml_corpus,
    extract_melody_and_chords_from_musicxml,
    filter_zero_duration_chords,
    quantize_timing_to_beat_grid,
    resolve_melody_overlaps,
    set_chord_symbol_parse_verbose,
    transform_wikifonia_chord_symbol,
)

_REPRESENTATIVE_KEY_PREFERENCE = "d"  # single canonical key used for every song


def _default_data_path() -> Path:
    return Path("data/chord-melody-dataset")


def discover_song_dirs(data_path: Path) -> List[Path]:
    """Find song folders, each containing one or more per-key .xml files."""
    return sorted(
        d
        for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and any(d.glob("*.xml"))
    )


def _pick_representative_file(xml_files: List[Path]) -> Path:
    """Choose one file per song, used for every split.

    Prefers the preference key (default "d") if present; otherwise falls back
    to the alphabetically-first available file (``xml_files`` is pre-sorted),
    so the choice is deterministic across runs. About 10% of songs (49/474)
    lack a file in the preference key and use the fallback.
    """
    for f in xml_files:
        if f.stem.lower() == _REPRESENTATIVE_KEY_PREFERENCE:
            return f
    return xml_files[0]


def discover_representative_xml_files(data_path: Path) -> List[Path]:
    """One representative MusicXML file per song folder (flat list, 1:1 with songs)."""
    xml_files = []
    for song_dir in discover_song_dirs(data_path):
        candidates = sorted(song_dir.glob("*.xml"), key=lambda p: p.stem)
        if candidates:
            xml_files.append(_pick_representative_file(candidates))
    return xml_files


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


def process_chord_melody_xml_file(xml_file: Path) -> Optional[Dict]:
    """`process_file` callable for `convert_musicxml_corpus`: derives the song
    slug from the parent folder name, since each folder is one song here.
    """
    return process_chord_melody_file(xml_file, song_slug=xml_file.parent.name)


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
        help="Maximum number of songs to process (for testing)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Create augmented (train-only, [-6, +6] semitone) dataset, same as Wikifonia/JAZZMUS",
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
    xml_files = discover_representative_xml_files(data_path)

    if args.report_only:
        if args.max_songs is not None:
            xml_files = xml_files[: args.max_songs]
        ok = 0
        failed: List[str] = []
        for xml_file in tqdm(xml_files, desc="Scanning Chord Melody Dataset"):
            try:
                song = process_chord_melody_xml_file(xml_file)
                if song:
                    ok += 1
                else:
                    failed.append(xml_file.parent.name)
            except Exception as exc:
                failed.append(f"{xml_file.parent.name}: {exc}")

        print(f"Found {len(xml_files)} songs (one representative key each)")
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

    stats = convert_musicxml_corpus(
        xml_files,
        output_dir,
        process_chord_melody_xml_file,
        augmentation=args.augmentation,
        max_files=args.max_songs,
        dataset_name="Chord Melody Dataset",
    )
    if stats["processed"]:
        print("Chord Melody Dataset conversion completed!")


if __name__ == "__main__":
    main()
