#!/usr/bin/env python3
"""Convert JAZZMUS MusicXML lead sheets to Hooktheory-compatible cache format.

Reuses the Wikifonia MusicXML pipeline in ``convert_wikifonia_to_cache.py``.
JAZZMUS files under ``data/jazzmus/*.musicxml`` are flat, uncompressed MusicXML
exports (many sourced from Wikifonia) with jazz-style chord spelling.

Per the JAZZMUS webpage, the 293 files are handwritten transcriptions of only
163 unique pieces -- many pieces have several independent transcriptions
under different filenames (confirmed: exactly 163 unique ``<movement-title>``
values across the 293 files). Splitting the raw 293 files directly let the
same piece land in both train and valid/test under different filenames --
found by comparing chord+melody content across splits: 34% of the "valid"
songs and 23% of "test" songs turned out to be exact-content duplicates of a
training song. We deduplicate by movement-title before splitting, so each
unique piece is used exactly once and can only ever end up in one split.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_wikifonia_to_cache import (
    _is_no_chord_symbol,
    _pedal_root_chord,
    convert_musicxml_corpus,
    process_musicxml_file,
    set_chord_symbol_parse_verbose,
    transform_wikifonia_chord_symbol,
)


def _extract_title(xml_file: Path) -> str:
    """Read <movement-title> via a lightweight XML parse (cheaper than a full
    music21 parse), used only for deduplication grouping.
    """
    title_el = ET.parse(xml_file).getroot().find("movement-title")
    if title_el is not None and title_el.text:
        return title_el.text.strip()
    return xml_file.stem


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


def deduplicate_by_title(xml_files: List[Path]) -> List[Path]:
    """Keep exactly one file per unique (normalized) movement-title.

    Files are pre-sorted by filename, so this deterministically keeps the
    lowest-numbered file for each piece and drops the rest.
    """
    seen = set()
    deduped = []
    for xml_file in xml_files:
        title = _normalize_title(_extract_title(xml_file))
        if title in seen:
            continue
        seen.add(title)
        deduped.append(xml_file)
    return deduped


def transform_jazzmus_chord_symbol(chord_symbol: str) -> str:
    """Normalize jazz lead-sheet chord spellings for note_seq."""
    if _is_no_chord_symbol(chord_symbol):
        return ""

    pedal_root = _pedal_root_chord(chord_symbol)
    if pedal_root is not None:
        return pedal_root

    chord = transform_wikifonia_chord_symbol(chord_symbol)
    if not chord:
        return chord

    chord = re.sub(r"Maj7", "maj7", chord)
    chord = re.sub(r"Min7", "m7", chord)
    chord = re.sub(r"Min", "m", chord)
    return chord


def process_jazzmus_file(xml_file: Path):
    return process_musicxml_file(
        xml_file,
        dataset_key="jazzmus",
        source_label="JAZZMUS Dataset",
        chord_symbol_transform=transform_jazzmus_chord_symbol,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jazzmus_path",
        type=str,
        default="data/jazzmus",
        help="Directory containing *.musicxml files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/jazzmus",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of XML files to process (for testing)",
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
        help="Print per-chord simplification warnings during parsing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_chord_symbol_parse_verbose(args.verbose_chord_warnings)
    jazzmus_path = Path(args.jazzmus_path)
    all_xml_files = sorted(jazzmus_path.glob("*.musicxml"))
    xml_files = deduplicate_by_title(all_xml_files)
    print(
        f"Found {len(all_xml_files)} files, "
        f"deduplicated to {len(xml_files)} unique pieces (by movement-title)"
    )

    if args.report_only:
        from tqdm import tqdm

        ok = 0
        failed: list[str] = []
        for xml_file in tqdm(xml_files, desc="Scanning JAZZMUS"):
            try:
                song = process_jazzmus_file(xml_file)
                if song:
                    ok += 1
                else:
                    failed.append(xml_file.name)
            except Exception as exc:
                failed.append(f"{xml_file.name}: {exc}")

        print(f"Found {len(xml_files)} MusicXML files")
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
        process_jazzmus_file,
        augmentation=args.augmentation,
        max_files=args.max_files,
        dataset_name="JAZZMUS",
    )
    if stats["processed"]:
        print("JAZZMUS dataset conversion completed!")


if __name__ == "__main__":
    main()
