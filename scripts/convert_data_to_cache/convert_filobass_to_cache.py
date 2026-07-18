#!/usr/bin/env python3
"""Convert FiloBass MusicXML transcriptions to Hooktheory-compatible cache format.

Data source: 48 manually verified jazz walking-bass transcriptions (Riley &
Dixon, ISMIR 2023), ``data/FiloBass_v1.0.0/FiloBass ISMIR Publication/musicxml/``.
Reuses the Wikifonia MusicXML pipeline in ``convert_wikifonia_to_cache.py``, the
same way JAZZMUS does -- but unlike JAZZMUS, FiloBass's chord symbols are
exported as a generic MusicXML ``<kind text="...">other</kind>``, so music21's
``ChordSymbol.figure`` only reflects the root (verified empirically: it drops
the quality entirely for this "other" kind). The real quality string survives
in ``ChordSymbol.chordKindStr``, so this converter supplies a custom
``chord_text_fn`` that reconstructs ``root + chordKindStr [+ /bass]`` instead
of relying on ``.figure``. Once reconstructed this way, the existing
``transform_wikifonia_chord_symbol`` parses every chord in the corpus with no
further normalization needed (verified: 0 of 15,910 chord instances across all
48 files fail to parse).

Usage::

    python scripts/convert_data_to_cache/convert_filobass_to_cache.py
                                            [--filobass_path ...]
                                            [--output_dir data/cache/filobass]
                                            [--augmentation]
                                            [--report_only]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_wikifonia_to_cache import (
    convert_musicxml_corpus,
    process_musicxml_file,
    set_chord_symbol_parse_verbose,
    transform_wikifonia_chord_symbol,
)

_TAGS = ["MELODY", "HARMONY", "SWING"]


def _filobass_chord_text(chord_symbol) -> str:
    """Build a parseable chord string from a music21 ChordSymbol.

    FiloBass exports chord quality as a generic ``other`` kind with the real
    label in ``chordKindStr`` (e.g. "m7", "7b9", "maj7", "mmaj7") -- music21's
    own ``.figure`` drops this and reflects only the root. Reconstructs
    ``root + kind [+ /bass]``, omitting the slash whenever the bass is the
    same as the root (also covers a handful of malformed source rows with an
    empty bass step, which music21 resolves to bass == root).
    """
    root = chord_symbol.root()
    bass = chord_symbol.bass()
    kind = chord_symbol.chordKindStr or ""
    text = f"{root.name}{kind}"
    if bass is not None and root is not None and bass.name != root.name:
        text += f"/{bass.name}"
    return text


def _load_soloist_lookup(filobass_root: Path) -> Dict[str, str]:
    """Best-effort ``file stem -> reference soloist`` lookup from Edit_Data.csv.

    "Soloist" here is the horn player on the original Aebersold reference
    recording the backing track imitates (e.g. "Getz", "Webster"), not the
    bassist -- kept as a descriptive source-recording hint, not "composer".
    """
    csv_path = filobass_root / "notebooks_and_scripts" / "Edit_Data.csv"
    lookup: Dict[str, str] = {}
    if not csv_path.exists():
        return lookup
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            title = row.get("Track Name", "").strip()
            soloist = row.get("Soloist", "").strip()
            if not title or not soloist:
                continue
            stem = title.replace(" ", "-")
            lookup[stem] = soloist
    return lookup


def process_filobass_file(xml_file: Path, soloist_lookup: Dict[str, str]):
    song = process_musicxml_file(
        xml_file,
        dataset_key="filobass",
        source_label="FiloBass Dataset",
        chord_symbol_transform=transform_wikifonia_chord_symbol,
        chord_text_fn=_filobass_chord_text,
        tags=_TAGS,
    )
    if song is None:
        return None
    # Prefer a clean, spaced title over the hyphenated file stem, and attach
    # the reference-recording soloist as metadata if we found one.
    song["filobass"]["title"] = xml_file.stem.replace("-", " ")
    soloist = soloist_lookup.get(xml_file.stem)
    if soloist:
        song["filobass"]["reference_soloist"] = soloist
    return song


def _default_filobass_path() -> Path:
    for candidate in (
        Path("data/FiloBass_v1.0.0/FiloBass ISMIR Publication"),
        Path("data/filobass"),
    ):
        if candidate.exists():
            return candidate
    return Path("data/FiloBass_v1.0.0/FiloBass ISMIR Publication")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filobass_path",
        type=str,
        default=None,
        help="Root directory of the FiloBass release (default: auto-detected)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/filobass",
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
    filobass_root = (
        Path(args.filobass_path) if args.filobass_path else _default_filobass_path()
    )
    xml_files = sorted((filobass_root / "musicxml").glob("*.xml"))
    soloist_lookup = _load_soloist_lookup(filobass_root)

    def _process(xml_file: Path):
        return process_filobass_file(xml_file, soloist_lookup)

    if args.report_only:
        from tqdm import tqdm

        ok = 0
        failed: list[str] = []
        for xml_file in tqdm(xml_files, desc="Scanning FiloBass"):
            try:
                song = _process(xml_file)
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
        _process,
        augmentation=args.augmentation,
        max_files=args.max_files,
        dataset_name="FiloBass",
    )
    if stats["processed"]:
        print("FiloBass dataset conversion completed!")


if __name__ == "__main__":
    main()
