#!/usr/bin/env python3
"""Batch-convert CoCoPops native Humdrum (.hum) files to lead-sheet MIDI.

Melody is rendered with the Verovio Humdrum Viewer (VHV) engine:
https://verovio.humdrum.org

Chord symbols from the ``**harte`` spine are realized as block chords on a
separate track (Verovio only plays ``**kern`` notes, not analytic chord labels).

Prerequisites (one-time):

    cd scripts/verovio_humdrum
    npm install

Examples:

    # All canonical */Data/*.hum files
    python scripts/convert_cocopops_hum_to_midi.py

    # Quick test on 5 files
    python scripts/convert_cocopops_hum_to_midi.py --max_files 5

    # Melody-only (no chord realization)
    python scripts/convert_cocopops_hum_to_midi.py --melody_only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import note_seq.chord_symbols_lib as chord_symbols_lib
import pretty_midi
from tqdm import tqdm

from realchords.constants import BASS_OCTAVE, CHORD_OCTAVE, CHORD_VELOCITY, MELODY_VELOCITY

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_cocopops_to_cache import (
    _HARM_EXCL,
    _HARTE_EXCL,
    _default_cocopops_path,
    _read_hum_rows,
    _select_kern_index,
    _uses_timebase,
    discover_hum_files,
    extract_harmony_from_spines,
    extract_melody_notes_timebase,
    transform_harte_chord_symbol,
)

VEROVIO_DIR = SCRIPT_DIR / "verovio_humdrum"
CONVERT_ONE = VEROVIO_DIR / "convert_one.mjs"
_MM_TEMPO_RE = re.compile(r"^\*MM(\d+(?:\.\d+)?)")


def _ensure_verovio_runtime() -> None:
    if shutil.which("node") is None:
        raise RuntimeError("Node.js is required but was not found on PATH.")
    if not CONVERT_ONE.exists():
        raise FileNotFoundError(f"Missing Verovio helper: {CONVERT_ONE}")
    if not (VEROVIO_DIR / "node_modules" / "verovio").exists():
        raise FileNotFoundError(
            "Verovio npm package not installed. Run:\n"
            f"  cd {VEROVIO_DIR} && npm install"
        )


def _verovio_cli_available() -> Optional[str]:
    return shutil.which("verovio")


def convert_with_verovio_cli(input_path: Path, output_path: Path, verovio_bin: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [verovio_bin, "-t", "midi", "-o", str(output_path), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(stderr or f"verovio CLI failed for {input_path.name}")


def convert_with_node_toolkit(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["node", str(CONVERT_ONE), str(input_path), str(output_path)],
        cwd=str(VEROVIO_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(stderr or f"Verovio toolkit failed for {input_path.name}")


def parse_tempo_bpm(rows: List[List[str]], default: float = 120.0) -> float:
    """Read the first ``*MM`` tandem tempo interpretation (quarter-note BPM)."""
    for row in rows:
        for token in row:
            match = _MM_TEMPO_RE.match(token.strip())
            if match:
                return float(match.group(1))
    return default


def extract_chord_events(hum_path: Path) -> List[Dict[str, float | str]]:
    spine_names, rows = _read_hum_rows(hum_path)
    if _HARM_EXCL not in spine_names or _HARTE_EXCL not in spine_names:
        return []
    harm_idx = spine_names.index(_HARM_EXCL)
    harte_idx = spine_names.index(_HARTE_EXCL)
    kern_idx = _select_kern_index(spine_names, rows)
    return extract_harmony_from_spines(rows, harm_idx, harte_idx, kern_idx)


def _dedup_pitches(pitches: List[int]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for pitch in pitches:
        if pitch not in seen:
            seen.add(pitch)
            out.append(pitch)
    return out


def realize_chord_pitches(
    harte_symbol: str,
    *,
    chord_octave: int = CHORD_OCTAVE,
    include_bass: bool = True,
) -> List[int]:
    lead_symbol = transform_harte_chord_symbol(harte_symbol)
    if not lead_symbol:
        return []
    try:
        chord_pcs = chord_symbols_lib.chord_symbol_pitches(lead_symbol)
    except Exception:
        return []
    if not chord_pcs:
        return []
    pitches = [pc % 12 + chord_octave * 12 for pc in chord_pcs]
    if include_bass:
        bass_pc = chord_symbols_lib.chord_symbol_bass(lead_symbol) % 12
        pitches.append(bass_pc + BASS_OCTAVE * 12)
    return _dedup_pitches(pitches)


def merge_chords_into_midi(
    midi_path: Path,
    chords: List[Dict[str, float | str]],
    *,
    bpm: float,
    chord_octave: int = CHORD_OCTAVE,
    include_bass: bool = True,
) -> None:
    """Add a block-chord accompaniment track to an existing melody MIDI file."""
    midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
    seconds_per_quarter = 60.0 / bpm

    if midi_obj.instruments:
        midi_obj.instruments[0].name = "Melody"
        for note in midi_obj.instruments[0].notes:
            note.velocity = MELODY_VELOCITY

    chord_instr = pretty_midi.Instrument(program=0, name="Chords")
    for chord in chords:
        pitches = realize_chord_pitches(
            str(chord["symbol"]),
            chord_octave=chord_octave,
            include_bass=include_bass,
        )
        if not pitches:
            continue
        start = float(chord["onset"]) * seconds_per_quarter
        end = float(chord["offset"]) * seconds_per_quarter
        if end <= start:
            continue
        for pitch in pitches:
            chord_instr.notes.append(
                pretty_midi.Note(
                    velocity=CHORD_VELOCITY,
                    pitch=pitch,
                    start=start,
                    end=end,
                )
            )

    midi_obj.instruments.append(chord_instr)
    midi_obj.write(str(midi_path))


def render_timebase_midi(
    input_path: Path,
    output_path: Path,
    *,
    bpm: float,
    melody_only: bool = False,
    chord_octave: int = CHORD_OCTAVE,
    include_bass: bool = True,
) -> None:
    """Render a *tb step-sequence file directly, bypassing Verovio.

    Verovio's Humdrum importer defaults every undecorated **kern token to a
    quarter note, since RollingStone files carry no **recip prefixes at all
    (rhythm comes from the *tb tandem interpretations instead) -- so its
    melody rendering for these files is flat-quarter-note wrong, not just its
    chord timing. Reconstruct both tracks from the *tb grid ourselves.
    """
    spine_names, rows = _read_hum_rows(input_path)
    kern_idx = _select_kern_index(spine_names, rows)
    seconds_per_quarter = 60.0 / bpm

    midi_obj = pretty_midi.PrettyMIDI()
    melody_instr = pretty_midi.Instrument(program=0, name="Melody")
    for note in extract_melody_notes_timebase(rows, kern_idx):
        melody_instr.notes.append(
            pretty_midi.Note(
                velocity=MELODY_VELOCITY,
                pitch=int(note["pitch"]),
                start=note["onset"] * seconds_per_quarter,
                end=note["offset"] * seconds_per_quarter,
            )
        )
    midi_obj.instruments.append(melody_instr)

    if not melody_only:
        harm_idx = spine_names.index(_HARM_EXCL)
        harte_idx = spine_names.index(_HARTE_EXCL)
        chords = extract_harmony_from_spines(rows, harm_idx, harte_idx, kern_idx)
        chord_instr = pretty_midi.Instrument(program=0, name="Chords")
        for chord in chords:
            pitches = realize_chord_pitches(
                str(chord["symbol"]),
                chord_octave=chord_octave,
                include_bass=include_bass,
            )
            if not pitches:
                continue
            start = float(chord["onset"]) * seconds_per_quarter
            end = float(chord["offset"]) * seconds_per_quarter
            if end <= start:
                continue
            for pitch in pitches:
                chord_instr.notes.append(
                    pretty_midi.Note(velocity=CHORD_VELOCITY, pitch=pitch, start=start, end=end)
                )
        midi_obj.instruments.append(chord_instr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_obj.write(str(output_path))


def convert_hum_to_midi(
    input_path: Path,
    output_path: Path,
    *,
    prefer_cli: bool,
    verovio_bin: Optional[str],
    melody_only: bool = False,
    chord_octave: int = CHORD_OCTAVE,
    include_bass: bool = True,
) -> str:
    _, rows = _read_hum_rows(input_path)
    bpm = parse_tempo_bpm(rows)

    if _uses_timebase(rows):
        render_timebase_midi(
            input_path,
            output_path,
            bpm=bpm,
            melody_only=melody_only,
            chord_octave=chord_octave,
            include_bass=include_bass,
        )
        return "native-timebase" if melody_only else "native-timebase+chords"

    if prefer_cli and verovio_bin:
        try:
            convert_with_verovio_cli(input_path, output_path, verovio_bin)
            backend = "verovio-cli"
        except RuntimeError:
            backend = ""
    else:
        backend = ""

    if not backend:
        convert_with_node_toolkit(input_path, output_path)
        backend = "verovio-node"

    if melody_only:
        return backend

    chords = extract_chord_events(input_path)
    if chords:
        merge_chords_into_midi(
            output_path,
            chords,
            bpm=bpm,
            chord_octave=chord_octave,
            include_bass=include_bass,
        )
        backend = f"{backend}+chords"

    return backend


def output_path_for(input_path: Path, data_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(data_root)
    return output_root / rel.with_suffix(".mid")


def convert_corpus(
    hum_files: List[Path],
    data_root: Path,
    output_root: Path,
    *,
    skip_existing: bool = True,
    prefer_cli: bool = False,
    melody_only: bool = False,
    chord_octave: int = CHORD_OCTAVE,
    include_bass: bool = True,
) -> Dict[str, object]:
    _ensure_verovio_runtime()
    verovio_bin = _verovio_cli_available()

    stats = {
        "total": len(hum_files),
        "converted": 0,
        "skipped": 0,
        "failed": 0,
        "backend": Counter(),
        "failures": [],
    }

    for hum_path in tqdm(hum_files, desc="Humdrum -> MIDI (Verovio)"):
        out_path = output_path_for(hum_path, data_root, output_root)
        if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            stats["skipped"] += 1
            continue
        try:
            backend = convert_hum_to_midi(
                hum_path,
                out_path,
                prefer_cli=prefer_cli,
                verovio_bin=verovio_bin,
                melody_only=melody_only,
                chord_octave=chord_octave,
                include_bass=include_bass,
            )
            stats["converted"] += 1
            stats["backend"][backend] += 1
        except Exception as exc:
            stats["failed"] += 1
            stats["failures"].append({"file": hum_path.name, "error": str(exc)})

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cocopops_path",
        type=str,
        default=None,
        help="CoCoPops repository root (default: data/CoCoPops if present)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cocopops_midi",
        help="Directory for rendered MIDI files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Limit number of files (for testing)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Re-render MIDI even if output already exists",
    )
    parser.add_argument(
        "--prefer_cli",
        action="store_true",
        help="Try native verovio CLI first when installed",
    )
    parser.add_argument(
        "--melody_only",
        action="store_true",
        help="Skip **harte chord realization (Verovio melody track only)",
    )
    parser.add_argument(
        "--no_chord_bass",
        action="store_true",
        help="Omit separate bass note from chord voicings",
    )
    parser.add_argument(
        "--chord_octave",
        type=int,
        default=CHORD_OCTAVE,
        help="Octave for chord tones (default: project CHORD_OCTAVE)",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional JSON report path (default: <output_dir>/conversion_report.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.cocopops_path) if args.cocopops_path else _default_cocopops_path()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    hum_files = discover_hum_files(data_root)
    if args.max_files is not None:
        hum_files = hum_files[: args.max_files]

    if not hum_files:
        print(f"No .hum files found under {data_root}")
        return

    print(f"CoCoPops root: {data_root}")
    print(f"Output dir:    {output_root}")
    print(f"Files:         {len(hum_files)}")
    print("Engine:        Verovio Humdrum toolkit + **harte chord realization\n")

    stats = convert_corpus(
        hum_files,
        data_root,
        output_root,
        skip_existing=not args.no_skip_existing,
        prefer_cli=args.prefer_cli,
        melody_only=args.melody_only,
        chord_octave=args.chord_octave,
        include_bass=not args.no_chord_bass,
    )

    report_path = (
        Path(args.report_path)
        if args.report_path
        else output_root / "conversion_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("\nConversion summary")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped:   {stats['skipped']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Backends:  {stats['backend']}")
    print(f"  Report:    {report_path}")

    failures = stats.get("failures", [])
    if failures:
        print("\nFailed examples:")
        for item in failures[:10]:
            print(f"  - {item['file']}: {item['error']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


if __name__ == "__main__":
    main()
