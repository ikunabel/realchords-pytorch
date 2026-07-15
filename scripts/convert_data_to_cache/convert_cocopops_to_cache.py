#!/usr/bin/env python3
"""Convert CoCoPops Humdrum (.hum) files to Hooktheory-compatible cache format.

Parses paired melody (**kern) and chord (**harte, timed via **harm) spines from
CoCoPops transcriptions and writes JSONL cache files compatible with
``HooktheoryDataset``.

Melody extraction uses music21 on a single selected **kern spine because full
CoCoPops files also contain **harm Roman numerals that music21 cannot parse.
Chord symbols come from the **harte spine with rhythmic durations taken from
the **harm spine (CoCoPops convention).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from music21 import converter
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_wikifonia_to_cache import (
    _is_no_chord_symbol,
    collect_chord_names,
    create_augmented_dataset,
    filter_zero_duration_chords,
    parse_chord_symbol_with_noteseq,
    quantize_timing_to_beat_grid,
    resolve_melody_overlaps,
    set_chord_symbol_parse_verbose,
    split_dataset,
)
from realchords.constants import ZERO_OCTAVE
from realchords.utils.data_utils import to_chord_name, update_global_chord_names
from realchords.utils.io_utils import save_jsonl

_NO_CHORD_TOKENS = {"", ".", "N", "n", "X", "x", "NC", "nc"}
_KERN_EXCL = "**kern"
_HARM_EXCL = "**harm"
_HARTE_EXCL = "**harte"
_RECIP_RE = re.compile(r"^(\d+)(\.+)?")
_REFERENCE_RE = re.compile(r"^!!!(\w+):\s*(.*)$")
_TB_RE = re.compile(r"^\*tb(\d+)$")
# Smallest quantized grid unit used elsewhere in the pipeline (FRAME_PER_BEAT=4).
# Used as a row-duration fallback when neither **harm nor **kern carries a
# recip-encoded duration on a row that still changes the **harte chord (e.g. an
# off-grid chord-only annotation), so the clock keeps advancing and the chord
# change isn't silently dropped.
_FALLBACK_ROW_DURATION = 0.25


def humdrum_root_to_leadsheet(root: str) -> str:
    """Convert a Humdrum root (e.g. ``B-``, ``F#``) to lead-sheet spelling."""
    if not root:
        return ""
    letter = root[0].upper()
    suffix = root[1:]
    out = letter
    while suffix.startswith("-"):
        out += "b"
        suffix = suffix[1:]
    while suffix.startswith("#"):
        out += "#"
        suffix = suffix[1:]
    return out


def transform_harte_chord_symbol(symbol: str) -> str:
    """Map CoCoPops **harte tokens to note_seq lead-sheet symbols."""
    token = symbol.strip()
    if _is_no_chord_symbol(token) or token in _NO_CHORD_TOKENS:
        return ""

    if ":" not in token:
        return humdrum_root_to_leadsheet(token)

    root, quality = token.split(":", 1)
    root = humdrum_root_to_leadsheet(root)
    if not root:
        return ""

    bass = None
    if "/" in quality:
        quality, bass_part = quality.split("/", 1)
        bass = humdrum_root_to_leadsheet(bass_part)

    quality = quality.strip().lower()
    quality_map = {
        "": "",
        "maj": "",
        "major": "",
        "min": "m",
        "m": "m",
        "minor": "m",
        "min7": "m7",
        "m7": "m7",
        "maj7": "maj7",
        "major7": "maj7",
        "7": "7",
        "dim": "dim",
        "dim7": "dim7",
        "hdim7": "m7b5",
        "aug": "aug",
        "sus4": "sus4",
        "sus2": "sus2",
        "1": "",
        "5": "",
    }
    suffix = quality_map.get(quality, quality.replace("min", "m").replace("maj", ""))
    lead = f"{root}{suffix}"
    if bass:
        lead = f"{lead}/{bass}"
    return lead


def parse_recip_duration(token: str) -> Optional[float]:
    """Return Humdrum **recip duration in quarter-note units."""
    if not token or token == ".":
        return None
    token = token.lstrip("[")
    match = _RECIP_RE.match(token)
    if not match:
        return None
    denom = int(match.group(1))
    if denom <= 0:
        return None
    duration = 4.0 / denom
    dots = match.group(2) or ""
    if dots:
        extra = duration / 2.0
        for _ in dots:
            duration += extra
            extra /= 2.0
    return duration


def parse_tb_value(token: str) -> Optional[int]:
    """Read a ``*tb<N>`` timebase tandem: step size is ``4/N`` quarter notes.

    RollingStone files encode rhythm as a step-sequence (see README.md), with
    ``*tb`` interpretations declaring how many rows make up a whole note --
    e.g. ``*tb8`` means every subsequent data row is one eighth note, until
    the next ``*tb`` change.
    """
    match = _TB_RE.match(token.strip())
    if not match:
        return None
    return int(match.group(1))


def _is_data_record(line: str) -> bool:
    if not line or line.startswith("!"):
        return False
    if line.startswith("*") and not line.startswith("**"):
        return False
    if line.startswith("="):
        return False
    if line.startswith("*>"):
        return False
    return True


def _read_hum_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    # Global reference records ("!!!KEY: value", e.g. RollingStone's leading
    # rank/title/composer block) legally precede the exclusive interpretation
    # line, so the spine-name row isn't always line 0 -- scan for the first
    # line whose first field actually starts with "**".
    header_idx = None
    for idx, line in enumerate(lines):
        first_field = line.split("\t", 1)[0]
        if first_field.startswith("**"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"No exclusive interpretation line found: {path}")
    spine_names = lines[header_idx].split("\t")
    rows: List[List[str]] = []
    for line in lines[header_idx + 1 :]:
        if line.startswith("!!!"):
            continue
        if line.startswith("*-"):
            break
        rows.append(line.split("\t"))
    return spine_names, rows


def _spine_indices(spine_names: Sequence[str], exclusive: str) -> List[int]:
    return [idx for idx, name in enumerate(spine_names) if name == exclusive]


def _select_kern_index(spine_names: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    kern_indices = _spine_indices(spine_names, _KERN_EXCL)
    if not kern_indices:
        raise ValueError("No **kern spine found")

    haupt = set()
    lead = set()
    for row in rows:
        if not row or row[0].startswith("*"):
            continue
        for idx in kern_indices:
            if idx >= len(row):
                continue
            token = row[idx]
            if token == "*Hstimme":
                haupt.add(idx)
            elif token == "*VRlead":
                lead.add(idx)

    if len(haupt) == 1:
        return next(iter(haupt))
    if len(haupt) > 1 and lead:
        for idx in haupt:
            if idx in lead:
                return idx
        return min(haupt)
    if lead:
        return min(lead)
    if len(kern_indices) == 1:
        return kern_indices[0]
    return kern_indices[0]


def _kern_spine_text(rows: Sequence[Sequence[str]], kern_idx: int) -> str:
    lines = ["**kern"]
    for row in rows:
        if len(row) <= kern_idx:
            continue
        token = row[kern_idx]
        if token == "*-":
            lines.append("*-")
            break
        lines.append(token)
    return "\n".join(lines) + "\n"


_KERN_PITCH_RE = re.compile(r"([A-Ga-g]+)([#\-n]*)")
_KERN_LETTER_TO_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def kern_token_to_midi_pitch(token: str) -> Optional[int]:
    """Convert a **kern pitch token (e.g. ``cc#``, ``B-``) to a MIDI pitch.

    Only used for RollingStone's *tb step-sequence **kern spine, which has no
    **recip prefix for music21/Verovio to key off of -- so we read pitch
    letters directly rather than routing through a Humdrum-aware parser.
    """
    match = _KERN_PITCH_RE.search(token.strip())
    if not match:
        return None
    letters, accidentals = match.groups()
    pitch_class = _KERN_LETTER_TO_PC[letters[0].lower()]
    for ch in accidentals:
        if ch == "#":
            pitch_class += 1
        elif ch == "-":
            pitch_class -= 1
    if letters[0].islower():
        octave = 4 + (len(letters) - 1)
    else:
        octave = 3 - (len(letters) - 1)
    return pitch_class + (octave + 1) * 12


def extract_melody_notes_timebase(
    rows: Sequence[Sequence[str]],
    kern_idx: int,
) -> List[Dict]:
    """Extract melody notes from a *tb step-sequence **kern spine.

    De Clercq/Temperley's step-sequence transcription records only note
    onsets (no durations, no rest tokens; see RollingStone/README.md), so
    each note is sustained until the next onset.
    """
    onsets: List[Tuple[float, int]] = []
    current_time = 0.0
    tb_value: Optional[int] = None

    for row in rows:
        if not row:
            continue
        line = "\t".join(row)
        if not _is_data_record(line):
            token = row[kern_idx] if kern_idx < len(row) else ""
            tb = parse_tb_value(token)
            if tb is not None:
                tb_value = tb
            continue

        if tb_value is None:
            continue

        token = row[kern_idx] if kern_idx < len(row) else "."
        if token not in ("", "."):
            pitch = kern_token_to_midi_pitch(token)
            if pitch is not None:
                onsets.append((current_time, pitch))
        current_time += 4.0 / tb_value

    notes: List[Dict] = []
    for i, (onset, pitch) in enumerate(onsets):
        offset = onsets[i + 1][0] if i + 1 < len(onsets) else current_time
        if offset <= onset:
            continue
        notes.append({"onset": onset, "offset": offset, "pitch": pitch})
    return notes


def _midi_to_hooktheory(midi_pitch: int) -> Tuple[int, int]:
    # Python's // already floors toward -infinity and % is non-negative, so
    # semitone == octave * 12 + pitch_class holds exactly -- matching
    # to_midi_pitch()'s inverse. No extra adjustment needed for negative
    # semitones (see scripts/convert_data_to_cache/convert_wikifonia_to_cache.py for the same
    # pattern without the redundant decrement).
    semitone = midi_pitch - ZERO_OCTAVE
    octave = semitone // 12
    pitch_class = semitone % 12
    return pitch_class, octave


def extract_melody_from_kern(kern_text: str) -> List[Dict]:
    """Parse a single-spine **kern Humdrum snippet with music21."""
    score = converter.parseData(kern_text, format="humdrum")
    if not score.parts:
        return []

    melody: List[Dict] = []
    for element in score.parts[0].flatten().notes:
        if not hasattr(element, "pitch"):
            continue
        onset = float(element.offset)
        offset = onset + float(element.quarterLength)
        pitch_class, octave = _midi_to_hooktheory(int(element.pitch.midi))
        melody.append(
            {
                "onset": onset,
                "offset": offset,
                "pitch_class": pitch_class,
                "octave": octave,
            }
        )
    return melody


def extract_harmony_from_spines(
    rows: Sequence[Sequence[str]],
    harm_idx: int,
    harte_idx: int,
    kern_idx: int,
) -> List[Dict]:
    """Extract chord segments using **harm rhythm and **harte labels.

    Handles both standard **recip-encoded files (Billboard) and RollingStone's
    ``*tb`` step-sequence encoding, where **harm/**harte/**kern tokens carry no
    recip prefix at all and row duration must be read from the active ``*tb``
    tandem interpretation instead (see :func:`parse_tb_value`).
    """
    chords: List[Dict] = []
    current_time = 0.0
    active_symbol: Optional[str] = None
    active_start: Optional[float] = None
    tb_value: Optional[int] = None

    def close_active(end_time: float) -> None:
        nonlocal active_symbol, active_start
        if active_symbol is None or active_start is None:
            return
        if end_time > active_start:
            chords.append(
                {
                    "onset": active_start,
                    "offset": end_time,
                    "symbol": active_symbol,
                }
            )
        active_symbol = None
        active_start = None

    for row in rows:
        if not row:
            continue
        line = "\t".join(row)
        if not _is_data_record(line):
            for idx in (harm_idx, kern_idx):
                token = row[idx] if idx < len(row) else ""
                tb = parse_tb_value(token)
                if tb is not None:
                    tb_value = tb
            continue

        harm_token = row[harm_idx] if harm_idx < len(row) else "."
        harte_token = row[harte_idx] if harte_idx < len(row) else "."
        kern_token = row[kern_idx] if kern_idx < len(row) else "."

        if tb_value is not None:
            row_dt = 4.0 / tb_value
        else:
            row_durations: List[float] = []
            for token in (harm_token, kern_token):
                duration = parse_recip_duration(token)
                if duration is not None:
                    row_durations.append(duration)
            row_dt = min(row_durations) if row_durations else _FALLBACK_ROW_DURATION

        if harte_token not in _NO_CHORD_TOKENS:
            if _is_no_chord_symbol(harte_token) or harte_token in {"N", "n", "X", "x", "NC", "nc"}:
                close_active(current_time)
            elif transform_harte_chord_symbol(harte_token):
                if active_symbol != harte_token:
                    close_active(current_time)
                    active_symbol = harte_token
                    active_start = current_time
            else:
                close_active(current_time)
        elif harte_token in {"N", "n", "X", "x", "NC", "nc"}:
            close_active(current_time)

        current_time += row_dt

    close_active(current_time)
    return chords


def _parse_reference_metadata(rows: Sequence[Sequence[str]], path: Path) -> Dict[str, Optional[str]]:
    metadata = {
        "title": path.stem,
        "composer": None,
        "source": "CoCoPops Dataset",
        "file": path.name,
    }
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = _REFERENCE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key == "OTL":
            metadata["title"] = value
        elif key == "COC":
            metadata["composer"] = value
    return metadata


def _parse_key_tonic(rows: Sequence[Sequence[str]], spine_names: Sequence[str]) -> Tuple[int, List[int]]:
    """Return tonic pitch class and major-scale intervals from tandem keys."""
    key_indices = [
        idx
        for idx, name in enumerate(spine_names)
        if name in {_HARM_EXCL, _HARTE_EXCL, _KERN_EXCL}
    ]
    tonic_name = "C"
    mode = "major"
    for row in rows:
        if not row or row[0].startswith("="):
            continue
        for idx in key_indices:
            if idx >= len(row):
                continue
            token = row[idx]
            if not token.startswith("*") or token.startswith("**"):
                continue
            if token.endswith(":"):
                tonic_name = token[1:-1]
                mode = "major"
            elif token.endswith(":min") or token.endswith(":minor"):
                tonic_name = token[1:].split(":")[0]
                mode = "minor"

    letter_to_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    tonic = humdrum_root_to_leadsheet(tonic_name)
    base_letter = tonic[0].upper()
    tonic_pc = letter_to_pc.get(base_letter, 0)
    if "b" in tonic[1:]:
        tonic_pc = (tonic_pc - tonic[1:].count("b")) % 12
    if "#" in tonic:
        tonic_pc = (tonic_pc + tonic.count("#")) % 12

    if mode == "minor":
        scale = [2, 1, 2, 2, 1, 2, 2]
    else:
        scale = [2, 2, 1, 2, 2, 2, 1]
    return tonic_pc, scale


def _uses_timebase(rows: Sequence[Sequence[str]]) -> bool:
    for row in rows:
        for token in row:
            if token.startswith("*tb"):
                return True
    return False


def _uses_non_four_four(rows: Sequence[Sequence[str]], spine_names: Sequence[str]) -> bool:
    meter_indices = [
        idx
        for idx, name in enumerate(spine_names)
        if name in {_HARM_EXCL, _HARTE_EXCL, _KERN_EXCL}
    ]
    for row in rows:
        for idx in meter_indices:
            if idx < len(row) and row[idx].startswith("*M") and row[idx] != "*M4/4":
                return True
    return False


def process_hum_file(
    hum_path: Path,
    *,
    dataset_key: str = "cocopops",
    source_label: str = "CoCoPops Dataset",
    allow_non_four_four: bool = False,
    allow_timebase: bool = False,
) -> Optional[Dict]:
    """Convert one CoCoPops ``.hum`` file into Hooktheory cache format."""
    spine_names, rows = _read_hum_rows(hum_path)
    if _HARM_EXCL not in spine_names or _HARTE_EXCL not in spine_names:
        return None
    if _KERN_EXCL not in spine_names:
        return None

    if _uses_timebase(rows) and not allow_timebase:
        return None
    if _uses_non_four_four(rows, spine_names) and not allow_non_four_four:
        return None

    harm_idx = spine_names.index(_HARM_EXCL)
    harte_idx = spine_names.index(_HARTE_EXCL)
    kern_idx = _select_kern_index(spine_names, rows)

    kern_text = _kern_spine_text(rows, kern_idx)
    raw_melody = extract_melody_from_kern(kern_text)
    raw_chords = extract_harmony_from_spines(rows, harm_idx, harte_idx, kern_idx)

    if not raw_melody or not raw_chords:
        return None

    harmony: List[Dict] = []
    for chord in raw_chords:
        root_pc, intervals, inversion = parse_chord_symbol_with_noteseq(
            chord["symbol"],
            chord_symbol_transform=transform_harte_chord_symbol,
        )
        if not intervals:
            continue
        harmony.append(
            {
                "onset": chord["onset"],
                "offset": chord["offset"],
                "root_pitch_class": root_pc,
                "root_position_intervals": intervals,
                "inversion": inversion,
            }
        )

    if not harmony:
        return None

    melody = quantize_timing_to_beat_grid(raw_melody, resolution=0.25)
    harmony = quantize_timing_to_beat_grid(harmony, resolution=0.25)
    melody = resolve_melody_overlaps(melody)
    harmony = filter_zero_duration_chords(harmony)

    if not melody or not harmony:
        return None

    max_offset = max(
        max(note["offset"] for note in melody),
        max(chord["offset"] for chord in harmony),
    )
    tonic_pc, scale_intervals = _parse_key_tonic(rows, spine_names)
    metadata = _parse_reference_metadata(rows, hum_path)

    return {
        "tags": ["MELODY", "HARMONY", "NO_SWING", "COCOPOPS"],
        "split": "TRAIN",
        dataset_key: {
            "id": hum_path.stem,
            "title": metadata["title"],
            "composer": metadata["composer"],
            "source": source_label,
            "file": metadata["file"],
            "time_signature": "4/4",
            "key_signature": None,
        },
        "annotations": {
            "num_beats": int(max_offset) if max_offset > 0 else 32,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "keys": [
                {
                    "beat": 0,
                    "tonic_pitch_class": tonic_pc,
                    "scale_degree_intervals": scale_intervals,
                }
            ],
            "melody": melody,
            "harmony": harmony,
        },
    }


def _default_cocopops_path() -> Path:
    """Prefer the cloned GitHub repo name over the lowercase samples folder."""
    for candidate in (Path("data/CoCoPops"), Path("data/cocopops")):
        if candidate.exists():
            return candidate
    return Path("data/CoCoPops")


def discover_hum_files(data_path: Path) -> List[Path]:
    """Find canonical CoCoPops ``.hum`` files under ``*/Data/`` directories."""
    files = sorted(
        path
        for path in data_path.rglob("*.hum")
        if not path.name.endswith(".varms.hum")
        and "/Resources/" not in path.as_posix()
        and path.parent.name == "Data"
    )
    return files


def convert_cocopops_corpus(
    hum_files: List[Path],
    output_dir: Path,
    *,
    augmentation: bool = False,
    max_files: Optional[int] = None,
    allow_non_four_four: bool = False,
    allow_timebase: bool = False,
) -> Dict[str, int]:
    """Convert CoCoPops ``.hum`` files to cache JSONL splits."""
    if max_files is not None:
        hum_files = hum_files[:max_files]
    if not hum_files:
        print("No .hum files found for CoCoPops")
        return {"total_files": 0, "processed": 0, "failed": 0}

    print(f"Found {len(hum_files)} CoCoPops .hum files to process")

    all_songs: List[Dict] = []
    failed = 0
    for hum_file in tqdm(hum_files, desc="Processing CoCoPops"):
        try:
            song = process_hum_file(
                hum_file,
                allow_non_four_four=allow_non_four_four,
                allow_timebase=allow_timebase,
            )
            if song:
                all_songs.append(song)
            else:
                failed += 1
        except Exception as exc:
            print(f"Error processing {hum_file.name}: {exc}")
            failed += 1

    print(f"Successfully processed {len(all_songs)} songs ({failed} failed/skipped)")
    if not all_songs:
        print("No songs were successfully processed!")
        return {
            "total_files": len(hum_files),
            "processed": 0,
            "failed": failed,
        }

    splits = split_dataset(all_songs)
    print("Dataset splits:")
    print(f"  Train: {len(splits['train'])} songs")
    print(f"  Valid: {len(splits['valid'])} songs")
    print(f"  Test: {len(splits['test'])} songs")

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
        "total_files": len(hum_files),
        "processed": len(all_songs),
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cocopops_path",
        type=str,
        default=None,
        help="Root directory of the CoCoPops repository (default: data/CoCoPops if present)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/cocopops",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of .hum files to process (for testing)",
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
        "--allow-non-four-four",
        action="store_true",
        help="Include songs whose meter is not 4/4",
    )
    parser.add_argument(
        "--allow-timebase",
        action="store_true",
        help="Include Rolling Stone *tb timebase files (experimental)",
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

    data_path = Path(args.cocopops_path) if args.cocopops_path else _default_cocopops_path()
    hum_files = discover_hum_files(data_path)
    if args.max_files is not None:
        hum_files = hum_files[: args.max_files]

    if args.report_only:
        ok = 0
        failed: List[str] = []
        for hum_file in tqdm(hum_files, desc="Scanning CoCoPops"):
            try:
                song = process_hum_file(
                    hum_file,
                    allow_non_four_four=args.allow_non_four_four,
                    allow_timebase=args.allow_timebase,
                )
                if song:
                    ok += 1
                else:
                    failed.append(hum_file.name)
            except Exception as exc:
                failed.append(f"{hum_file.name}: {exc}")

        print(f"Found {len(hum_files)} .hum files")
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

    stats = convert_cocopops_corpus(
        hum_files,
        output_dir,
        augmentation=args.augmentation,
        max_files=args.max_files,
        allow_non_four_four=args.allow_non_four_four,
        allow_timebase=args.allow_timebase,
    )
    if stats["processed"]:
        print("CoCoPops dataset conversion completed!")


if __name__ == "__main__":
    main()
