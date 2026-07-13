#!/usr/bin/env python3
"""Convert generated sequence tensors (.pt) to MIDI files.

Loads the interleaved chord/melody token tensors saved by
``scripts/generate_sequences.py`` and converts them to MIDI the same way the
rest of the codebase does (see ``output_to_midi`` in
``realchords/lit_module/decoder_only.py``, ``enc_dec.py``, and
``realchords/rl/rl_trainer.py``): split the alternating sequence into chord
and melody frames, then call ``HooktheoryTokenizer.decode_to_midi`` directly.

Example:
    python scripts/to_midi/convert_generated_sequences_to_midi.py \
        logs/generated/hooktheory_gt \
        --max-sequences 4

Output layout (default): MIDI files are written next to the input tensors,
inside the system directory itself, so each system is self-contained:
    <system_dir>/midi/<tensor_stem>/seq_0000.mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import note_seq.chord_symbols_lib as _chord_lib
import pretty_midi

from realchords.constants import (
    BASS_OCTAVE,
    BASS_VELOCITY,
    CHORD_OCTAVE,
    CHORD_VELOCITY,
    MELODY_VELOCITY,
    WJD_CHORD_OCTAVE,
)
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer, to_midi_pitch
from realchords.utils.sequence_penalty_analysis import (
    DEFAULT_CHORD_NAMES_PATH,
    DEFAULT_GENERATED_ROOT,
    SEQUENCE_ORDER_CHOICES,
    collect_sequence_files,
    find_vocab_for_dir,
    load_sequences,
    load_tokenizer,
    resolve_generated_system_dir,
    split_tokens_by_order,
    strip_bos,
)
from realchords.utils.voicing_selector import VoicingSelector

DEFAULT_MIDI_SUBDIR = "midi"


def _melody_pitch_at_beat(
    melody_annotations: List[dict], beat: float
) -> Optional[int]:
    """Return the MIDI pitch of whatever melody note is sounding at *beat*."""
    for note in melody_annotations:
        if note["onset"] <= beat < note["offset"]:
            return to_midi_pitch(note["octave"], note["pitch_class"])
    return None


def _dedup_pitches(pitches: List[int]) -> List[int]:
    """Remove duplicate MIDI pitches while preserving order."""
    seen: set = set()
    out = []
    for p in pitches:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _dedup_by_pc(pitches: List[int]) -> List[int]:
    """One note per pitch class; keep the lowest occurrence (preserves bass register)."""
    best: dict = {}
    for p in pitches:
        pc = p % 12
        if pc not in best or p < best[pc]:
            best[pc] = p
    return sorted(best.values())


def _naive_pitches(
    chord_name: str,
    *,
    include_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> List[int]:
    pitches = [
        p % 12 + chord_octave * 12
        for p in _chord_lib.chord_symbol_pitches(chord_name)
    ]
    if include_bass:
        pitches.append(
            _chord_lib.chord_symbol_bass(chord_name) % 12 + BASS_OCTAVE * 12
        )
    return _dedup_pitches(pitches)


def row_to_midi_combined(
    tokens,
    tokenizer: HooktheoryTokenizer,
    selector: VoicingSelector,
    *,
    sequence_order: str = "chord_first",
    bpm: int = 120,
    melody_role: str = "top",
    pause_bars: float = 1.0,
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> pretty_midi.PrettyMIDI:
    """Full naive song → pause → full voiced song in a single MIDI file.

    Layout:
        [full song, naive root-position chords]
        [1-bar pause]
        [full song, VoicingSelector chords]

    Two instruments: Melody (identical in both halves) and Chords.
    """
    if tokens.dim() != 1:
        raise ValueError(f"Expected rank-1 token sequence, got shape {tuple(tokens.shape)}")

    spb = 60.0 / bpm
    pause_sec = pause_bars * 4 * spb  # 4 beats per bar

    chord_frames, melody_frames = split_tokens_by_order(tokens, sequence_order)
    chord_annotations = tokenizer.decode_chord_frames(chord_frames.numpy())
    melody_annotations = tokenizer.decode_melody_frames(melody_frames.numpy())

    # Duration of the naive section = end of last chord or melody note
    all_offsets = (
        [c["offset"] for c in chord_annotations]
        + [m["offset"] for m in melody_annotations]
    )
    section_beats = max(all_offsets) if all_offsets else 0.0
    offset = section_beats * spb + pause_sec  # start of voiced section

    midi = pretty_midi.PrettyMIDI()
    melody_instr = pretty_midi.Instrument(program=0, name="Melody")
    chord_instr = pretty_midi.Instrument(program=0, name="Chords")

    # ---- Section 1: naive ------------------------------------------------
    for note in melody_annotations:
        melody_instr.notes.append(pretty_midi.Note(
            velocity=MELODY_VELOCITY,
            pitch=to_midi_pitch(note["octave"], note["pitch_class"]),
            start=note["onset"] * spb,
            end=note["offset"] * spb,
        ))
    for chord in chord_annotations:
        t = chord["onset"] * spb
        midi.lyrics.append(pretty_midi.Lyric(text=chord["chord_name"], time=t))
        for p in _naive_pitches(
            chord["chord_name"],
            include_bass=include_chord_bass,
            chord_octave=chord_octave,
        ):
            chord_instr.notes.append(pretty_midi.Note(
                velocity=CHORD_VELOCITY, pitch=p,
                start=t,
                end=chord["offset"] * spb,
            ))

    # ---- Section 2: voiced -----------------------------------------------
    for note in melody_annotations:
        melody_instr.notes.append(pretty_midi.Note(
            velocity=MELODY_VELOCITY,
            pitch=to_midi_pitch(note["octave"], note["pitch_class"]),
            start=note["onset"] * spb + offset,
            end=note["offset"] * spb + offset,
        ))

    selector.reset()
    for chord in chord_annotations:
        t = chord["onset"] * spb + offset
        midi.lyrics.append(pretty_midi.Lyric(text=chord["chord_name"], time=t))
        pitches = selector.select(chord["chord_name"])
        if pitches is None:
            pitches = _naive_pitches(
                chord["chord_name"],
                include_bass=include_chord_bass,
                chord_octave=chord_octave,
            )
        else:
            pitches = _dedup_pitches(list(pitches))
        for p in pitches:
            chord_instr.notes.append(pretty_midi.Note(
                velocity=CHORD_VELOCITY,
                pitch=max(0, min(127, p)),
                start=t,
                end=chord["offset"] * spb + offset,
            ))

    midi.instruments.extend([melody_instr, chord_instr])
    return midi


def row_to_midi(
    tokens,
    tokenizer: HooktheoryTokenizer,
    *,
    sequence_order: str = "chord_first",
    bpm: int = 120,
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> pretty_midi.PrettyMIDI:
    """Decode one interleaved token row into PrettyMIDI.

    Mirrors ``output_to_midi`` in ``realchords/lit_module/decoder_only.py``:
    split the alternating sequence and hand the frames straight to the
    tokenizer's own decoder.
    """
    if tokens.dim() != 1:
        raise ValueError(f"Expected rank-1 token sequence, got shape {tuple(tokens.shape)}")

    chord_frames, melody_frames = split_tokens_by_order(tokens, sequence_order)
    return tokenizer.decode_to_midi(
        chord_frames=chord_frames.numpy(),
        melody_frames=melody_frames.numpy(),
        bpm=bpm,
        include_chord_bass=include_chord_bass,
        chord_octave=chord_octave,
    )


def _select_sequence_indices(
    num_sequences: int,
    *,
    max_sequences: Optional[int],
    sequence_indices: Optional[Sequence[int]],
) -> List[int]:
    if sequence_indices is not None:
        indices = [int(index) for index in sequence_indices]
        for index in indices:
            if index < 0 or index >= num_sequences:
                raise IndexError(
                    f"Sequence index {index} out of range for {num_sequences} sequences"
                )
        return indices

    if max_sequences is None or max_sequences < 0:
        return list(range(num_sequences))
    return list(range(min(max_sequences, num_sequences)))


def convert_sequence_file_to_midi(
    input_path: Path,
    output_dir: Path,
    tokenizer: HooktheoryTokenizer,
    *,
    sequence_order: str = "chord_first",
    bpm: int = 120,
    strip_bos_token: bool = True,
    max_sequences: Optional[int] = 8,
    sequence_indices: Optional[Sequence[int]] = None,
    skip_existing: bool = False,
    voicing_selector: Optional[VoicingSelector] = None,
    melody_role: str = "top",
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> List[Path]:
    """Write MIDI files for selected rows in one generated tensor file.

    When ``voicing_selector`` is given, each sequence also produces a
    ``seq_NNNN_voiced.mid`` alongside the standard ``seq_NNNN.mid``.
    """
    sequences = load_sequences(input_path)
    if strip_bos_token:
        sequences = strip_bos(sequences, tokenizer)

    indices = _select_sequence_indices(
        sequences.size(0),
        max_sequences=max_sequences,
        sequence_indices=sequence_indices,
    )

    file_output_dir = output_dir / input_path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for index in indices:
        output_path = file_output_dir / f"seq_{index:04d}.mid"

        if skip_existing and output_path.exists():
            written.append(output_path)
            continue

        try:
            if voicing_selector is not None:
                midi = row_to_midi_combined(
                    sequences[index],
                    tokenizer,
                    voicing_selector,
                    sequence_order=sequence_order,
                    bpm=bpm,
                    melody_role=melody_role,
                    include_chord_bass=include_chord_bass,
                    chord_octave=chord_octave,
                )
            else:
                midi = row_to_midi(
                    sequences[index],
                    tokenizer,
                    sequence_order=sequence_order,
                    bpm=bpm,
                    include_chord_bass=include_chord_bass,
                    chord_octave=chord_octave,
                )
        except Exception as e:  # noqa: BLE001 - match decoder_only.py's own handling
            print(f"  seq {index}: skipped ({e})")
            continue

        midi.write(str(output_path))
        written.append(output_path)

    return written


def convert_generated_system_to_midi(
    input_dir: Path,
    tokenizer: HooktheoryTokenizer,
    *,
    output_dir: Optional[Path] = None,
    system_name: Optional[str] = None,
    sequence_order: str = "chord_first",
    bpm: int = 120,
    strip_bos_token: bool = True,
    max_sequences: Optional[int] = 8,
    sequence_indices: Optional[Sequence[int]] = None,
    skip_existing: bool = False,
    voicing_selector: Optional[VoicingSelector] = None,
    melody_role: str = "top",
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> tuple[str, Path, List[Path]]:
    """Convert all generated sequence files under one system directory to MIDI.

    By default, MIDI files are written inside the system directory itself
    (``<input_dir>/midi/<tensor_stem>/seq_0000.mid``) so each system's outputs
    stay self-contained next to its ``.pt`` tensors. Pass ``output_dir`` to
    write elsewhere instead (files are then nested under a subfolder named
    after the system, to avoid collisions between systems).
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    resolved_name = system_name or input_dir.name
    if output_dir is not None:
        system_output_dir = output_dir.expanduser().resolve() / resolved_name
    else:
        system_output_dir = input_dir / DEFAULT_MIDI_SUBDIR

    written: List[Path] = []
    for sequence_file in collect_sequence_files(input_dir):
        written.extend(
            convert_sequence_file_to_midi(
                sequence_file,
                system_output_dir,
                tokenizer,
                sequence_order=sequence_order,
                bpm=bpm,
                strip_bos_token=strip_bos_token,
                max_sequences=max_sequences,
                sequence_indices=sequence_indices,
                skip_existing=skip_existing,
                voicing_selector=voicing_selector,
                melody_role=melody_role,
                include_chord_bass=include_chord_bass,
                chord_octave=chord_octave,
            )
        )
    return resolved_name, system_output_dir, written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "system_dirs",
        nargs="+",
        help=(
            "Generated system directory (e.g. logs/generated/hooktheory_gt), "
            "or a bare system name resolved under --generated-root. Repeatable."
        ),
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help="Root directory used to resolve bare system names (not needed when passing a path).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory where MIDI files will be written. Defaults to "
            "<system_dir>/midi, keeping each system's output next to its tensors."
        ),
    )
    parser.add_argument(
        "--chord-names-path",
        type=Path,
        default=DEFAULT_CHORD_NAMES_PATH,
        help="Tokenizer chord-name mapping JSON.",
    )
    parser.add_argument(
        "--sequence-order",
        choices=SEQUENCE_ORDER_CHOICES,
        default="chord_first",
        help="Interleaving order used when the sequences were generated.",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Tempo for MIDI export.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=8,
        help="Maximum number of sequences to export per .pt file (-1 for all).",
    )
    parser.add_argument(
        "--sequence-index",
        action="append",
        type=int,
        default=None,
        help="Explicit sequence row index to export (repeatable). Overrides --max-sequences.",
    )
    parser.add_argument(
        "--keep-bos",
        action="store_true",
        help="Keep the leading BOS token instead of stripping it before decoding.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip MIDI files that already exist.",
    )
    parser.add_argument(
        "--voicings",
        type=Path,
        default=None,
        help=(
            "Path to chord_voicings.json lookup table.  When provided, each "
            "sequence also produces a seq_NNNN_voiced.mid with real-world "
            "voicings chosen via voice-leading-aware selection."
        ),
    )
    parser.add_argument(
        "--melody-role",
        choices=["top", "bass"],
        default="top",
        help=(
            "Melody role for voicing selection. 'top': chord notes stay below "
            "melody (default). 'bass': chord notes stay above melody (e.g. WJD)."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON path recording conversion outputs.",
    )
    parser.add_argument(
        "--no-chord-bass",
        action="store_true",
        help="Omit the separate bass note from chord voicings.",
    )
    parser.add_argument(
        "--include-chord-bass",
        action="store_true",
        help="Force the separate chord bass note even for wjd systems.",
    )
    return parser.parse_args()


def _resolve_include_chord_bass(args: argparse.Namespace, system_name: str) -> bool:
    if args.include_chord_bass:
        return True
    if args.no_chord_bass:
        return False
    return "wjd" not in system_name.lower()


def _resolve_chord_octave(args: argparse.Namespace, system_name: str) -> int:
    if "wjd" in system_name.lower():
        return WJD_CHORD_OCTAVE
    return CHORD_OCTAVE


def main() -> None:
    args = parse_args()

    max_sequences = None if args.max_sequences < 0 else args.max_sequences
    fallback_chord_names_path = args.chord_names_path.resolve()

    selector: Optional[VoicingSelector] = None
    if args.voicings is not None:
        voicings_path = args.voicings.expanduser().resolve()
        print(f"Loading voicings lookup: {voicings_path}")
        selector = VoicingSelector(voicings_path)

    results: Dict[str, Dict[str, object]] = {}
    for system in args.system_dirs:
        input_dir = resolve_generated_system_dir(system, args.generated_root)
        vocab_path = find_vocab_for_dir(input_dir, fallback_chord_names_path)
        tokenizer = load_tokenizer(vocab_path)
        name, output_dir, midi_paths = convert_generated_system_to_midi(
            input_dir,
            tokenizer,
            output_dir=args.output_dir,
            system_name=input_dir.name,
            sequence_order=args.sequence_order,
            bpm=args.bpm,
            strip_bos_token=not args.keep_bos,
            max_sequences=max_sequences,
            sequence_indices=args.sequence_index,
            skip_existing=args.skip_existing,
            voicing_selector=selector,
            melody_role=args.melody_role,
            include_chord_bass=_resolve_include_chord_bass(args, input_dir.name),
            chord_octave=_resolve_chord_octave(args, input_dir.name),
        )
        print(
            f"{name}: wrote {len(midi_paths)} MIDI files -> {output_dir}"
        )
        results[name] = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_sequences_written": len(midi_paths),
            "midi_paths": [str(path) for path in midi_paths],
        }

    if args.summary_path is not None:
        summary_path = args.summary_path.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_root": str(args.generated_root.resolve()),
            "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
            "sequence_order": args.sequence_order,
            "bpm": args.bpm,
            "systems": results,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
