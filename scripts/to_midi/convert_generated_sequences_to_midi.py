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

import pretty_midi

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
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

DEFAULT_MIDI_SUBDIR = "midi"


def row_to_midi(
    tokens,
    tokenizer: HooktheoryTokenizer,
    *,
    sequence_order: str = "chord_first",
    bpm: int = 120,
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
) -> List[Path]:
    """Write MIDI files for selected rows in one generated tensor file."""
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
            midi = row_to_midi(
                sequences[index],
                tokenizer,
                sequence_order=sequence_order,
                bpm=bpm,
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
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON path recording conversion outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_sequences = None if args.max_sequences < 0 else args.max_sequences
    fallback_chord_names_path = args.chord_names_path.resolve()

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
