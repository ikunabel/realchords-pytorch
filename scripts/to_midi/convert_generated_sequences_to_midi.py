#!/usr/bin/env python3
"""Convert generated sequence tensors (.pt) to MIDI files.

Thin CLI over ``realchords.utils.sequence_penalty_analysis`` helpers, which
reuse ``HooktheoryTokenizer.decode_to_midi`` and the same sequence loading path
as ``scripts/evaluate_generated_sequences.py``.

Example:
    python scripts/to_midi/convert_generated_sequences_to_midi.py \
        --system hooktheory_gt \
        --max-sequences 4

Output layout (default):
    scripts/to_midi/output/<system_name>/<tensor_stem>/seq_0000.mid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from realchords.utils.sequence_penalty_analysis import (
    DEFAULT_CHORD_NAMES_PATH,
    DEFAULT_GENERATED_ROOT,
    DEFAULT_MIDI_OUTPUT_ROOT,
    SEQUENCE_ORDER_CHOICES,
    convert_generated_system_to_midi,
    load_tokenizer,
    resolve_generated_system_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        help="Generated system name under --generated-root, or an explicit input directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Single generated system directory (alternative to --system).",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help="Root directory containing generated system folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MIDI_OUTPUT_ROOT,
        help="Root directory where MIDI files will be written.",
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
    systems: list[str] = []
    if args.input_dir is not None:
        systems.append(str(args.input_dir))
    if args.system:
        systems.extend(args.system)
    if not systems:
        raise SystemExit("Provide at least one --system or --input-dir.")

    max_sequences = None if args.max_sequences < 0 else args.max_sequences
    tokenizer = load_tokenizer(args.chord_names_path.resolve())

    results: dict[str, dict[str, object]] = {}
    for system in systems:
        input_dir = resolve_generated_system_dir(system, args.generated_root)
        name, output_dir, midi_paths = convert_generated_system_to_midi(
            input_dir,
            args.output_dir,
            tokenizer,
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
            "output_dir": str(args.output_dir.resolve()),
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
