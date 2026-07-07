#!/usr/bin/env python3
"""Evaluate note-in-chord and note-in-mode metrics.

Supports two input sources:
  - hooktheory: songs from the Hooktheory cache dataloader
  - generated: interleaved sequence tensors from ``scripts/generate_sequences.py``

Examples:
    python scripts/note_in_mode/evaluate_note_in_mode.py hooktheory \\
        --split valid --num-songs 100

    python scripts/note_in_mode/evaluate_note_in_mode.py generated \\
        --input-dir logs/generated/realchords --model-part melody
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Allow running without installing the package or setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "cache" / "hooktheory"
_DEFAULT_CHORD_NAMES_PATH = _REPO_ROOT / "data" / "cache" / "chord_names_augmented.json"

import numpy as np
import torch

from realchords.dataset.hooktheory_dataloader import HooktheoryDataset
from realchords.utils.eval_utils import (
    evaluate_melody_mode_fit_ratio,
    evaluate_note_in_chord_ratio,
)
from realchords.utils.sequence_penalty_analysis import (
    MODEL_PART_CHOICES,
    SEQUENCE_ORDER_CHOICES,
    collect_sequence_files,
    load_sequences,
    load_tokenizer,
    strip_bos,
)


def sequence_order_for_model_part(model_part: str) -> str:
    return "chord_first" if model_part == "melody" else "melody_first"


def infer_sequence_order(path: Path, model_part: str) -> str:
    name = path.name.lower()
    if "melody_order" in name:
        return "melody_first"
    if "chord_order" in name:
        return "chord_first"
    return sequence_order_for_model_part(model_part)


def encode_interleaved_song(
    item: Dict[str, object],
    tokenizer,
    model_part: str,
) -> torch.Tensor:
    encoded = tokenizer.encode(item)
    melody = encoded["melody"].tolist()
    chord = encoded["chord"].tolist()
    if len(melody) != len(chord):
        length = min(len(melody), len(chord))
        melody = melody[:length]
        chord = chord[:length]

    interleaved: List[int] = []
    for melody_token, chord_token in zip(melody, chord):
        if model_part == "melody":
            interleaved.extend([chord_token, melody_token])
        else:
            interleaved.extend([melody_token, chord_token])
    return torch.tensor([interleaved], dtype=torch.long)


def evaluate_sequences(
    sequences: torch.Tensor,
    tokenizer,
    model_part: str,
    sequence_order: str,
    scoring: str,
    sigma: float,
) -> Dict[str, object]:
    note_in_chord, valid_counts, correct_counts = evaluate_note_in_chord_ratio(
        sequences,
        tokenizer,
        model_part=model_part,
        return_count=True,
        sequence_order=sequence_order,
    )
    mode_fit, melody_weights, segment_counts = evaluate_melody_mode_fit_ratio(
        sequences,
        tokenizer,
        model_part=model_part,
        sequence_order=sequence_order,
        scoring=scoring,
        sigma=sigma,
        return_count=True,
    )

    per_sequence: List[Dict[str, object]] = []
    note_in_chord_values: List[float] = []
    mode_fit_values: List[float] = []

    for index in range(sequences.size(0)):
        nic = float(note_in_chord[index].item())
        mmf = float(mode_fit[index].item())
        if not np.isnan(nic):
            note_in_chord_values.append(nic)
        if not np.isnan(mmf):
            mode_fit_values.append(mmf)
        per_sequence.append(
            {
                "index": index,
                "note_in_chord_ratio": None if np.isnan(nic) else nic,
                "mode_fit_ratio": None if np.isnan(mmf) else mmf,
                "mode_fit_segments": int(segment_counts[index].item()),
                "mode_fit_melody_weight": float(melody_weights[index].item()),
                "valid_frames": int(valid_counts[index].item()),
                "correct_frames": int(correct_counts[index].item()),
            }
        )

    total_valid_frames = int(valid_counts.sum().item())
    total_correct_frames = int(correct_counts.sum().item())
    total_melody_weight = float(melody_weights.sum().item())
    weighted_mode_fit = 0.0
    if total_melody_weight > 0:
        weighted_mode_fit = float(
            (mode_fit * melody_weights).sum().item() / total_melody_weight
        )

    summary = {
        "num_sequences": int(sequences.size(0)),
        "num_sequences_with_note_in_chord": len(note_in_chord_values),
        "num_sequences_with_mode_fit": len(mode_fit_values),
        "mean_note_in_chord_ratio": (
            float(np.mean(note_in_chord_values)) if note_in_chord_values else None
        ),
        "mean_mode_fit_ratio": (
            float(np.mean(mode_fit_values)) if mode_fit_values else None
        ),
        "median_note_in_chord_ratio": (
            float(np.median(note_in_chord_values)) if note_in_chord_values else None
        ),
        "median_mode_fit_ratio": (
            float(np.median(mode_fit_values)) if mode_fit_values else None
        ),
        "overall_note_in_chord_ratio": (
            float(total_correct_frames / total_valid_frames)
            if total_valid_frames
            else None
        ),
        "overall_mode_fit_ratio": (
            weighted_mode_fit if total_melody_weight > 0 else None
        ),
        "total_valid_frames": total_valid_frames,
        "total_correct_frames": total_correct_frames,
        "total_mode_fit_melody_weight": total_melody_weight,
        "total_mode_fit_segments": int(segment_counts.sum().item()),
        "scoring": scoring,
        "sigma": sigma,
        "model_part": model_part,
        "sequence_order": sequence_order,
    }
    return {"summary": summary, "sequences": per_sequence}


def print_summary(summary: Dict[str, object], label: str) -> None:
    print(label)
    if "split" in summary:
        print(f"  Split: {summary['split']}")
    print(f"  Sequences evaluated: {summary['num_sequences']}")
    if summary.get("overall_note_in_chord_ratio") is not None:
        print(
            "  Overall note-in-chord: "
            f"{summary['overall_note_in_chord_ratio']:.4f}"
        )
    if summary.get("overall_mode_fit_ratio") is not None:
        print(
            f"  Overall note-in-mode ({summary['scoring']}): "
            f"{summary['overall_mode_fit_ratio']:.4f}"
        )
    if summary.get("mean_note_in_chord_ratio") is not None:
        print(f"  Mean note-in-chord: {summary['mean_note_in_chord_ratio']:.4f}")
    if summary.get("mean_mode_fit_ratio") is not None:
        print(
            f"  Mean note-in-mode ({summary['scoring']}): "
            f"{summary['mean_mode_fit_ratio']:.4f}"
        )
    if summary.get("median_note_in_chord_ratio") is not None:
        print(f"  Median note-in-chord: {summary['median_note_in_chord_ratio']:.4f}")
    if summary.get("median_mode_fit_ratio") is not None:
        print(f"  Median note-in-mode: {summary['median_mode_fit_ratio']:.4f}")


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chord-names-path",
        type=Path,
        default=_DEFAULT_CHORD_NAMES_PATH,
        help="Path to chord_names JSON vocabulary.",
    )
    parser.add_argument(
        "--model-part",
        choices=MODEL_PART_CHOICES,
        default="melody",
        help="Model part used when interleaving generated sequences.",
    )
    parser.add_argument(
        "--sequence-order",
        choices=SEQUENCE_ORDER_CHOICES,
        default=None,
        help="Interleaving order. Defaults from model_part or .pt filename.",
    )
    parser.add_argument(
        "--scoring",
        choices=("coverage", "distance"),
        default="coverage",
        help="Note-in-mode scoring method.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian kernel width for distance scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )


def build_hooktheory_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "hooktheory",
        help="Evaluate Hooktheory cache songs.",
    )
    add_shared_args(parser)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Hooktheory cache directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="valid",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--num-songs",
        type=int,
        default=100,
        help="Number of songs to evaluate from the start of the split.",
    )


def build_generated_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "generated",
        help="Evaluate generated interleaved .pt sequence files.",
    )
    add_shared_args(parser)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing generated .pt tensors.",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=None,
        help="Optional single .pt file to evaluate instead of scanning input-dir.",
    )


def load_hooktheory_dataset(args: argparse.Namespace) -> HooktheoryDataset:
    return HooktheoryDataset(
        cache_dir=str(args.cache_dir),
        split=args.split,
        model_type="decoder_only",
        model_part=args.model_part,
        max_len=512,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=str(args.chord_names_path),
    )


def resolve_sequence_order(args: argparse.Namespace, path: Path | None = None) -> str:
    if args.sequence_order is not None:
        return args.sequence_order
    if path is not None:
        return infer_sequence_order(path, args.model_part)
    return sequence_order_for_model_part(args.model_part)


def evaluate_hooktheory(args: argparse.Namespace) -> Dict[str, object]:
    dataset = load_hooktheory_dataset(args)
    tokenizer = dataset.tokenizer
    sequence_order = resolve_sequence_order(args)
    num_songs = min(args.num_songs, len(dataset))

    sources: List[Dict[str, object]] = []
    note_in_chord_values: List[float] = []
    mode_fit_values: List[float] = []
    total_valid_frames = 0
    total_correct_frames = 0
    total_melody_weight = 0.0
    weighted_mode_fit_sum = 0.0

    for index in range(num_songs):
        item = dataset.data[index]
        sequences = encode_interleaved_song(item, tokenizer, args.model_part)
        result = evaluate_sequences(
            sequences,
            tokenizer,
            model_part=args.model_part,
            sequence_order=sequence_order,
            scoring=args.scoring,
            sigma=args.sigma,
        )
        summary = result["summary"]
        seq = result["sequences"][0]

        if summary["mean_note_in_chord_ratio"] is not None:
            note_in_chord_values.append(summary["mean_note_in_chord_ratio"])
        if summary["mean_mode_fit_ratio"] is not None:
            mode_fit_values.append(summary["mean_mode_fit_ratio"])
        total_valid_frames += int(summary["total_valid_frames"])
        total_correct_frames += int(summary["total_correct_frames"])
        total_melody_weight += float(summary["total_mode_fit_melody_weight"])
        if summary["overall_mode_fit_ratio"] is not None:
            weighted_mode_fit_sum += (
                summary["overall_mode_fit_ratio"]
                * summary["total_mode_fit_melody_weight"]
            )

        song_url = "unknown"
        if isinstance(item, dict) and "hooktheory" in item:
            song_url = item["hooktheory"]["urls"]["song"]

        sources.append(
            {
                "index": index,
                "song_url": song_url,
                "note_in_chord_ratio": seq["note_in_chord_ratio"],
                "mode_fit_ratio": seq["mode_fit_ratio"],
                "mode_fit_segments": seq["mode_fit_segments"],
                "mode_fit_melody_weight": seq["mode_fit_melody_weight"],
                "num_frames": int(sequences.size(1) // 2),
            }
        )

    overall_summary = {
        "source": "hooktheory",
        "split": args.split,
        "num_songs_requested": args.num_songs,
        "num_sequences": num_songs,
        "mean_note_in_chord_ratio": (
            float(np.mean(note_in_chord_values)) if note_in_chord_values else None
        ),
        "mean_mode_fit_ratio": (
            float(np.mean(mode_fit_values)) if mode_fit_values else None
        ),
        "median_note_in_chord_ratio": (
            float(np.median(note_in_chord_values)) if note_in_chord_values else None
        ),
        "median_mode_fit_ratio": (
            float(np.median(mode_fit_values)) if mode_fit_values else None
        ),
        "overall_note_in_chord_ratio": (
            float(total_correct_frames / total_valid_frames)
            if total_valid_frames
            else None
        ),
        "overall_mode_fit_ratio": (
            float(weighted_mode_fit_sum / total_melody_weight)
            if total_melody_weight > 0
            else None
        ),
        "scoring": args.scoring,
        "sigma": args.sigma,
        "model_part": args.model_part,
        "sequence_order": sequence_order,
        "sources": sources,
    }
    return {"summary": overall_summary}


def collect_generated_files(args: argparse.Namespace) -> List[Path]:
    if args.sequences is not None:
        path = args.sequences.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")
        return [path]
    input_dir = args.input_dir.expanduser().resolve()
    return collect_sequence_files(input_dir)


def evaluate_generated(args: argparse.Namespace) -> Dict[str, object]:
    tokenizer = load_tokenizer(args.chord_names_path)
    sequence_files = collect_generated_files(args)

    file_results: List[Dict[str, object]] = []
    total_sequences = 0
    total_valid_frames = 0
    total_correct_frames = 0
    total_melody_weight = 0.0
    weighted_mode_fit_sum = 0.0
    note_in_chord_values: List[float] = []
    mode_fit_values: List[float] = []

    for path in sequence_files:
        sequences = strip_bos(load_sequences(path), tokenizer)
        sequence_order = resolve_sequence_order(args, path)
        result = evaluate_sequences(
            sequences,
            tokenizer,
            model_part=args.model_part,
            sequence_order=sequence_order,
            scoring=args.scoring,
            sigma=args.sigma,
        )
        summary = result["summary"]
        total_sequences += int(summary["num_sequences"])
        total_valid_frames += int(summary["total_valid_frames"])
        total_correct_frames += int(summary["total_correct_frames"])
        total_melody_weight += float(summary["total_mode_fit_melody_weight"])
        if summary["overall_mode_fit_ratio"] is not None:
            weighted_mode_fit_sum += (
                summary["overall_mode_fit_ratio"]
                * summary["total_mode_fit_melody_weight"]
            )
        for seq in result["sequences"]:
            if seq["note_in_chord_ratio"] is not None:
                note_in_chord_values.append(seq["note_in_chord_ratio"])
            if seq["mode_fit_ratio"] is not None:
                mode_fit_values.append(seq["mode_fit_ratio"])

        file_results.append(
            {
                "path": str(path),
                "sequence_order": sequence_order,
                **summary,
                "sequences": result["sequences"],
            }
        )

    overall_summary = {
        "source": "generated",
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "num_sequence_files": len(sequence_files),
        "num_sequences": total_sequences,
        "mean_note_in_chord_ratio": (
            float(np.mean(note_in_chord_values)) if note_in_chord_values else None
        ),
        "mean_mode_fit_ratio": (
            float(np.mean(mode_fit_values)) if mode_fit_values else None
        ),
        "median_note_in_chord_ratio": (
            float(np.median(note_in_chord_values)) if note_in_chord_values else None
        ),
        "median_mode_fit_ratio": (
            float(np.median(mode_fit_values)) if mode_fit_values else None
        ),
        "overall_note_in_chord_ratio": (
            float(total_correct_frames / total_valid_frames)
            if total_valid_frames
            else None
        ),
        "overall_mode_fit_ratio": (
            float(weighted_mode_fit_sum / total_melody_weight)
            if total_melody_weight > 0
            else None
        ),
        "scoring": args.scoring,
        "sigma": args.sigma,
        "model_part": args.model_part,
        "files": file_results,
    }
    return {"summary": overall_summary}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)
    build_hooktheory_parser(subparsers)
    build_generated_parser(subparsers)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.source == "hooktheory":
        results = evaluate_hooktheory(args)
    elif args.source == "generated":
        results = evaluate_generated(args)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    print_summary(results["summary"], label=f"Source: {results['summary']['source']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
