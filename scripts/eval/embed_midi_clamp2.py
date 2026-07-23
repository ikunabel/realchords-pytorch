#!/usr/bin/env python3
"""Embed a GT MIDI file with CLaMP2, as a first pilot step for the
CLaMP2-as-judge idea (see NotaGen's CLaMP-DPO: cosine similarity between a
candidate's embedding and a ground-truth centroid as a data-likeness score).

Uses the ``frechet_music_distance`` package's ``CLaMP2Extractor``, which wraps
CLaMP2 (Wu et al., 2024, "CLaMP 2: Multimodal Music Information Retrieval
Across 101 Languages") end-to-end (MIDI -> MTF -> M3 patch encoder -> pooled
embedding) -- no need to reimplement the MIDI-to-MTF conversion by hand. The
checkpoint is downloaded once to ``~/.cache/frechet_music_distance/checkpoints/clamp2/``
and reused on subsequent runs.

By default, embeds the first (alphabetically sorted) MIDI file under
hooktheory's full-songs GT export from ``custom_evaluation.py``.

Usage::

    python scripts/eval/embed_midi_clamp2.py
    python scripts/eval/embed_midi_clamp2.py --midi_dir path/to/midi --index 0
    python scripts/eval/embed_midi_clamp2.py --midi_path path/to/song.mid --output embedding.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--midi_dir",
        type=str,
        default="logs/paired_eval/gt/hooktheory_all/full_songs/midi",
        help="Directory of GT MIDI files to pick from (ignored if --midi_path is set).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which file to embed, by alphabetical sort order within --midi_dir.",
    )
    parser.add_argument(
        "--midi_path",
        type=str,
        default=None,
        help="Embed this specific MIDI file directly, instead of picking from --midi_dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, save the embedding as a .npy file at this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.midi_path:
        midi_path = Path(args.midi_path)
    else:
        midi_dir = Path(args.midi_dir)
        midi_files = sorted(midi_dir.glob("*.mid"))
        if not midi_files:
            raise SystemExit(f"No .mid files found in {midi_dir}")
        if args.index >= len(midi_files):
            raise SystemExit(
                f"--index {args.index} out of range ({len(midi_files)} files in {midi_dir})"
            )
        midi_path = midi_files[args.index]

    print(f"Embedding: {midi_path}")

    from frechet_music_distance.models import CLaMP2Extractor

    extractor = CLaMP2Extractor(verbose=True)
    embedding = extractor.extract_feature(str(midi_path))

    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding.flatten()[:5]}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
        print(f"Saved embedding to {output_path}")


if __name__ == "__main__":
    main()
