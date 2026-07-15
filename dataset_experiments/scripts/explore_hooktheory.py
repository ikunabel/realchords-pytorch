#!/usr/bin/env python3
"""Explore one Hooktheory cache song: JSON, MIDI, and tokenized output.

Loads from data/cache/hooktheory, which only contains songs that passed the
cache filter in scripts/convert_data_to_cache/convert_hooktheory_to_cache.py (MELODY + HARMONY tags,
no TEMPO_CHANGES). See hooktheory_cache_filter.py for details.
"""

from __future__ import annotations

import argparse
from pprint import pformat

import torch

from _common import PROJECT_ROOT, ensure_output, setup_imports, write_json, write_text
from hooktheory_cache_filter import assert_passes_hooktheory_cache_filter

setup_imports()

from dataset_experiments.dataviewer_utils import hooktheory_to_midi
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset


def tensor_dict_to_jsonable(item: dict) -> dict:
    out = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            out[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "values": value.tolist(),
            }
        else:
            out[key] = value
    return out


def decode_tokens(tokenizer, token_ids: list[int], limit: int = 64) -> list[str]:
    return [
        tokenizer.id_to_name.get(int(token_id), f"UNK_{token_id}")
        for token_id in token_ids[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--song-index", type=int, default=7)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data/cache/hooktheory"),
    )
    parser.add_argument(
        "--model-type",
        default="decoder_only",
        choices=["decoder_only", "encoder_decoder", "decoder_only_single"],
    )
    args = parser.parse_args()

    # Non-augmented cache matches convert_hooktheory_to_cache.py without --augmentation.
    dataset = HooktheoryDataset(
        split=args.split,
        model_type=args.model_type,
        max_len=512,
        cache_dir=args.cache_dir,
        data_augmentation=False,
    )

    idx = args.song_index % len(dataset)
    song = dataset.data[idx]
    assert_passes_hooktheory_cache_filter(song)
    tokenized = dataset[idx]

    targets = tokenized["targets"].tolist()
    token_names = decode_tokens(dataset.tokenizer, targets)

    midi_dir = ensure_output("hooktheory/midis")
    midi_path = hooktheory_to_midi(song, output_dir=midi_dir)

    meta = {
        "split": args.split,
        "song_index": idx,
        "dataset_len": len(dataset),
        "cache_dir": args.cache_dir,
        "model_type": args.model_type,
        "cache_filter": "MELODY + HARMONY, no TEMPO_CHANGES",
        "artist": song["hooktheory"]["artist"],
        "song": song["hooktheory"]["song"],
        "id": song["hooktheory"]["id"],
        "tags": song.get("tags", []),
        "num_harmony_events": len(song["annotations"]["harmony"]),
        "num_melody_events": len(song["annotations"]["melody"]),
        "num_beats": song["annotations"].get("num_beats"),
        "midi_path": str(midi_path),
        "num_tokens": len(targets),
        "song_url": tokenized.get("song_url"),
    }

    write_json("hooktheory", "sample_song_meta.json", meta)
    write_json("hooktheory", "sample_song_full.json", song)
    write_json("hooktheory", "tokenized_item.json", tensor_dict_to_jsonable(tokenized))
    write_text(
        "hooktheory",
        "sample_song_summary.txt",
        pformat(meta)
        + "\n\n--- hooktheory ---\n"
        + pformat(song["hooktheory"])
        + "\n\n--- first harmony events ---\n"
        + pformat(song["annotations"]["harmony"][:8])
        + "\n\n--- first melody events ---\n"
        + pformat(song["annotations"]["melody"][:8]),
    )
    write_text(
        "hooktheory",
        "tokenization_report.txt",
        "\n".join(
            [
                f"Target shape: {list(tokenized['targets'].shape)}",
                f"First {len(token_names)} tokens:",
                *[
                    f"  {tid:5d}  {name}"
                    for tid, name in zip(targets[: len(token_names)], token_names)
                ],
            ]
        )
        + "\n",
    )

    summary = write_text(
        "hooktheory",
        "run_summary.txt",
        f"Song index {idx}: {meta['artist']} - {meta['song']} ({meta['id']})\n"
        f"Cache filter: {meta['cache_filter']}\n"
        f"JSON:  output/hooktheory/sample_song_full.json\n"
        f"MIDI:  {midi_path}  (same song as sample JSON)\n"
        f"Tokens: output/hooktheory/tokenization_report.txt\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
