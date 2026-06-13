#!/usr/bin/env python3
"""Export Hookpad-native JSON for the same filtered cache song as explore_hooktheory.

Looks up one song from data/cache/hooktheory (same filter as training data:
MELODY + HARMONY, no TEMPO_CHANGES) and writes the matching entry from
Hooktheory_Raw.json. See hooktheory_cache_filter.py for details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import PROJECT_ROOT, setup_imports, write_json, write_text
from hooktheory_cache_filter import assert_passes_hooktheory_cache_filter

setup_imports()

from realchords.dataset.hooktheory_dataloader import HooktheoryDataset


def load_raw_song(raw_path: Path, song_id: str) -> dict:
    with raw_path.open(encoding="utf-8") as f:
        raw_data = json.load(f)
    if song_id not in raw_data:
        raise KeyError(
            f"Song id {song_id!r} not found in {raw_path}. "
            "Cache and Hooktheory_Raw may be out of sync."
        )
    return raw_data[song_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=PROJECT_ROOT / "data/hooktheory/Hooktheory_Raw.json",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--song-index", type=int, default=7)
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data/cache/hooktheory"),
    )
    args = parser.parse_args()

    if not args.raw_json.exists():
        msg = f"Raw JSON not found: {args.raw_json}\n"
        write_text("raw_json", "run_summary.txt", msg)
        print(msg)
        return

    dataset = HooktheoryDataset(
        split=args.split,
        max_len=512,
        cache_dir=args.cache_dir,
        data_augmentation=False,
    )
    idx = args.song_index % len(dataset)
    cache_song = dataset.data[idx]
    assert_passes_hooktheory_cache_filter(cache_song)

    song_id = cache_song["hooktheory"]["id"]
    raw_song = load_raw_song(args.raw_json, song_id)

    meta = {
        "split": args.split,
        "song_index": idx,
        "cache_filter": "MELODY + HARMONY, no TEMPO_CHANGES",
        "song_id": song_id,
        "artist": cache_song["hooktheory"]["artist"],
        "song": cache_song["hooktheory"]["song"],
        "raw_json_path": str(args.raw_json),
    }

    write_json("raw_json", "sample_meta.json", meta)
    write_json("raw_json", f"raw_hookpad_{song_id}.json", raw_song)

    summary = write_text(
        "raw_json",
        "run_summary.txt",
        f"Cache index {idx}: {meta['artist']} - {meta['song']} ({song_id})\n"
        f"Cache filter: {meta['cache_filter']}\n"
        f"Wrote output/raw_json/raw_hookpad_{song_id}.json\n"
        f"(Same song as output/hooktheory/ when run with matching --split/--song-index)\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
