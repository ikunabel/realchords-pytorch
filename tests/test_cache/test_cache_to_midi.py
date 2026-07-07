#!/usr/bin/env python3
"""Verify all cached JSONL examples for one dataset round-trip to MIDI."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import pytest

from realchords.constants import CACHE_DIR, CHORD_NAMES_AUG_PATH
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.io_utils import JSONLIndexer

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATASETS = ("hooktheory", "pop909", "nottingham", "wikifonia", "jazzmus")
SPLITS = ("train", "valid", "test")
DATASET_METADATA_KEYS = DATASETS


def load_tokenizer() -> HooktheoryTokenizer:
    with Path(CHORD_NAMES_AUG_PATH).open("r", encoding="utf-8") as handle:
        chord_names = json.load(handle)
    return HooktheoryTokenizer(chord_names=chord_names)


def cache_split_path(dataset_name: str, split: str) -> Path:
    return Path(CACHE_DIR) / dataset_name / f"{split}.jsonl"


def dataset_metadata(item: dict) -> tuple[str, dict]:
    for key in DATASET_METADATA_KEYS:
        if key in item:
            return key, item[key]
    return "unknown", {}


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "unknown"


def midi_basename(dataset_name: str, split: str, item: dict, row_index: int) -> str:
    """Build a MIDI filename that traces back to the source file or song id."""
    meta_key, meta = dataset_metadata(item)

    if meta.get("file"):
        source_name = Path(str(meta["file"])).name
        return f"{split}__{safe_filename(source_name)}"

    if meta_key == "hooktheory":
        artist = safe_filename(str(meta.get("artist", "unknown")))
        song = safe_filename(str(meta.get("song", "unknown")))
        song_id = safe_filename(str(meta.get("id", row_index)))
        return f"{split}__{artist}__{song}__{song_id}"

    song_id = meta.get("id") or meta.get("index") or row_index
    title = meta.get("title")
    if title:
        return f"{split}__{safe_filename(str(song_id))}__{safe_filename(str(title))}"
    return f"{split}__{safe_filename(str(song_id))}"


def cache_item_to_midi(tokenizer: HooktheoryTokenizer, item: dict):
    encoded = tokenizer.encode(item)
    return tokenizer.decode_to_midi(
        chord_frames=encoded["chord"],
        melody_frames=encoded["melody"],
    )


def export_cache_split_to_midi(
    *,
    dataset_name: str,
    split: str,
    tokenizer: HooktheoryTokenizer,
    output_dir: Path,
) -> tuple[list[Path], list[dict]]:
    cache_path = cache_split_path(dataset_name, split)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache split: {cache_path}")

    indexer = JSONLIndexer(str(cache_path))
    if len(indexer) == 0:
        raise ValueError(f"Empty cache split: {cache_path}")

    split_dir = output_dir / dataset_name / split
    split_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    manifest_rows: list[dict] = []
    used_names: set[str] = set()

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a project dependency
        tqdm = lambda x, **_: x  # type: ignore

    for row_index in tqdm(
        range(len(indexer)),
        desc=f"{dataset_name}/{split}",
        unit="song",
    ):
        item = indexer[row_index]
        midi = cache_item_to_midi(tokenizer, item)
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        if not midi.instruments or total_notes == 0:
            raise ValueError(
                f"{dataset_name}/{split}[{row_index}] produced empty MIDI "
                f"(instruments={len(midi.instruments)}, notes={total_notes})"
            )

        base_name = midi_basename(dataset_name, split, item, row_index)
        if base_name in used_names:
            base_name = f"{base_name}__row{row_index:05d}"
        used_names.add(base_name)

        midi_path = split_dir / f"{base_name}.mid"
        midi.write(str(midi_path))
        written.append(midi_path)

        meta_key, meta = dataset_metadata(item)
        manifest_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "row_index": row_index,
                "midi_path": str(midi_path),
                "metadata_key": meta_key,
                "source_file": meta.get("file"),
                "source_id": meta.get("id") or meta.get("index"),
                "title": meta.get("title"),
                "artist": meta.get("artist"),
                "song": meta.get("song"),
            }
        )

    return written, manifest_rows


def export_cache_dataset_to_midi(
    *,
    dataset_name: str,
    tokenizer: HooktheoryTokenizer,
    output_dir: Path,
    splits: Iterable[str] = SPLITS,
) -> tuple[list[Path], list[dict]]:
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Expected one of: {', '.join(DATASETS)}"
        )

    all_written: list[Path] = []
    all_manifest: list[dict] = []
    for split in splits:
        written, manifest_rows = export_cache_split_to_midi(
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            output_dir=output_dir,
        )
        all_written.extend(written)
        all_manifest.extend(manifest_rows)
    return all_written, all_manifest


def write_manifest(output_dir: Path, dataset_name: str, rows: list[dict]) -> Path:
    manifest_path = output_dir / dataset_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "num_midis": len(rows),
        "entries": rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return manifest_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=DATASETS,
        help="Cached dataset to export and verify.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory for exported MIDI files.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SPLITS,
        default=None,
        help="Restrict to specific split(s). Default: train, valid, test.",
    )
    return parser.parse_args(argv)


def run_dataset_export(dataset_name: str, output_dir: Path, splits: Iterable[str]) -> None:
    if not Path(CHORD_NAMES_AUG_PATH).exists():
        raise FileNotFoundError(f"Missing chord names file: {CHORD_NAMES_AUG_PATH}")

    tokenizer = load_tokenizer()
    written, manifest_rows = export_cache_dataset_to_midi(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        output_dir=output_dir,
        splits=splits,
    )
    manifest_path = write_manifest(output_dir, dataset_name, manifest_rows)
    print(
        f"OK {dataset_name}: wrote {len(written)} MIDI files under "
        f"{output_dir / dataset_name}"
    )
    print(f"Manifest: {manifest_path}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cache-dataset",
        action="store",
        default=None,
        choices=DATASETS,
        help="Run cache-to-MIDI export for one dataset.",
    )


@pytest.fixture(scope="module")
def tokenizer() -> HooktheoryTokenizer:
    if not Path(CHORD_NAMES_AUG_PATH).exists():
        pytest.skip(f"Missing chord names file: {CHORD_NAMES_AUG_PATH}")
    return load_tokenizer()


def test_cache_dataset_converts_to_midi(
    request: pytest.FixtureRequest,
    tokenizer: HooktheoryTokenizer,
) -> None:
    dataset_name = request.config.getoption("--cache-dataset")
    if dataset_name is None:
        pytest.skip("Pass --cache-dataset=<name> to run this test.")

    written, manifest_rows = export_cache_dataset_to_midi(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        output_dir=OUTPUT_DIR,
    )
    assert written, f"No MIDI files written for {dataset_name}"
    manifest_path = write_manifest(OUTPUT_DIR, dataset_name, manifest_rows)
    assert manifest_path.exists()
    for midi_path in written:
        assert midi_path.exists()
        assert midi_path.stat().st_size > 0


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    splits = args.split or list(SPLITS)
    try:
        run_dataset_export(args.dataset, args.output_dir, splits)
    except Exception as exc:
        print(f"FAIL {args.dataset}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
