#!/usr/bin/env python3
"""Try loading all cached datasets for each split (train, valid, test, all)."""

import sys
from pathlib import Path

from realchords.constants import CACHE_DIR, CHORD_NAMES_AUG_PATH
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset

DATASETS = ("hooktheory", "pop909", "nottingham", "wikifonia", "jazzmus")
SPLITS = ("train", "valid", "test", "all")


def try_load(dataset_name: str, split: str) -> int:
    dataset = HooktheoryDataset(
        cache_dir=str(Path(CACHE_DIR) / dataset_name),
        split=split,
        model_type="decoder_only",
        model_part="chord",
        max_len=512,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=CHORD_NAMES_AUG_PATH,
    )
    _ = dataset[0]
    return len(dataset)


def main() -> None:
    failures = []

    for dataset_name in DATASETS:
        for split in SPLITS:
            label = f"{dataset_name}/{split}"
            try:
                num_items = try_load(dataset_name, split)
                print(f"OK   {label}: {num_items} items")
            except Exception as exc:
                print(f"FAIL {label}: {exc}")
                failures.append(label)

    if failures:
        print(f"\n{len(failures)} load(s) failed.")
        sys.exit(1)

    print("\nAll datasets loaded successfully.")


if __name__ == "__main__":
    main()
