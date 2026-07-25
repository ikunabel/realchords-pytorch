#!/usr/bin/env python3
"""Compute per-dataset, per-split song/beat/frame statistics from the cache,
cross-checked against the GT tensors already exported to logs/paired_eval/.

For each dataset under data/cache/ (any directory with train/valid/test.jsonl)
and for each split plus the combined "all" (train+valid+test), computes:
    - num_songs
    - avg_beats_per_song    (from annotations.num_beats)
    - avg_frames_per_song   (num_beats * FRAME_PER_BEAT)
    - total_frames

Uses the *non-augmented* {split}.jsonl (not {split}_augmented.jsonl) -- these
are one row per underlying song, matching what the GT MIDI export in
logs/paired_eval/gt/<dataset>_all/full_songs/ was built from (confirmed by
the cross-check below), whereas the augmented files contain ~12-13x
duplicated (key-transposed) copies of the same songs.

Cross-check: logs/paired_eval/gt/<dataset>_all/full_songs/gt_num_frames.pt
holds the actual per-song frame count tensor for the combined ("all" split)
dataset, computed independently via the MIDI/tokenizer pipeline rather than
from annotations.num_beats. Comparing against it catches any drift between
the two representations (e.g. rounding at fractional-beat boundaries) --
in spot checks they agree to within ~0.01%, not exactly, so a small
nonzero relative difference here is expected, not a bug. This is a sanity
check only -- printed as a warning on mismatch, not written to the output
JSON (which reflects the JSONL cache alone).

Writes data/cache/dataset_stats.json.

Usage::

    python scripts/write_dataset_statistics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from realchords.constants import CACHE_DIR, FRAME_PER_BEAT

SPLITS = ["train", "valid", "test"]
PAIRED_EVAL_GT_DIR = Path("logs/paired_eval/gt")

# Relative difference above which the cross-check is flagged as a real mismatch
# rather than expected floating-point/rounding noise.
CROSS_CHECK_TOLERANCE = 0.01


def discover_datasets(cache_dir: Path) -> List[str]:
    """Find dataset dirs under `cache_dir` that have all three raw split files."""
    names = []
    for d in sorted(cache_dir.iterdir()):
        if d.is_dir() and all((d / f"{split}.jsonl").exists() for split in SPLITS):
            names.append(d.name)
    return names


def split_stats(jsonl_path: Path) -> Dict:
    """path / num_songs / avg_beats / avg_frames / total_frames for one split file."""
    num_beats_list = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            num_beats_list.append(item["annotations"]["num_beats"])

    num_songs = len(num_beats_list)
    total_beats = sum(num_beats_list)
    total_frames = total_beats * FRAME_PER_BEAT

    return {
        "path": str(jsonl_path),
        "num_songs": num_songs,
        "avg_beats_per_song": round(total_beats / num_songs, 2) if num_songs else 0.0,
        "avg_frames_per_song": round(total_frames / num_songs, 2) if num_songs else 0.0,
        "total_frames": total_frames,
    }


def combine_stats(per_split: Dict[str, Dict]) -> Dict:
    """Combine per-split stats into a train+valid+test "all" aggregate."""
    num_songs = sum(s["num_songs"] for s in per_split.values())
    total_frames = sum(s["total_frames"] for s in per_split.values())
    total_beats = total_frames / FRAME_PER_BEAT

    return {
        "num_songs": num_songs,
        "avg_beats_per_song": round(total_beats / num_songs, 2) if num_songs else 0.0,
        "avg_frames_per_song": round(total_frames / num_songs, 2) if num_songs else 0.0,
        "total_frames": total_frames,
    }


def cross_check(dataset: str, computed: Dict[str, float]) -> Optional[Dict]:
    """Compare `computed` (the "all" aggregate) against the GT num_frames tensor."""
    pt_path = PAIRED_EVAL_GT_DIR / f"{dataset}_all" / "full_songs" / "gt_num_frames.pt"
    if not pt_path.exists():
        return None

    import torch

    frames = torch.load(pt_path, weights_only=False)
    pt_num_songs = int(frames.shape[0])
    pt_total_frames = int(frames.sum().item())

    frames_diff = abs(pt_total_frames - computed["total_frames"])
    frames_rel_diff = frames_diff / max(pt_total_frames, 1)

    return {
        "pt_path": str(pt_path),
        "pt_num_songs": pt_num_songs,
        "pt_total_frames": pt_total_frames,
        "num_songs_match": pt_num_songs == computed["num_songs"],
        "total_frames_relative_diff": frames_rel_diff,
        "within_tolerance": frames_rel_diff <= CROSS_CHECK_TOLERANCE,
    }


def main() -> None:
    cache_dir = Path(CACHE_DIR)
    datasets = discover_datasets(cache_dir)
    print(f"Found {len(datasets)} datasets under {cache_dir}: {datasets}")

    stats = {}
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        per_split = {}
        for split in SPLITS:
            jsonl_path = cache_dir / dataset / f"{split}.jsonl"
            per_split[split] = split_stats(jsonl_path)
            s = per_split[split]
            print(
                f"  {split:6s}: {s['num_songs']:6d} songs, "
                f"avg_beats={s['avg_beats_per_song']:.2f}, "
                f"avg_frames={s['avg_frames_per_song']:.2f}, "
                f"total_frames={s['total_frames']:.0f}"
            )

        all_stats = combine_stats(per_split)
        print(
            f"  {'all':6s}: {all_stats['num_songs']:6d} songs, "
            f"avg_beats={all_stats['avg_beats_per_song']:.2f}, "
            f"avg_frames={all_stats['avg_frames_per_song']:.2f}, "
            f"total_frames={all_stats['total_frames']:.0f}"
        )

        # Sanity check only -- not written to the output JSON.
        check = cross_check(dataset, all_stats)
        if check is None:
            print(f"  (no GT export found at {PAIRED_EVAL_GT_DIR}/{dataset}_all/... -- skipping cross-check)")
        else:
            status = "OK" if check["within_tolerance"] and check["num_songs_match"] else "MISMATCH"
            print(
                f"  cross-check vs {check['pt_path']}: [{status}] "
                f"songs {all_stats['num_songs']} vs {check['pt_num_songs']}, "
                f"total_frames relative diff {check['total_frames_relative_diff']:.4%}"
            )
            if status == "MISMATCH":
                print(
                    "  WARNING: cross-check exceeds tolerance "
                    f"({CROSS_CHECK_TOLERANCE:.2%}) -- investigate before trusting these stats."
                )

        stats[dataset] = {
            "cache_dir": str(cache_dir / dataset),
            **per_split,
            "all": all_stats,
        }

    out_path = cache_dir / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote stats to {out_path}")


if __name__ == "__main__":
    main()
