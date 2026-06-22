"""
Creates a genre-sorted view of aria-midi-v1-deduped-ext using symlinks.
Output structure:
  dataset_experiments/output/aria/deduped-by-genre/{genre}/{id}_0.mid -> original file

The original dataset is not modified.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
SOURCE_DIR = DATA_DIR / "aria-midi-v1-deduped-ext"
METADATA = SOURCE_DIR / "metadata.json"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "aria" / "deduped-by-genre"

print("Loading metadata...")
with open(METADATA) as f:
    metadata = json.load(f)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

missing = 0
created = 0
skipped = 0

for entry_id, entry in metadata.items():
    genre = entry.get("metadata", {}).get("genre") or "unknown"
    genre_dir = OUTPUT_DIR / genre
    genre_dir.mkdir(exist_ok=True)

    # Files are sharded: data/{shard}/{id:06d}_0.mid
    # Find the actual file by searching the shard dirs
    padded = f"{int(entry_id):06d}_0.mid"
    shard = None
    for d in (SOURCE_DIR / "data").iterdir():
        candidate = d / padded
        if candidate.exists():
            shard = candidate
            break

    if shard is None:
        missing += 1
        continue

    link = genre_dir / padded
    if link.exists() or link.is_symlink():
        skipped += 1
        continue

    link.symlink_to(shard.resolve())
    created += 1

print(f"\nDone.")
print(f"  Symlinks created : {created:,}")
print(f"  Already existed  : {skipped:,}")
print(f"  Source not found : {missing:,}")
print(f"\nOutput: {OUTPUT_DIR}")

# Print genre summary
genres = sorted(OUTPUT_DIR.iterdir(), key=lambda p: -len(list(p.glob("*.mid"))))
print(f"\nGenre folders ({len(genres)} genres):")
for g in genres:
    count = len(list(g.glob("*.mid")))
    print(f"  {g.name:25s} {count:>7,}")
