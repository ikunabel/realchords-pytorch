"""
Compare aria-midi-v1-deduped-ext and aria-midi-v1-unique-ext datasets.
Shows counts, sample entries, key differences, and genre breakdowns.
Writes a summary JSON to dataset_experiments/output/aria/comparison.json.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "aria"
DEDUPED = DATA_DIR / "aria-midi-v1-deduped-ext" / "metadata.json"
UNIQUE = DATA_DIR / "aria-midi-v1-unique-ext" / "metadata.json"

print("Loading metadata files...")
with open(DEDUPED) as f:
    deduped = json.load(f)
with open(UNIQUE) as f:
    unique = json.load(f)

print(f"\nEntry counts:")
print(f"  deduped : {len(deduped)}")
print(f"  unique  : {len(unique)}")

# Key overlap
deduped_keys = set(deduped.keys())
unique_keys = set(unique.keys())
print(f"\nKey overlap:")
print(f"  in both     : {len(deduped_keys & unique_keys)}")
print(f"  only deduped: {len(deduped_keys - unique_keys)}")
print(f"  only unique : {len(unique_keys - deduped_keys)}")

def genre_counts(data):
    genres = Counter()
    for entry in data.values():
        genre = entry.get("metadata", {}).get("genre", "unknown")
        genres[genre] += 1
    return dict(sorted(genres.items(), key=lambda x: -x[1]))

deduped_genres = genre_counts(deduped)
unique_genres = genre_counts(unique)

print(f"\nGenre breakdown — deduped:")
for g, n in deduped_genres.items():
    print(f"  {g:20s} {n:>7,}")

print(f"\nGenre breakdown — unique:")
for g, n in unique_genres.items():
    print(f"  {g:20s} {n:>7,}")

result = {
    "deduped": {
        "total": len(deduped),
        "genres": deduped_genres,
    },
    "unique": {
        "total": len(unique),
        "genres": unique_genres,
    },
    "key_overlap": {
        "in_both": len(deduped_keys & unique_keys),
        "only_deduped": len(deduped_keys - unique_keys),
        "only_unique": len(unique_keys - deduped_keys),
    },
}

OUTPUT_DIR.mkdir(exist_ok=True)
out_path = OUTPUT_DIR / "comparison.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {out_path}")
