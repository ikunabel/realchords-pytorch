"""Filter rules shared with scripts/convert_data_to_cache/convert_hooktheory_to_cache.py.

Songs are written to data/cache/hooktheory/*.jsonl only when they pass
passes_hooktheory_cache_filter(). Viewer scripts should inspect the same set.

Filter (per song):
  - "MELODY" in tags
  - "HARMONY" in tags
  - "TEMPO_CHANGES" not in tags

Not filtered: alignment tags (USER_ALIGNMENT, REFINED_ALIGNMENT), swing,
KEY_CHANGES, AUDIO_AVAILABLE, etc. Songs without refined alignment can still
be in the cache if they have melody + harmony and no tempo changes.
"""

from __future__ import annotations

from typing import Iterable


def passes_hooktheory_cache_filter(song: dict) -> bool:
    """Return True if *song* would be included in the Hooktheory cache."""
    tags = song.get("tags") or []
    return (
        "MELODY" in tags
        and "HARMONY" in tags
        and "TEMPO_CHANGES" not in tags
    )


def assert_passes_hooktheory_cache_filter(song: dict) -> None:
    """Raise ValueError if *song* would not be in the Hooktheory cache."""
    if not passes_hooktheory_cache_filter(song):
        tags = song.get("tags") or []
        raise ValueError(
            "Song does not pass the Hooktheory cache filter "
            f"(tags={tags}). See hooktheory_cache_filter.py."
        )


def filter_hooktheory_songs(songs: Iterable[dict]) -> list[dict]:
    """Keep only songs that would appear in the Hooktheory cache."""
    return [song for song in songs if passes_hooktheory_cache_filter(song)]
