"""Chord events for the progression metrics (root motion, progression n-gram).

Turns a model's or ground truth's token lane into the sequence of chords it played.

A chord is (root pitch class, quality token). Quality is either the *function* class --
(third, fifth, seventh), extensions dropped, so C7/C9/C13 are one token -- or, with
quality="full", the root-relative interval set verbatim, keeping them distinct.

Only chord *changes* are events: a chord held four bars and the same chord for one beat
are identical, a re-articulation of the same symbol is not a new event, and a rest does
not break the progression. Duration is therefore absent from the representation. That is
deliberate -- harmonic rhythm is measured separately -- but it means a passage played at
half speed looks like one played at full speed, and the number of chord changes per window
should be reported alongside.

The divergence metrics that used to live here (root-motion and chord-vocabulary
Jensen-Shannon against a reference histogram, plus the higher-resolution joint views and
the named conditionals) were removed on 2026-10-13, and progression matching on 2026-11-01
in favour of the n-gram (journal/reference/METRICS.md).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from realchords.utils.eval_utils import chord_symbols_lib, _parse_chord_symbol

ChordEvent = Tuple[int, str]          # (root pitch class, quality class)
OTHER = "other"
NUM_ROOTS = 12

# Function classes, most frequent first across the corpora. Used only as a stable
# ordering; the active list for a corpus comes from its reference file.
KNOWN_CLASSES = [
    "MP", "mP", "mP7", "MP7", "MPM7", "sP", "md7", "sP7", "md", "mdd7", "mPM7", "MA",
]


def quality_class(intervals: frozenset) -> str:
    """Function triple (third, fifth, seventh) from root-relative pitch classes.

    third:  M major (4) / m minor (3) / s neither, i.e. sus or power chord
    fifth:  P perfect (7) / d diminished (6) / A augmented (8) / - absent
    seventh: 7 minor (10) / M7 major (11) / d7 diminished (9, with a flat fifth) / none
    """
    third = "M" if 4 in intervals else "m" if 3 in intervals else "s"
    fifth = "P" if 7 in intervals else "d" if 6 in intervals else "A" if 8 in intervals else "-"
    if 10 in intervals:
        seventh = "7"
    elif 11 in intervals:
        seventh = "M7"
    elif 9 in intervals and 6 in intervals:
        seventh = "d7"
    else:
        seventh = ""
    return f"{third}{fifth}{seventh}"


def chord_symbol_to_event(symbol: str, quality: str = "function") -> ChordEvent | None:
    """(root, quality token) for a chord symbol, or None if unparseable.

    quality="function"  -> (third, fifth, seventh) class; extensions dropped, so C7,
                           C9, C11 and C13 are one token. 24 tokens on Hooktheory.
    quality="full"      -> the root-relative interval set verbatim, as a sorted tuple,
                           so extensions are kept distinct. 106 tokens on Hooktheory.

    The cache stores "full" and collapses on load, so the finer view never has to be
    re-extracted from the dataloader.
    """
    try:
        root = chord_symbols_lib.chord_symbol_root(symbol)
        pitches = chord_symbols_lib.chord_symbol_pitches(symbol)
    except Exception:
        return None
    intervals = frozenset((p - root) % 12 for p in pitches)
    token = quality_class(intervals) if quality == "function" else _set_token(intervals)
    return root % 12, token


def _set_token(intervals: frozenset) -> str:
    """An interval set as a stable string token, e.g. '0,4,7,10'."""
    return ",".join(str(i) for i in sorted(intervals))


def collapse_to_function(token: str) -> str:
    """Map a 'full' interval-set token back to its function class."""
    if not token or not token[0].isdigit():
        return token          # already a function class
    return quality_class(frozenset(int(x) for x in token.split(",")))


def chord_events(
    chord_tokens: Sequence[int],
    id_to_name: Dict[int, str],
    classes: Sequence[str] | None = None,
    quality: str = "function",
) -> List[ChordEvent]:
    """Chord-change events from one sequence's chord lane.

    Non-chord tokens (silence, padding, specials) are skipped, and a repeated chord
    is not an event -- only a change in (root, quality) is.
    """
    allowed = set(classes) if classes is not None else None
    events: List[ChordEvent] = []
    previous: ChordEvent | None = None
    for token in chord_tokens:
        symbol = _parse_chord_symbol(id_to_name.get(int(token), ""))
        if symbol is None:
            continue
        event = chord_symbol_to_event(symbol, quality=quality)
        if event is None:
            continue
        if allowed is not None and event[1] not in allowed:
            event = (event[0], OTHER)
        if event != previous:
            events.append(event)
        previous = event
    return events


def chord_lane(interleaved: Sequence[int], chord_first: bool = True) -> List[int]:
    """The chord half of an interleaved sequence, dropping the leading BOS.

    Verified against both the dataloader and custom_evaluation's dumps: the token
    after BOS is a chord (BOS, CHORD_ON_F, NOTE_ON_69, CHORD_F, ...), so the chord
    lane is the *even* offsets of the body. Padding and specials stay in and are
    dropped later by symbol parsing -- filtering them here would shift the parity
    and silently yield the melody lane.
    """
    body = list(interleaved)[1:]
    return body[0::2] if chord_first else body[1::2]


# --------------------------------------------------------------------------- #
# event cache
# --------------------------------------------------------------------------- #
# Extracting chord events means walking the dataloader, which is ~4 minutes for
# hooktheory's training split and dominates every run. The events themselves are
# small (a few MB per corpus), so they are cached and everything downstream --
# new n-gram orders, different quality groupings, corpus-to-corpus distances --
# becomes instant. It also decouples the metrics from the dataloader, which is
# where both of this module's early bugs lived (lane parity, mask filtering).

EVENTS_CACHE_DIR = "data/cache/progression_events"


def events_cache_path(dataset_name: str, split: str, root: str = EVENTS_CACHE_DIR,
                      quality: str = "function") -> "Path":
    from pathlib import Path
    suffix = "" if quality == "function" else f".{quality}"
    return Path(root) / f"{dataset_name}.{split}{suffix}.json"


def save_events(path, sequences, meta: Dict) -> None:
    """Write event sequences plus the provenance needed to detect a stale cache."""
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "sequences": [[[r, q] for r, q in seq] for seq in sequences],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_events(path, meta: Dict | None = None):
    """Cached event sequences, or None if absent or built under different settings.

    `meta` is compared field by field; a mismatch returns None rather than silently
    serving events extracted at a different window length or chord vocabulary.
    """
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta is not None:
        cached = payload.get("meta", {})
        for key, value in meta.items():
            if cached.get(key) != value:
                return None
    return [[(int(r), str(q)) for r, q in seq] for seq in payload["sequences"]]
