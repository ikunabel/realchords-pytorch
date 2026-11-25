"""Chord events for the progression metrics (root motion on the line of fifths, chord placement).

Turns a model's or ground truth's token lane into the sequence of chords it played.

A chord is (root pitch class, chord type), the chord type being the root-relative interval set
verbatim: the vocabulary's chord token with the slash bass dropped (C/E = C), so extensions stay
distinct (C7, C9, C13). A coarser (third, fifth, seventh) class existed until 2026-11-25 and was the
n-gram's default; removed so every metric reads chords the same way. The n-gram itself was removed the
same day (journal/reference/METRICS.md).

Only chord *changes* are events: a chord held four bars and the same chord for one beat
are identical, a re-articulation of the same symbol is not a new event, and a rest does
not break the progression. Duration is therefore absent from the representation. That is
deliberate -- harmonic rhythm is measured separately -- but it means a passage played at
half speed looks like one played at full speed, and the number of chord changes per window
should be reported alongside.

The divergence metrics that used to live here (root-motion and chord-vocabulary
Jensen-Shannon against a reference histogram, plus the higher-resolution joint views and
the named conditionals) were removed on 2026-10-13, and progression matching on 2026-11-01
in favour of the n-gram, itself removed on 2026-11-25 (journal/reference/METRICS.md).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import note_seq.chord_symbols_lib as chord_symbols_lib
from realchords.utils.eval_utils import _parse_chord_symbol
from realchords.utils.data_utils import chord_symbol_pitches  # note_seq's reader without the add7 flattening

ChordEvent = Tuple[int, str]          # (root pitch class, chord type as an interval set)
NUM_ROOTS = 12


def chord_symbol_to_event(symbol: str) -> ChordEvent | None:
    """(root, chord type) for a chord symbol, or None if unparseable. The chord type is the
    root-relative interval set as a string, e.g. '0,4,7,10' (153 types in Hooktheory's training split)."""
    try:
        root = chord_symbols_lib.chord_symbol_root(symbol)
        pitches = chord_symbol_pitches(symbol)
    except Exception:
        return None
    intervals = frozenset((p - root) % 12 for p in pitches)
    return root % 12, _set_token(intervals)


def _set_token(intervals: frozenset) -> str:
    """An interval set as a stable string token, e.g. '0,4,7,10'."""
    return ",".join(str(i) for i in sorted(intervals))


def chord_events(
    chord_tokens: Sequence[int],
    id_to_name: Dict[int, str],
) -> List[ChordEvent]:
    """Chord-change events from one sequence's chord lane.

    Non-chord tokens (silence, padding, specials) are skipped, and a repeated chord
    is not an event -- only a change in (root, chord type) is.
    """
    events: List[ChordEvent] = []
    previous: ChordEvent | None = None
    for token in chord_tokens:
        symbol = _parse_chord_symbol(id_to_name.get(int(token), ""))
        if symbol is None:
            continue
        event = chord_symbol_to_event(symbol)
        if event is None:
            continue
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
