"""How well the performer's keys fit each chord's stored voicings (complete mode).

For ReaLJam's complete mode (2026-11-18). The performer holds some keys; the
agent completes the chord with a stored voicing that contains those keys at
their exact pitches. For every chord of the model's vocabulary this module
gives the voicing likelihood

    fit(chord) = (count of the chord's playable stored voicings that contain
                  the keys at their pitches) / (count of all its stored voicings)

where "playable" is the completion's own rule (play_from_midi_input.js,
completionVoicing + voicingRegisterOk): the voicing meets the register rules,
and the notes it adds lie on no other held key and not above the melody
ceiling. The page draws the chord from p_model^(1/T) * fit, so the shape of
the hand (bass note, spacing, register) counts, and a chord is only chosen if
the agent can complete it (journal/reference/INTERACTION_MODES.md, "using the
keys in the chord choice", rule C2).

Voicings are indexed once as 128-bit pitch masks (two uint64 halves), so one
request is a few vectorised numpy operations over all stored voicings.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# The register rules -- must match play_from_midi_input.js (voicingRegisterOk)
VOICING_LOWEST = 36           # C2: nothing lower
VOICING_MIN_SPAN = 12         # lowest to highest at least an octave
VOICING_LOW_REGISTER = 48     # below C3 ...
VOICING_LOW_MIN_INTERVAL = 7  # ... adjacent notes at least a fifth apart


def register_exempt_notes(pitches: Sequence[int]) -> List[int]:
    """The notes that must all be the performer's own keys for this voicing to be
    allowed, or [] if it meets the register rules as it is.

    The rules judge only what the agent adds (2026-11-23: a performer playing
    close intervals below C3 got no completion at all, since every stored
    voicing containing their keys broke the rule): a note below C2, and both
    notes of a pair closer than a fifth below C3, are fine if the performer
    plays them; a voicing narrower than an octave is fine if it adds nothing.
    """
    p = sorted(set(pitches))
    need = {x for x in p if x < VOICING_LOWEST}
    for a, b in zip(p, p[1:]):
        if a < VOICING_LOW_REGISTER and b - a < VOICING_LOW_MIN_INTERVAL:
            need |= {a, b}
    if p and p[-1] - p[0] < VOICING_MIN_SPAN:
        need |= set(p)
    return sorted(need)


def register_ok(pitches: Sequence[int], keys: Sequence[int] = ()) -> bool:
    """The register rules on one voicing (as voicingRegisterOk), with the
    performer's `keys` exempt (register_exempt_notes)."""
    return set(register_exempt_notes(pitches)) <= set(keys)


def _mask(pitches: Sequence[int]) -> tuple:
    lo = hi = 0
    for p in pitches:
        if 0 <= p < 64:
            lo |= 1 << p
        elif 64 <= p < 128:
            hi |= 1 << (p - 64)
    return lo, hi


class VoicingFitIndex:
    """All stored voicings of every vocabulary chord, as pitch masks."""

    def __init__(self, symbols: List[str], lookup: Dict[str, List[Dict]]):
        """
        Args:
          symbols: the vocabulary's chord symbols, in the order fit() answers in
            (Agent.chord_vocab_entries)
          lookup: chord symbol -> [{pitches, count, ...}] (the voicing lookup).
            A slash chord without voicings uses its base chord's, as the page
            does (completionVoicing).
        """
        chord, count, lo, hi, need_lo, need_hi = [], [], [], [], [], []
        total = np.zeros(len(symbols))
        for i, symbol in enumerate(symbols):
            voicings = lookup.get(symbol) or []
            if not voicings and "/" in symbol:
                voicings = lookup.get(symbol.split("/")[0]) or []
            for v in voicings:
                if not v.get("count") or v.get("count") <= 0:
                    continue  # as loadVoicingCandidates
                m = _mask(v["pitches"])
                chord.append(i)
                count.append(v["count"])
                lo.append(m[0])
                hi.append(m[1])
                n = _mask(register_exempt_notes(v["pitches"]))
                need_lo.append(n[0])
                need_hi.append(n[1])
                total[i] += v["count"]
        self.n = len(symbols)
        self.chord = np.array(chord, dtype=np.int64)
        self.count = np.array(count, dtype=np.float64)
        self.lo = np.array(lo, dtype=np.uint64)
        self.hi = np.array(hi, dtype=np.uint64)
        self.need_lo = np.array(need_lo, dtype=np.uint64)  # notes that must be keys (register rules)
        self.need_hi = np.array(need_hi, dtype=np.uint64)
        self.total = total

    def fit(self, keys: Sequence[int], held: Sequence[int], ceiling: Optional[int]) -> List[float]:
        """fit(chord) for every vocabulary chord, in the symbols' order.

        Args:
          keys: the keys the voicing must contain, at their pitches (the chord
            body: the chord keys without a top note read as melody)
          held: every key held now; the voicing's other notes may not be on them
          ceiling: the highest pitch an added note may have (None: no limit)
        """
        key_lo, key_hi = _mask(keys)
        forbid = set(held) - set(keys)
        if ceiling is not None:
            forbid |= {p for p in range(int(ceiling) + 1, 128)}
        forbid_lo, forbid_hi = _mask(forbid - set(keys))
        key_lo, key_hi = np.uint64(key_lo), np.uint64(key_hi)
        forbid_lo, forbid_hi = np.uint64(forbid_lo), np.uint64(forbid_hi)
        register = ((self.need_lo & key_lo) == self.need_lo) & ((self.need_hi & key_hi) == self.need_hi)
        playable = (register
                    & ((self.lo & key_lo) == key_lo) & ((self.hi & key_hi) == key_hi)
                    & ((self.lo & forbid_lo) == 0) & ((self.hi & forbid_hi) == 0))
        weight = np.bincount(self.chord, weights=self.count * playable, minlength=self.n)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(self.total > 0, weight / self.total, 0.0)
        return [round(float(x), 6) for x in share]
