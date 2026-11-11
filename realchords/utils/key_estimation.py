"""Key estimation for the interleaved token format, delegating to music21's implementations.

The token lanes are turned into music21 streams (notes and chords with their durations, one frame =
one 16th note) and handed to music21's key-profile analysers -- by default
``KrumhanslSchmuckler``, the Krumhansl-Schmuckler algorithm with the Krumhansl-Kessler profiles,
the reference implementation cited in the literature. These analysers correlate the
duration-weighted pitch-class distribution with the profile rotated to all 24 keys. A numpy
re-implementation with the same profiles agreed on 309/309 real Hooktheory melodies, but music21 is
fast enough (~20 ms per 16-bar window) that it is used directly, so the estimate is music21's.

Scoring against a reference key uses ``mir_eval.key.weighted_score`` (MIREX: exact 1.0, the fifth
*above* 0.5, relative 0.3, parallel 0.2, else 0). ``key_relation`` gives the same categories, plus
"fifth below", for reporting.

Keys are indexed 0-23: tonic pitch class + 12 * (minor). Validated against Hooktheory's key labels
in scripts/eval/custom_eval/key_estimator_validation.py (journal/METRICS.md).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from music21 import chord as m21chord
from music21 import note as m21note
from music21 import stream as m21stream
from music21.analysis import discrete

from realchords.utils import progression_metrics as pm
from realchords.utils.eval_utils import _parse_chord_symbol

ANALYSERS = {
    "krumhansl": discrete.KrumhanslSchmuckler,
    "aarden": discrete.AardenEssen,
    "bellman": discrete.BellmanBudge,
    "temperley": discrete.TemperleyKostkaPayne,
    "simple": discrete.SimpleWeights,
}
SILENCE = 3
MELODY_HOLD = (4, 131)      # NOTE_<p>     -> pitch p = id - 4
MELODY_ONSET = (132, 259)   # NOTE_ON_<p>  -> pitch p = id - 132
_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def key_name(k: int) -> str:
    return _NAMES[k % 12] + ("m" if k >= 12 else "")


def mir_eval_key(k: int) -> str:
    """Key index -> the 'tonic mode' string mir_eval expects, e.g. 'Eb major', 'A minor'."""
    return f"{_NAMES[k % 12]} {'minor' if k >= 12 else 'major'}"


def melody_stream(melody_tokens: Sequence[int]) -> m21stream.Stream:
    """Notes from a melody lane: an onset starts a note, holds of it extend it, anything else
    ends it."""
    s = m21stream.Stream()
    pitch, dur = None, 0

    def flush():
        if pitch is not None and dur > 0:
            n = m21note.Note(pitch)
            n.quarterLength = dur / 4
            s.append(n)

    for t in melody_tokens:
        t = int(t)
        if MELODY_ONSET[0] <= t <= MELODY_ONSET[1]:
            flush()
            pitch, dur = t - MELODY_ONSET[0], 1
        elif MELODY_HOLD[0] <= t <= MELODY_HOLD[1] and pitch == t - MELODY_HOLD[0]:
            dur += 1
        else:
            flush()
            pitch, dur = None, 0
    flush()
    return s


@lru_cache(maxsize=None)
def _chord_pcs(name: str) -> Tuple[int, ...]:
    symbol = _parse_chord_symbol(name)
    if symbol is None:
        return ()
    ev = pm.chord_symbol_to_event(symbol, quality="full")
    if ev is None:
        return ()
    root, intervals = ev
    return tuple(sorted({(root + int(i)) % 12 for i in intervals.split(",")}))


def chord_stream(chord_tokens: Sequence[int], id_to_name: Dict[int, str]) -> m21stream.Stream:
    """Chords from a chord lane: consecutive frames with the same chord tones form one chord.
    Pitches are placed in octave 4; only pitch classes matter to the key analysers."""
    s = m21stream.Stream()
    pcs, dur = (), 0

    def flush():
        if pcs and dur > 0:
            c = m21chord.Chord([60 + p for p in pcs])
            c.quarterLength = dur / 4
            s.append(c)

    for t in chord_tokens:
        cur = _chord_pcs(id_to_name.get(int(t), ""))
        if cur and cur == pcs:
            dur += 1
            continue
        flush()
        pcs, dur = cur, (1 if cur else 0)
    flush()
    return s


def combined_stream(melody: m21stream.Stream, chords: m21stream.Stream) -> m21stream.Stream:
    """Both parts in one stream. The analysers sum durations per pitch class, so element order is
    irrelevant; note that each chord frame contributes all its tones, so chords weigh more."""
    s = m21stream.Stream()
    for el in list(melody.notes) + list(chords.notes):
        s.append(el.__deepcopy__())
    return s


def estimate_key(s: m21stream.Stream, analyser: str = "krumhansl") -> Optional[int]:
    """music21's estimate for a stream, as a key index; None if the stream has no pitches."""
    if not len(s.notes):
        return None
    try:
        k = ANALYSERS[analyser]().getSolution(s)
    except Exception:
        return None
    return k.tonic.pitchClass + (12 if k.mode == "minor" else 0)


def key_profile(s: m21stream.Stream, analyser: str = "krumhansl") -> Optional[np.ndarray]:
    """All 24 key correlations music21 computes for a stream (its best key plus the 23
    alternateInterpretations), indexed tonic + 12 * minor. None if the stream has no pitches."""
    if not len(s.notes):
        return None
    try:
        best = ANALYSERS[analyser]().getSolution(s)
    except Exception:
        return None
    v = np.full(24, np.nan)
    for k in [best] + list(best.alternateInterpretations):
        v[k.tonic.pitchClass + (12 if k.mode == "minor" else 0)] = k.correlationCoefficient
    return None if np.isnan(v).any() else v


def transpose_profile(v: np.ndarray, semitones: int) -> np.ndarray:
    """Profile of the same music transposed by `semitones`. Profile correlation is transposition-
    equivariant, so this is a rotation within the major and the minor half -- no re-estimation."""
    return np.concatenate([np.roll(v[:12], semitones), np.roll(v[12:], semitones)])


def tonal_agreement(melody_profile: Optional[np.ndarray],
                    chord_profile: Optional[np.ndarray]) -> Optional[float]:
    """Pearson correlation between two 24-key profiles: do the chords favour the same keys as the
    melody? Compares the shape of the profiles, not their level; no hard key decision."""
    if melody_profile is None or chord_profile is None:
        return None
    if melody_profile.std() == 0 or chord_profile.std() == 0:
        return None
    return float(np.corrcoef(melody_profile, chord_profile)[0, 1])


# --- Fast path: identical numbers, vectorised ---------------------------------------------
# music21 builds a duration-weighted pitch-class distribution and takes its Pearson correlation
# with the profile rotated to each key. The same computation in numpy is ~100x faster, which the
# full-grid metric needs (one CPU core on the login node). It is NOT a second implementation of
# record: scripts that use it check it against key_profile() on a sample every run.


@lru_cache(maxsize=None)
def _profile_matrix(analyser: str) -> np.ndarray:
    """[24, 12] z-scored profiles from music21's own weights: row k = key k."""
    obj = ANALYSERS[analyser]()
    rows = [np.roll(np.asarray(obj.getWeights(mode), dtype=float), tonic)
            for mode in ("major", "minor") for tonic in range(12)]
    m = np.stack(rows)
    return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True)


def melody_histogram(melody_tokens: Sequence[int]) -> np.ndarray:
    """Frames per pitch class over sounding melody notes -- melody_stream()'s durations."""
    h, pitch = np.zeros(12), None
    for t in melody_tokens:
        t = int(t)
        if MELODY_ONSET[0] <= t <= MELODY_ONSET[1]:
            pitch = t - MELODY_ONSET[0]
            h[pitch % 12] += 1
        elif MELODY_HOLD[0] <= t <= MELODY_HOLD[1] and pitch == t - MELODY_HOLD[0]:
            h[pitch % 12] += 1
        else:
            pitch = None
    return h


def chord_histogram(chord_tokens: Sequence[int], id_to_name: Dict[int, str]) -> np.ndarray:
    """Frames per pitch class over sounding chord tones -- chord_stream()'s durations."""
    h = np.zeros(12)
    for t in chord_tokens:
        for pc in _chord_pcs(id_to_name.get(int(t), "")):
            h[pc] += 1
    return h


def fast_profile(hist: np.ndarray, analyser: str = "krumhansl") -> Optional[np.ndarray]:
    """key_profile() computed from a histogram; equal to it to floating-point precision."""
    if hist.sum() == 0 or hist.std() == 0:
        return None
    return _profile_matrix(analyser) @ ((hist - hist.mean()) / hist.std()) / 12.0


def key_relation(est: int, ref: int) -> str:
    """exact / fifth above / fifth below / relative / parallel / other (same categories as MIREX,
    with the fifth split by direction; MIREX credits only the fifth above)."""
    et, em, rt, rm = est % 12, est >= 12, ref % 12, ref >= 12
    if est == ref:
        return "exact"
    if em == rm and (et - rt) % 12 == 7:
        return "fifth above"
    if em == rm and (et - rt) % 12 == 5:
        return "fifth below"
    if (not rm and em and (et - rt) % 12 == 9) or (rm and not em and (et - rt) % 12 == 3):
        return "relative"
    if et == rt and em != rm:
        return "parallel"
    return "other"


def circle_of_fifths_distance(a: int, b: int) -> int:
    """Steps between two keys' tonics on the circle of fifths, after mapping each minor key to its
    relative major (so a key and its relative are 0 apart). 0-6."""
    ta = (a % 12 + 3) % 12 if a >= 12 else a % 12
    tb = (b % 12 + 3) % 12 if b >= 12 else b % 12
    k = (7 * (tb - ta)) % 12
    return min(k, 12 - k)
