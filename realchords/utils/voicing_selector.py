"""Music-theory-aware voicing selector.

Given a chord symbol and a lookup table (``chord_voicings.json``), selects the
best candidate voicing for the current musical context by jointly optimising:

  1. **Voice leading** (weight: ``vl_weight``, default 0.6)
       Minimise total absolute pitch movement from the previous voicing using a
       greedy nearest-note assignment.  The candidate can be shifted by whole
       octaves (±``max_octave_shift`` octaves) to search for the best
       registration automatically.

  2. **Register** (weight: ``reg_weight``, default 0.3)
       Penalise voicings whose centroid deviates from ``target_mid`` (default
       MIDI 60, middle C).  The penalty is Gaussian with σ = ``reg_sigma``
       semitones (default 14).

  3. **Frequency prior** (weight: ``count_weight``, default 0.1)
       Prefer voicings that appeared more often in the source dataset (acts as a
       stylistic tiebreaker).

Hard constraints applied before scoring (violating candidates are dropped):

  * **Melody ceiling**: if ``melody_pitch`` is given and ``melody_role`` is
    ``"top"``, the highest chord note must be *strictly below* ``melody_pitch``.
  * **Melody floor**: if ``melody_role`` is ``"bass"``, the lowest chord note
    must be *strictly above* ``melody_pitch``.
  * **Absolute register bounds**: all pitches must stay in
    [``pitch_lo``, ``pitch_hi``] (defaults: 28–100).

If all candidates fail the hard constraints the melody constraint is silently
relaxed (register bounds are never relaxed).  Returns ``None`` only when the
lookup table has no entry for the requested chord.

Usage::

    from realchords.utils.voicing_selector import VoicingSelector

    sel = VoicingSelector("data/voicings/merged/chord_voicings.json")
    sel.reset()                                      # fresh state per song
    pitches = sel.select("Cmaj7")
    pitches = sel.select("Am7",  melody_pitch=72)    # melody ceiling at C5
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Voice-leading helpers
# ---------------------------------------------------------------------------

def _vl_cost(from_pitches: List[int], to_pitches: List[int]) -> float:
    """Greedy nearest-note voice-leading cost (average semitone movement).

    Each note in *from_pitches* is paired with its nearest available partner in
    *to_pitches*; unpaired notes from the larger set incur zero cost (the voice
    simply appears or disappears).  The result is normalised by the maximum of
    the two set sizes so it stays comparable across chords of different sizes.
    """
    if not from_pitches or not to_pitches:
        return 0.0

    remaining = list(to_pitches)
    total = 0
    pairs = 0
    for p in from_pitches:
        if not remaining:
            break
        nearest = min(remaining, key=lambda n, p=p: abs(n - p))
        total += abs(p - nearest)
        remaining.remove(nearest)
        pairs += 1

    return total / max(len(from_pitches), len(to_pitches))


def _shift(pitches: List[int], semitones: int) -> List[int]:
    return [p + semitones for p in pitches]


# ---------------------------------------------------------------------------
# VoicingSelector
# ---------------------------------------------------------------------------

class VoicingSelector:
    """Stateful selector that remembers the previous voicing for voice leading."""

    def __init__(
        self,
        voicings_path: str | Path,
        *,
        target_mid: int = 60,
        reg_sigma: float = 14.0,
        max_octave_shift: int = 3,
        pitch_lo: int = 28,
        pitch_hi: int = 100,
        vl_weight: float = 0.6,
        reg_weight: float = 0.3,
        count_weight: float = 0.1,
    ) -> None:
        with open(voicings_path, encoding="utf-8") as f:
            self._lookup: Dict[str, List[Dict]] = json.load(f)

        self.target_mid = target_mid
        self.reg_sigma = reg_sigma
        self.max_octave_shift = max_octave_shift
        self.pitch_lo = pitch_lo
        self.pitch_hi = pitch_hi
        self.vl_weight = vl_weight
        self.reg_weight = reg_weight
        self.count_weight = count_weight

        self._prev_voicing: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the previous voicing (call at the start of each new song)."""
        self._prev_voicing = None

    def select(
        self,
        chord_name: str,
        *,
        prev_voicing: Optional[List[int]] = None,
        melody_pitch: Optional[int] = None,
        melody_role: str = "top",
        update_state: bool = True,
    ) -> Optional[List[int]]:
        """Select the best voicing for *chord_name* given the current context.

        Parameters
        ----------
        chord_name:
            Key into the voicings lookup table (e.g. ``"Cmaj7"``).
        prev_voicing:
            Override the internally tracked previous voicing.  If ``None``,
            the selector uses its own internal state.
        melody_pitch:
            MIDI pitch of the current melody note, used as a hard constraint.
        melody_role:
            ``"top"``  → chord notes must all be *below* ``melody_pitch``
            ``"bass"`` → chord notes must all be *above* ``melody_pitch``
        update_state:
            If ``True`` (default), stores the returned voicing so the next
            call to :meth:`select` uses it for voice leading.

        Returns
        -------
        List[int] or None
            Chosen MIDI pitches, or ``None`` if no candidates exist.
        """
        candidates = self._lookup.get(chord_name, [])
        if not candidates:
            return None

        prev = prev_voicing if prev_voicing is not None else self._prev_voicing
        result = self._pick(candidates, prev, melody_pitch, melody_role)

        if update_state and result is not None:
            self._prev_voicing = result

        return result

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _candidates_with_shifts(
        self, candidates: List[Dict]
    ) -> List[Tuple[List[int], int, float]]:
        """Enumerate (shifted_pitches, shift_semitones, count) for all valid
        (candidate, octave-shift) combinations within register bounds."""
        out = []
        shifts = range(
            -self.max_octave_shift * 12,
            self.max_octave_shift * 12 + 1,
            12,
        )
        for entry in candidates:
            raw = entry["pitches"]
            count = entry["count"]
            for semitones in shifts:
                shifted = _shift(raw, semitones)
                if min(shifted) >= self.pitch_lo and max(shifted) <= self.pitch_hi:
                    out.append((shifted, semitones, count))
        return out

    def _score(
        self,
        pitches: List[int],
        count: float,
        prev: Optional[List[int]],
        max_log_count: float,
    ) -> float:
        """Lower score = better candidate."""
        # 1. Voice leading
        if prev is not None:
            vl = _vl_cost(prev, pitches) / 12.0  # normalise: 12 st = 1 octave
        else:
            vl = 0.0

        # 2. Register: Gaussian penalty centred on target_mid
        centroid = sum(pitches) / len(pitches)
        reg = ((centroid - self.target_mid) / self.reg_sigma) ** 2

        # 3. Frequency prior (negative: higher count → lower score)
        cnt = -math.log1p(count) / (max_log_count + 1e-9)

        return self.vl_weight * vl + self.reg_weight * reg + self.count_weight * cnt

    def _pick(
        self,
        candidates: List[Dict],
        prev: Optional[List[int]],
        melody_pitch: Optional[int],
        melody_role: str,
    ) -> Optional[List[int]]:
        all_shifted = self._candidates_with_shifts(candidates)
        if not all_shifted:
            return None

        max_log_count = math.log1p(max(c for _, _, c in all_shifted))

        def _melody_ok(pitches: List[int]) -> bool:
            if melody_pitch is None:
                return True
            if melody_role == "top":
                return max(pitches) < melody_pitch
            if melody_role == "bass":
                return min(pitches) > melody_pitch
            return True

        # First pass: apply melody hard constraint
        constrained = [t for t in all_shifted if _melody_ok(t[0])]

        # Fallback: drop melody constraint if nothing passes
        pool = constrained if constrained else all_shifted

        best_pitches, _, best_count = min(
            pool,
            key=lambda t: self._score(t[0], t[2], prev, max_log_count),
        )
        return best_pitches


