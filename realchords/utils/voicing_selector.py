"""Music-theory-aware voicing selector.

Given a chord symbol and a lookup table (``chord_voicings.json``), selects the
best candidate voicing for the current musical context by jointly optimising
up to four weighted terms. Only the first two are exposed as sliders in the
live ReaLJam UI (kept deliberately minimal); all four remain available
through this API and the CLI demo (``scripts/extract_voicings/voicing_selector.py``)
for offline experimentation:

  1. **Voice leading** (weight: ``vl_weight``, default 0.5) -- exposed in the UI
       Minimise total absolute pitch movement from the previous voicing using a
       greedy nearest-note assignment. The candidate can be shifted by whole
       octaves (±``max_octave_shift`` octaves) to search for the best
       registration automatically. Clamped to [0, 1] (capped at one octave of
       movement) so it stays comparable to the other terms -- see below.

  2. **Register** (weight: ``reg_weight``, default 0.2; target: ``target_mid``,
     default MIDI 60 / middle C) -- weight exposed in the UI, target fixed
       Penalise voicings whose centroid deviates from ``target_mid``. Gaussian
       with σ = ``reg_sigma`` semitones (default 14), clamped to [0, 1].

  3. **Note count** (weight: ``note_count_weight``, default 0.0 -- off unless
     explicitly set) -- not exposed in the UI
       One-directional preference for fewer notes. Score is
       ``n_notes / max_notes`` across all candidates for the chord, so a
       3-note voicing scores 0 relative to the densest-in-note-count
       candidate available. Already naturally in [0, 1].

  4. **Density** (weight: ``density_weight``, default 0.0 -- off unless
     explicitly set; target: ``density_target``, default 0.5) -- not exposed
     in the UI
       How tightly packed the voicing's notes are in *pitch space*,
       independent of how many notes there are -- e.g. a 4-note voicing
       spanning a single octave (dense/clustered) vs. the same 4 notes spread
       across 3 octaves (spread out). Measured as the average semitone gap
       between adjacent sorted pitches, ``(max - min) / (n - 1)``, capped at 2
       octaves and inverted so 1.0 = maximally clustered, 0.0 = maximally
       spread. Scored as squared distance from ``density_target``, in [0, 1].

All four raw weights are re-normalised to sum to 1 before scoring (see
``select``), so setting any subset of sliders to the *same* value -- whether
that's all at maximum or any other equal point -- always means "weight these
terms equally," regardless of the absolute values chosen. With note-count and
density weights defaulted to 0, the live UI's two sliders (voice leading,
register) are the only terms that actually influence scoring unless a caller
explicitly overrides the other two.

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


_MAX_SPREAD_SEMITONES = 24.0  # 2 octaves average gap treated as "fully spread"


def _density(pitches: List[int]) -> float:
    """How tightly packed pitches are, in [0, 1]: 1.0 = clustered (adjacent
    sorted pitches close together), 0.0 = spread out (wide average gap).

    Uses the average gap between adjacent sorted pitches, which for any list
    telescopes to (max - min) / (n - 1) regardless of the individual gaps'
    distribution. A single-note voicing has no spacing to measure; treated as
    neutral (0.5) rather than favoring either extreme.
    """
    if len(pitches) < 2:
        return 0.5
    avg_gap = (max(pitches) - min(pitches)) / (len(pitches) - 1)
    spread = min(avg_gap / _MAX_SPREAD_SEMITONES, 1.0)
    return 1.0 - spread


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
        vl_weight: float = 0.5,
        reg_weight: float = 0.2,
        note_count_weight: float = 0.0,
        density_weight: float = 0.0,
        density_target: float = 0.5,
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
        self.note_count_weight = note_count_weight
        self.density_weight = density_weight
        self.density_target = density_target

        self._prev_voicing: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the previous voicing (call at the start of each new song)."""
        self._prev_voicing = None

    def list_chords(self) -> List[str]:
        """All chord symbols covered by the lookup table, sorted."""
        return sorted(self._lookup.keys())

    def get_voicings(self, chord_name: str) -> List[Dict]:
        """Raw stored voicing list for one chord (already sorted by count
        descending, see match_voicings_to_chords.py), for browsing/auditioning
        -- independent of select()'s scoring/context-aware picking."""
        return self._lookup.get(chord_name, [])

    def select(
        self,
        chord_name: str,
        *,
        prev_voicing: Optional[List[int]] = None,
        melody_pitch: Optional[int] = None,
        melody_role: str = "top",
        update_state: bool = True,
        vl_weight: Optional[float] = None,
        reg_weight: Optional[float] = None,
        note_count_weight: Optional[float] = None,
        density_weight: Optional[float] = None,
        target_mid: Optional[float] = None,
        density_target: Optional[float] = None,
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
        vl_weight, reg_weight, note_count_weight, density_weight, target_mid,
        density_target:
            Per-call overrides for the corresponding constructor defaults.
            ``None`` (default) falls back to whatever this instance was
            constructed with. Passed explicitly rather than mutating
            instance state, so one shared VoicingSelector stays safe to use
            concurrently with different settings per caller (e.g. one
            instance shared across live-jam clients with different UI
            slider values). The four weights are re-normalised to sum to 1
            before scoring -- see module docstring.

        Returns
        -------
        List[int] or None
            Chosen MIDI pitches, or ``None`` if no candidates exist.
        """
        candidates = self._lookup.get(chord_name, [])
        if not candidates:
            return None

        weights = {
            "vl": self.vl_weight if vl_weight is None else vl_weight,
            "reg": self.reg_weight if reg_weight is None else reg_weight,
            "note_count": (
                self.note_count_weight if note_count_weight is None else note_count_weight
            ),
            "density": self.density_weight if density_weight is None else density_weight,
        }
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 0.25 for k in weights}

        prev = prev_voicing if prev_voicing is not None else self._prev_voicing
        result = self._pick(
            candidates, prev, melody_pitch, melody_role,
            vl_weight=weights["vl"],
            reg_weight=weights["reg"],
            note_count_weight=weights["note_count"],
            density_weight=weights["density"],
            target_mid=self.target_mid if target_mid is None else target_mid,
            density_target=self.density_target if density_target is None else density_target,
        )

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
        prev: Optional[List[int]],
        max_notes: int,
        *,
        vl_weight: float,
        reg_weight: float,
        note_count_weight: float,
        density_weight: float,
        target_mid: float,
        density_target: float,
    ) -> float:
        """Lower score = better candidate. Every term is clamped to [0, 1]
        so that (post weight-normalisation) equal weights mean equal
        influence, not just equal coefficients on differently-scaled terms.
        """
        # 1. Voice leading: capped at one octave of movement
        if prev is not None:
            vl = min(_vl_cost(prev, pitches) / 12.0, 1.0)
        else:
            vl = 0.0

        # 2. Register: Gaussian penalty centred on target_mid, capped at 1
        centroid = sum(pitches) / len(pitches)
        reg = min(((centroid - target_mid) / self.reg_sigma) ** 2, 1.0)

        # 3. Note count: prefer fewer notes (ratio relative to worst candidate)
        note_count = len(pitches) / max_notes if max_notes > 0 else 0.0

        # 4. Density: squared distance from density_target, already in [0, 1]
        density = (_density(pitches) - density_target) ** 2

        return (
            vl_weight * vl
            + reg_weight * reg
            + note_count_weight * note_count
            + density_weight * density
        )

    def _pick(
        self,
        candidates: List[Dict],
        prev: Optional[List[int]],
        melody_pitch: Optional[int],
        melody_role: str,
        *,
        vl_weight: float,
        reg_weight: float,
        note_count_weight: float,
        density_weight: float,
        target_mid: float,
        density_target: float,
    ) -> Optional[List[int]]:
        all_shifted = self._candidates_with_shifts(candidates)
        if not all_shifted:
            return None

        max_notes = max(len(p) for p, _, _ in all_shifted)

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

        best_pitches, _, _ = min(
            pool,
            key=lambda t: self._score(
                t[0], prev, max_notes,
                vl_weight=vl_weight, reg_weight=reg_weight,
                note_count_weight=note_count_weight, density_weight=density_weight,
                target_mid=target_mid, density_target=density_target,
            ),
        )
        return best_pitches


