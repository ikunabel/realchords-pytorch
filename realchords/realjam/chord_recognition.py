"""Name a played chord with music21, mapped onto the model's chord vocabulary.

For ReaLJam's input mode (2026-11-17). The vocabulary's names come from
note_seq's ``pitches_to_chord_symbol``, which has no notion of omitted chord
tones: A C# G (an A7 without its fifth) only exists as "Dbped(add#5)(#11)" /
"Gped(add2)(addb5)". music21 (Cuthbert & Ariza 2010) finds the root by stacking
thirds and classifies the pitch-class set, including incomplete chords
("incomplete dominant-seventh chord", root A). On the training data its root
agrees with the annotated root for 97.1% of chord occurrences (journal/
reference/INTERACTION_MODES.md, 2026-11-17).

Reading, in order:
  1. incomplete: music21 says "incomplete <type>" -> root + that type's
     suffix (A E G -> A7, not the exact but odd "A5(add7)");
  2. exact: a vocabulary chord with music21's root and exactly the played
     pitch classes;
  3. nearest: the vocabulary chord with that root containing every played
     pitch class with the fewest extra notes (dyads, unusual sets).
A slash version is preferred when the lowest note is not the root and the
vocabulary has one.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# music21 common name (without "incomplete " / "enharmonic equivalent to ") ->
# our vocabulary's suffix
TYPE_SUFFIX = {
    "major triad": "", "minor triad": "m", "diminished triad": "o", "augmented triad": "+",
    "dominant seventh chord": "7", "major seventh chord": "maj7", "minor seventh chord": "m7",
    "half diminished seventh chord": "m7b5", "diminished seventh chord": "o7",
    "augmented seventh chord": "+7", "minor augmented tetrachord": "mmaj7",
    "dominant ninth": "9", "major ninth chord": "maj9", "minor ninth chord": "m9",
    "dominant eleventh": "11",
}
PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# Interval above the root -> its name, for "A7 (omit 5)"
INTERVAL_NAMES = {0: "1", 1: "b9", 2: "9", 3: "b3", 4: "3", 5: "11", 6: "b5", 7: "5",
                  8: "#5", 9: "6", 10: "7", 11: "maj7"}


def _base_type(common_name: str) -> str:
    """music21's type without "incomplete " / "enharmonic equivalent to ",
    hyphens as spaces (it writes "incomplete dominant-seventh chord")."""
    for prefix in ("incomplete ", "enharmonic equivalent to "):
        if common_name.startswith(prefix):
            common_name = common_name[len(prefix):]
    return common_name.replace("-", " ")


def recognize(pitches: List[int], vocab: List[Dict]) -> Optional[Dict]:
    """music21's reading of `pitches` (MIDI), mapped onto `vocab`.

    Args:
      pitches: the played chord's MIDI pitches
      vocab: [{symbol, pitchClasses, root, ...}] -- the model's chord onsets
        (Agent._onset_chord_info, with their token ids added by the server)

    Returns:
      the vocabulary entry chosen (so its token ids too) plus commonName,
      how ('exact' | 'incomplete' | 'nearest'), omitted (names of the chord
      tones not played, e.g. ["5"]) and display ("A7 (omit 5)"); None if
      nothing in the vocabulary fits. Always a chord of `vocab`, so it can go
      into the model's history as is.
    """
    from music21 import chord as m21chord  # imported lazily: slow first import

    if not pitches:
        return None
    ch = m21chord.Chord(sorted(pitches))
    root = ch.root().pitchClass
    common = ch.commonName
    played = {p % 12 for p in pitches}
    bass = min(pitches) % 12
    with_root = [c for c in vocab if c["root"] == root]

    def result(entry: Dict, how: str) -> Dict:
        if bass != root:  # the slash version, if the vocabulary has it
            slash = next((c for c in vocab if c["symbol"] == f"{entry['symbol']}/{PC_NAMES[bass]}"), None)
            entry = slash or entry
        omitted = sorted((set(entry["pitchClasses"]) - played), key=lambda pc: (pc - root) % 12)
        names = [INTERVAL_NAMES[(pc - root) % 12] for pc in omitted]
        display = entry["symbol"] + (f" (omit {', '.join(names)})" if names else "")
        return {**entry, "commonName": common, "how": how, "omitted": names, "display": display}

    suffix = TYPE_SUFFIX.get(_base_type(common))
    if common.startswith("incomplete ") and suffix is not None:
        symbol = PC_NAMES[root] + suffix
        match = next((c for c in with_root if c["symbol"] == symbol), None)
        if match and played <= set(match["pitchClasses"]):
            return result(match, "incomplete")

    exact = [c for c in with_root if set(c["pitchClasses"]) == played and "/" not in c["symbol"]]
    if exact:
        return result(min(exact, key=lambda c: len(c["symbol"])), "exact")

    supersets = [c for c in with_root if played <= set(c["pitchClasses"]) and "/" not in c["symbol"]]
    if not supersets:
        return None
    return result(min(supersets, key=lambda c: (len(c["pitchClasses"]), len(c["symbol"]))), "nearest")
