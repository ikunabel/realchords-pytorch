"""Hookpad (Hooktheory raw) chord objects -> pitch classes and chord labels.

Hookpad stores a chord symbolically, relative to the key (Hooktheory_Raw.json,
``json.chords[]``): ``root`` (scale degree 1-7), ``type`` (5 = triad, 7, 9, 11,
13), ``inversion`` (0-3), ``applied`` (secondary chords: the chord is the
``applied``-th degree chord of the key whose tonic is ``root``), ``borrowed``
(a mode name, or 7 semitone offsets, to take the scale from), ``adds``,
``omits``, ``alterations``, ``suspensions``. Sheet Sage's processed form
(root pitch class + stacked intervals + inversion) was derived from these, but
this module renders the chord with documented rules and
build_chord_conversion_db.py checks it against Sheet Sage's result for every
chord. (Sheet Sage's own conversion is TheorytabChord.as_chord in
chrisdonahue/sheetsage, sheetsage/theory/theorytab.py; found 2026-11-23.)

Rules (refined against the processed data; see journal/reference/CHORD_CONVERSION_DB.md):
- scale: ``borrowed`` mode (or offsets) if given, else the key's mode;
- chord tones: diatonic thirds stacked from the root degree, ``type`` decides
  how many (triad 3 tones, 7th 4, 9th 5, 11th 6, 13th 7);
- applied chords: the chord is built on degree ``applied`` of the major scale
  on the target's pitch (V/x, vii/x, IV/x ...);
- suspensions replace the third (sus2 / sus4, both if both); adds append
  the 2nd/4th/6th (as 9/4/6 major intervals); omits drop the 3rd/5th;
  alterations shift a chord degree (b5, #5, b9, #9, #11, b13);
- inversion: the n-th tone of the (suspended) stack is the bass.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11], "ionian": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10], "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11], "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10], "harmonicMinor": [0, 2, 3, 5, 7, 8, 11],
    "phrygianDominant": [0, 1, 4, 5, 7, 8, 10],
}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
# chord degree (1, 3, 5, 7, 9, 11, 13) -> index in the stack of thirds
DEGREE_INDEX = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6}
TYPE_TONES = {5: 3, 7: 4, 9: 5, 11: 6, 13: 7}


def tonic_pc(tonic: str) -> int:
    """'F', 'Bb', 'F#' -> pitch class."""
    pc = LETTER_PC[tonic[0].upper()]
    for ch in tonic[1:]:
        pc += 1 if ch == "#" else -1 if ch == "b" else 0
    return pc % 12


def key_at(keys: List[Dict], beat: float) -> Optional[Dict]:
    """The key in force at `beat` (raw keys list, sorted by beat)."""
    current = None
    for k in sorted(keys or [], key=lambda k: k.get("beat", 0)):
        if k.get("beat", 0) <= beat + 1e-9:
            current = k
    return current or (keys[0] if keys else None)


def scale_of(key: Dict, borrowed) -> List[int]:
    if isinstance(borrowed, list) and len(borrowed) == 7:
        return [int(x) for x in borrowed]
    if isinstance(borrowed, str) and borrowed in SCALES:
        return SCALES[borrowed]
    return SCALES.get(key.get("scale", "major"), SCALES["major"])


def degree_pc(scale: List[int], tonic: int, degree: int) -> int:
    """Pitch class of scale degree `degree` (1-based, may exceed 7 or be < 1)."""
    d = degree - 1
    octave, idx = divmod(d, 7)
    return (tonic + scale[idx] + 12 * octave) % 12


def degree_semitones(scale: List[int], degree: int) -> int:
    """Semitones from degree 1 to `degree` (1-based) going up."""
    d = degree - 1
    octave, idx = divmod(d, 7)
    return scale[idx] + 12 * octave


def render(chord: Dict, key: Dict) -> Dict:
    """One Hookpad chord -> {'root_pc', 'pcs', 'bass_pc', 'tones' (semitones
    above the chord root, in stack order), 'roman', 'symbol', 'notes'}.
    Raises ValueError for chords it can't render (rests, root 0)."""
    if chord.get("isRest"):
        raise ValueError("rest")
    root_deg = int(chord.get("root", 0))
    if root_deg < 1 or root_deg > 7:
        raise ValueError(f"root degree {root_deg}")
    ctype = int(chord.get("type", 5))
    n_tones = TYPE_TONES.get(ctype, 3)
    tonic = tonic_pc(key.get("tonic", "C"))
    applied = int(chord.get("applied") or 0)
    borrowed = chord.get("borrowed")
    scale = scale_of(key, borrowed)

    if applied:
        # secondary chord: degree `applied` of the target's major scale; a
        # leading-tone chord (vii/x) from its harmonic minor, so vii7/x is
        # fully diminished (as Sheet Sage has it)
        target_pc = degree_pc(scale, tonic, root_deg)
        chord_scale = SCALES["harmonicMinor"] if applied == 7 else SCALES["major"]
        chord_tonic, chord_deg = target_pc, applied
    else:
        chord_scale, chord_tonic, chord_deg = scale, tonic, root_deg

    root_pc = degree_pc(chord_scale, chord_tonic, chord_deg)
    base = degree_semitones(chord_scale, chord_deg)
    # stack of thirds: chord degrees 1, 3, 5, 7, 9, 11, 13 -> semitones above the root
    stack = {}
    for i, cdeg in enumerate([1, 3, 5, 7, 9, 11, 13][:n_tones]):
        stack[cdeg] = degree_semitones(chord_scale, chord_deg + 2 * i) - base

    def diatonic(step: int) -> int:
        """Semitones above the chord root of the chord scale's degree `step` above it."""
        return degree_semitones(chord_scale, chord_deg + step) - base

    # suspensions and added notes are diatonic in the chord's scale
    sus = chord.get("suspensions") or []
    if sus:
        stack.pop(3, None)
        for s in sus:
            stack[{2: "s2", 4: "s4"}[int(s)]] = diatonic({2: 1, 4: 3}[int(s)])
    for a in chord.get("adds") or []:
        stack[f"a{a}"] = diatonic({9: 8, 2: 1, 4: 3, 11: 10, 6: 5, 13: 12}[int(a)])
    for o in chord.get("omits") or []:
        stack.pop(int(o), None)
    # an alteration shifts the chord degree if it is there, else adds it
    # altered from its natural (major-scale) interval
    for alt in chord.get("alterations") or []:
        acc, num = alt[0], int(alt[1:])
        shift = -1 if acc == "b" else 1
        if num in stack:
            stack[num] += shift
        else:
            stack[num] = {5: 7, 9: 14, 11: 17, 13: 21}[num] + shift

    order = [k for k in [1, "s2", 3, "s4", 5, 7, 9, 11, 13] if k in stack] + \
            [k for k in stack if k not in (1, "s2", 3, "s4", 5, 7, 9, 11, 13)]
    tones = [stack[k] for k in order]
    pcs = sorted({(root_pc + t) % 12 for t in tones})

    inv = int(chord.get("inversion") or 0)
    # bass = the inv-th tone of the basic stack (root, third or its suspension, fifth, seventh)
    basic = [k for k in [1, "s2" if 3 not in stack and "s2" in stack and "s4" not in stack else 3,
                         5, 7] if k in stack]
    if 3 not in stack and "s4" in stack:
        basic = [k for k in [1, "s4", 5, 7] if k in stack]
    bass_pc = (root_pc + stack[basic[inv]]) % 12 if inv < len(basic) else root_pc

    return {"root_pc": root_pc, "pcs": pcs, "bass_pc": bass_pc, "tones": tones,
            "roman": roman_numeral(chord, stack, ctype), "symbol": symbol(root_pc, stack, ctype, bass_pc)}


def _quality(stack: Dict) -> str:
    third, fifth = stack.get(3), stack.get(5)
    if third == 4 and fifth == 8:
        return "aug"
    if third == 3 and fifth == 6:
        return "dim"
    if third == 3:
        return "min"
    return "maj"


def roman_numeral(chord: Dict, stack: Dict, ctype: int) -> str:
    """V9sus4, ii7, viio7/V, bVI(maj) ... in Hookpad's spirit (for reading, not parsing)."""
    deg = int(chord["root"])
    q = _quality(stack)
    applied = int(chord.get("applied") or 0)
    num = ROMAN[(applied or deg) - 1]
    num = num.lower() if q in ("min", "dim") else num
    mark = {"dim": "o", "aug": "+"}.get(q, "")
    if q == "dim" and stack.get(7) == 10:
        mark = "ø"
    ext = "" if ctype == 5 else str(ctype)
    sus = "".join(f"sus{s}" for s in (chord.get("suspensions") or []))
    adds = "".join(f"add{a}" for a in (chord.get("adds") or []))
    omits = "".join(f"no{o}" for o in (chord.get("omits") or []))
    alts = "".join(f"({a})" for a in (chord.get("alterations") or []))
    inv = ["", "6" if ctype == 5 else "65", "64" if ctype == 5 else "43", "42"][int(chord.get("inversion") or 0)]
    out = f"{num}{mark}{ext}{inv}{sus}{adds}{omits}{alts}"
    if applied:
        out += f"/{ROMAN[deg - 1]}"
    b = chord.get("borrowed")
    if b:
        out += f" [{b if isinstance(b, str) else 'custom'}]"
    return out


def symbol(root_pc: int, stack: Dict, ctype: int, bass_pc: int) -> str:
    """Lead-sheet style symbol from the rendered stack (for reading, not parsing):
    root + quality + 7th + extensions + sus/add/alterations + /bass."""
    q = _quality(stack)
    seventh = stack.get(7)
    base = PC_NAMES[root_pc]
    if 3 not in stack and ("s2" in stack or "s4" in stack):
        qual = ""
    else:
        qual = {"maj": "", "min": "m", "dim": "dim", "aug": "aug"}[q]
    if seventh is not None:
        if q == "dim" and seventh == 9:
            qual, sev = "dim", "7"
        elif q == "dim" and seventh == 10:
            qual, sev = "m", "7b5"
        elif seventh == 11:
            sev = "maj7" if qual in ("", "aug") else "maj7"
        else:
            sev = "7"
        top = {9: "9", 11: "11", 13: "13"}.get(ctype)
        if top:
            sev = sev.replace("7", top) if sev in ("7", "maj7") else sev + f"({top})"
        name = base + qual + sev
    else:
        name = base + qual
    if "s2" in stack and 3 not in stack:
        name += "sus2"
    if "s4" in stack and 3 not in stack:
        name += "sus4"
    for k, v in stack.items():
        if isinstance(k, str) and k.startswith("a"):
            name += f"add{k[1:]}"
    if bass_pc != root_pc:
        name += f"/{PC_NAMES[bass_pc]}"
    return name


def processed_pcs(root_pc: int, intervals: List[int]) -> Tuple[List[int], List[int]]:
    """Sheet Sage form -> (pitch classes, root-position stack in semitones)."""
    stack = [0]
    for iv in intervals:
        stack.append(stack[-1] + iv)
    return sorted({(root_pc + s) % 12 for s in stack}), stack
