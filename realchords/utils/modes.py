"""Pitch-class set utilities for mode and chord-scale analysis.

Builds reference tables for chord symbols, the 21 parent-scale modes, and
chord-quality-to-mode mappings. The curated map is the working definition for
a modal **note-in-mode** metric: at each frame, look up the chord quality,
take the primary mode, and test whether melody pitch classes lie in that mode.

Curated pairings follow chord-scale pedagogy (Nettles & Graf, Levine, Berklee).
The exhaustive map lists all subset matches; the curated map keeps idiomatic
modal choices. No song key or Roman-numeral function — harmony is treated
modally, chord by chord.

References: Russell 1953/2001; Nettles & Graf 2015; Levine 1995; Mulholland &
Hojnacki 2013; Haerle 1982. See journal/modes.md.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Sequence, Tuple

from note_seq import chord_symbols_lib

DEFAULT_PITCH_CLASS_CHORD_MAP_PATH = (
    Path(__file__).resolve().parent / "pitch_class_chord_map.jsonl"
)
DEFAULT_CHORD_QUALITY_MODE_MAP_PATH = (
    Path(__file__).resolve().parent / "chord_quality_mode_map.jsonl"
)
DEFAULT_CHORD_QUALITY_MODE_MAP_ALL_PATH = (
    Path(__file__).resolve().parent / "chord_quality_mode_map_all.jsonl"
)
DEFAULT_CHORD_NAMES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "cache" / "chord_names.json"
)

PitchClassSet = FrozenSet[int]
PitchClassChordMap = Dict[PitchClassSet, str]
ModeInfo = Dict[str, Any]

PARENT_SCALE_INTERVALS: Dict[str, List[int]] = {
    "major": [2, 2, 1, 2, 2, 2, 1],
    "harmonic_minor": [2, 1, 2, 2, 1, 3, 1],
    "melodic_minor": [2, 1, 2, 2, 2, 2, 1],
}

PARENT_SCALE_MODE_NAMES: Dict[str, List[str]] = {
    "major": [
        "Ionian",
        "Dorian",
        "Phrygian",
        "Lydian",
        "Mixolydian",
        "Aeolian",
        "Locrian",
    ],
    "harmonic_minor": [
        "Harmonic minor",
        "Locrian natural 6",
        "Ionian augmented",
        "Romanian minor",
        "Phrygian dominant",
        "Lydian sharp 2",
        "Ultralocrian",
    ],
    "melodic_minor": [
        "Melodic minor",
        "Dorian flat 2",
        "Lydian augmented",
        "Lydian dominant",
        "Mixolydian flat 6",
        "Locrian sharp 2",
        "Altered",
    ],
}

ModeRef = Tuple[str, int]
MAX_CURATED_MODES = 3
UNDERDETERMINED_MAX_PITCHES = 2

# Bibliography key: [N&G] Nettles & Graf; [Levine] Levine; [Berklee] Mulholland & Hojnacki; [Russell] Russell

# Chord quality → modal scale(s), priority order. See journal/modes.md.
CURATED_CHORD_QUALITY_MODES: Dict[str, List[ModeRef]] = {
    # Major sonorities [Levine, N&G, Russell]
    "": [("major", 1), ("major", 4), ("major", 5)],
    "6": [("major", 1), ("major", 5)],
    "maj7": [("major", 1), ("major", 4)],
    "maj9": [("major", 1), ("major", 4)],
    "maj11": [("major", 1)],
    "maj13": [("major", 1)],
    # Dominant family [Levine, Berklee]
    "7": [("major", 5), ("melodic_minor", 4)],
    "9": [("major", 5), ("melodic_minor", 4)],
    "11": [("major", 5)],
    "13": [("major", 5)],
    # Minor sonorities [N&G, Levine]
    "m": [("major", 2), ("major", 6), ("major", 3)],
    "m6": [("major", 2), ("melodic_minor", 1)],
    "m7": [("major", 2), ("major", 6)],
    "m9": [("major", 2), ("major", 6)],
    "m11": [("major", 2), ("major", 6)],
    "m13": [("major", 2)],
    # Diminished / half-dim [N&G, Berklee]
    "m7b5": [("major", 7), ("melodic_minor", 6)],
    "o": [("major", 7)],
    "o7": [("harmonic_minor", 2)],
    "mmaj7": [("melodic_minor", 1), ("harmonic_minor", 1)],
    # Augmented / suspended [Levine]
    "+": [("melodic_minor", 3), ("harmonic_minor", 3)],
    "+7": [("melodic_minor", 7), ("harmonic_minor", 5)],
    "sus": [("major", 5), ("major", 2)],
    "sus2": [("major", 2), ("major", 5)],
    "sus7": [("major", 5), ("major", 2)],
    # Too few chord tones to pick a mode from the sonority alone
    "5": [],
    "ped": [],
}

# Fallback priority when no explicit curation exists but subset matches do.
# Mirrors the families above; see journal/modes.md § "Curated mapping".
FAMILY_MODE_PRIORITY: Dict[str, List[ModeRef]] = {
    "major": [("major", 1), ("major", 4), ("major", 5)],
    "maj7": [("major", 1), ("major", 4)],
    "minor": [("major", 2), ("major", 6), ("major", 3)],
    "min7": [("major", 2), ("major", 6)],
    "dominant": [("major", 5), ("melodic_minor", 4), ("melodic_minor", 7)],
    "half_dim": [("major", 7), ("melodic_minor", 6)],
    "dim": [("major", 7), ("harmonic_minor", 2)],
    "aug": [("melodic_minor", 3), ("harmonic_minor", 3)],
    "min_maj7": [("melodic_minor", 1), ("harmonic_minor", 1)],
}

OBSCURE_MODE_REFS: FrozenSet[ModeRef] = frozenset(
    {
        # Ultralocrian / Altered / Lydian #2: valid subset matches but rarely
        # taught as default options without explicit alterations [Levine, N&G]
        ("harmonic_minor", 7),
        ("harmonic_minor", 6),
        ("melodic_minor", 7),
    }
)


def map_octave_pitch_combinations_to_chords(
    min_pitches: int = 3,
    max_pitches: int = 7,
    return_unresolved: bool = False,
) -> PitchClassChordMap | Tuple[PitchClassChordMap, List[PitchClassSet]]:
    """Map every pitch-class combination within one octave to a chord symbol.

    Enumerates all combinations of 3–7 distinct pitch classes from {0, …, 11}
    and resolves each set to a lead-sheet chord name via
    ``note_seq.chord_symbols_lib.pitches_to_chord_symbol`` — the same helper
    used in ``realchords.utils.data_utils.to_chord_name``.

    Pitch classes are passed in ascending order within the octave (0–11). Sets
    that ``note_seq`` cannot interpret are skipped.

    Args:
        min_pitches: Minimum number of pitch classes per combination.
        max_pitches: Maximum number of pitch classes per combination.
        return_unresolved: If True, also return pitch-class sets that could not
            be mapped to a chord symbol.

    Returns:
        A dict mapping each resolved pitch-class set (as a ``frozenset``) to its
        chord symbol string. With ``return_unresolved=True``, returns
        ``(mapping, unresolved_sets)``.

    Raises:
        ValueError: If ``min_pitches`` or ``max_pitches`` are out of range.
    """
    if min_pitches < 1 or max_pitches > 12:
        raise ValueError("Pitch counts must satisfy 1 <= min_pitches <= max_pitches <= 12.")
    if min_pitches > max_pitches:
        raise ValueError("min_pitches must not exceed max_pitches.")

    mapping: PitchClassChordMap = {}
    unresolved: List[PitchClassSet] = []

    for size in range(min_pitches, max_pitches + 1):
        for combo in combinations(range(12), size):
            pitch_classes = list(combo)
            pitch_class_set = frozenset(pitch_classes)
            try:
                chord_name = chord_symbols_lib.pitches_to_chord_symbol(pitch_classes)
            except Exception:
                unresolved.append(pitch_class_set)
                continue
            mapping[pitch_class_set] = chord_name

    if return_unresolved:
        return mapping, unresolved
    return mapping


def _rotate_intervals(intervals: List[int], steps: int) -> List[int]:
    steps %= len(intervals)
    return intervals[steps:] + intervals[:steps]


def _intervals_to_pitch_classes(root_pc: int, intervals: List[int]) -> List[int]:
    pitch_classes = [root_pc % 12]
    current = root_pc % 12
    for interval in intervals[:-1]:
        current = (current + interval) % 12
        pitch_classes.append(current)
    return pitch_classes


def _half_steps_from_intervals(intervals: List[int]) -> List[Dict[str, int]]:
    """Return 1-indexed scale-degree pairs separated by a half step."""
    half_steps = []
    for index, semitones in enumerate(intervals):
        if semitones != 1:
            continue
        from_degree = index + 1
        to_degree = index + 2 if index < 6 else 1
        half_steps.append({"from_degree": from_degree, "to_degree": to_degree})
    return half_steps


def _build_mode_info(
    name: str,
    mode_index: int,
    intervals: List[int],
    root_pc: int,
) -> ModeInfo:
    return {
        "name": name,
        "mode_index": mode_index,
        "intervals": intervals,
        "pitch_classes": _intervals_to_pitch_classes(root_pc, intervals),
        "half_steps": _half_steps_from_intervals(intervals),
    }


def list_scale_modes(root_pc: int = 0) -> Dict[str, List[ModeInfo]]:
    """List the seven modes of the major, harmonic minor, and melodic minor scales.

    Each mode is a rotation of the parent scale's interval pattern. The result
    includes semitone steps between consecutive scale degrees (the final interval
    wraps from degree 7 back to degree 1) and explicit half-step locations using
    1-indexed scale degrees.

    Mode names and interval rotations follow standard modal theory; see
    journal/modes.md § "Twenty-one parent-scale modes".

    Args:
        root_pc: Root pitch class (0–11) for the parent scale of each family.
            Defaults to 0 (C).

    Returns:
        Dict with keys ``major``, ``harmonic_minor``, and ``melodic_minor``. Each
        value is a list of seven mode records.
    """
    modes: Dict[str, List[ModeInfo]] = {}
    for parent_name, parent_intervals in PARENT_SCALE_INTERVALS.items():
        mode_names = PARENT_SCALE_MODE_NAMES[parent_name]
        modes[parent_name] = [
            _build_mode_info(
                name=mode_name,
                mode_index=index + 1,
                intervals=_rotate_intervals(parent_intervals, index),
                root_pc=root_pc,
            )
            for index, mode_name in enumerate(mode_names)
        ]
    return modes


def _strip_quality_extensions(quality: str) -> str:
    return re.sub(r"\([^)]*\)", "", quality)


def _mode_ref(mode: Dict[str, Any]) -> ModeRef:
    return (mode["parent_scale"], mode["mode_index"])


def _mode_from_ref(
    mode_ref: ModeRef,
    modes: Dict[str, List[ModeInfo]],
) -> Dict[str, Any] | None:
    parent_scale, mode_index = mode_ref
    for mode in modes[parent_scale]:
        if mode["mode_index"] == mode_index:
            return {
                "parent_scale": parent_scale,
                "mode_index": mode_index,
                "name": mode["name"],
            }
    return None


def _modes_from_refs(
    mode_refs: Sequence[ModeRef],
    modes: Dict[str, List[ModeInfo]],
) -> List[Dict[str, Any]]:
    resolved = []
    for mode_ref in mode_refs:
        mode = _mode_from_ref(mode_ref, modes)
        if mode is not None:
            resolved.append(mode)
    return resolved


def _annotate_mode_roles(modes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotated = []
    for index, mode in enumerate(modes):
        annotated.append(
            {
                **mode,
                "role": "primary" if index == 0 else "alternative",
            }
        )
    return annotated


def _classify_quality_family(quality: str) -> str:
    base = _strip_quality_extensions(quality)
    if base in {"", "6"} or base.startswith("maj"):
        return "maj7" if "maj7" in base or base.startswith("maj") else "major"
    if base in {"m", "m6"} or (base.startswith("m") and "maj" not in base and "m7b5" not in base):
        return "min7" if "7" in base or "9" in base or "11" in base or "13" in base else "minor"
    if base in {"m7b5"}:
        return "half_dim"
    if base in {"o", "o7"}:
        return "dim"
    if base in {"+", "+7"} or base.startswith("+"):
        return "aug"
    if base == "mmaj7" or base.startswith("mmaj"):
        return "min_maj7"
    if "7" in base or "9" in base or "11" in base or "13" in base or "sus7" in base:
        return "dominant"
    return "major"


def _narrow_modes_by_extensions(
    quality: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Narrow candidates using alteration-specific chord-scale rules.

    #11 → Lydian family [Russell, Levine]; b13 → Mixolydian b6 / Phrygian dom.
    [Levine, N&G]; #9 → Altered / Phrygian dominant [Levine]; addb2 → Phrygian
    family [Berklee]. See journal/modes.md § "Extension narrowing".
    """
    if not candidates:
        return candidates

    if "(#11)" in quality or "#11" in quality:
        if "maj" in quality:
            filtered = [mode for mode in candidates if mode["name"] == "Lydian"]
            if filtered:
                return filtered
        if quality.startswith(("7", "9", "11", "13")) or "+7" in quality:
            filtered = [
                mode
                for mode in candidates
                if mode["name"] in {"Lydian dominant", "Lydian"}
            ]
            if filtered:
                return filtered
        if "m" in quality[:2]:
            filtered = [mode for mode in candidates if mode["name"] == "Romanian minor"]
            if filtered:
                return filtered

    if "(b13)" in quality or "b13" in quality:
        filtered = [
            mode
            for mode in candidates
            if mode["name"]
            in {
                "Mixolydian flat 6",
                "Phrygian dominant",
                "Aeolian",
                "Phrygian",
                "Harmonic minor",
            }
        ]
        if filtered:
            return filtered

    if "(#9)" in quality or "#9" in quality:
        filtered = [
            mode
            for mode in candidates
            if mode["name"] in {"Altered", "Phrygian dominant", "Lydian sharp 2"}
        ]
        if filtered:
            return filtered

    if "(addb2)" in quality or "addb2" in quality:
        filtered = [
            mode
            for mode in candidates
            if mode["name"] in {"Phrygian dominant", "Dorian flat 2", "Phrygian"}
        ]
        if filtered:
            return filtered

    return candidates


def _filter_obscure_modes(
    candidates: List[Dict[str, Any]],
    quality: str,
) -> List[Dict[str, Any]]:
    if "o7" in quality or "Ultralocrian" in {mode["name"] for mode in candidates}:
        return candidates
    return [mode for mode in candidates if _mode_ref(mode) not in OBSCURE_MODE_REFS]


def _intersect_with_exhaustive(
    curated: List[Dict[str, Any]],
    exhaustive: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    exhaustive_refs = {_mode_ref(mode) for mode in exhaustive}
    return [mode for mode in curated if _mode_ref(mode) in exhaustive_refs]


def _pick_by_family_priority(
    quality: str,
    exhaustive: List[Dict[str, Any]],
    modes: Dict[str, List[ModeInfo]],
) -> List[Dict[str, Any]]:
    family = _classify_quality_family(quality)
    priority_refs = FAMILY_MODE_PRIORITY.get(family, FAMILY_MODE_PRIORITY["major"])
    exhaustive_refs = {_mode_ref(mode) for mode in exhaustive}
    selected = [
        mode
        for mode_ref in priority_refs
        if mode_ref in exhaustive_refs
        for mode in [_mode_from_ref(mode_ref, modes)]
        if mode is not None
    ]
    return selected


def _all_mode_dicts(modes: Dict[str, List[ModeInfo]]) -> List[Dict[str, Any]]:
    return [
        {
            "parent_scale": parent_scale,
            "mode_index": mode["mode_index"],
            "name": mode["name"],
        }
        for parent_scale, mode_list in modes.items()
        for mode in mode_list
    ]


def _mode_pitch_classes(
    mode: Dict[str, Any],
    modes: Dict[str, List[ModeInfo]],
) -> PitchClassSet:
    for entry in modes[mode["parent_scale"]]:
        if entry["mode_index"] == mode["mode_index"]:
            return frozenset(entry["pitch_classes"])
    return frozenset()


def _combinatorial_fallback_modes(
    exhaustive_matches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """All exhaustive subset matches (no obscure filtering)."""
    return _annotate_mode_roles(exhaustive_matches)


def _extension_hint_modes(
    quality: str,
    modes: Dict[str, List[ModeInfo]],
) -> List[Dict[str, Any]]:
    """Pick modes from alteration rules when subset matching fails."""
    if not re.search(r"#9|#11|b13|addb2", quality):
        return []
    return _narrow_modes_by_extensions(quality, _all_mode_dicts(modes))


def _best_overlap_modes(
    pitch_classes: PitchClassSet,
    modes: Dict[str, List[ModeInfo]],
) -> List[Dict[str, Any]]:
    """Modes sharing the largest number of chord pitch classes."""
    best_score = 0
    best_modes: List[Dict[str, Any]] = []
    for mode in _all_mode_dicts(modes):
        overlap = len(pitch_classes & _mode_pitch_classes(mode, modes))
        if overlap > best_score:
            best_score = overlap
            best_modes = [mode]
        elif overlap == best_score and overlap > 0:
            best_modes.append(mode)
    return best_modes


def _family_default_modes(
    quality: str,
    modes: Dict[str, List[ModeInfo]],
) -> List[Dict[str, Any]]:
    family = _classify_quality_family(quality)
    priority_refs = FAMILY_MODE_PRIORITY.get(family, FAMILY_MODE_PRIORITY["major"])
    return _modes_from_refs(priority_refs, modes)


def _resolve_mode_fallback(
    quality: str,
    exhaustive_matches: List[Dict[str, Any]],
    modes: Dict[str, List[ModeInfo]],
    pitch_classes: PitchClassSet,
) -> Tuple[List[Dict[str, Any]], str | None]:
    if exhaustive_matches:
        return _combinatorial_fallback_modes(exhaustive_matches), "combinatorial"

    hinted = _extension_hint_modes(quality, modes)
    if hinted:
        return _annotate_mode_roles(hinted[:MAX_CURATED_MODES]), "extension_hint"

    overlap = _best_overlap_modes(pitch_classes, modes)
    if overlap:
        return _annotate_mode_roles(overlap[:MAX_CURATED_MODES]), "best_overlap"

    family = _family_default_modes(quality, modes)
    if family:
        return _annotate_mode_roles(family[:MAX_CURATED_MODES]), "family_default"

    return [], None


def _is_underdetermined_quality(
    quality: str,
    pitch_classes: PitchClassSet,
) -> bool:
    return len(pitch_classes) <= UNDERDETERMINED_MAX_PITCHES and (
        quality in {"", "5", "ped"} or quality.startswith("ped")
    )


def curate_modes_for_chord_quality(
    quality: str,
    exhaustive_matches: List[Dict[str, Any]],
    modes: Dict[str, List[ModeInfo]],
    pitch_classes: PitchClassSet,
) -> Tuple[List[Dict[str, Any]], bool, str | None]:
    """Return curated modal scale choices for a chord quality.

    Uses ``CURATED_CHORD_QUALITY_MODES`` and extension narrowing. When curation
    yields no modes, falls back to combinatorial matches, then extension hints,
    best pitch-class overlap, or family defaults.
    """
    if _is_underdetermined_quality(quality, pitch_classes):
        fallback, fallback_type = _resolve_mode_fallback(
            quality, exhaustive_matches, modes, pitch_classes
        )
        return fallback, True, fallback_type

    if quality in CURATED_CHORD_QUALITY_MODES:
        curated = _modes_from_refs(CURATED_CHORD_QUALITY_MODES[quality], modes)
    else:
        base = _strip_quality_extensions(quality)
        if base in CURATED_CHORD_QUALITY_MODES:
            curated = _modes_from_refs(CURATED_CHORD_QUALITY_MODES[base], modes)
        else:
            curated = _pick_by_family_priority(quality, exhaustive_matches, modes)

    curated = _intersect_with_exhaustive(curated, exhaustive_matches)
    curated = _narrow_modes_by_extensions(quality, curated)
    curated = _filter_obscure_modes(curated, quality)

    if not curated:
        fallback, fallback_type = _resolve_mode_fallback(
            quality, exhaustive_matches, modes, pitch_classes
        )
        if fallback:
            return fallback, False, fallback_type

    return _annotate_mode_roles(curated[:MAX_CURATED_MODES]), False, None


def extract_chord_quality(chord_symbol: str) -> str:
    """Return the root-independent chord quality suffix from a lead-sheet symbol.

    Slash bass and root pitch class are stripped. Examples:
    ``Cmaj7`` → ``maj7``, ``Am7/E`` → ``m7``, ``C`` → ``""`` (major triad).
    """
    base_symbol = chord_symbol.split("/")[0]
    _root, quality, extension, _slash = chord_symbols_lib._split_chord_symbol(base_symbol)
    return quality + extension


def chord_quality_to_pitch_classes(quality: str) -> PitchClassSet:
    """Map a chord quality to pitch classes using ``C`` as the reference root."""
    chord_symbol = f"C{quality}"
    return frozenset(
        pitch % 12 for pitch in chord_symbols_lib.chord_symbol_pitches(chord_symbol)
    )


def collect_chord_qualities_from_names(
    chord_names_path: Path | str = DEFAULT_CHORD_NAMES_PATH,
) -> List[str]:
    """Collect unique chord qualities from a ``chord_names.json`` vocabulary file."""
    chord_names_path = Path(chord_names_path)
    with chord_names_path.open("r", encoding="utf-8") as handle:
        chord_names = json.load(handle)
    qualities = {extract_chord_quality(name) for name in chord_names}
    return sorted(qualities, key=lambda quality: (len(quality), quality))


def _iter_mode_entries(
    modes: Dict[str, List[ModeInfo]],
) -> Iterator[Tuple[str, ModeInfo]]:
    for parent_scale, mode_list in modes.items():
        for mode in mode_list:
            yield parent_scale, mode


def find_modes_containing_pitch_classes(
    pitch_classes: PitchClassSet,
    modes: Dict[str, List[ModeInfo]] | None = None,
    root_pc: int = 0,
) -> List[Dict[str, Any]]:
    """Return modes whose pitch classes are a superset of ``pitch_classes``."""
    if modes is None:
        modes = list_scale_modes(root_pc=root_pc)

    matches = []
    for parent_scale, mode in _iter_mode_entries(modes):
        mode_pitch_classes = frozenset(mode["pitch_classes"])
        if pitch_classes <= mode_pitch_classes:
            matches.append(
                {
                    "parent_scale": parent_scale,
                    "mode_index": mode["mode_index"],
                    "name": mode["name"],
                }
            )
    return matches


def map_chord_qualities_to_modes(
    qualities: Sequence[str] | None = None,
    *,
    chord_names_path: Path | str = DEFAULT_CHORD_NAMES_PATH,
    root_pc: int = 0,
    curated: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Map chord qualities to compatible modes.

    When ``curated`` is True (default), returns pedagogically narrowed mode
    choices with primary/alternative roles. When False, returns every mode whose
    pitch-class set is a superset of the chord tones.
    """
    if qualities is None:
        qualities = collect_chord_qualities_from_names(chord_names_path)

    modes = list_scale_modes(root_pc=root_pc)
    mapping: Dict[str, Dict[str, Any]] = {}
    for quality in qualities:
        pitch_classes = chord_quality_to_pitch_classes(quality)
        exhaustive = find_modes_containing_pitch_classes(
            pitch_classes,
            modes=modes,
        )
        if curated:
            selected, underdetermined, fallback = curate_modes_for_chord_quality(
                quality,
                exhaustive,
                modes,
                pitch_classes,
            )
            entry: Dict[str, Any] = {
                "pitch_classes": sorted(pitch_classes),
                "modes": selected,
                "modes_all": exhaustive,
                "underdetermined": underdetermined,
            }
            if fallback is not None:
                entry["fallback"] = fallback
            mapping[quality] = entry
        else:
            mapping[quality] = {
                "pitch_classes": sorted(pitch_classes),
                "modes": exhaustive,
            }
    return mapping


def chord_quality_mode_map_to_jsonl_records(
    mapping: Dict[str, Dict[str, Any]],
    *,
    include_all_modes: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records for a chord-quality-to-modes mapping."""
    for quality in sorted(mapping.keys(), key=lambda value: (len(value), value)):
        entry = mapping[quality]
        record = {
            "chord_quality": quality,
            "pitch_classes": entry["pitch_classes"],
            "modes": entry["modes"],
        }
        if entry.get("underdetermined"):
            record["underdetermined"] = True
        if entry.get("fallback"):
            record["fallback"] = entry["fallback"]
        if include_all_modes and "modes_all" in entry:
            record["modes_all"] = entry["modes_all"]
        yield record


def write_chord_quality_mode_map_jsonl(
    output_path: Path | str = DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
    qualities: Sequence[str] | None = None,
    chord_names_path: Path | str = DEFAULT_CHORD_NAMES_PATH,
    root_pc: int = 0,
    *,
    write_all_modes_path: Path | str | None = DEFAULT_CHORD_QUALITY_MODE_MAP_ALL_PATH,
) -> Path:
    """Build the curated chord-quality-to-modes map and write it to JSONL."""
    mapping = map_chord_qualities_to_modes(
        qualities=qualities,
        chord_names_path=chord_names_path,
        root_pc=root_pc,
        curated=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in chord_quality_mode_map_to_jsonl_records(mapping):
            handle.write(json.dumps(record) + "\n")

    if write_all_modes_path is not None:
        exhaustive_mapping = map_chord_qualities_to_modes(
            qualities=qualities,
            chord_names_path=chord_names_path,
            root_pc=root_pc,
            curated=False,
        )
        all_path = Path(write_all_modes_path)
        with all_path.open("w", encoding="utf-8") as handle:
            for record in chord_quality_mode_map_to_jsonl_records(
                exhaustive_mapping,
                include_all_modes=False,
            ):
                handle.write(json.dumps(record) + "\n")

    return output_path


def pitch_class_chord_map_to_jsonl_records(
    mapping: PitchClassChordMap,
    unresolved: List[PitchClassSet] | None = None,
) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records for mapped and unresolved pitch-class sets."""
    for pitch_class_set, chord_name in sorted(
        mapping.items(),
        key=lambda item: (len(item[0]), sorted(item[0])),
    ):
        yield {
            "pitch_classes": sorted(pitch_class_set),
            "chord_name": chord_name,
        }
    if unresolved is not None:
        for pitch_class_set in sorted(unresolved, key=lambda pcs: (len(pcs), sorted(pcs))):
            yield {
                "pitch_classes": sorted(pitch_class_set),
                "chord_name": None,
            }


def write_pitch_class_chord_map_jsonl(
    output_path: Path | str = DEFAULT_PITCH_CLASS_CHORD_MAP_PATH,
    min_pitches: int = 3,
    max_pitches: int = 7,
) -> Path:
    """Build the pitch-class chord map and write it to JSONL.

    Each line is one pitch-class set. Resolved sets include a chord symbol;
    unresolved sets have ``chord_name: null``.

    Args:
        output_path: Destination JSONL file.
        min_pitches: Minimum number of pitch classes per combination.
        max_pitches: Maximum number of pitch classes per combination.

    Returns:
        Path to the written JSONL file.
    """
    mapping, unresolved = map_octave_pitch_combinations_to_chords(
        min_pitches=min_pitches,
        max_pitches=max_pitches,
        return_unresolved=True,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in pitch_class_chord_map_to_jsonl_records(mapping, unresolved):
            handle.write(json.dumps(record) + "\n")
    return output_path


if __name__ == "__main__":
    pitch_class_path = write_pitch_class_chord_map_jsonl()
    print(f"Wrote pitch-class chord map to {pitch_class_path}")
    quality_mode_path = write_chord_quality_mode_map_jsonl()
    print(f"Wrote curated chord-quality mode map to {quality_mode_path}")
    print(f"Wrote exhaustive map to {DEFAULT_CHORD_QUALITY_MODE_MAP_ALL_PATH}")
