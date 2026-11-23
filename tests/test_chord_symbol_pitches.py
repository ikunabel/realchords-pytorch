"""The project's chord_symbol_pitches: note_seq's reader with added 7ths read as written (note_seq bug A)."""
import json
import re
from pathlib import Path

import pytest
from note_seq import chord_symbols_lib

from realchords.utils.data_utils import chord_symbol_pitches, to_chord_name

VOCAB = Path(__file__).resolve().parents[1] / "checkpoints" / "chord_names_augmented.json"


@pytest.mark.parametrize("name,expected", [
    ("Asus(add7)", {9, 2, 4, 8}),      # A D E G#: Amaj7sus4
    ("Gsus2(addb7)", {7, 9, 2, 5}),    # G A D F: G7sus2
    ("C+(add7)", {0, 4, 8, 11}),       # C E G# B
    ("Asus7", {9, 2, 4, 7}),           # no added 7th: unchanged
    ("Dm7b5", {2, 5, 8, 0}),
])
def test_examples(name, expected):
    assert set(chord_symbol_pitches(name)) == expected


def test_round_trip_of_written_add7():
    # Amaj7sus4 as Hookpad's I7sus4 in A major: root 9, stacked intervals [5, 2, 4]
    name = to_chord_name(9, [5, 2, 4], 0)
    assert name == "Asus(add7)"
    assert set(chord_symbol_pitches(name)) == {9, 2, 4, 8}


@pytest.mark.skipif(not VOCAB.exists(), reason="old vocabulary not available")
def test_same_as_note_seq_without_added_7th():
    names = json.loads(VOCAB.read_text())
    names = names if isinstance(names, list) else list(names)
    for name in names:
        if re.search(r"add[#b]?7", name):
            continue
        try:
            expected = chord_symbols_lib.chord_symbol_pitches(name)
        except chord_symbols_lib.ChordSymbolError:
            with pytest.raises(chord_symbols_lib.ChordSymbolError):
                chord_symbol_pitches(name)
            continue
        assert chord_symbol_pitches(name) == expected, name


@pytest.mark.parametrize("name,expected", [
    ("C7", (0, {1: 0, 3: 0, 5: 0, 7: -1}, 0)),
    ("Cm7b5", (0, {1: 0, 3: -1, 5: -1, 7: -1}, 0)),
    ("C9/E", (0, {1: 0, 3: 0, 5: 0, 7: -1, 9: 0}, 4)),
    ("F6/9/D", (5, {1: 0, 3: 0, 5: 0, 6: 0, 9: 0}, 2)),
    ("Asus(add7)", (9, {1: 0, 4: 0, 5: 0, 7: 0}, 9)),       # note_seq: 7: -1
    ("Csus2(addb7)", (0, {1: 0, 2: 0, 5: 0, 7: -1}, 0)),    # note_seq: 7: -2
])
def test_degrees(name, expected):
    from realchords.utils.data_utils import chord_symbol_degrees
    assert chord_symbol_degrees(name) == expected


@pytest.mark.skipif(not VOCAB.exists(), reason="old vocabulary not available")
def test_degrees_match_note_seq_without_added_7th():
    from realchords.utils.data_utils import chord_symbol_degrees
    names = json.loads(VOCAB.read_text())
    names = names if isinstance(names, list) else list(names)
    for name in names:
        if re.search(r"add[#b]?7", name):
            continue
        root, degrees, bass = chord_symbols_lib._parse_chord_symbol(name)
        assert chord_symbol_degrees(name) == (
            chord_symbols_lib._pitch_class_to_midi(*root), degrees,
            chord_symbols_lib._pitch_class_to_midi(*bass)), name
