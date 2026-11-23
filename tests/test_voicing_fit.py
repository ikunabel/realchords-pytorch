"""VoicingFitIndex (realchords/realjam/voicing_fit.py) against a brute-force
version of the completion's rule (play_from_midi_input.js, completionVoicing).

Run: python -m pytest tests/test_voicing_fit.py
"""
import random

from realchords.realjam.voicing_fit import VoicingFitIndex, register_ok


def brute_fit(voicings, keys, held, ceiling):
    total = sum(v["count"] for v in voicings if v["count"] > 0)
    if not total:
        return 0.0
    good = 0
    for v in voicings:
        if v["count"] <= 0 or not set(keys) <= set(v["pitches"]) or not register_ok(v["pitches"], keys):
            continue
        added = [p for p in v["pitches"] if p not in keys]
        if all(p not in held and (ceiling is None or p <= ceiling) for p in added):
            good += v["count"]
    return good / total


LOOKUP = {
    "C": [{"pitches": [48, 55, 64], "count": 100},      # C3 G3 E4
          {"pitches": [52, 60, 67], "count": 50},       # E3 C4 G4
          {"pitches": [60, 64, 67], "count": 400},      # close triad: span < octave, never playable
          {"pitches": [36, 43, 52, 60], "count": 30},   # C2 G2 E3 C4
          {"pitches": [36, 40, 43, 48], "count": 999}], # muddy below C3, never playable
    "C/E": [{"pitches": [40, 52, 60, 67], "count": 20},  # E2 E3 C4 G4
            {"pitches": [52, 60, 67], "count": 80}],
    "Am": [{"pitches": [45, 52, 60, 64], "count": 60}],
    "G/B": [],  # a slash chord without voicings: falls back to G
    "G": [{"pitches": [43, 50, 59, 67], "count": 10}],
    "F": [],
}
SYMBOLS = ["C", "C/E", "Am", "G/B", "G", "F", "Dm"]  # Dm: not in the lookup at all


def test_examples():
    index = VoicingFitIndex(SYMBOLS, LOOKUP)
    # E3 C4 held, melody high: C has 50 + 30 of 1579, C/E 20 + 80 of 100
    fit = dict(zip(SYMBOLS, index.fit([52, 60], [52, 60], None)))
    assert abs(fit["C"] - 80 / 1579) < 1e-6
    assert fit["C/E"] == 1.0
    assert abs(fit["Am"] - 1.0) < 1e-6
    assert fit["F"] == 0.0 and fit["Dm"] == 0.0
    # a ceiling at B3 rules out C/E's G4 voicings and C's G4 one
    fit = dict(zip(SYMBOLS, index.fit([52, 60], [52, 60], 59)))
    assert fit["C/E"] == 0.0 and abs(fit["C"] - 30 / 1579) < 1e-6
    # an added note may not lie on another held key (G4 held as melody)
    fit = dict(zip(SYMBOLS, index.fit([52, 60], [52, 60, 67], None)))
    assert abs(fit["C"] - 30 / 1579) < 1e-6
    # slash fallback: G/B uses G's voicings
    fit = dict(zip(SYMBOLS, index.fit([43], [43], None)))
    assert fit["G/B"] == fit["G"] == 1.0


def test_random_against_brute_force():
    rng = random.Random(0)
    lookup = {}
    symbols = [f"X{i}" for i in range(40)]
    for s in symbols:
        lookup[s] = [{"pitches": sorted(rng.sample(range(30, 90), rng.randint(3, 6))),
                      "count": rng.choice([0, 1, 5, 50, 300])} for _ in range(rng.randint(0, 25))]
    index = VoicingFitIndex(symbols, lookup)
    for _ in range(200):
        keys = sorted(rng.sample(range(36, 80), rng.randint(1, 3)))
        held = sorted(set(keys) | set(rng.sample(range(36, 90), rng.randint(0, 2))))
        ceiling = rng.choice([None, 60, 67, 72])
        got = index.fit(keys, held, ceiling)
        for s, g in zip(symbols, got):
            assert abs(g - brute_fit(lookup[s], keys, held, ceiling)) < 1e-6, (s, keys, held, ceiling)


def test_register_rules_judge_only_added_notes():
    # 2026-11-23: a performer playing close low intervals got no completion
    muddy = [36, 40, 43, 48]                    # C2 E2 G2 C3
    assert not register_ok(muddy)               # the agent may not add it ...
    assert register_ok(muddy, [36, 40, 43, 48])  # ... but the performer may play it
    assert not register_ok(muddy, [36, 40, 43])  # the agent's C3 is too close above G2
    assert register_ok([60, 64, 67], [60, 64, 67])  # a close triad the performer plays, nothing added
    assert not register_ok([60, 64, 67], [60, 64])  # the agent would complete it to a close triad
    assert register_ok([33, 45, 52, 60], [33])  # below C2 only if the performer plays it
    assert not register_ok([33, 45, 52, 60], [45])


def test_index_lets_performer_play_low_close_intervals():
    idx = VoicingFitIndex(SYMBOLS, LOOKUP)
    fit = idx.fit([36, 40, 43, 48], [36, 40, 43, 48], None)
    assert fit[SYMBOLS.index("C")] > 0
    assert abs(fit[SYMBOLS.index("C")] - brute_fit(LOOKUP["C"], [36, 40, 43, 48], [36, 40, 43, 48], None)) < 1e-6
