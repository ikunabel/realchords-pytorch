"""Calibration demo for a Disklavier, listed in the robot-melody song search.

``disklavier_sixteenths`` -- middle E (E4) as sixteenth notes back to back,
for four bars on a loop: the densest repetition ReaLJam can send, since the
model's chords and the played-back melodies sit on a sixteenth-note grid and
two notes on neighbouring frames leave no time between them in the schedule.
Every repeat therefore depends entirely on the re-strike handling (MIDI
Release before repeat settings) or, with the piano's MIDI IN Delay on, on the
instrument itself. Play it with "Play Ground Truth" ticked (no model) and the
Disklavier selected as Output; the Disklavier presets describe what each
configuration does to it.

The gap settings and velocities are read per note or chord, just before it
is played, so they can be changed while a demo plays. Send Early applies from the
next loop pass.

Records use the same schema as the data cache (see convert_*_to_cache.py), so
the existing melody/reference endpoints serve them unchanged. Times are in
beats, so the demo follows whatever tempo the session is set to.

``disklavier_e3-b3-halves`` -- E3 and B3 together (an E5 chord, given as
exact keys so no bass note is added) held for 8 frames (a half note) and
struck again immediately, eight times, on a loop, with no melody: the
long-hold case, where every key is shared with the next chord and has come
to rest on the backcheck before it must repeat.

``disklavier_fifths-sweep-halves`` / ``-quarters`` / ``-eighths`` /
``-sixteenths`` -- the same fifth moved up a semitone at a time from E3 + B3
to E4 + B4 (13 positions), each held for that note value and struck four
times back to back: repetition across
an octave of keys, since it was found to differ between keys.

Earlier demos (disklavier_send_early, disklavier_re-strike-gap,
disklavier_hold-time) were removed after the calibration they served; see
journal/DISKLAVIER.md.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from realchords.constants import ZERO_OCTAVE

DEMO_DATASET = "demo"
DEMO_SPLIT = "demo"
DEMO_ARTIST = "Disklavier calibration"


def _note(pitch: int, onset: float, offset: float) -> dict:
    octave, pitch_class = divmod(pitch - ZERO_OCTAVE, 12)
    return {"onset": onset, "offset": offset, "octave": octave, "pitch_class": pitch_class}


def _chord(root_pitch_class: int, intervals: List[int], onset: float, offset: float,
           pitches: Optional[List[int]] = None) -> dict:
    chord = {
        "onset": onset,
        "offset": offset,
        "root_pitch_class": root_pitch_class,
        "root_position_intervals": intervals,
        "inversion": 0,
    }
    if pitches is not None:
        # Exact keys to play, bypassing voicing (which would add a bass note)
        chord["pitches"] = pitches
    return chord


def _record(song_id: str, num_beats: int, melody: List[dict], harmony: List[dict]) -> dict:
    return {
        DEMO_DATASET: {"id": song_id, "title": song_id, "artist": DEMO_ARTIST},
        "annotations": {
            "num_beats": num_beats,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "melody": melody,
            "harmony": harmony,
        },
    }


def sixteenths_demo() -> dict:
    """E4 sixteenths, legato, four bars."""
    melody = [_note(64, i / 4, (i + 1) / 4) for i in range(16 * 4)]
    return _record("disklavier_sixteenths", 16, melody, [])


def e3_b3_halves_demo() -> dict:
    """E3 + B3 held for 8 frames (a half note), struck again at once, x8."""
    harmony = [_chord(4, [7], 2 * i, 2 * i + 2, pitches=[52, 59]) for i in range(8)]
    return _record("disklavier_e3-b3-halves", 16, [], harmony)


def fifths_sweep_demo(name: str, beats: float) -> dict:
    """The E3 + B3 fifth moved up a semitone at a time to E4 + B4, each
    position held for `beats` and struck four times back to back."""
    harmony = []
    for step in range(13):                       # E5, F5, F#5, ... E5 an octave up
        for repeat in range(4):
            onset = beats * (4 * step + repeat)
            harmony.append(_chord((4 + step) % 12, [7], onset, onset + beats,
                                  pitches=[52 + step, 59 + step]))
    return _record(name, round(beats * 4 * 13), [], harmony)


def demo_songs() -> List[Tuple[dict, dict]]:
    """(catalogue entry, record) for every demo, ready for build_catalogue."""
    out = []
    for record in (sixteenths_demo(), e3_b3_halves_demo(),
                   fifths_sweep_demo("disklavier_fifths-sweep-halves", 2),
                   fifths_sweep_demo("disklavier_fifths-sweep-quarters", 1),
                   fifths_sweep_demo("disklavier_fifths-sweep-eighths", 0.5),
                   fifths_sweep_demo("disklavier_fifths-sweep-sixteenths", 0.25)):
        meta = record[DEMO_DATASET]
        entry = {
            "dataset": DEMO_DATASET,
            "split": DEMO_SPLIT,
            "id": meta["id"],
            "title": meta["title"],
            "artist": DEMO_ARTIST,
        }
        out.append((entry, record))
    return out
