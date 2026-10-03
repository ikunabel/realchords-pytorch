"""Calibration demos for a Disklavier, listed in the robot-melody song search.

Each demo isolates one of the MIDI output settings so that changing it is
plainly audible. Play them with "Play Ground Truth" ticked (no model), the
Disklavier selected as Output, and the metronome on.

* ``disklavier_send_early`` -- short notes exactly on every beat, alternating
  pitches so the re-strike gap never applies. The metronome still clicks on
  the laptop, which makes it the reference: with the right "MIDI Send Early"
  the piano lands with the click, too little and it drags behind, too much
  and it rushes ahead.
* ``disklavier_re-strike-gap`` -- a C major chord repeated legato in
  quarters, eighths, then sixteenths. A "MIDI Re-strike Gap" in the
  50-100 ms range makes repetitions fail (keys pressed again mid-return);
  far too large and the short notes don't reach the bottom.
* ``disklavier_hold-time`` -- middle C alone, repeated legato in shrinking
  note lengths, so at a fixed gap the time each key is held before its
  release steps down bar by bar.

Records use the same schema as the data cache (see convert_*_to_cache.py), so
the existing melody/reference endpoints serve them unchanged. Times are in
beats, so the demos follow whatever tempo the session is set to. Slider
changes take effect from the next repetition of the loop, since a whole loop
is scheduled when it starts.
"""

from __future__ import annotations

from typing import List, Tuple

from realchords.constants import ZERO_OCTAVE

DEMO_DATASET = "demo"
DEMO_SPLIT = "demo"
DEMO_ARTIST = "Disklavier calibration"


def _note(pitch: int, onset: float, offset: float) -> dict:
    octave, pitch_class = divmod(pitch - ZERO_OCTAVE, 12)
    return {"onset": onset, "offset": offset, "octave": octave, "pitch_class": pitch_class}


def _chord(root_pitch_class: int, intervals: List[int], onset: float, offset: float) -> dict:
    return {
        "onset": onset,
        "offset": offset,
        "root_pitch_class": root_pitch_class,
        "root_position_intervals": intervals,
        "inversion": 0,
    }


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


def send_early_demo() -> dict:
    """Two bars of staccato notes on every beat.

    Bar lines on C6, other beats alternating G4 / C5, so no pitch repeats on
    consecutive beats and the re-strike gap stays out of the picture. Notes
    last half a beat: long enough to hear clearly, short enough that each
    onset is a distinct event against the click.
    """
    melody = []
    for beat in range(8):
        pitch = 84 if beat % 4 == 0 else (67 if beat % 2 else 72)
        melody.append(_note(pitch, beat, beat + 0.5))
    return _record("disklavier_send_early", 8, melody, [])


def restrike_gap_demo() -> dict:
    """A C major chord repeated legato: a bar each of quarters, eighths and
    sixteenths.

    Every chord starts exactly where the previous one ends, so each
    repetition is a release and a press of the same keys. Measured on the
    ENSPIRE U1 at 80 BPM, velocity 40: a gap of 0 (release and press sent
    together) re-strikes cleanly; 50-100 ms fails, the press arriving while
    the key is still on its way up; 150 ms works again; 300 ms leaves the
    eighths too short a hold to reach the bottom.
    """
    harmony = []
    for bar, per_beat in enumerate((1, 2, 4)):
        step = 1 / per_beat
        for i in range(4 * per_beat):
            onset = 4 * bar + i * step
            harmony.append(_chord(0, [4, 3], onset, onset + step))
    return _record("disklavier_re-strike-gap", 12, [], harmony)


def hold_time_demo() -> dict:
    """Middle C repeated legato, one note length per bar, shortest last.

    Each bar plays three beats of repeats -- quarters, dotted eighths,
    eighths, sixteenths -- then rests a beat so the action settles between
    groups. At a fixed re-strike gap, each key is held for (note length -
    gap), so the bars step the hold time down while the release time stays
    the same: for testing whether a repeat depends on how long the key was
    held before it, not just on the gap (e.g. at 80 BPM and a 50 ms gap the
    holds are 700, 512, 325 and 137 ms). Change the BPM to move the steps.
    """
    melody = []
    for bar, length in enumerate((1, 0.75, 0.5, 0.25)):
        for i in range(round(3 / length)):
            onset = 4 * bar + i * length
            melody.append(_note(60, onset, onset + length))
    return _record("disklavier_hold-time", 16, melody, [])


def demo_songs() -> List[Tuple[dict, dict]]:
    """(catalogue entry, record) for every demo, ready for build_catalogue."""
    out = []
    for record in (send_early_demo(), restrike_gap_demo(), hold_time_demo()):
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
