"""Utilities for exploring Hooktheory examples as MIDI.

Expects songs from data/cache/hooktheory (MELODY + HARMONY, no TEMPO_CHANGES).
See dataset_experiments/scripts/hooktheory_cache_filter.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi
from scipy.interpolate import interp1d


def hooktheory_to_midi(
    example: dict[str, Any],
    chord_octave: int = 4,
    melody_octave: int = 5,
    output_dir: Path | str | None = None,
) -> Path:
    """Convert a Hooktheory song dict to MIDI and write to disk.

    Args:
        example: One Hooktheory song from cache or Hooktheory.json.
        chord_octave: Octave for harmony pitches.
        melody_octave: Base octave for melody pitches.
        output_dir: Directory for the MIDI file. Defaults to ./midis.

    Returns:
        Path to the written MIDI file.
    """
    beat_to_time_fn = interp1d(
        example["alignment"]["refined"]["beats"],
        example["alignment"]["refined"]["times"],
        kind="linear",
        fill_value="extrapolate",
    )

    midi = pretty_midi.PrettyMIDI()

    harmony = pretty_midi.Instrument(program=0)
    midi.instruments.append(harmony)
    for c in example["annotations"]["harmony"]:
        root_position_pitches = [c["root_pitch_class"]]
        for interval in c["root_position_intervals"]:
            root_position_pitches.append(root_position_pitches[-1] + interval)
        for pitch in root_position_pitches:
            harmony.notes.append(
                pretty_midi.Note(
                    67,
                    pitch + chord_octave * 12,
                    beat_to_time_fn(c["onset"]),
                    beat_to_time_fn(c["offset"]),
                )
            )

    melody = pretty_midi.Instrument(program=0)
    midi.instruments.append(melody)
    for n in example["annotations"]["melody"]:
        melody.notes.append(
            pretty_midi.Note(
                100,
                n["pitch_class"] + (melody_octave + n["octave"]) * 12,
                beat_to_time_fn(n["onset"]),
                beat_to_time_fn(n["offset"]),
            )
        )

    if output_dir is None:
        output_dir = Path("midis")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        output_dir
        / f"annotations_{example['hooktheory']['song']}_{example['hooktheory']['id']}.midi"
    )
    midi.write(str(filename))
    return filename
