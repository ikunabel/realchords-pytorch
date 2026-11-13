"""Render Chord Melody Dataset songs from the cache, as-is and with the melody dropped an octave.

Why: CMD is engraved entirely for nylon guitar, which is written an octave above sounding pitch.
216 of its 378 training files declare `<transpose><octave-change>-1</octave-change>`, which
`score.toSoundingPitch()` applies; the other 162 omit the tag, so their melodies stay at written
pitch -- a median of MIDI 76 against MIDI 62 for the tagged ones (other corpora sit near 69). The
converter is faithful to the metadata; the metadata is absent in 43% of the corpus.

This writes, for each selected song, the melody+chords exactly as the cache holds them (what every
model has been trained on) and the same with the melody transposed down 12 semitones (the proposed
fix), so the diagnosis can be checked by ear before the converter is changed. Control songs that
*do* declare the transposition are rendered once, as the target register.

Chords are realised plainly (root position stack from the annotation, low register) -- this is a
listening aid, not the voicing model.

Usage:
    python scripts/data/render_cmd_octave_ab.py --out data/cmd_octave_ab
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from realchords.constants import ZERO_OCTAVE

# (song folder, kind): "wrong" = no <octave-change> tag, "control" = tag present and applied
SELECTION = [
    ("nothin_gonna", "wrong"),
    ("livin_on_a_prayer", "wrong"),
    ("river_of_dreams", "wrong"),
    ("please_dont_talk_about_me", "wrong"),
    ("blues_in_the_night", "control"),
    ("good_morning_heartache", "control"),
]


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_") or "song"


def chord_pitches(ann: dict, base: int = 48) -> list:
    """Root-position stack from a harmony annotation, then `inversion` notes lifted an octave."""
    pitches = [base + ann["root_pitch_class"]]
    for step in ann.get("root_position_intervals", []):
        pitches.append(pitches[-1] + step)
    for _ in range(ann.get("inversion", 0) or 0):
        pitches = pitches[1:] + [pitches[0] + 12]
    return pitches


def write_midi(path: Path, melody: list, harmony: list, shift: int, bpm: float) -> None:
    import pretty_midi

    spb = 60.0 / bpm
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    lead = pretty_midi.Instrument(program=24, name="melody")      # nylon guitar
    comp = pretty_midi.Instrument(program=25, name="chords")
    for n in melody:
        pitch = ZERO_OCTAVE + n["octave"] * 12 + n["pitch_class"] + shift
        if 0 <= pitch < 128:
            lead.notes.append(pretty_midi.Note(velocity=96, pitch=int(pitch),
                                               start=n["onset"] * spb, end=max(n["offset"], n["onset"] + 0.25) * spb))
    for h in harmony:
        for pitch in chord_pitches(h):
            if 0 <= pitch < 128:
                comp.notes.append(pretty_midi.Note(velocity=64, pitch=int(pitch),
                                                   start=h["onset"] * spb, end=max(h["offset"], h["onset"] + 0.25) * spb))
    pm.instruments.extend([lead, comp])
    pm.write(str(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/cache/chord_melody_dataset__rebuild")
    ap.add_argument("--out", default="data/cmd_octave_ab")
    ap.add_argument("--bpm", type=float, default=90.0)
    args = ap.parse_args()

    wanted = dict(SELECTION)
    records = {}
    for split in ("train", "valid", "test"):
        path = Path(args.cache) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            d = json.loads(line)
            stem = d["chord_melody_dataset"]["id"].rsplit("__", 1)[0]
            if stem in wanted:
                records[stem] = d

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, kind in SELECTION:
        d = records.get(stem)
        if d is None:
            print(f"  !! {stem} not found in {args.cache}")
            continue
        title = slug(d["chord_melody_dataset"]["title"])
        melody, harmony = d["annotations"]["melody"], d["annotations"]["harmony"]
        med = sorted(ZERO_OCTAVE + n["octave"] * 12 + n["pitch_class"] for n in melody)[len(melody) // 2]
        if kind == "wrong":
            write_midi(out_dir / f"{title}__1_as_cached_MIDI{med}.mid", melody, harmony, 0, args.bpm)
            write_midi(out_dir / f"{title}__2_octave_fixed_MIDI{med - 12}.mid", melody, harmony, -12, args.bpm)
            print(f"  {title:32s} untagged: as-cached median MIDI {med}, fixed {med - 12}")
        else:
            write_midi(out_dir / f"{title}__0_control_correct_MIDI{med}.mid", melody, harmony, 0, args.bpm)
            print(f"  {title:32s} control (tag present): median MIDI {med}")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
