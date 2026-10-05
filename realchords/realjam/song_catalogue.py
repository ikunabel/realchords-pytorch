"""Song catalogue for ReaLJam's "robot melody" playback feature.

Lets the UI offer a fuzzy-searchable song picker and play back a real
ground-truth melody through the live generation pipeline, so
chord-generation/voicing settings can be A/B'd hands-free instead of
needing live MIDI input for every test.

Only includes datasets where the cache's "melody" track is a genuine melody
line and usable title/artist metadata exists:

- Included: Hooktheory, POP909, Nottingham, Wikifonia, Chord Melody Dataset,
  JAZZMUS.
- Excluded: FiloBass and WJD (their "melody" is actually a walking bass
  line, not a melody -- see journal/DATASET_SUMMARY.md). EMOPIA+ (no usable
  title/artist metadata -- only an emotion-quadrant label and a YouTube ID).

Reads only the plain (non-augmented) train/valid/test.jsonl splits -- the
augmented files are ±6-semitone transpositions of the same songs and would
otherwise show up as near-duplicate catalogue entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from urllib.parse import unquote

from realchords.constants import ZERO_OCTAVE
from realchords.realjam.demo_songs import demo_songs
from realchords.utils.data_utils import to_chord_name


class MelodyNote(TypedDict):
    pitch: int
    onset: float   # quarter-note (beat) units, matching FRAME_PER_BEAT=4 frames/beat
    offset: float


class SongEntry(TypedDict):
    dataset: str
    split: str
    id: str
    title: str
    artist: str


# (cache_dir_name, dataset_key_in_json, artist_field_or_None, title_field)
# artist_field=None means this dataset has no usable per-song artist/composer
# field -- all its songs are grouped under one synthetic artist bucket below.
_DATASET_CONFIGS: List[Tuple[str, str, Optional[str], str]] = [
    ("hooktheory", "hooktheory", "artist", "song"),
    ("pop909", "pop909", "artist", "title"),
    ("nottingham", "nottingham", "source", "title"),
    ("wikifonia", "wikifonia", "composer", "title"),
    ("chord_melody_dataset", "chord_melody_dataset", None, "title"),
    ("jazzmus", "jazzmus", "composer", "title"),
]

_SYNTHETIC_ARTIST = {
    "chord_melody_dataset": "Chord Melody Dataset (Jazz Standards)",
}


def _prettify(text: str) -> str:
    """Hooktheory stores artist/song as lowercase-hyphenated slugs
    (e.g. "adam-lambert" / "whataya-want-from-me") -- clean those up for
    display. A no-op for datasets that already store human-readable text.

    Being URL slugs, some also carry percent escapes ("%28break%29 In Case
    Of"), which would otherwise be both displayed and fuzzy-searched as
    literal text.
    """
    if not text:
        return text
    if "%" in text:
        text = unquote(text)
    text = text.strip()
    if "-" in text and text == text.lower():
        # Skip empty parts, so leading/trailing/doubled hyphens in the slug
        # don't come back as stray spaces
        return " ".join(w.capitalize() for w in text.split("-") if w)
    return text


def _load_split(cache_root: Path, dataset_dir: str, split: str) -> List[dict]:
    path = cache_root / dataset_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_catalogue(
    cache_root: str | Path = "data/cache",
) -> Tuple[List[SongEntry], Dict[Tuple[str, str, str], dict]]:
    """Scan the included datasets' cache files and build the catalogue.

    Returns
    -------
    catalogue:
        Flat ``[SongEntry, ...]`` sorted by title, for the UI's fuzzy song
        search, which matches against title and artist alike.
    song_index:
        ``{(dataset, split, id): full_cache_record}`` -- lets the server look
        up a specific song's full annotations (melody, harmony, ...) in O(1)
        once a user picks it, without re-scanning any files.
    """
    cache_root = Path(cache_root)
    catalogue: List[SongEntry] = []
    song_index: Dict[Tuple[str, str, str], dict] = {}

    for dataset_dir, dataset_key, artist_field, title_field in _DATASET_CONFIGS:
        for split in ("train", "valid", "test"):
            for record in _load_split(cache_root, dataset_dir, split):
                meta = record.get(dataset_key, {})
                song_id = meta.get("id")
                if song_id is None:
                    continue

                title = _prettify(str(meta.get(title_field) or song_id))
                if artist_field is not None and meta.get(artist_field):
                    artist = _prettify(str(meta[artist_field]))
                else:
                    artist = _SYNTHETIC_ARTIST.get(dataset_dir, dataset_dir)

                catalogue.append({
                    "dataset": dataset_dir,
                    "split": split,
                    "id": song_id,
                    "title": title,
                    "artist": artist,
                })
                song_index[(dataset_dir, split, song_id)] = record

    # Synthetic calibration songs for driving a Disklavier (see demo_songs.py)
    for entry, record in demo_songs():
        catalogue.append(entry)
        song_index[(entry["dataset"], entry["split"], entry["id"])] = record

    catalogue.sort(key=lambda e: (e["title"].lower(), e["artist"].lower()))
    return catalogue, song_index


class ReferenceChord(TypedDict):
    symbol: str
    onset: float   # quarter-note (beat) units, as in the cache
    offset: float


def extract_reference_chords(record: dict) -> List[ReferenceChord]:
    """Convert a cache record's harmony annotation to chord symbols.

    The cache stores harmony as root pitch class + intervals rather than a
    symbol (see convert_hooktheory_to_cache.py), so this runs the same
    `to_chord_name` conversion the tokenizer uses -- meaning the symbols here
    are exactly the ones the model was trained to predict for this song,
    which is the point of playing them back as a reference.

    Chords whose spelling `note_seq` can't render are skipped rather than
    failing the whole song; they'd have no pitches to play anyway.
    """
    chords: List[ReferenceChord] = []
    for chord in record.get("annotations", {}).get("harmony", []):
        try:
            symbol = to_chord_name(
                chord["root_pitch_class"], chord["root_position_intervals"]
            )
        except Exception:
            continue
        entry = {"symbol": symbol, "onset": chord["onset"], "offset": chord["offset"]}
        if "pitches" in chord:
            # Exact keys, for synthetic demo songs (demo_songs.py); real
            # songs never carry this and are voiced from the symbol
            entry["pitches"] = list(chord["pitches"])
        chords.append(entry)
    return chords


def extract_melody_notes(record: dict) -> List[MelodyNote]:
    """Convert a cache record's melody annotation to raw-MIDI-pitch notes.

    ``record["annotations"]["melody"]`` stores each note as
    ``{onset, offset, pitch_class, octave}`` in Hooktheory quarter-note units
    (octave relative to ZERO_OCTAVE, the same convention used throughout the
    conversion scripts, e.g. convert_pop909_to_cache.py's
    extract_melody_from_midi). This just inverts that back to a single raw
    MIDI pitch per note, for the frontend to schedule directly.
    """
    notes = record.get("annotations", {}).get("melody", [])
    return [
        {
            "pitch": ZERO_OCTAVE + note["octave"] * 12 + note["pitch_class"],
            "onset": note["onset"],
            "offset": note["offset"],
        }
        for note in notes
    ]
