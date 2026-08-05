"""Shared MIDI-rendering helpers for custom-eval tensors.

Used by both realchords/utils/custom_evaluation.py (auto-renders a small
sample every run) and scripts/eval/custom_eval/export_paired_midis.py
(re-exports a saved run's tensors, any sample count, without re-running
generation or metrics) -- one rendering implementation, not two.

Each selected song is written as one MIDI file *per source* (GT, and each
model), not concatenated into a single multi-section file -- this makes it
trivial to A/B a single source in isolation, and to diff the same song
across sources by filename (identical `{out_idx:04d}_seq{i:04d}_{song}.mid`
name in each source's own subfolder).
"""

import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import note_seq.chord_symbols_lib as _chord_lib
import pretty_midi
import torch
from tqdm import tqdm

from realchords.dataset.hooktheory_tokenizer import to_midi_pitch

CHORD_OCTAVE = 4   # default chord-tone octave (MIDI 48-59)
BASS_OCTAVE = 3    # bass note one octave below chord root voicing
MELODY_VELOCITY = 90
CHORD_VELOCITY = 64


def _slugify(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "model"


def _dedup(pitches: List[int]) -> List[int]:
    seen: set = set()
    out = []
    for p in pitches:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _naive_pitches(
    chord_name: str,
    *,
    include_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
) -> List[int]:
    chord_pcs = _chord_lib.chord_symbol_pitches(chord_name)
    pitches = [p % 12 + chord_octave * 12 for p in chord_pcs]
    if include_bass:
        bass_pc = _chord_lib.chord_symbol_bass(chord_name) % 12
        pitches.append(bass_pc + BASS_OCTAVE * 12)
    return _dedup(pitches)


def _lenient_decode_chord_frames(chord_frames, tokenizer):
    """Decode chord frames leniently: only CHORD_ON tokens start a chord.

    Unlike ``tokenizer.decode_chord_frames``, this never raises for
    hold-only transitions or crops that start mid-chord.  Hold tokens
    that don't belong to the current ongoing chord are simply skipped,
    so the output starts cleanly from the first genuine chord onset.
    """
    fpb = tokenizer.frame_per_beat
    chords = []
    ongoing = None
    for i, tok in enumerate(chord_frames):
        name = tokenizer.id_to_name.get(int(tok), "")
        if "CHORD_ON_" in name:
            if ongoing is not None:
                ongoing["offset"] = i / fpb
                chords.append(ongoing)
            ongoing = {
                "chord_name": name[len("CHORD_ON_"):],
                "onset": i / fpb,
            }
        elif name == "SILENCE" and ongoing is not None:
            ongoing["offset"] = i / fpb
            chords.append(ongoing)
            ongoing = None
        # CHORD_HOLD tokens for a different chord, PAD, BOS, EOS -> silently skip
    if ongoing is not None:
        ongoing["offset"] = len(chord_frames) / fpb
        chords.append(ongoing)
    return chords


def _decode_chord_anns(chord_frames, tokenizer, *, strict: bool):
    """Decode chord frames, using strict tokenizer decode when possible."""
    if strict:
        try:
            return tokenizer.decode_chord_frames(chord_frames)
        except ValueError:
            pass
    return _lenient_decode_chord_frames(chord_frames, tokenizer)


def _append_section(
    seq: torch.Tensor,          # 1-D, BOS already stripped
    tokenizer,
    spb: float,
    t0: float,
    melody_instr: pretty_midi.Instrument,
    chord_instr: pretty_midi.Instrument,
    midi_obj: pretty_midi.PrettyMIDI,
    *,
    strict_chords: bool = False,
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
    melody_octave: int = 0,
) -> None:
    """Render one source's melody + chords into instruments, starting at t0."""
    chord_frames = seq[0::2].numpy()
    melody_frames = seq[1::2].numpy()

    chord_anns = _decode_chord_anns(chord_frames, tokenizer, strict=strict_chords)
    try:
        melody_anns = tokenizer.decode_melody_frames(melody_frames)
    except ValueError:
        melody_anns = []

    for note in melody_anns:
        melody_instr.notes.append(pretty_midi.Note(
            velocity=MELODY_VELOCITY,
            pitch=to_midi_pitch(note["octave"] + melody_octave, note["pitch_class"]),
            start=note["onset"] * spb + t0,
            end=note["offset"] * spb + t0,
        ))

    for chord in chord_anns:
        for p in _naive_pitches(
            chord["chord_name"],
            include_bass=include_chord_bass,
            chord_octave=chord_octave,
        ):
            chord_instr.notes.append(pretty_midi.Note(
                velocity=CHORD_VELOCITY,
                pitch=max(0, min(127, p)),
                start=chord["onset"] * spb + t0,
                end=chord["offset"] * spb + t0,
            ))
        midi_obj.lyrics.append(pretty_midi.Lyric(
            text=chord["chord_name"],
            time=chord["onset"] * spb + t0,
        ))


def select_midi_indices(
    num_sequences: int,
    midi_samples: Optional[int],
    *,
    default_all: bool,
    seed: int,
) -> List[int]:
    """Pick which sequence indices to export as MIDI.

    midi_samples: None -> default_all ? every sequence : 10. Negative -> every
    sequence. Non-negative -> exactly that many (clamped to num_sequences).
    """
    if num_sequences <= 0:
        return []
    if midi_samples is None:
        target = num_sequences if default_all else 10
    elif midi_samples < 0:
        target = num_sequences
    else:
        target = midi_samples
    target = min(target, num_sequences)
    if target >= num_sequences:
        return list(range(num_sequences))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_sequences), target))


def resolve_include_chord_bass(
    include_chord_bass: bool, no_chord_bass: bool, dataset_name: Optional[str]
) -> bool:
    if include_chord_bass:
        return True
    if no_chord_bass:
        return False
    return dataset_name != "wjd"


def write_source_midis(
    tensor: torch.Tensor,
    tokenizer,
    midi_dir: Path,
    source_name: str,
    metadata: List[Dict],
    indices: List[int],
    *,
    bpm: int = 120,
    strict_chords: bool = False,
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
    melody_octave: int = 0,
    quiet: bool = False,
) -> None:
    """Write one MIDI file per song for a single source (GT or one model),
    to midi_dir/source_name/{out_idx:04d}_seq{i:04d}_{song}.mid.
    """
    out_dir = midi_dir / (source_name if source_name == "gt" else _slugify(source_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    spb = 60.0 / bpm

    iterator = indices if quiet else tqdm(indices, desc=f"Writing {source_name} MIDIs")
    for out_idx, i in enumerate(iterator):
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        melody_instr = pretty_midi.Instrument(program=0, name="Melody")
        chord_instr = pretty_midi.Instrument(program=0, name="Chords")

        seq = tensor[i, 1:]
        _append_section(
            seq, tokenizer, spb, 0.0, melody_instr, chord_instr, midi_obj,
            strict_chords=strict_chords,
            include_chord_bass=include_chord_bass,
            chord_octave=chord_octave,
            melody_octave=melody_octave,
        )
        midi_obj.instruments.extend([melody_instr, chord_instr])

        song_url = metadata[i].get("song_url", "unknown") if i < len(metadata) else "unknown"
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", song_url)[-60:]
        midi_obj.write(str(out_dir / f"{out_idx:04d}_seq{i:04d}_{safe}.mid"))

    if not quiet:
        print(f"  Wrote {len(indices)} MIDI files to {out_dir}")


def write_all_source_midis(
    gt_tensor: torch.Tensor,
    model_tensors: Dict[str, torch.Tensor],
    ordered_labels: List[str],
    metadata: List[Dict],
    gt_tokenizer,
    model_tokenizer,
    midi_dir: Path,
    indices: List[int],
    *,
    bpm: int = 120,
    include_chord_bass: bool = True,
    chord_octave: int = CHORD_OCTAVE,
    melody_octave: int = 0,
    quiet: bool = False,
) -> None:
    """Write GT + every model's MIDI, all using the *same* song indices (so
    the same songs are directly comparable file-for-file across source
    subfolders)."""
    write_source_midis(
        gt_tensor, gt_tokenizer, midi_dir, "gt", metadata, indices,
        bpm=bpm, strict_chords=True, include_chord_bass=include_chord_bass,
        chord_octave=chord_octave, melody_octave=melody_octave, quiet=quiet,
    )
    for label in ordered_labels:
        write_source_midis(
            model_tensors[label], model_tokenizer, midi_dir, label, metadata, indices,
            bpm=bpm, strict_chords=False, include_chord_bass=include_chord_bass,
            chord_octave=chord_octave, melody_octave=melody_octave, quiet=quiet,
        )
