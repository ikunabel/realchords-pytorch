#!/usr/bin/env python3
"""Write MIDI files from a `custom_evaluation.py` run, without re-running
generation or metrics.

`realchords/utils/custom_evaluation.py` used to write MIDI itself at the end
of every run, which meant that regenerating a large full-dataset MIDI export
(e.g. all ~23k Hooktheory songs) required re-running the whole batch loop
(generation + every metric) as well. This script decouples the two: point it
at a `--save_dir` already populated by `custom_evaluation.py` (i.e. a
`cropped_songs/` or `full_songs/` directory containing `gt.pt`,
`metadata.jsonl`, `model_labels.json`, and `chord_names_augmented.json`) and
it writes MIDI on its own, as many times / with as many samples as you like,
without touching the metrics.

Usage::

    python scripts/eval/export_paired_midis.py \
        --save_dir logs/paired_eval/gt/hooktheory_all/full_songs

    python scripts/eval/export_paired_midis.py \
        --save_dir logs/paired_eval/hooktheory_test/full_songs \
        --midi_samples -1 --midi_dir logs/paired_eval/hooktheory_test/full_songs/midi_all
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import note_seq.chord_symbols_lib as _chord_lib
import pretty_midi
import torch
from tqdm import tqdm

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer, to_midi_pitch

# ---------------------------------------------------------------------------
# MIDI rendering constants and helpers (moved from custom_evaluation.py)
# ---------------------------------------------------------------------------

_CHORD_OCTAVE = 4   # default chord-tone octave (MIDI 48-59)
_BASS_OCTAVE = 3    # bass note one octave below chord root voicing
_MELODY_VEL = 90
_CHORD_VEL = 64


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
    chord_octave: int = _CHORD_OCTAVE,
) -> List[int]:
    chord_pcs = _chord_lib.chord_symbol_pitches(chord_name)
    pitches = [p % 12 + chord_octave * 12 for p in chord_pcs]
    if include_bass:
        bass_pc = _chord_lib.chord_symbol_bass(chord_name) % 12
        pitches.append(bass_pc + _BASS_OCTAVE * 12)
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
    label: str,
    melody_instr: pretty_midi.Instrument,
    chord_instr: pretty_midi.Instrument,
    midi_obj: pretty_midi.PrettyMIDI,
    *,
    strict_chords: bool = False,
    include_chord_bass: bool = True,
    chord_octave: int = _CHORD_OCTAVE,
    melody_octave: int = 0,
) -> float:
    """Render one section (melody + chords) into instruments.  Returns section duration (s)."""
    chord_frames = seq[0::2].numpy()
    melody_frames = seq[1::2].numpy()

    chord_anns = _decode_chord_anns(chord_frames, tokenizer, strict=strict_chords)
    try:
        melody_anns = tokenizer.decode_melody_frames(melody_frames)
    except ValueError:
        melody_anns = []

    for note in melody_anns:
        melody_instr.notes.append(pretty_midi.Note(
            velocity=_MELODY_VEL,
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
                velocity=_CHORD_VEL,
                pitch=max(0, min(127, p)),
                start=chord["onset"] * spb + t0,
                end=chord["offset"] * spb + t0,
            ))
        midi_obj.lyrics.append(pretty_midi.Lyric(
            text=chord["chord_name"],
            time=chord["onset"] * spb + t0,
        ))

    all_offsets = [c["offset"] for c in chord_anns] + [m["offset"] for m in melody_anns]
    return max(all_offsets) * spb if all_offsets else 0.0


def _select_midi_indices(
    num_sequences: int,
    midi_samples: Optional[int],
    *,
    gt_only: bool,
    seed: int,
) -> List[int]:
    """Pick which sequence indices to export as MIDI."""
    if num_sequences <= 0:
        return []
    if midi_samples is None:
        target = 10 if gt_only else num_sequences
    elif midi_samples < 0:
        target = num_sequences
    else:
        target = midi_samples
    target = min(target, num_sequences)
    if target >= num_sequences:
        return list(range(num_sequences))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_sequences), target))


def write_paired_midis(
    gt_tensor: torch.Tensor,
    model_tensors: Dict[str, torch.Tensor],
    ordered_labels: List[str],
    metadata: List[Dict],
    gt_tokenizer,
    model_tokenizer,
    midi_dir: Path,
    bpm: int = 120,
    pause_bars: float = 0.5,
    indices: Optional[List[int]] = None,
    include_chord_bass: bool = True,
    chord_octave: int = _CHORD_OCTAVE,
    melody_octave: int = 0,
) -> None:
    """Write one MIDI file per song.

    Layout:
        [GT melody + GT chords]
        [pause]
        [GT melody + Model-A chords]
        [pause]
        [GT melody + Model-B chords]  ...

    The melody track is identical in every section (GT melody used as
    conditioning for all models).  Naive fixed-octave voicings are used
    throughout so the only difference between sections is the chord symbols.
    """
    midi_dir.mkdir(parents=True, exist_ok=True)
    spb = 60.0 / bpm
    pause_sec = pause_bars * 4 * spb  # 4 beats per bar

    export_indices = indices if indices is not None else list(range(gt_tensor.size(0)))
    for out_idx, i in enumerate(tqdm(export_indices, desc="Writing MIDIs")):
        midi_obj = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        melody_instr = pretty_midi.Instrument(program=0, name="Melody")
        chord_instr = pretty_midi.Instrument(program=0, name="Chords")

        t_cursor = 0.0

        # GT section: decode with the dataset vocab that encoded gt.pt
        gt_seq = gt_tensor[i, 1:]
        dur = _append_section(
            gt_seq, gt_tokenizer, spb, t_cursor, "GT",
            melody_instr, chord_instr, midi_obj, strict_chords=True,
            include_chord_bass=include_chord_bass,
            chord_octave=chord_octave,
            melody_octave=melody_octave,
        )
        t_cursor += dur + pause_sec

        # Model sections: decode with the model checkpoint vocab
        for label in ordered_labels:
            model_seq = model_tensors[label][i, 1:]
            dur = _append_section(
                model_seq, model_tokenizer, spb, t_cursor, label,
                melody_instr, chord_instr, midi_obj, strict_chords=False,
                include_chord_bass=include_chord_bass,
                chord_octave=chord_octave,
                melody_octave=melody_octave,
            )
            t_cursor += dur + pause_sec

        midi_obj.instruments.extend([melody_instr, chord_instr])

        song_url = metadata[i].get("song_url", "unknown")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", song_url)[-60:]
        midi_obj.write(str(midi_dir / f"{out_idx:04d}_seq{i:04d}_{safe}.mid"))

    print(f"  Wrote {len(export_indices)} MIDI files to {midi_dir}")


# ---------------------------------------------------------------------------
# Loading a custom_evaluation.py save_dir
# ---------------------------------------------------------------------------

def _load_metadata(save_dir: Path) -> List[Dict]:
    metadata: List[Dict] = []
    with (save_dir / "metadata.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            metadata.append(json.loads(line))
    return metadata


def _load_model_labels(save_dir: Path) -> Dict[str, object]:
    path = save_dir / "model_labels.json"
    if not path.exists():
        return {"dataset_name": None, "labels": {}}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _build_tokenizer(save_dir: Path) -> HooktheoryTokenizer:
    snapshot_path = save_dir / "chord_names_augmented.json"
    if not snapshot_path.exists():
        raise SystemExit(
            f"No chord_names_augmented.json under {save_dir} -- was this "
            "directory produced by realchords/utils/custom_evaluation.py?"
        )
    with snapshot_path.open(encoding="utf-8") as fh:
        chord_names = json.load(fh)
    return HooktheoryTokenizer(chord_names=chord_names)


def _resolve_include_chord_bass(
    include_chord_bass: bool, no_chord_bass: bool, dataset_name: Optional[str]
) -> bool:
    if include_chord_bass:
        return True
    if no_chord_bass:
        return False
    return dataset_name != "wjd"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="A cropped_songs/ or full_songs/ directory written by "
             "realchords/utils/custom_evaluation.py (must contain gt.pt, "
             "metadata.jsonl, model_labels.json, chord_names_augmented.json).",
    )
    parser.add_argument(
        "--midi_dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory for MIDI output. Defaults to <save_dir>/midi.",
    )
    parser.add_argument(
        "--midi_samples",
        type=int,
        default=None,
        metavar="N",
        help="Export N randomly chosen sequences as MIDI (seeded by --seed). "
             "Default: 10 for GT-only runs, all sequences otherwise. Use -1 for all.",
    )
    parser.add_argument("--bpm", type=int, default=120,
                        help="Tempo for MIDI rendering (default: 120).")
    parser.add_argument("--pause_bars", type=float, default=0.5,
                        help="Silence between sections in bars (default: 0.5).")
    parser.add_argument("--melody_octave", type=int, default=0,
                        help="Offset added to each melody note's stored octave for MIDI export.")
    parser.add_argument("--chord_octave", type=int, default=_CHORD_OCTAVE,
                        help="Octave for naive chord-tone voicings (default: 4).")
    parser.add_argument("--no_chord_bass", action="store_true",
                        help="Omit the separate bass note from chord voicings (default for wjd).")
    parser.add_argument("--include_chord_bass", action="store_true",
                        help="Force the separate chord bass note even for wjd.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    save_dir: Path = args.save_dir

    gt_tensor = torch.load(save_dir / "gt.pt")
    metadata = _load_metadata(save_dir)
    model_info = _load_model_labels(save_dir)
    labels: Dict[str, str] = model_info["labels"]
    dataset_name = model_info["dataset_name"]
    gt_only = len(labels) == 0

    tokenizer = _build_tokenizer(save_dir)

    ordered_labels = list(labels.keys())
    model_tensors = {
        label: torch.load(save_dir / f"{slug}.pt") for label, slug in labels.items()
    }

    midi_dir = args.midi_dir if args.midi_dir is not None else save_dir / "midi"
    midi_indices = _select_midi_indices(
        gt_tensor.size(0), args.midi_samples, gt_only=gt_only, seed=args.seed
    )
    include_chord_bass = _resolve_include_chord_bass(
        args.include_chord_bass, args.no_chord_bass, dataset_name
    )

    print(f"Loaded {gt_tensor.size(0)} sequences from {save_dir}")
    print(f"Models: {ordered_labels or '(none, GT only)'}")
    print(f"Writing {len(midi_indices)} MIDI files to {midi_dir} ...")

    write_paired_midis(
        gt_tensor=gt_tensor,
        model_tensors=model_tensors,
        ordered_labels=ordered_labels,
        metadata=metadata,
        gt_tokenizer=tokenizer,
        model_tokenizer=tokenizer,
        midi_dir=midi_dir,
        bpm=args.bpm,
        pause_bars=args.pause_bars,
        indices=midi_indices,
        include_chord_bass=include_chord_bass,
        chord_octave=args.chord_octave,
        melody_octave=args.melody_octave,
    )


if __name__ == "__main__":
    main()
