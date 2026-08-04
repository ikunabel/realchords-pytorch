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

--save_dir is the *parent* directory written by custom_evaluation.py (i.e.
the one containing cropped_songs/ and full_songs/ subdirectories) -- this
script writes MIDI for both variants in one call, mirroring how
custom_evaluation.py itself always produces both.

Usage (argbind, config-driven -- reuses the *same* config file the
custom_evaluation.py run itself used, e.g. configs/custom_eval/
gt_hooktheory.yml, since this script's own argbind binding only reads the
keys it declares (save_dir, midi_samples, chord_octave, ...) and ignores
the rest; see scripts/eval/custom_eval/run_custom_eval.py for the `midi_gt_<dataset>`
presets that wrap this):

    python scripts/eval/custom_eval/export_paired_midis.py \
        --args.load configs/custom_eval/gt_hooktheory.yml

Or equivalently, without a config file::

    python scripts/eval/custom_eval/export_paired_midis.py \
        --save_dir logs/custom_eval/gt/hooktheory_all

    python scripts/eval/custom_eval/export_paired_midis.py \
        --save_dir logs/custom_eval/hooktheory_test \
        --midi_samples -1 --midi_dir logs/custom_eval/hooktheory_test/midi_all
"""

import json
import random
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import argbind
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


GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

_MODES = ("cropped_songs", "full_songs")


def _export_one_mode(
    mode_dir: Path,
    *,
    midi_dir_override: str,
    midi_samples: int,
    bpm: int,
    pause_bars: float,
    melody_octave: int,
    chord_octave: int,
    no_chord_bass: bool,
    include_chord_bass: bool,
    seed: int,
) -> None:
    gt_tensor = torch.load(mode_dir / "gt.pt")
    metadata = _load_metadata(mode_dir)
    model_info = _load_model_labels(mode_dir)
    labels: Dict[str, str] = model_info["labels"]
    dataset_name = model_info["dataset_name"]
    gt_only = len(labels) == 0

    tokenizer = _build_tokenizer(mode_dir)

    ordered_labels = list(labels.keys())
    model_tensors = {
        label: torch.load(mode_dir / f"{slug}.pt") for label, slug in labels.items()
    }

    midi_dir = (
        Path(midi_dir_override) / mode_dir.name
        if midi_dir_override
        else mode_dir / "midi"
    )
    midi_indices = _select_midi_indices(
        gt_tensor.size(0), midi_samples, gt_only=gt_only, seed=seed
    )
    resolved_include_chord_bass = _resolve_include_chord_bass(
        include_chord_bass, no_chord_bass, dataset_name
    )

    print(f"Loaded {gt_tensor.size(0)} sequences from {mode_dir}")
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
        bpm=bpm,
        pause_bars=pause_bars,
        indices=midi_indices,
        include_chord_bass=resolved_include_chord_bass,
        chord_octave=chord_octave,
        melody_octave=melody_octave,
    )


@bind(without_prefix=True)
def main(
    args,
    save_dir: str = "",
    midi_dir: str = "",
    midi_samples: int = -1,
    bpm: int = 120,
    pause_bars: float = 0.5,
    melody_octave: int = 0,
    chord_octave: int = _CHORD_OCTAVE,
    no_chord_bass: bool = False,
    include_chord_bass: bool = False,
    seed: int = 42,
) -> None:
    """
    Args:
        save_dir: Parent directory written by custom_evaluation.py,
            containing cropped_songs/ and/or full_songs/ subdirectories.
        midi_dir: Directory for MIDI output. Defaults to <mode_dir>/midi
            for each mode found under save_dir; if given, MIDI for each
            mode is written under midi_dir/<mode>.
        midi_samples: Export N randomly chosen sequences as MIDI (seeded
            by seed). -1 = all sequences.
        no_chord_bass: Omit the separate bass note from chord voicings
            (default behavior for the wjd dataset).
        include_chord_bass: Force the separate chord bass note even for
            wjd.
    """
    if not save_dir:
        raise ValueError("save_dir must be provided.")
    save_dir_path = Path(save_dir)

    found_any = False
    for mode in _MODES:
        mode_dir = save_dir_path / mode
        if not (mode_dir / "gt.pt").exists():
            continue
        found_any = True
        _export_one_mode(
            mode_dir,
            midi_dir_override=midi_dir,
            midi_samples=midi_samples,
            bpm=bpm,
            pause_bars=pause_bars,
            melody_octave=melody_octave,
            chord_octave=chord_octave,
            no_chord_bass=no_chord_bass,
            include_chord_bass=include_chord_bass,
            seed=seed,
        )
    if not found_any:
        raise SystemExit(
            f"No {'/'.join(_MODES)} subdirectory with gt.pt found under {save_dir_path} "
            "-- was this produced by realchords/utils/custom_evaluation.py?"
        )


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    with argbind.scope(args):
        main(args)
