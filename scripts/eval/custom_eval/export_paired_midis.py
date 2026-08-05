#!/usr/bin/env python3
"""Write MIDI files from a `custom_evaluation.py` run, without re-running
generation or metrics.

`custom_evaluation.py` itself already auto-renders a small MIDI sample
(`--midi_samples`, default 10) at the end of every run -- see
`realchords/utils/midi_export.py`, which this script shares. This script is
for re-exporting a *different* sample count from an already-completed run
(most commonly: everything, `--midi_samples -1`, for a large full-dataset
export) without re-running the whole batch loop (generation + every metric).

--save_dir is the *parent* directory written by custom_evaluation.py (i.e.
the one containing cropped_songs/ and full_songs/ subdirectories) -- this
script writes MIDI for both variants in one call, mirroring how
custom_evaluation.py itself always produces both.

Each selected song is written as one MIDI file *per source* (GT, and each
model) to <mode_dir>/midi/<source>/{out_idx:04d}_seq{i:04d}_{song}.mid --
the same song has the same filename across every source's subfolder, so
they're trivial to diff/A-B against each other.

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
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import argbind
import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.midi_export import (
    CHORD_OCTAVE,
    resolve_include_chord_bass,
    select_midi_indices,
    write_all_source_midis,
)

# ---------------------------------------------------------------------------
# Loading a custom_evaluation.py save_dir
# ---------------------------------------------------------------------------

def _load_metadata(save_dir: Path) -> list:
    metadata = []
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


GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

_MODES = ("cropped_songs", "full_songs")


def _export_one_mode(
    mode_dir: Path,
    *,
    midi_dir_override: str,
    midi_samples: int,
    bpm: int,
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

    midi_dir = Path(midi_dir_override) / mode_dir.name if midi_dir_override else mode_dir / "midi"
    midi_indices = select_midi_indices(
        gt_tensor.size(0), midi_samples, default_all=not gt_only, seed=seed
    )
    resolved_include_chord_bass = resolve_include_chord_bass(
        include_chord_bass, no_chord_bass, dataset_name
    )

    print(f"Loaded {gt_tensor.size(0)} sequences from {mode_dir}")
    print(f"Models: {ordered_labels or '(none, GT only)'}")
    print(f"Writing {len(midi_indices)} song(s) x {1 + len(ordered_labels)} source(s) to {midi_dir} ...")

    write_all_source_midis(
        gt_tensor=gt_tensor,
        model_tensors=model_tensors,
        ordered_labels=ordered_labels,
        metadata=metadata,
        gt_tokenizer=tokenizer,
        model_tokenizer=tokenizer,
        midi_dir=midi_dir,
        indices=midi_indices,
        bpm=bpm,
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
    melody_octave: int = 0,
    chord_octave: int = CHORD_OCTAVE,
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
        midi_samples: Export N randomly chosen songs as MIDI (seeded by
            seed), one file per song per source. -1 = every song.
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
