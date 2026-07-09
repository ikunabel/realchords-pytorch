"""Tests for converting generated interleaved tensors to MIDI.

The conversion logic under test lives entirely in
``scripts/to_midi/convert_generated_sequences_to_midi.py`` (loaded here by
file path since ``scripts/`` is not an importable package). It mirrors the
``output_to_midi`` pattern used in ``realchords/lit_module/decoder_only.py``:
split the interleaved sequence and call ``HooktheoryTokenizer.decode_to_midi``
directly, with no extra decoding logic. The tokenizer and
``realchords.utils.sequence_penalty_analysis`` are used unmodified.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from realchords.constants import CHORD_NAMES_AUG_PATH
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset, get_dataloader
from realchords.utils.sequence_penalty_analysis import load_tokenizer, strip_bos

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "to_midi"
    / "convert_generated_sequences_to_midi.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "convert_generated_sequences_to_midi", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


convert_module = _load_script_module()

HOOKTHEORY_GT_PT = (
    Path("logs/generated/hooktheory_gt")
    / "melody_data_vs_chord_data_generated_chord_order.pt"
)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(Path(CHORD_NAMES_AUG_PATH))


def test_row_to_midi_matches_tokenizer_decode_directly(tokenizer):
    """row_to_midi should be a thin wrapper: split lanes, call decode_to_midi."""
    np.random.seed(0)
    dataset = HooktheoryDataset(
        cache_dir="data/cache/hooktheory",
        split="test",
        model_type="decoder_only",
        model_part="chord",
        max_len=512,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=CHORD_NAMES_AUG_PATH,
    )
    dataloader = get_dataloader(dataset, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    sequences = strip_bos(batch["targets"][:, :-1], tokenizer)

    for row in sequences[:4]:
        midi = convert_module.row_to_midi(row, tokenizer, sequence_order="chord_first")

        chord_frames, melody_frames = convert_module.split_tokens_by_order(
            row, "chord_first"
        )
        expected = tokenizer.decode_to_midi(
            chord_frames=chord_frames.numpy(),
            melody_frames=melody_frames.numpy(),
        )

        def note_signature(midi_obj):
            return [
                sorted((n.start, n.end, n.pitch) for n in instr.notes)
                for instr in midi_obj.instruments
            ]

        assert note_signature(midi) == note_signature(expected)
        assert sum(len(instr.notes) for instr in midi.instruments) > 0


@pytest.mark.skipif(not HOOKTHEORY_GT_PT.exists(), reason="hooktheory_gt tensor missing")
def test_hooktheory_gt_pt_conversion_does_not_crash(tmp_path, tokenizer):
    """``.pt`` rows are random 256-frame crops (see ``HooktheoryDataset.random_crop``),
    so a crop boundary can land mid-chord and fail the tokenizer's strict
    CHORD_ON grammar (this is a property of the crop, not a tokenizer bug -
    full, uncropped songs from the cache decode without error, see
    ``tests/test_cache/test_cache_to_midi.py``). Conversion should skip such
    rows gracefully, matching ``decoder_only.py``'s own try/except around
    ``decode_to_midi``, rather than crash the whole run.
    """
    written = convert_module.convert_sequence_file_to_midi(
        HOOKTHEORY_GT_PT,
        tmp_path,
        tokenizer,
        sequence_order="chord_first",
        max_sequences=20,
    )
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_convert_generated_system_to_midi_default_output_dir(tmp_path, tokenizer):
    """Default output should live inside the system directory as <dir>/midi."""
    system_dir = tmp_path / "some_system"
    system_dir.mkdir()

    np.random.seed(0)
    dataset = HooktheoryDataset(
        cache_dir="data/cache/hooktheory",
        split="test",
        model_type="decoder_only",
        model_part="chord",
        max_len=512,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=CHORD_NAMES_AUG_PATH,
    )
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    torch.save(batch["targets"][:, :-1], system_dir / "fake_generated_chord_order.pt")

    name, output_dir, midi_paths = convert_module.convert_generated_system_to_midi(
        system_dir,
        tokenizer,
        max_sequences=2,
    )

    assert output_dir == system_dir / "midi"
    assert len(midi_paths) == 2
    for path in midi_paths:
        assert path.exists()
