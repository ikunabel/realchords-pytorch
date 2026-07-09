"""Common utilities for sequence generation experiments."""

import json
import os
import shutil
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from realchords.constants import CACHE_DIR, CHORD_NAMES_AUG_PATH
from realchords.dataset.hooktheory_dataloader import (
    HooktheoryDataset,
    get_dataloader as get_hooktheory_dataloader,
)
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.decoder_only import LitDecoder
from realchords.rl.utils import compute_full_kl
from realchords.utils.experiment_utils_data_perturbation import (
    apply_data_perturbation,
)
from realchords.utils.inference_utils import load_lit_model, load_rl_model
from realchords.utils.sequence_utils import (
    add_bos_to_sequence,
    log_probs_from_online_model,
    sequences_order_to_counterpart,
)

DATASET_CACHE_DIRS = {
    "hooktheory": os.path.join(CACHE_DIR, "hooktheory"),
    "pop909": os.path.join(CACHE_DIR, "pop909"),
    "nottingham": os.path.join(CACHE_DIR, "nottingham"),
    "wikifonia": os.path.join(CACHE_DIR, "wikifonia"),
    "jazzmus": os.path.join(CACHE_DIR, "jazzmus"),
}


def create_dataset_dataloaders(
    dataset_name: str,
    dataset_split: str,
    model_part: str,
    batch_size: int,
    max_len: int,
    num_workers: int = 0,
) -> Tuple[Optional[torch.utils.data.DataLoader], torch.utils.data.DataLoader]:
    """Create dataloaders for a specific dataset/split using HooktheoryDataset."""

    dataset_key = dataset_name.lower()
    split_key = dataset_split.lower()

    if dataset_key not in DATASET_CACHE_DIRS:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. Expected one of: {list(DATASET_CACHE_DIRS.keys())}"
        )

    cache_dir = DATASET_CACHE_DIRS[dataset_key]
    chord_names_path = CHORD_NAMES_AUG_PATH
    if not os.path.exists(chord_names_path):
        raise FileNotFoundError(
            f"Expected augmented chord vocab at {chord_names_path}"
        )

    dataset = HooktheoryDataset(
        cache_dir=cache_dir,
        split=split_key,
        model_type="decoder_only",
        model_part=model_part,
        max_len=max_len,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=chord_names_path,
    )

    dataloader = get_hooktheory_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (None, dataloader)


def ensure_bos_token(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Ensure every row starts with a BOS token."""
    bos = tokenizer.bos_token
    first_col = sequences[:, 0]
    all_have_bos = bool((first_col == bos).all())
    none_have_bos = bool((first_col != bos).all())
    if all_have_bos:
        return sequences
    if none_have_bos:
        return add_bos_to_sequence(sequences, bos)
    raise ValueError(
        "Inconsistent BOS handling: some rows start with BOS and some do not."
    )


def replace_eos_with_pad(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Replace EOS tokens with PAD tokens in sequences."""
    sequences[sequences == tokenizer.eos_token] = tokenizer.pad_token
    return sequences


VOCAB_SNAPSHOT_FILENAME = "chord_names_augmented.json"


def save_vocab_snapshot(save_dir: str, chord_names_path: Optional[str] = None) -> Path:
    """Copy the active chord-names vocab into *save_dir* so that sequence
    tensors can always be decoded with the exact vocab that was used to encode
    them, regardless of future changes to the global vocab file.

    The snapshot is written to ``<save_dir>/chord_names_augmented.json``.
    The MIDI conversion script and evaluation pipeline pick it up
    automatically when it is present.

    Args:
        save_dir: Directory where the ``.pt`` tensors are being saved.
        chord_names_path: Path to the chord names JSON to snapshot.
            Defaults to ``CHORD_NAMES_AUG_PATH``.

    Returns:
        Path to the written snapshot file.
    """
    src = Path(chord_names_path) if chord_names_path else Path(CHORD_NAMES_AUG_PATH)
    dst = Path(save_dir) / VOCAB_SNAPSHOT_FILENAME
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def generate_save_paths(
    save_dir: str, mode: str, generation_type: str = "data_conditioned"
) -> Dict[str, Path]:
    """Generate save paths for different types of outputs."""
    save_dir_path = Path(save_dir)

    if generation_type in {"model_vs_model", "agent_switching"}:
        return {
            "seq1": save_dir_path / f"{mode}_generated_chord_order.pt",
            "seq2": save_dir_path / f"{mode}_generated_melody_order.pt",
            "kl_chord": save_dir_path / f"{mode}_kl_chord.pt",
            "kl_melody": save_dir_path / f"{mode}_kl_melody.pt",
        }
    if generation_type == "data_conditioned":
        return {
            "sequences": save_dir_path / f"{mode}_generated.pt",
            "kl": save_dir_path / f"{mode}_kl.pt",
        }
    if generation_type == "data_only":
        return {
            "chord_data": save_dir_path / f"{mode}_generated_chord_order.pt"
        }
    raise ValueError(f"Unknown generation_type: {generation_type}")


def _random_crop_on_chord_onset(
    self: "HooktheoryDataset", melody: torch.Tensor, chord: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like ``HooktheoryDataset.random_crop``, but restricts the crop start to
    frames where a chord onset begins.

    ``HooktheoryDataset.random_crop`` only aligns crop starts to beat
    boundaries, so a crop can start in the middle of a held chord. That is
    fine for training, but it means the resulting window has no
    ``CHORD_ON`` for whatever chord is already sounding, which the
    tokenizer's strict ``decode_to_midi``/``decode_chord_frames`` cannot
    represent (it raises "Chord off without chord on"). Restricting starts
    to chord-onset frames keeps every exported ground-truth crop decodable.
    """
    assert melody.shape[0] == chord.shape[0]

    if melody.shape[0] <= self.max_len_per_part:
        return melody, chord

    max_len = min(self.max_len_per_part, melody.shape[0])
    start_allowed = list(
        range(0, melody.shape[0] - max_len + 1, self.frame_per_beat)
    )
    onset_starts = [
        s for s in start_allowed if self.tokenizer.is_chord_on(int(chord[s]))
    ]
    start = np.random.choice(onset_starts if onset_starts else start_allowed)
    end = start + max_len
    return melody[start:end], chord[start:end]


def handle_data_only_mode(
    args: Any,
    chord_dataloaders: Tuple,
    device: torch.device,
    data_perturbation: str = "none",
) -> None:
    """Handle the special case where both sources are data."""
    print("No sequences were generated. Save the data.")
    chord_data_all = []

    _, chord_val_dataloader = chord_dataloaders
    # Patch this dataloader's dataset instance (not the shared
    # HooktheoryDataset class) so ground-truth crops always start on a
    # chord onset, without changing crop behavior used for training.
    chord_val_dataloader.dataset.random_crop = types.MethodType(
        _random_crop_on_chord_onset, chord_val_dataloader.dataset
    )
    print(f"Collecting chord data from {len(chord_val_dataloader)} batches.")

    for batch in tqdm(chord_val_dataloader, desc="Collecting chord data"):
        targets = batch["targets"].to(device)
        chord_data_all.append(targets.cpu())

    if chord_data_all:
        os.makedirs(args.save_dir, exist_ok=True)
        chord_data_tensor = torch.cat(chord_data_all, dim=0)[:, :-1]

        if data_perturbation != "none":
            chord_tokenizer = chord_val_dataloader.dataset.tokenizer
            chord_data_tensor = apply_data_perturbation(
                chord_data_tensor, data_perturbation, "chord", chord_tokenizer
            )

        save_paths = generate_save_paths(args.save_dir, args.mode, "data_only")
        torch.save(chord_data_tensor, save_paths["chord_data"])
        print(f"Saved chord data to {save_paths['chord_data']}")
        print(f"Chord data shape: {chord_data_tensor.shape}")
        snapshot = save_vocab_snapshot(args.save_dir)
        print(f"Saved vocab snapshot to {snapshot}")


def save_generated_sequences(
    generated_all: list, args: Any, mel_source: str, chord_source: str
) -> None:
    """Save generated sequences and KL values to files."""
    os.makedirs(args.save_dir, exist_ok=True)

    if not generated_all:
        print("No sequences were generated. Nothing to save.")
        return

    is_switching_mode = mel_source in [
        "switching_melody",
        "switching_chord",
    ] or chord_source in ["switching_melody", "switching_chord"]

    if (
        "data" not in mel_source and "data" not in chord_source
    ) or is_switching_mode:
        seq1_all = torch.cat([x[0] for x in generated_all], dim=0)
        seq2_all = torch.cat([x[1] for x in generated_all], dim=0)
        kl_chord_all = torch.cat([x[2] for x in generated_all], dim=0)
        kl_melody_all = torch.cat([x[3] for x in generated_all], dim=0)

        generation_type = (
            "agent_switching" if is_switching_mode else "model_vs_model"
        )
        save_paths = generate_save_paths(
            args.save_dir, args.mode, generation_type
        )

        torch.save(seq1_all, save_paths["seq1"])
        torch.save(seq2_all, save_paths["seq2"])
        torch.save(kl_chord_all, save_paths["kl_chord"])
        torch.save(kl_melody_all, save_paths["kl_melody"])
        save_vocab_snapshot(args.save_dir)

        print(f"Saved generated chord sequences to {save_paths['seq1']}")
        print(f"Chord sequences shape: {seq1_all.shape}")
        print(f"Saved generated melody sequences to {save_paths['seq2']}")
        print(f"Melody sequences shape: {seq2_all.shape}")
        print(f"Saved chord KL values to {save_paths['kl_chord']}")
        print(f"Chord KL values shape: {kl_chord_all.shape}")
        print(f"Saved melody KL values to {save_paths['kl_melody']}")
        print(f"Melody KL values shape: {kl_melody_all.shape}")
    else:
        sequences_all = torch.cat([x[0] for x in generated_all], dim=0)
        kl_all = torch.cat([x[1] for x in generated_all], dim=0)

        save_paths = generate_save_paths(
            args.save_dir, args.mode, "data_conditioned"
        )

        torch.save(sequences_all, save_paths["sequences"])
        torch.save(kl_all, save_paths["kl"])
        save_vocab_snapshot(args.save_dir)

        print(f"Saved generated sequences to {save_paths['sequences']}")
        print(f"Generated sequences shape: {sequences_all.shape}")
        print(f"Saved KL values to {save_paths['kl']}")
        print(f"KL values shape: {kl_all.shape}")


def load_models(
    args: Any, mel_source: str, chord_source: str, device: torch.device
) -> Tuple[
    Optional[torch.nn.Module],
    Optional[torch.nn.Module],
    Optional[HooktheoryTokenizer],
    Optional[HooktheoryTokenizer],
    Optional[Tuple],
    Optional[Tuple],
    Optional[torch.nn.Module],
    Optional[torch.nn.Module],
]:
    """Load the MLE baselines, optional RL checkpoints, and dataloaders."""
    melody_model, chord_model = None, None
    melody_tokenizer, chord_tokenizer = None, None
    mle_melody_model, mle_chord_model = None, None

    mle_melody_model, melody_tokenizer, _ = load_lit_model(
        model_path=args.mle_melody_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.batch_size,
        compile=False,
    )

    mle_chord_model, chord_tokenizer, _ = load_lit_model(
        model_path=args.mle_chord_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.batch_size,
        compile=False,
    )

    melody_dataloaders = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="melody",
        batch_size=args.batch_size,
        max_len=args.target_seq_len,
    )
    chord_dataloaders = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="chord",
        batch_size=args.batch_size,
        max_len=args.target_seq_len,
    )

    if mel_source in ["rl_melody", "mle_melody"]:
        if mel_source == "rl_melody":
            if not args.rl_melody_model_path:
                raise ValueError(
                    "rl_melody_model_path must be provided for rl_melody mode"
                )
            melody_model = load_rl_model(
                model_path=args.rl_melody_model_path,
                model=deepcopy(mle_melody_model),
                compile=False,
            )
        else:
            melody_model = mle_melody_model

    if chord_source in ["rl_chord", "mle_chord"]:
        if chord_source == "rl_chord":
            if not args.rl_chord_model_path:
                raise ValueError(
                    "rl_chord_model_path must be provided for rl_chord mode"
                )
            chord_model = load_rl_model(
                model_path=args.rl_chord_model_path,
                model=deepcopy(mle_chord_model),
                compile=False,
            )
        else:
            chord_model = mle_chord_model

    if melody_tokenizer is None:
        melody_tokenizer = chord_tokenizer
    if chord_tokenizer is None:
        chord_tokenizer = melody_tokenizer
    if melody_model is not None:
        melody_model.eval().to(device)
    if chord_model is not None:
        chord_model.eval().to(device)
    if mle_melody_model is not None:
        mle_melody_model.eval().to(device)
    if mle_chord_model is not None:
        mle_chord_model.eval().to(device)

    return (
        melody_model,
        chord_model,
        melody_tokenizer,
        chord_tokenizer,
        melody_dataloaders,
        chord_dataloaders,
        mle_melody_model,
        mle_chord_model,
    )


def compute_social_influence_kl(
    sequence: torch.Tensor,
    online_model: torch.nn.Module,
    semi_online_model: torch.nn.Module,
) -> torch.Tensor:
    """Compute the social influence KL used by the original evaluation code."""
    sequence_counterpart = sequences_order_to_counterpart(sequence)
    _, logits_online = log_probs_from_online_model(
        online_model,
        sequence,
    )
    _, logits_semi_online = log_probs_from_online_model(
        semi_online_model,
        sequence_counterpart,
    )

    logits_online = logits_online[:, ::2, :]
    logits_semi_online = logits_semi_online[:, 1::2, :]

    return compute_full_kl(logits_online, logits_semi_online)


def get_online_and_semi_online_model(
    model_part: str,
    mle_melody_model: torch.nn.Module,
    mle_chord_model: torch.nn.Module,
    melody_model: Optional[torch.nn.Module] = None,
    chord_model: Optional[torch.nn.Module] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Get online and semi-online models based on the model part."""
    if model_part == "melody":
        if melody_model is None:
            raise ValueError("melody_model must be provided for melody part")
        return melody_model, mle_chord_model
    if model_part == "chord":
        if chord_model is None:
            raise ValueError("chord_model must be provided for chord part")
        return chord_model, mle_melody_model
    raise ValueError(f"Invalid model_part: {model_part}")
