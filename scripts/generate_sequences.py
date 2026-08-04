#!/usr/bin/env python3
"""Generate evaluation sequences from trained checkpoints.

This CLI supports three generation regimes:

1. Model-vs-model MARL generation
   Example:
       python scripts/generate_sequences.py \
           --args.load configs/generate_sequences/model_vs_model/mle_melody_vs_mle_chord_free_generation.yml

   Or equivalently, without a config file:
       python scripts/generate_sequences.py \
           --mode rl_melody_vs_rl_chord \
           --rl_melody_model_path logs/melody_rl/actor.pth \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --save_dir logs/generated/rl_vs_rl \
           --num_batches 16

2. Data-conditioned generation with optional perturbation
   Example:
       python scripts/generate_sequences.py \
           --args.load configs/generate_sequences/gt_vs_gapt/hooktheory.yml

   Or equivalently, without a config file:
       python scripts/generate_sequences.py \
           --mode melody_data_vs_rl_chord \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --dataset_name wikifonia \
           --dataset_split all \
           --data_perturbation multiple_transpose \
           --save_dir logs/generated/ood \
           --num_batches -1

3. Agent-switching generation (no saved config preset yet -- CLI/YAML only)
   Example:
       python scripts/generate_sequences.py \
           --mode rl_chord_vs_switching_melody \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --rl_melody_model_paths "logs/melody_a/actor.pth logs/melody_b/actor.pth" \
           --agent_switch_frames 64 \
           --target_seq_len 257 \
           --save_dir logs/generated/switching \
           --num_batches 8

See configs/generate_sequences/{gt,gt_vs_mle,gt_vs_realchords,gt_vs_gapt,model_vs_model}/ for every
saved preset, and scripts/eval/generate_sequences/run_generate_sequences.py to run one or more of them.

Generated artifacts:
  - model-vs-model and agent switching modes:
      <mode>_generated_chord_order.pt
      <mode>_generated_melody_order.pt
      <mode>_kl_chord.pt
      <mode>_kl_melody.pt
  - data-conditioned modes:
      <mode>_generated.pt
      <mode>_kl.pt
  - data-only mode:
      <mode>_chord_order.pt

All sequence tensors are rank-2 integer tensors with one sequence per row.
Outputs intended for downstream evaluation keep BOS at column 0 when generation
produces it, matching the private evaluation pipeline.
"""

from functools import partial
from pathlib import Path
from typing import List

import argbind
import torch
from lightning import seed_everything

from realchords.utils.experiment_utils import (
    create_dataset_dataloaders,
    handle_data_only_mode,
    load_models,
    save_generated_sequences,
)
from realchords.utils.experiment_utils_data_perturbation import (
    validate_perturbation_args,
)
from realchords.utils.experiment_utils_model_data import (
    MAX_GEN_STEPS,
    handle_data_conditioned_generation,
)
from realchords.utils.experiment_utils_model_model import (
    handle_agent_switching_generation,
    handle_model_vs_model_generation,
    load_models_for_switching,
)
from realchords.utils.train_utils import AttrDict

GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

_VALID_DATA_PERTURBATIONS = {"none", "multiple_transpose", "single_transpose_6"}
_VALID_DATASET_NAMES = {"hooktheory", "pop909", "nottingham", "wikifonia", "jazzmus", "wjd"}
_VALID_DATASET_SPLITS = {"train", "valid", "test", "all"}


def validate_args(args, mode_parts: list[str]) -> None:
    is_data_mode = "data" in mode_parts[0] or "data" in mode_parts[1]
    if not is_data_mode and args.num_batches < 1:
        raise ValueError(
            "num_batches must be >= 1 for model-vs-model and switching modes. Use -1 only with data-conditioned generation."
        )


@bind(without_prefix=True)
def main(
    args,
    rl_melody_model_path: str = "",
    rl_chord_model_path: str = "",
    mle_melody_model_path: str = "",
    mle_chord_model_path: str = "",
    mode: str = "",
    save_dir: str = "",
    batch_size: int = 64,
    num_batches: int = 1,
    target_seq_len: int = MAX_GEN_STEPS,
    prompt_steps: int = 0,
    rl_melody_model_paths: List[str] = [],
    mle_melody_model_paths: List[str] = [],
    rl_chord_model_paths: List[str] = [],
    mle_chord_model_paths: List[str] = [],
    agent_switch_steps: List[int] = [],
    seed: int = 42,
    data_perturbation: str = "none",
    dataset_name: str = "hooktheory",
    dataset_split: str = "valid",
) -> None:
    """
    Args:
        rl_melody_model_path: Path to the RL melody checkpoint (.pth).
        rl_chord_model_path: Path to the RL chord checkpoint (.pth).
        mle_melody_model_path: Path to the baseline MLE melody Lightning
            checkpoint. Not required for mode melody_data_vs_chord_data (GT dump).
        mle_chord_model_path: Path to the baseline MLE chord Lightning
            checkpoint. Not required for mode melody_data_vs_chord_data (GT dump).
        mode: Generation mode, e.g. rl_melody_vs_rl_chord or melody_data_vs_rl_chord.
        save_dir: Directory where generated tensors and KL artifacts are written.
        num_batches: Number of batches to process. Use -1 only for
            data-conditioned modes, to process the entire selected split.
        target_seq_len: Target length of the final sequence in tokens,
            including prompts.
        prompt_steps: Number of data frames used as prompts before free
            generation (CLI/YAML key: prompt_steps; was --prompt_frames
            under the old argparse CLI).
        rl_melody_model_paths: RL melody checkpoints for agent switching.
        mle_melody_model_paths: MLE melody checkpoints for agent switching.
        rl_chord_model_paths: RL chord checkpoints for agent switching.
        mle_chord_model_paths: MLE chord checkpoints for agent switching.
        agent_switch_steps: Frame counts for each switching segment (one
            frame = one chord+melody pair). Required for switching modes.
        data_perturbation: One of none / multiple_transpose / single_transpose_6.
        dataset_name: One of hooktheory / pop909 / nottingham / wikifonia / jazzmus / wjd.
        dataset_split: One of train / valid / test / all ('all' loads
            train+valid+test combined).
    """
    if not mode:
        raise ValueError(
            "mode must be provided, e.g. rl_melody_vs_rl_chord or melody_data_vs_rl_chord."
        )
    if not save_dir:
        raise ValueError("save_dir must be provided.")
    if data_perturbation not in _VALID_DATA_PERTURBATIONS:
        raise ValueError(
            f"Invalid data_perturbation '{data_perturbation}'. "
            f"Expected one of: {sorted(_VALID_DATA_PERTURBATIONS)}"
        )
    if dataset_name not in _VALID_DATASET_NAMES:
        raise ValueError(
            f"Invalid dataset_name '{dataset_name}'. Expected one of: {sorted(_VALID_DATASET_NAMES)}"
        )
    if dataset_split not in _VALID_DATASET_SPLITS:
        raise ValueError(
            f"Invalid dataset_split '{dataset_split}'. Expected one of: {sorted(_VALID_DATASET_SPLITS)}"
        )

    # Downstream helpers (experiment_utils*.py) all expect a namespace-like
    # `args` with plain attribute access -- rebuild it here from the typed
    # params above so the rest of this function (and every helper it calls)
    # is otherwise unchanged from the pre-argbind version.
    args = AttrDict(
        rl_melody_model_path=rl_melody_model_path or None,
        rl_chord_model_path=rl_chord_model_path or None,
        mle_melody_model_path=mle_melody_model_path or None,
        mle_chord_model_path=mle_chord_model_path or None,
        mode=mode,
        save_dir=save_dir,
        batch_size=batch_size,
        num_batches=num_batches,
        target_seq_len=target_seq_len,
        prompt_steps=prompt_steps,
        rl_melody_model_paths=rl_melody_model_paths or None,
        mle_melody_model_paths=mle_melody_model_paths or None,
        rl_chord_model_paths=rl_chord_model_paths or None,
        mle_chord_model_paths=mle_chord_model_paths or None,
        agent_switch_steps=agent_switch_steps or None,
        seed=seed,
        data_perturbation=data_perturbation,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )

    seed_everything(args.seed)

    mode_parts = args.mode.split("_vs_")
    if len(mode_parts) != 2:
        raise ValueError(
            "Mode should be in the format 'mel_source_vs_chord_source'"
        )

    validate_args(args, mode_parts)

    is_switching_mode = mode_parts[0] in [
        "switching_melody",
        "switching_chord",
    ] or mode_parts[1] in ["switching_melody", "switching_chord"]

    if is_switching_mode:
        if args.agent_switch_steps is None:
            raise ValueError(
                "agent_switch_steps must be provided for switching modes"
            )

        if (
            mode_parts[0] == "switching_melody"
            or mode_parts[1] == "switching_melody"
        ):
            if (
                not args.rl_melody_model_paths
                and not args.mle_melody_model_paths
            ):
                raise ValueError(
                    "rl_melody_model_paths or mle_melody_model_paths must be provided for switching_melody mode"
                )
        if (
            mode_parts[0] == "switching_chord"
            or mode_parts[1] == "switching_chord"
        ):
            if not args.rl_chord_model_paths and not args.mle_chord_model_paths:
                raise ValueError(
                    "rl_chord_model_paths or mle_chord_model_paths must be provided for switching_chord mode"
                )

    valid_sources = [
        "rl_melody",
        "mle_melody",
        "melody_data",
        "switching_melody",
        "rl_chord",
        "mle_chord",
        "switching_chord",
    ]
    valid_targets = [
        "rl_chord",
        "mle_chord",
        "chord_data",
        "switching_chord",
        "rl_melody",
        "mle_melody",
        "switching_melody",
    ]

    if mode_parts[0] not in valid_sources or mode_parts[1] not in valid_targets:
        raise ValueError(
            "Invalid mode. See --help for supported mode families."
        )

    validate_perturbation_args(args.mode, args.data_perturbation)

    mel_source, chord_source = mode_parts[0], mode_parts[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "switching_melody" in mode_parts:
        switching_type = "melody"
        fixed_source = (
            mode_parts[1]
            if mode_parts[0] == "switching_melody"
            else mode_parts[0]
        )
    elif "switching_chord" in mode_parts:
        switching_type = "chord"
        fixed_source = (
            mode_parts[1]
            if mode_parts[0] == "switching_chord"
            else mode_parts[0]
        )
    else:
        switching_type = None
        fixed_source = None

    print(f"Mode: {args.mode}")
    print(f"Switching mode: {switching_type is not None}")
    print(f"Mel source: {mel_source}")
    print(f"Chord source: {chord_source}")

    if switching_type is not None:
        (
            switching_models,
            fixed_model,
            melody_tokenizer,
            chord_tokenizer,
            melody_dataloaders,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
        ) = load_models_for_switching(
            args, switching_type, fixed_source, device
        )

        generated_all = handle_agent_switching_generation(
            args,
            switching_models,
            fixed_model,
            melody_tokenizer,
            chord_tokenizer,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
            switching_type,
            fixed_source,
            device,
        )
    elif mel_source == "melody_data" and chord_source == "chord_data":
        # GT / data-only mode: no model needed, just dump the dataset.
        chord_dataloaders = create_dataset_dataloaders(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            model_part="chord",
            batch_size=args.batch_size,
            max_len=args.target_seq_len,
        )
        handle_data_only_mode(args, chord_dataloaders, device, args.data_perturbation)
        return
    else:
        (
            melody_model,
            chord_model,
            melody_tokenizer,
            chord_tokenizer,
            melody_dataloaders,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
        ) = load_models(args, mel_source, chord_source, device)
        if "data" not in mel_source and "data" not in chord_source:
            generated_all = handle_model_vs_model_generation(
                args,
                chord_model,
                melody_model,
                chord_tokenizer,
                melody_tokenizer,
                chord_dataloaders,
                mle_melody_model,
                mle_chord_model,
                device,
            )
        else:
            generated_all = handle_data_conditioned_generation(
                args,
                mel_source,
                chord_source,
                melody_model,
                chord_model,
                melody_tokenizer,
                chord_tokenizer,
                melody_dataloaders,
                chord_dataloaders,
                mle_melody_model,
                mle_chord_model,
                device,
                data_perturbation=args.data_perturbation,
            )

    if switching_type is not None:
        if switching_type == "melody":
            save_generated_sequences(
                generated_all, args, "switching_melody", fixed_source
            )
        else:
            save_generated_sequences(
                generated_all, args, fixed_source, "switching_chord"
            )
    else:
        save_generated_sequences(generated_all, args, mel_source, chord_source)


if __name__ == "__main__":
    parsed_args = argbind.parse_args(group=GROUP)
    if parsed_args.get("save_dir"):
        argbind.dump_args(parsed_args, Path(parsed_args["save_dir"]) / "args.yml")
    with argbind.scope(parsed_args):
        main(parsed_args)
