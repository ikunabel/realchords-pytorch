"""Utilities for model-vs-data sequence generation experiments."""

from typing import Any, List, Optional, Tuple

import torch
from tqdm import tqdm

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.experiment_utils import (
    compute_social_influence_kl,
    get_online_and_semi_online_model,
    replace_eos_with_pad,
)
from realchords.utils.experiment_utils_data_perturbation import (
    apply_data_perturbation,
)
from realchords.utils.sequence_utils import (
    add_bos_to_sequence,
    add_eos_to_sequence,
    sequences_order_to_counterpart,
)

MAX_GEN_STEPS = 512


def extract_prompts_from_data(
    sequences: torch.Tensor, prompt_steps: int, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Extract BOS-prefixed prompts from interleaved data sequences."""
    if prompt_steps == 0:
        return torch.full(
            (sequences.shape[0], 1),
            tokenizer.bos_token,
            dtype=torch.long,
            device=sequences.device,
        )

    prompt_tokens = sequences[:, : prompt_steps * 2]
    bos_tokens = torch.full(
        (sequences.shape[0], 1),
        tokenizer.bos_token,
        dtype=torch.long,
        device=sequences.device,
    )

    return torch.cat([bos_tokens, prompt_tokens], dim=1)


def adjust_conditions_for_prompts(
    sequences: torch.Tensor,
    prompt_steps: int,
) -> torch.Tensor:
    """Adjust conditions to account for the prompted portion."""
    if prompt_steps == 0:
        return sequences[:, 1::2]

    start_idx = prompt_steps * 2
    remaining_seq = sequences[:, start_idx:]
    return remaining_seq[:, 1::2]


def convert_sequence_order_for_model(
    sequences: torch.Tensor, target_model_type: str, source_model_type: str
) -> torch.Tensor:
    """Convert sequence order between chord-order and melody-order."""
    if target_model_type == source_model_type:
        return sequences
    return sequences_order_to_counterpart(sequences)


def generate_from_data(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    prompt_steps: int = 0,
    target_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Generate sequences from data conditions and return prompt + generated tokens."""
    if target_seq_len is None:
        target_seq_len = MAX_GEN_STEPS

    if target_seq_len % 2 == 0:
        target_seq_len += 1

    prompts = extract_prompts_from_data(sequences, prompt_steps, tokenizer)
    prompt_len = prompts.shape[1]
    tokens_to_generate = target_seq_len - prompt_len

    if tokens_to_generate <= 0:
        raise ValueError(
            f"Invalid configuration: target_seq_len ({target_seq_len}) must be greater than prompt length ({prompt_len})."
        )
    if tokens_to_generate % 2 != 0:
        raise ValueError(
            f"Internal error: tokens_to_generate ({tokens_to_generate}) should be even."
        )

    conditions = adjust_conditions_for_prompts(sequences, prompt_steps)

    full_conditions = sequences[:, 1::2]
    conditions_mask = full_conditions != tokenizer.pad_token

    output_mask = torch.zeros(
        (conditions.shape[0], target_seq_len - 1),
        dtype=torch.bool,
        device=sequences.device,
    )
    output_mask[:, 0::2][:, : conditions_mask.shape[1]] = conditions_mask
    output_mask[:, 1::2][:, : conditions_mask.shape[1]] = conditions_mask
    output_mask = torch.cat(
        [
            torch.ones(
                (conditions.shape[0], 1),
                dtype=torch.bool,
                device=sequences.device,
            ),
            output_mask,
        ],
        dim=1,
    )

    with torch.no_grad():
        generated_part = model.generate_online(
            prompts,
            conditions=conditions,
            seq_len=tokens_to_generate,
            cache_kv=True,
        )
        complete_sequence = torch.cat([prompts, generated_part], dim=1)
        complete_sequence.masked_fill_(~output_mask, tokenizer.pad_token)

    return complete_sequence


def generate_from_data_enc_dec(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    target_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Encoder-decoder counterpart of generate_from_data.

    Encodes the *entire* melody once (encoder), then autoregressively
    decodes chord predictions conditioned on that fixed encoding -- a
    fundamentally different procedure from decoder-only's frame-by-frame
    interleaved online generation (generate_online), since encoder-decoder
    models see the whole melody up front rather than causally.

    Despite the different generation mechanism, returns an identically
    shaped/masked tensor to generate_from_data -- [BOS, chord_0, melody_0,
    chord_1, melody_1, ...], with content past the real (non-PAD) melody
    length masked back to PAD, matching generate_from_data's own
    output_mask convention -- so every downstream consumer (saving to
    {slug}.pt, strip_bos, every metric in eval_utils.py) is unchanged and
    doesn't need to know or care which architecture generated a given
    tensor.

    Args:
        model: The raw EncoderDecoderTransformer (i.e. lit_module.model from
            load_lit_model with lit_module_cls=LitEncoderDecoder), not the
            Lightning wrapper.
        sequences: [B, S] interleaved chord-first GT sequence (BOS/EOS
            already stripped, EOS already replaced with PAD -- same
            `gt_seq_stripped` generate_from_data itself takes). Only the
            melody lane (odd positions) is used; the chord lane is ignored
            since it's the target being predicted, not conditioned on.
        tokenizer: Shared tokenizer for pad/bos/eos ids.
        target_seq_len: Same convention as generate_from_data -- total
            output length including the leading BOS.
    """
    if target_seq_len is None:
        target_seq_len = MAX_GEN_STEPS
    if target_seq_len % 2 == 0:
        target_seq_len += 1

    num_frames = (target_seq_len - 1) // 2
    melody_frames = sequences[:, 1::2][:, :num_frames]
    conditions_mask = melody_frames != tokenizer.pad_token

    enc_inputs = add_bos_to_sequence(melody_frames, tokenizer.bos_token)
    enc_inputs = add_eos_to_sequence(enc_inputs, tokenizer.pad_token, tokenizer.eos_token)
    enc_inputs_mask = enc_inputs != tokenizer.pad_token

    gen_start = torch.full(
        (melody_frames.shape[0], 1), tokenizer.bos_token,
        dtype=torch.long, device=sequences.device,
    )

    with torch.no_grad():
        chord_preds = model.generate(
            enc_inputs, gen_start, seq_len=num_frames,
            mask=enc_inputs_mask, cache_kv=True,
        )
    chord_preds = chord_preds[:, :num_frames].masked_fill(~conditions_mask, tokenizer.pad_token)

    interleaved = torch.empty(
        (melody_frames.shape[0], 2 * num_frames),
        dtype=torch.long, device=sequences.device,
    )
    interleaved[:, 0::2] = chord_preds
    interleaved[:, 1::2] = melody_frames

    bos_col = torch.full(
        (melody_frames.shape[0], 1), tokenizer.bos_token,
        dtype=torch.long, device=sequences.device,
    )
    return torch.cat([bos_col, interleaved], dim=1)


def handle_data_conditioned_generation(
    args: Any,
    mel_source: str,
    chord_source: str,
    melody_model: Optional[torch.nn.Module],
    chord_model: Optional[torch.nn.Module],
    melody_tokenizer: HooktheoryTokenizer,
    chord_tokenizer: HooktheoryTokenizer,
    melody_dataloaders: Tuple,
    chord_dataloaders: Tuple,
    mle_melody_model: torch.nn.Module,
    mle_chord_model: torch.nn.Module,
    device: torch.device,
    data_perturbation: str = "none",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Handle data-conditioned generation (e.g. MLE chord conditioned on melody data)."""
    generated_all = []

    condition_type = "melody" if mel_source == "melody_data" else "chord"
    model_part = "melody" if condition_type == "chord" else "chord"
    model_to_use = chord_model if mel_source == "melody_data" else melody_model
    tokenizer_to_use = (
        chord_tokenizer if mel_source == "melody_data" else melody_tokenizer
    )
    dataloaders_to_use = (
        melody_dataloaders if model_part == "melody" else chord_dataloaders
    )

    _, val_dataloader = dataloaders_to_use
    print(f"Validation dataloader has {len(val_dataloader)} batches.")

    if args.num_batches == -1:
        num_batches_to_process = len(val_dataloader)
        print(
            f"Processing all {num_batches_to_process} batches from validation dataloader"
        )
    elif args.num_batches > len(val_dataloader):
        raise ValueError(
            f"num_batches ({args.num_batches}) cannot be greater than validation dataloader length ({len(val_dataloader)})"
        )
    else:
        num_batches_to_process = args.num_batches
        print(
            f"Processing {num_batches_to_process} batches from validation dataloader"
        )

    print(
        f"Generating {chord_source if mel_source == 'melody_data' else mel_source} conditioned on {condition_type} data"
    )

    if args.prompt_steps > 0:
        print(f"Using {args.prompt_steps} frames as prompts from data")

    for batch_idx, batch in enumerate(
        tqdm(val_dataloader, total=num_batches_to_process)
    ):
        if batch_idx >= num_batches_to_process:
            break

        sequences = batch["targets"].to(device)[:, 1:-1]
        sequences = replace_eos_with_pad(sequences, tokenizer_to_use)

        if data_perturbation != "none":
            sequences = apply_data_perturbation(
                sequences, data_perturbation, condition_type, tokenizer_to_use
            )

        decoder_preds = generate_from_data(
            model_to_use,
            sequences,
            tokenizer_to_use,
            prompt_steps=args.prompt_steps,
            target_seq_len=args.target_seq_len,
        )

        assert (
            decoder_preds[0, 0] == tokenizer_to_use.bos_token
        ), "BOS token should be at the beginning"

        online_model, semi_online_model = get_online_and_semi_online_model(
            model_part,
            mle_melody_model,
            mle_chord_model,
            melody_model,
            chord_model,
        )
        kl = compute_social_influence_kl(
            decoder_preds, online_model, semi_online_model
        )

        generated_all.append((decoder_preds.cpu(), kl.cpu()))

    return generated_all
