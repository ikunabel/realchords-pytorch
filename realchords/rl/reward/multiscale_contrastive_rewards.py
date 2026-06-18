"""Multiscale contrastive reward for RL rollouts.

Scores each on-policy trajectory with:
- Legacy (256-frame) contrastive models on the full rollout
- Multiscale contrastive models via 50%-overlap sliding windows, averaged per scale
- Final reward = mean of legacy + w16 + w32 + w64 + w128 scale scores
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from realchords.model.reward_model import ContrastiveReward
from realchords.rl.experience_maker import Samples
from realchords.rl.reward.base import BaseRewardModel
from realchords.rl.utils import assign_reward_to_last_token
from realchords.utils.sequence_utils import add_bos_to_sequence, add_eos_to_sequence

MULTISCALE_OVERLAP_FRACTION = 0.5


def window_starts(num_frames: int, window_len: int, stride: int) -> List[int]:
    """Return frame start indices for sliding windows (matches segment dataset)."""
    if num_frames <= 0:
        return []
    if num_frames <= window_len:
        return [0]

    starts = list(range(0, num_frames - window_len + 1, stride))
    last_start = num_frames - window_len
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def split_interleaved_lanes(
    sequence: torch.Tensor,
    _pad_token_id: int,
    _bos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split interleaved rollout into model and context lanes (drop leading BOS)."""
    sequence = sequence[:, 1:].clone()
    model_tokens = sequence[:, ::2]
    context_tokens = sequence[:, 1::2]
    return model_tokens, context_tokens


def valid_frame_lengths(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Number of non-pad frames per batch row."""
    return (tokens != pad_token_id).sum(dim=1)


def encode_contrastive_pair(
    model_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add BOS/EOS and build masks (same as ContrastiveRewardFn)."""
    model_tokens = add_eos_to_sequence(
        model_tokens, pad_token_id, eos_token_id
    )
    context_tokens = add_eos_to_sequence(
        context_tokens, pad_token_id, eos_token_id
    )
    model_tokens = add_bos_to_sequence(model_tokens, bos_token_id)
    context_tokens = add_bos_to_sequence(context_tokens, bos_token_id)
    model_mask = model_tokens != pad_token_id
    context_mask = context_tokens != pad_token_id
    return model_tokens, context_tokens, model_mask, context_mask


def lanes_to_chord_melody(
    model_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    model_mask: torch.Tensor,
    context_mask: torch.Tensor,
    model_part: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_part == "chord":
        return model_tokens, context_tokens, model_mask, context_mask
    if model_part == "melody":
        return context_tokens, model_tokens, context_mask, model_mask
    raise ValueError(f"model_part must be 'chord' or 'melody', got {model_part}")


@torch.no_grad()
def contrastive_similarity(
    model: ContrastiveReward,
    chord_tokens: torch.Tensor,
    melody_tokens: torch.Tensor,
    chord_mask: torch.Tensor,
    melody_mask: torch.Tensor,
) -> torch.Tensor:
    """Dot product of normalized melody/chord embeddings. Shape: [B]."""
    device = next(model.parameters()).device
    chord_embed, melody_embed, _ = model(
        chord=chord_tokens.to(device),
        melody=melody_tokens.to(device),
        chord_mask=chord_mask.to(device),
        melody_mask=melody_mask.to(device),
    )
    return (chord_embed * melody_embed).sum(-1)


def gather_sliding_windows(
    model_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    valid_lens: torch.Tensor,
    window_len: int,
    stride: int,
    pad_token_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Stack all sliding windows in the batch for one contrastive forward pass.

    Returns:
        sample_idx: [N] source row per window
        model_windows: [N, window_len]
        context_windows: [N, window_len]
        or (None, None, None) when every sequence is empty.
    """
    batch_size, max_frames = model_tokens.shape
    device = model_tokens.device

    sample_indices: List[int] = []
    start_indices: List[int] = []
    valid_lens_list = valid_lens.tolist()
    for idx in range(batch_size):
        valid_len = int(valid_lens_list[idx])
        if valid_len <= 0:
            continue
        for start in window_starts(valid_len, window_len, stride):
            sample_indices.append(idx)
            start_indices.append(start)

    if not sample_indices:
        return None, None, None

    sample_idx = torch.tensor(sample_indices, device=device, dtype=torch.long)
    starts = torch.tensor(start_indices, device=device, dtype=torch.long)
    crop_lens = torch.minimum(
        torch.full_like(starts, window_len),
        valid_lens[sample_idx] - starts,
    )

    positions = starts.unsqueeze(1) + torch.arange(
        window_len, device=device, dtype=torch.long
    ).unsqueeze(0)
    valid_pos = torch.arange(window_len, device=device).unsqueeze(0) < crop_lens.unsqueeze(
        1
    )

    source_model = model_tokens.index_select(0, sample_idx)
    source_context = context_tokens.index_select(0, sample_idx)
    positions = positions.clamp(max=max_frames - 1)

    model_windows = source_model.gather(1, positions)
    context_windows = source_context.gather(1, positions)
    model_windows = model_windows.masked_fill(~valid_pos, pad_token_id)
    context_windows = context_windows.masked_fill(~valid_pos, pad_token_id)
    return sample_idx, model_windows, context_windows


class MultiscaleContrastiveRewardFn(BaseRewardModel):
    """Combine legacy full-rollout and multiscale sliding-window contrastive scores."""

    def __init__(
        self,
        legacy_models: Sequence[ContrastiveReward],
        multiscale_models: Sequence[ContrastiveReward],
        window_lens: Sequence[int],
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        model_part: str,
        max_windows_per_forward: int = 8192,
    ):
        super().__init__()
        if len(multiscale_models) != len(window_lens):
            raise ValueError(
                "multiscale_models and window_lens must have the same length."
            )
        if not legacy_models:
            raise ValueError("At least one legacy contrastive model is required.")
        if not multiscale_models:
            raise ValueError("At least one multiscale contrastive model is required.")

        self.legacy_models = nn.ModuleList(list(legacy_models))
        self.multiscale_models = nn.ModuleList(list(multiscale_models))
        self.window_lens = list(window_lens)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_part = model_part
        self.max_windows_per_forward = max_windows_per_forward

    @property
    def device(self) -> torch.device:
        return next(self.legacy_models[0].parameters()).device

    @torch.no_grad()
    def _score_encoded_batch(
        self,
        model: ContrastiveReward,
        chord_tokens: torch.Tensor,
        melody_tokens: torch.Tensor,
        chord_mask: torch.Tensor,
        melody_mask: torch.Tensor,
    ) -> torch.Tensor:
        return contrastive_similarity(
            model, chord_tokens, melody_tokens, chord_mask, melody_mask
        )

    @torch.no_grad()
    def _score_full_legacy(
        self,
        model: ContrastiveReward,
        model_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full rollout in one batched forward (same as ContrastiveRewardFn)."""
        m_enc, c_enc, m_mask, c_mask = encode_contrastive_pair(
            model_tokens,
            context_tokens,
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        )
        chord_t, melody_t, chord_mask, melody_mask = lanes_to_chord_melody(
            m_enc, c_enc, m_mask, c_mask, self.model_part
        )
        scores = self._score_encoded_batch(
            model, chord_t, melody_t, chord_mask, melody_mask
        )
        return scores * (valid_lens > 0).to(scores.dtype)

    @torch.no_grad()
    def _score_sliding_average(
        self,
        model: ContrastiveReward,
        model_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        valid_lens: torch.Tensor,
        window_len: int,
    ) -> torch.Tensor:
        stride = max(1, int(window_len * MULTISCALE_OVERLAP_FRACTION))
        batch_size = model_tokens.shape[0]
        device = model_tokens.device
        scores = torch.zeros(batch_size, device=device, dtype=torch.float32)

        sample_idx, model_windows, context_windows = gather_sliding_windows(
            model_tokens,
            context_tokens,
            valid_lens,
            window_len=window_len,
            stride=stride,
            pad_token_id=self.pad_token_id,
        )
        if sample_idx is None:
            return scores

        num_windows = sample_idx.shape[0]
        chunk_starts = range(0, num_windows, self.max_windows_per_forward)
        per_sample_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        per_sample_count = torch.zeros(batch_size, device=device, dtype=torch.float32)

        for chunk_start in chunk_starts:
            chunk_end = min(chunk_start + self.max_windows_per_forward, num_windows)
            chunk_sample_idx = sample_idx[chunk_start:chunk_end]
            chunk_model = model_windows[chunk_start:chunk_end]
            chunk_context = context_windows[chunk_start:chunk_end]

            m_enc, c_enc, m_mask, c_mask = encode_contrastive_pair(
                chunk_model,
                chunk_context,
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            )
            chord_t, melody_t, chord_mask, melody_mask = lanes_to_chord_melody(
                m_enc, c_enc, m_mask, c_mask, self.model_part
            )
            window_scores = self._score_encoded_batch(
                model, chord_t, melody_t, chord_mask, melody_mask
            )
            ones = torch.ones_like(window_scores)
            per_sample_sum.scatter_add_(0, chunk_sample_idx, window_scores)
            per_sample_count.scatter_add_(0, chunk_sample_idx, ones)

        scores = per_sample_sum / per_sample_count.clamp(min=1)
        return scores

    @torch.no_grad()
    def forward(self, samples: Samples) -> Dict[str, torch.Tensor]:
        sequence = samples.sequences
        action_mask = samples.action_mask

        model_tokens, context_tokens = split_interleaved_lanes(
            sequence, self.pad_token_id, self.bos_token_id
        )
        valid_lens = valid_frame_lengths(model_tokens, self.pad_token_id)

        legacy_scores = []
        for model in self.legacy_models:
            legacy_scores.append(
                self._score_full_legacy(model, model_tokens, context_tokens, valid_lens)
            )
        legacy_mean = torch.stack(legacy_scores, dim=0).mean(dim=0)

        scale_scores = [legacy_mean]
        metrics: Dict[str, torch.Tensor] = {
            "multiscale_contrastive_w256": legacy_mean.detach(),
        }

        for window_len, model in zip(self.window_lens, self.multiscale_models):
            scale_score = self._score_sliding_average(
                model,
                model_tokens,
                context_tokens,
                valid_lens,
                window_len=window_len,
            )
            scale_scores.append(scale_score)
            metrics[f"multiscale_contrastive_w{window_len}"] = scale_score.detach()

        combined = torch.stack(scale_scores, dim=0).mean(dim=0)
        metrics["multiscale_contrastive_combined"] = combined.detach()

        return {
            "reward": assign_reward_to_last_token(
                combined.to(sequence.device), action_mask
            ),
            **metrics,
        }
