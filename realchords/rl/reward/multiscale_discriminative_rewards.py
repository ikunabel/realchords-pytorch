"""Multiscale discriminative reward for RL rollouts.

Scores each on-policy trajectory with:
- Legacy (256-frame) discriminative models on the full rollout
- Multiscale discriminative models via 50%-overlap sliding windows, averaged per scale
- Final reward = mean of legacy + w16 + w32 + w64 + w128 scale scores

Mirrors realchords/rl/reward/multiscale_contrastive_rewards.py's structure exactly
(same sliding-window gathering, same legacy+per-scale averaging), reusing its
generic (model-agnostic) helpers. The only real difference is how a pair is
scored: contrastive dot-products two separately-encoded embeddings, while
discriminative concatenates [melody, bos (sep), chord] into one sequence and
reads off softmax(logits)[:, 1] (probability of being a real pair) from a
single encoder -- same input construction as DiscriminativeRewardFn.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from realchords.model.reward_model import DiscriminativeReward
from realchords.rl.experience_maker import Samples
from realchords.rl.reward.base import BaseRewardModel
from realchords.rl.reward.multiscale_contrastive_rewards import (
    gather_sliding_windows,
    lanes_to_chord_melody,
    split_interleaved_lanes,
    valid_frame_lengths,
)
from realchords.rl.utils import assign_reward_to_last_token
from realchords.utils.sequence_utils import add_bos_to_sequence, add_eos_to_sequence

MULTISCALE_OVERLAP_FRACTION = 0.5


def encode_discriminative_pair(
    model_tokens: torch.Tensor,
    context_tokens: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    model_part: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add BOS/EOS and concatenate [melody, bos (sep), chord] (same as DiscriminativeRewardFn)."""
    model_tokens = add_eos_to_sequence(model_tokens, pad_token_id, eos_token_id)
    context_tokens = add_eos_to_sequence(context_tokens, pad_token_id, eos_token_id)
    model_tokens = add_bos_to_sequence(model_tokens, bos_token_id)
    context_tokens = add_bos_to_sequence(context_tokens, bos_token_id)

    chord_tokens, melody_tokens, _, _ = lanes_to_chord_melody(
        model_tokens,
        context_tokens,
        model_tokens != pad_token_id,
        context_tokens != pad_token_id,
        model_part,
    )

    bos_token = torch.full_like(
        melody_tokens[:, :1], bos_token_id, dtype=melody_tokens.dtype
    )
    input_tokens = torch.cat([melody_tokens, bos_token, chord_tokens], dim=1)
    input_mask = input_tokens != pad_token_id
    return input_tokens, input_mask


class MultiscaleDiscriminativeRewardFn(BaseRewardModel):
    """Combine legacy full-rollout and multiscale sliding-window discriminative scores."""

    def __init__(
        self,
        legacy_models: Sequence[DiscriminativeReward],
        multiscale_models: Sequence[DiscriminativeReward],
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
            raise ValueError("At least one legacy discriminative model is required.")
        if not multiscale_models:
            raise ValueError("At least one multiscale discriminative model is required.")

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
        self, model: DiscriminativeReward, input_tokens: torch.Tensor, input_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = model(input_tokens, input_mask)
        # logits come out bf16 (out_proj is a plain nn.Linear inside the model's
        # own bf16 autocast context) -- cast up front so downstream scatter_add_
        # against a float32 accumulator (_score_sliding_average) doesn't dtype-mismatch.
        return F.softmax(logits, dim=-1)[:, 1].float()

    @torch.no_grad()
    def _score_full_legacy(
        self,
        model: DiscriminativeReward,
        model_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full rollout in one batched forward (same as DiscriminativeRewardFn)."""
        input_tokens, input_mask = encode_discriminative_pair(
            model_tokens,
            context_tokens,
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.model_part,
        )
        scores = self._score_encoded_batch(model, input_tokens, input_mask)
        return scores * (valid_lens > 0).to(scores.dtype)

    @torch.no_grad()
    def _score_sliding_average(
        self,
        model: DiscriminativeReward,
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

            input_tokens, input_mask = encode_discriminative_pair(
                chunk_model,
                chunk_context,
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.model_part,
            )
            window_scores = self._score_encoded_batch(model, input_tokens, input_mask)
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
            "multiscale_discriminative_w256": legacy_mean.detach(),
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
            metrics[f"multiscale_discriminative_w{window_len}"] = scale_score.detach()

        combined = torch.stack(scale_scores, dim=0).mean(dim=0)
        metrics["multiscale_discriminative_combined"] = combined.detach()

        return {
            "reward": assign_reward_to_last_token(
                combined.to(sequence.device), action_mask
            ),
            **metrics,
        }
