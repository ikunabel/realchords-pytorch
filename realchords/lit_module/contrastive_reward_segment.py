"""Lightning module for segment-based contrastive reward training."""

import argbind
import torch
import torch.nn.functional as F

from functools import partial

from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.model.reward_model import ContrastiveReward
from realchords.dataset.segment_hooktheory import (
    create_segment_weighted_joint_dataset,
)
from realchords.dataset.weighted_joint_dataset import get_dataloader

GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

ContrastiveReward = bind(ContrastiveReward)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)
create_segment_weighted_joint_dataset = bind(
    create_segment_weighted_joint_dataset, without_prefix=True
)


@bind(without_prefix=True)
class LitContrastiveRewardSegment(LitContrastiveReward):
    """Contrastive reward training on fixed or sliding song segments."""

    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        mask_same_song_negatives: bool = False,
        mask_overlapping_negatives: bool = False,
    ):
        # Skip LitContrastiveReward.__init__ — it uses the random-crop joint dataset.
        super(LitContrastiveReward, self).__init__()

        train_dataset = create_segment_weighted_joint_dataset(split="train")
        val_dataset = create_segment_weighted_joint_dataset(split="valid")
        self.train_dataloader = get_dataloader(train_dataset)
        self.val_dataloader = get_dataloader(val_dataset, shuffle=False)

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples
        self.mask_same_song_negatives = mask_same_song_negatives
        self.mask_overlapping_negatives = mask_overlapping_negatives

        tokenizer = train_dataset.tokenizer
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.model = ContrastiveReward(num_tokens=self.num_tokens)

        if compile:
            self.model = torch.compile(self.model)

    def _song_ids_from_batch(self, batch, device: torch.device) -> torch.Tensor:
        urls = batch["song_url"]
        if isinstance(urls, str):
            urls = [urls]
        id_map: dict[str, int] = {}
        ids = []
        for url in urls:
            if url not in id_map:
                id_map[url] = len(id_map)
            ids.append(id_map[url])
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _negative_mask_from_batch(
        self, batch, device: torch.device, num_logits: int
    ) -> torch.Tensor | None:
        """Build off-diagonal mask for pairs that should not be treated as negatives."""
        if not self.mask_same_song_negatives and not self.mask_overlapping_negatives:
            return None

        off_diag = ~torch.eye(num_logits, dtype=torch.bool, device=device)
        negative_mask = torch.zeros(
            num_logits, num_logits, dtype=torch.bool, device=device
        )

        song_ids = self._song_ids_from_batch(batch, device)
        same_song = song_ids.unsqueeze(0) == song_ids.unsqueeze(1)

        if self.mask_overlapping_negatives and "segment_start" in batch:
            starts = torch.as_tensor(batch["segment_start"], device=device)
            ends = torch.as_tensor(batch["segment_end"], device=device)
            overlaps = (starts[:, None] < ends[None, :]) & (
                starts[None, :] < ends[:, None]
            )
            negative_mask |= off_diag & same_song & overlaps

        if self.mask_same_song_negatives:
            negative_mask |= off_diag & same_song

        return negative_mask if negative_mask.any() else None

    def contrastive_loss(
        self,
        melody_embed,
        chord_embed,
        logit_scale,
        negative_mask: torch.Tensor | None = None,
    ):
        """In-batch InfoNCE with optional masking of invalid negative pairs."""
        logits_per_melody = logit_scale * melody_embed @ chord_embed.T
        num_logits = logits_per_melody.shape[0]
        labels = torch.arange(
            num_logits, device=melody_embed.device, dtype=torch.long
        )

        if negative_mask is not None:
            logits_per_melody = logits_per_melody.masked_fill(
                negative_mask,
                float("-inf"),
            )

        loss_melody = F.cross_entropy(logits_per_melody, labels)
        loss_chord = F.cross_entropy(logits_per_melody.T, labels)
        return (loss_melody + loss_chord) / 2

    def training_step(self, batch, batch_idx):
        melody_tokens, chord_tokens, melody_mask, chord_mask = self.get_inputs(
            batch
        )
        chord_embed, melody_embed, logit_scale = self.model(
            chord=chord_tokens,
            melody=melody_tokens,
            chord_mask=chord_mask,
            melody_mask=melody_mask,
        )

        negative_mask = self._negative_mask_from_batch(
            batch, melody_embed.device, melody_embed.shape[0]
        )
        loss = self.contrastive_loss(
            melody_embed, chord_embed, logit_scale, negative_mask=negative_mask
        )
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        melody_tokens, chord_tokens, melody_mask, chord_mask = self.get_inputs(
            batch
        )
        chord_embed, melody_embed, logit_scale = self.model(
            chord=chord_tokens,
            melody=melody_tokens,
            chord_mask=chord_mask,
            melody_mask=melody_mask,
        )

        negative_mask = self._negative_mask_from_batch(
            batch, melody_embed.device, melody_embed.shape[0]
        )
        loss = self.contrastive_loss(
            melody_embed, chord_embed, logit_scale, negative_mask=negative_mask
        )
        self._log_dict({"val/loss": loss})
        return loss
