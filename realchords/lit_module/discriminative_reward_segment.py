"""Lightning module for segment-based discriminative reward training."""

import argbind
import torch

from functools import partial

from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.model.reward_model import DiscriminativeReward
from realchords.dataset.segment_hooktheory import (
    create_segment_weighted_joint_dataset,
)
from realchords.dataset.weighted_joint_dataset import get_dataloader

GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

DiscriminativeReward = bind(DiscriminativeReward)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)
create_segment_weighted_joint_dataset = bind(
    create_segment_weighted_joint_dataset, without_prefix=True
)


@bind(without_prefix=True)
class LitDiscriminativeRewardSegment(LitDiscriminativeReward):
    """Discriminative reward training on fixed or sliding song segments."""

    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
    ):
        # Skip LitDiscriminativeReward.__init__ — it uses the random-crop joint dataset.
        super(LitDiscriminativeReward, self).__init__()

        train_dataset = create_segment_weighted_joint_dataset(split="train")
        val_dataset = create_segment_weighted_joint_dataset(split="valid")
        self.train_dataloader = get_dataloader(train_dataset)
        self.val_dataloader = get_dataloader(val_dataset, shuffle=False)

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer = train_dataset.tokenizer
        self.num_tokens = tokenizer.num_tokens
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.model = DiscriminativeReward(num_tokens=self.num_tokens)

        self.ce_loss = torch.nn.CrossEntropyLoss()

        if compile:
            self.model = torch.compile(self.model)
