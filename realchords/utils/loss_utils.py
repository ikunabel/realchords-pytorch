"""Loss helpers shared across lit modules."""

import torch
import torch.nn.functional as F


def per_sample_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Per-sample mean cross-entropy loss, masking out `ignore_index` positions.

    Args:
        logits: (B, T, V)
        targets: (B, T)

    Returns:
        (B,) mean loss per sample, over its non-ignored positions.
    """
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(targets.shape)
    mask = (targets != ignore_index).float()
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
