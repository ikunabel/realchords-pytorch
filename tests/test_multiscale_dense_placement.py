"""Checks the dense reward placement of the multiscale reward functions.

Stub reward models return deterministic scores, so only the placement logic is
tested:
  1. dense: each sequence's summed reward == the combined score (sparse mode);
  2. dense: window rewards sit on model-lane positions (even action indices)
     inside the sequence, legacy share on the last action token;
  3. default (sparse) mode: a single reward on the last token, as before.

Run: python tests/test_multiscale_dense_placement.py
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from realchords.rl.reward.multiscale_contrastive_rewards import MultiscaleContrastiveRewardFn
from realchords.rl.reward.multiscale_discriminative_rewards import MultiscaleDiscriminativeRewardFn
from realchords.rl.utils import assign_reward_to_last_token

PAD, BOS, EOS = 0, 1, 2


class StubContrastive(MultiscaleContrastiveRewardFn):
    def _score_encoded_batch(self, model, chord_tokens, melody_tokens, chord_mask, melody_mask):
        # depends on the window's content, so different windows get different scores
        return (chord_tokens * chord_mask).float().sum(1) * 1e-3 + model.weight.sum() * 0


class StubDiscriminative(MultiscaleDiscriminativeRewardFn):
    def _score_encoded_batch(self, model, input_tokens, input_mask):
        return torch.sigmoid((input_tokens * input_mask).float().sum(1) * 1e-3)


def make_samples(valid_frames=(64, 40, 17)):
    """Interleaved [BOS, c0, m0, c1, m1, ...], padded; action mask over sequence[1:]."""
    max_frames = max(valid_frames)
    seq_len = 1 + 2 * max_frames
    sequences = torch.full((len(valid_frames), seq_len), PAD, dtype=torch.long)
    sequences[:, 0] = BOS
    for row, n in enumerate(valid_frames):
        chords = 10 + torch.randint(0, 50, (n,))
        melody = 100 + torch.randint(0, 50, (n,))
        sequences[row, 1:1 + 2 * n:2] = chords
        sequences[row, 2:2 + 2 * n:2] = melody
    action_mask = sequences[:, 1:] != PAD
    return SimpleNamespace(sequences=sequences, action_mask=action_mask), valid_frames


def check(cls, name):
    torch.manual_seed(0)
    samples, valid_frames = make_samples()
    stub = lambda: nn.Linear(1, 1)
    kwargs = dict(
        legacy_models=[stub(), stub()], multiscale_models=[stub(), stub()],
        window_lens=[16, 32], pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS,
        model_part="chord",
    )
    sparse = cls(**kwargs)(samples)
    dense = cls(**kwargs, dense_placement=True)(samples)
    combined = sparse[[k for k in sparse if k.endswith("_combined")][0]]

    # 3. default mode unchanged: one reward on the last token
    expected_sparse = assign_reward_to_last_token(combined, samples.action_mask)
    assert torch.allclose(sparse["reward"], expected_sparse), f"{name}: sparse mode changed"
    # 1. dense sums equal the combined score
    assert torch.allclose(dense["reward"].sum(1), combined, atol=1e-5), (
        f"{name}: dense sums {dense['reward'].sum(1)} != combined {combined}")
    # 2. placement: all rewarded positions are model-lane frames within the sequence,
    #    except the last action token (legacy share)
    last = samples.action_mask.size(1) - 1 - samples.action_mask.long().fliplr().argmax(1)
    for row, n in enumerate(valid_frames):
        pos = dense["reward"][row].nonzero().flatten()
        window_pos = pos[pos != last[row]]
        assert (window_pos % 2 == 0).all(), f"{name}: reward on a non-model token {window_pos}"
        assert (window_pos <= 2 * (n - 1)).all(), f"{name}: reward beyond the sequence"
        assert len(window_pos) > 1, f"{name}: rewards not spread over windows"
        print(f"  {name} row {row} ({n} frames): {len(pos)} rewarded positions, "
              f"sum {dense['reward'][row].sum():.5f} == combined {combined[row]:.5f}")
    print(f"{name}: OK")


if __name__ == "__main__":
    check(StubContrastive, "contrastive")
    check(StubDiscriminative, "discriminative")
    print("all checks passed")
