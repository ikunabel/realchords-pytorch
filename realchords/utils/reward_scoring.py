"""Score evaluated sequences with frozen reward models (custom_evaluation.py's reward metrics).

The contrastive and discriminative reward models used for RL are also usable as *metrics*: for
each sequence they answer "do these chords fit this melody?" (contrastive: cosine similarity of
the melody and chord embeddings) and "does this pair look real?" (discriminative: probability of
being a real pair). Scoring the ground truth alongside the models gives the reference level, so a
model's score can be read as a fraction of it -- and a model scoring *above* ground truth is a
reward-hacking signal rather than a quality signal (relevant for the RL policies, which were
trained against reward models of this kind; for MLE models the score is independent).

Vocabulary: a reward model's chord embedding table is fixed to the vocabulary it was trained on,
while the sequences being scored use their own model's vocabulary. Token ids differ between the
two (the chord list is re-sorted whenever it grows), so ids are translated through chord *names*,
which are stable: id -> name in the source vocabulary -> id in the reward model's vocabulary.
This is exact as long as every name exists on both sides, which is checked; otherwise scoring is
refused rather than silently scoring the wrong chords.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import argbind
import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.rl.reward.model_based_rewards import (
    ContrastiveRewardFn,
    DiscriminativeRewardFn,
)

_LIT_CLS = {"contrastive": LitContrastiveReward, "discriminative": LitDiscriminativeReward}
_FN_CLS = {"contrastive": ContrastiveRewardFn, "discriminative": DiscriminativeRewardFn}


def load_reward_model(checkpoint: str, kind: str, device: torch.device,
                      fallback_tokenizer: Optional[HooktheoryTokenizer] = None):
    """Load a reward checkpoint and the tokenizer of the vocabulary it was trained on.

    Avoids load_lit_model, which would build the run's training dataloaders (minutes for large
    mixes) just to read the weights.

    fallback_tokenizer: used when the checkpoint's args.yml has no chord_names_path (older reward
    runs predate that key) and its embedding table has exactly as many rows -- then the evaluated
    vocabulary *is* the checkpoint's vocabulary and no remapping is needed.
    """
    ckpt = Path(checkpoint)
    args = argbind.load_args(ckpt.parent / "args.yml")
    args["compile"] = False
    with argbind.scope(args):
        lit = _LIT_CLS[kind]()
    state = torch.load(ckpt, weights_only=True, map_location="cpu")["state_dict"]
    lit.load_state_dict({k.replace("_orig_mod.", "").replace("._orig_mod", ""): v
                         for k, v in state.items()})
    model = lit.model.to(device).eval()
    if not hasattr(model, "device"):
        model.device = device
    embed_rows = next(v.shape[0] for k, v in state.items() if k.endswith("token_emb.emb.weight"))
    chord_names_path = args.get("chord_names_path")
    if not chord_names_path or not Path(chord_names_path).exists():
        if fallback_tokenizer is not None and fallback_tokenizer.num_tokens == embed_rows:
            return model, fallback_tokenizer
        raise FileNotFoundError(
            f"{checkpoint}: its args.yml chord_names_path ({chord_names_path}) is missing and the "
            f"evaluated vocabulary ({getattr(fallback_tokenizer, 'num_tokens', None)} tokens) does "
            f"not match the checkpoint's embedding table ({embed_rows}), so the vocabulary it was "
            "trained on can't be reconstructed."
        )
    with open(chord_names_path, encoding="utf-8") as fh:
        tokenizer = HooktheoryTokenizer(chord_names=json.load(fh))
    if tokenizer.num_tokens != embed_rows:
        raise ValueError(
            f"{checkpoint}: chord vocab at {chord_names_path} has {tokenizer.num_tokens} tokens but "
            f"the checkpoint's embedding table has {embed_rows} -- the vocab file changed since "
            "training, so its chord names can no longer be matched to the model's token ids."
        )
    return model, tokenizer


def build_token_remap(src: HooktheoryTokenizer, dst: HooktheoryTokenizer) -> Optional[torch.Tensor]:
    """Lookup table mapping every ``src`` token id to the id of the same name in ``dst``.

    None when the vocabularies are identical (no remapping needed). Raises if a name is missing
    on the ``dst`` side, i.e. the reward model never saw that chord.
    """
    if src.name_to_id == dst.name_to_id:
        return None
    missing = [name for name in src.name_to_id if name not in dst.name_to_id]
    if missing:
        raise ValueError(
            f"{len(missing)} token(s) of the evaluated vocabulary are absent from the reward "
            f"model's vocabulary (e.g. {missing[:3]}), so its scores would not be comparable."
        )
    table = torch.zeros(src.num_tokens, dtype=torch.long)
    for name, src_id in src.name_to_id.items():
        table[src_id] = dst.name_to_id[name]
    return table


@torch.no_grad()
def score_sequences(
    tensor: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    checkpoints: Sequence[str],
    kind: str,
    device: torch.device,
    batch_size: int = 128,
) -> List[float]:
    """Mean score per checkpoint over all sequences (BOS + interleaved chord/melody tokens)."""
    means: List[float] = []
    for checkpoint in checkpoints:
        model, reward_tokenizer = load_reward_model(checkpoint, kind, device, tokenizer)
        table = build_token_remap(tokenizer, reward_tokenizer)
        fn = _FN_CLS[kind](
            model=model,
            pad_token_id=reward_tokenizer.pad_token,
            bos_token_id=reward_tokenizer.bos_token,
            eos_token_id=reward_tokenizer.eos_token,
            model_part="chord",
        )
        total, count = 0.0, 0
        for start in range(0, tensor.size(0), batch_size):
            batch = tensor[start : start + batch_size]
            if batch.size(1) == 0:
                continue
            if table is not None:
                batch = table[batch]
            samples = SimpleNamespace(
                sequences=batch.to(device),
                action_mask=torch.ones_like(batch[:, 1:], dtype=torch.bool, device=device),
            )
            scores = fn(samples)["reward"].sum(dim=1).float().cpu()
            total += float(scores.sum())
            count += scores.numel()
        del model, fn
        torch.cuda.empty_cache()
        means.append(total / max(1, count))
    return means


def reward_metrics(
    tensor: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    contrastive_checkpoints: Sequence[str],
    discriminative_checkpoints: Sequence[str],
    device: torch.device,
) -> Dict[str, float]:
    """{reward_contrastive_score, reward_discriminative_score} -- mean over sequences, then over
    the checkpoints of that kind (they are seed replicates of one model)."""
    out: Dict[str, float] = {}
    for kind, checkpoints in (("contrastive", contrastive_checkpoints),
                              ("discriminative", discriminative_checkpoints)):
        if not checkpoints:
            continue
        means = score_sequences(tensor, tokenizer, checkpoints, kind, device)
        out[f"reward_{kind}_score"] = sum(means) / len(means)
    return out
