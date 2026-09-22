#!/usr/bin/env python3
"""How well do contrastive reward models separate true from mismatched pairs?

Retrieval (eval_reward_recall.py, R@K) asks whether a model can rank the one
true chord window first among ~all test songs -- a hard task for short windows,
since many songs contain similar short fragments. During RL the contrastive
reward only scores the given pair (dot product of the chord and melody
embeddings, see ContrastiveRewardFn), so what matters there is whether true
pairs score higher than mismatched ones. This script measures that, on the
same test windows as eval_reward_recall.py (one deterministic crop per test
song at the checkpoint's own max_len), with every other song's chord window as
a mismatched pair:

  - pair_auc:  P(score of a true pair > score of a random mismatched pair)
               over all pairs (0.5 = no separation, 1 = perfect). Rewards are
               compared across rollouts during RL, so this global version is
               the one closest to how the reward is used.
  - mean_rank: per melody, the fraction of mismatched chord windows scoring
               below the true one, averaged over melodies (a soft R@1; 0.5 =
               chance).
  - d_prime:   (mean true - mean mismatched) / sqrt(mean of both variances).

Key-matched negatives (Hooktheory only, via the section's annotated key):
  - *_same_key:  mismatched pairs restricted to chord windows of other test
                 songs annotated in the *same key* (tonic + mode). The model
                 can't reject these just by noticing a key clash, so this
                 measures whether it judges fit to the specific melody.
  - *_diff_key:  restricted to other songs in a different key.
If *_diff_key is much higher than *_same_key, the model mostly detects key
mismatch. Songs with key changes or without a key partner are left out of the
same-key numbers.

Each --model path is resolved to its run's lowest-val-loss checkpoint by
default (same as eval_reward_recall.py). See journal/REWARD_MODELS.md.

Usage (from the repo root):
    python scripts/eval/recall/eval_reward_pair_separation.py --eval_augmentation off \\
        --model "Full (256)=.../contrastive_reward/step=8000.ckpt" \\
        --model "w16 sliding=.../contrastive_reward_w16_sliding_2o1vixac/step=18500.ckpt"
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import argbind
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reward_recall import (  # noqa: E402
    build_test_dataloader,
    parse_model_arg,
    resolve_best_checkpoint,
)
from realchords.lit_module.contrastive_reward import LitContrastiveReward  # noqa: E402
from realchords.utils.experiment_utils import DATASET_CACHE_DIRS  # noqa: E402
from realchords.utils.inference_utils import load_lit_model  # noqa: E402


def load_hooktheory_keys(split: str = "test") -> Dict[str, tuple]:
    """Hooktheory section id -> (tonic, scale intervals), single-key sections only."""
    keys = {}
    path = Path(DATASET_CACHE_DIRS["hooktheory"]) / f"{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            ks = item["annotations"]["keys"]
            if len(ks) == 1:
                keys[item["hooktheory"]["id"]] = (
                    ks[0]["tonic_pitch_class"], tuple(ks[0]["scale_degree_intervals"])
                )
    return keys


@torch.no_grad()
def encode_with_ids(lit_module, dataloader, device):
    """Like eval_reward_recall.encode_contrastive, but also returns each row's song_id."""
    model = lit_module.model.to(device).eval()
    melody_embeds, chord_embeds, song_ids = [], [], []
    for batch in dataloader:
        melody_tokens, chord_tokens, melody_mask, chord_mask = lit_module.get_inputs(batch)
        chord_embed, melody_embed, _ = model(
            chord=chord_tokens.to(device),
            melody=melody_tokens.to(device),
            chord_mask=chord_mask.to(device),
            melody_mask=melody_mask.to(device),
        )
        melody_embeds.append(melody_embed.float().cpu())
        chord_embeds.append(chord_embed.float().cpu())
        song_ids.extend(batch.get("song_id", [None] * melody_tokens.shape[0]))
    return torch.cat(melody_embeds), torch.cat(chord_embeds), song_ids


def restricted_metrics(scores: torch.Tensor, negative_mask: torch.Tensor, suffix: str) -> Dict[str, float]:
    """pair_auc / mean_rank / d_prime using only the negatives in negative_mask
    (rows without any such negative are skipped)."""
    rows = negative_mask.any(dim=1)
    if not rows.any():
        return {}
    true = scores.diagonal()[rows]
    neg = scores[rows][negative_mask[rows]]
    sorted_neg = neg.sort().values
    below = torch.searchsorted(sorted_neg, true.contiguous(), right=False).float()
    ties = torch.searchsorted(sorted_neg, true.contiguous(), right=True).float() - below
    per_row_below = ((scores[rows] < true.unsqueeze(1)) & negative_mask[rows]).sum(dim=1).float()
    per_row_n = negative_mask[rows].sum(dim=1).float()
    return {
        f"pair_auc_{suffix}": ((below + 0.5 * ties) / neg.numel()).mean().item(),
        f"mean_rank_{suffix}": (per_row_below / per_row_n).mean().item(),
        f"d_prime_{suffix}": (
            (true.mean() - neg.mean()) / torch.sqrt(0.5 * (true.var() + neg.var()))
        ).item(),
        f"rows_{suffix}": int(rows.sum().item()),
        f"negatives_per_row_{suffix}": per_row_n.mean().item(),
    }


def separation_metrics(melody_embeds: torch.Tensor, chord_embeds: torch.Tensor) -> Dict[str, float]:
    """Scores are the RL reward: dot(chord_embed, melody_embed). Row i = melody i."""
    scores = melody_embeds @ chord_embeds.T
    n = scores.shape[0]
    true = scores.diagonal()
    off_diag = ~torch.eye(n, dtype=torch.bool)
    mismatched = scores[off_diag]

    sorted_mismatched = mismatched.sort().values
    below = torch.searchsorted(sorted_mismatched, true.contiguous(), right=False).float()
    ties = torch.searchsorted(sorted_mismatched, true.contiguous(), right=True).float() - below
    pair_auc = ((below + 0.5 * ties) / mismatched.numel()).mean().item()

    per_melody_below = (scores < true.unsqueeze(1)).sum(dim=1).float()
    mean_rank = (per_melody_below / (n - 1)).mean().item()

    d_prime = (
        (true.mean() - mismatched.mean())
        / torch.sqrt(0.5 * (true.var() + mismatched.var()))
    ).item()
    return {
        "pair_auc": pair_auc,
        "mean_rank": mean_rank,
        "d_prime": d_prime,
        "mean_true": true.mean().item(),
        "mean_mismatched": mismatched.mean().item(),
        "num_songs": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=parse_model_arg, action="append", required=True,
                        help="LABEL=PATH to a contrastive checkpoint (or run dir). Repeatable.")
    parser.add_argument("--eval_datasets", nargs="+", default=None,
                        help="Override the test gallery datasets for every checkpoint.")
    parser.add_argument("--eval_augmentation", choices=["on", "off"], default=None)
    parser.add_argument("--no_best_checkpoint", action="store_true",
                        help="Use exactly the given checkpoint files.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="logs/eval/recall")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_aug = None if args.eval_augmentation is None else args.eval_augmentation == "on"

    hooktheory_keys = load_hooktheory_keys("test")
    results, paths = {}, {}
    for label, path in args.model:
        path = path if args.no_best_checkpoint else resolve_best_checkpoint(path)
        paths[label] = path
        print(f"\n=== {label}: {path}")
        ckpt_args = argbind.load_args(Path(path).parent / "args.yml")
        lit_module = load_lit_model(path, lit_module_cls=LitContrastiveReward, compile=False,
                                    return_lit_module=True)
        dataloader = build_test_dataloader(ckpt_args, args.batch_size, args.num_workers,
                                           args.eval_datasets, eval_aug)
        melody_embeds, chord_embeds, song_ids = encode_with_ids(lit_module, dataloader, device)
        results[label] = separation_metrics(melody_embeds, chord_embeds)
        row_keys = [hooktheory_keys.get(sid) for sid in song_ids]
        if any(k is not None for k in row_keys):
            scores = melody_embeds @ chord_embeds.T
            n = len(row_keys)
            key_ids = {k: i for i, k in enumerate(sorted({k for k in row_keys if k is not None}))}
            kid = torch.tensor([key_ids[k] if k is not None else -1 for k in row_keys])
            keyed = kid >= 0
            both = keyed.unsqueeze(1) & keyed.unsqueeze(0) & ~torch.eye(n, dtype=torch.bool)
            same = both & (kid.unsqueeze(1) == kid.unsqueeze(0))
            results[label].update(restricted_metrics(scores, same, "same_key"))
            results[label].update(restricted_metrics(scores, both & ~same, "diff_key"))
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in results[label].items()))

    cols = ["pair_auc", "pair_auc_diff_key", "pair_auc_same_key", "mean_rank_same_key",
            "d_prime", "d_prime_diff_key", "d_prime_same_key", "rows_same_key",
            "negatives_per_row_same_key", "num_songs"]
    print(f"\n{'model':28s}" + "".join(f"{c:>16s}" for c in cols))
    for label, m in results.items():
        print(f"{label:28s}" + "".join(
            f"{'-':>16s}" if c not in m else f"{m[c]:>16.4f}" if isinstance(m[c], float) else f"{m[c]:>16d}"
            for c in cols))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pair_separation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "model_path"] + cols)
        for label, m in results.items():
            writer.writerow([label, paths[label]] + [m.get(c) for c in cols])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
