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

Each --model path is resolved to its run's lowest-val-loss checkpoint by
default (same as eval_reward_recall.py). See journal/REWARD_MODELS.md.

Usage (from the repo root):
    python scripts/eval/recall/eval_reward_pair_separation.py --eval_augmentation off \\
        --model "Full (256)=.../contrastive_reward/step=8000.ckpt" \\
        --model "w16 sliding=.../contrastive_reward_w16_sliding_2o1vixac/step=18500.ckpt"
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import argbind
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_reward_recall import (  # noqa: E402
    build_test_dataloader,
    encode_contrastive,
    parse_model_arg,
    resolve_best_checkpoint,
)
from realchords.lit_module.contrastive_reward import LitContrastiveReward  # noqa: E402
from realchords.utils.inference_utils import load_lit_model  # noqa: E402


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
        melody_embeds, chord_embeds = encode_contrastive(lit_module, dataloader, device)
        results[label] = separation_metrics(melody_embeds, chord_embeds)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in results[label].items()))

    cols = ["pair_auc", "mean_rank", "d_prime", "mean_true", "mean_mismatched", "num_songs"]
    print(f"\n{'model':28s}" + "".join(f"{c:>16s}" for c in cols))
    for label, m in results.items():
        print(f"{label:28s}" + "".join(f"{m[c]:>16.4f}" if isinstance(m[c], float) else f"{m[c]:>16d}"
                                       for c in cols))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pair_separation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "model_path"] + cols)
        for label, m in results.items():
            writer.writerow([label, paths[label]] + [m[c] for c in cols])
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
