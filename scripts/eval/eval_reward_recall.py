#!/usr/bin/env python3
"""Evaluate contrastive/discriminative reward-model checkpoints on the test set.

Reproduces the reward-model test-set metrics reported in the ReaLchords paper
(Appendix G, Table 4 for contrastive, Table 5 for discriminative):
  - Contrastive: corpus-level retrieval R@1/R@5/R@10/mAP@10, both directions
    (Note to Chord, Chord to Note).
  - Discriminative: precision/recall/F1 on real-vs-randomly-repaired-negative
    classification, aggregated over the whole test set.

Each checkpoint is evaluated using its *own* training args.yml (datasets,
weights, chord_names_path, max_len, ...), so a mix of full-length and
segment/sliding-window checkpoints can be compared directly in one table.

Retrieval/classification is always computed over one deterministic crop per
test song (via the plain, non-segment dataset loader, using the checkpoint's
own `max_len`) rather than over every overlapping sliding-window segment —
matching the paper's methodology (Appendix G: the candidate pool is test
*songs*, not windows) and keeping the corpus-level retrieval matrix a
tractable size regardless of whether the checkpoint was itself trained on
sliding windows. The reward *model* is loaded via the plain lit_module
classes too (`LitContrastiveReward`/`LitDiscriminativeReward`) — the
underlying network architecture is fully determined by
`ContrastiveReward.max_seq_len`/`DiscriminativeReward.max_seq_len` in the
checkpoint's own args.yml regardless of which lit_module trained it, so no
segment-specific lit_module is needed for eval.

By default each checkpoint's eval gallery uses whatever datasets/augmentation it was
itself trained with. Pass --eval_datasets/--eval_augmentation to override both for
every checkpoint uniformly (e.g. to match the paper's own eval setup: hooktheory-only
gallery, no augmentation), independent of what each checkpoint trained on.

By default, the ReaLchords paper's own Table 4/5 numbers (Appendix G) and this repo's
HuggingFace-downloaded reference checkpoint(s) are always included as extra rows for
direct comparison -- pass --no_paper_reference / --no_hf_reference to skip either.

Results are printed as a table and also written as a CSV to --output_dir (default
logs/eval/recall/<reward_type>_<timestamp>.csv), one row per checkpoint.

Usage:
    python scripts/eval/eval_reward_recall.py --reward_type contrastive \\
        --model "Full (256)=.../contrastive_reward_3_datasets/step=8000.ckpt" \\
        --model "w16 fixed=.../contrastive_reward_w16_3_datasets/step=....ckpt" \\
        --model "w16 sliding=.../contrastive_reward_w16_sliding_3_datasets/step=....ckpt" \\
        --eval_datasets hooktheory --eval_augmentation off

    python scripts/eval/eval_reward_recall.py --reward_type discriminative \\
        --model "Full (256)=.../discriminative_reward_128_bs/step=1500.ckpt" \\
        --model "w16 sliding=.../discriminative_reward_w16_sliding_3_datasets/step=....ckpt"
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import argbind
import torch

from realchords.dataset.weighted_joint_dataset import (
    create_weighted_joint_dataset,
    get_dataloader,
)
from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.utils.inference_utils import load_lit_model

LIT_MODULE_CLASSES = {
    "contrastive": LitContrastiveReward,
    "discriminative": LitDiscriminativeReward,
}

# ReaLchords paper, Appendix G, Table 4 (contrastive) / Table 5 (discriminative).
# Included as fixed reference rows -- no model loading, just the published numbers.
PAPER_CONTRASTIVE_REFERENCE = {
    "Paper: Full (256)": {
        "note_to_chord": {"R@1": 0.17, "R@5": 0.39, "R@10": 0.49, "mAP@10": 0.26},
        "chord_to_note": {"R@1": 0.17, "R@5": 0.39, "R@10": 0.51, "mAP@10": 0.27},
    },
    "Paper: 1/2 (128)": {
        "note_to_chord": {"R@1": 0.05, "R@5": 0.14, "R@10": 0.21, "mAP@10": 0.09},
        "chord_to_note": {"R@1": 0.05, "R@5": 0.15, "R@10": 0.21, "mAP@10": 0.09},
    },
    "Paper: 1/4 (64)": {
        "note_to_chord": {"R@1": 0.02, "R@5": 0.08, "R@10": 0.13, "mAP@10": 0.05},
        "chord_to_note": {"R@1": 0.02, "R@5": 0.07, "R@10": 0.12, "mAP@10": 0.05},
    },
    "Paper: 1/8 (32)": {
        "note_to_chord": {"R@1": 0.02, "R@5": 0.06, "R@10": 0.10, "mAP@10": 0.04},
        "chord_to_note": {"R@1": 0.02, "R@5": 0.05, "R@10": 0.09, "mAP@10": 0.03},
    },
    "Paper: 1/16 (16)": {
        "note_to_chord": {"R@1": 0.01, "R@5": 0.04, "R@10": 0.07, "mAP@10": 0.03},
        "chord_to_note": {"R@1": 0.01, "R@5": 0.03, "R@10": 0.06, "mAP@10": 0.02},
    },
}

PAPER_DISCRIMINATIVE_REFERENCE = {
    "Paper: Full (256)": {"precision": 0.69, "recall": 0.91, "f1": 0.79},
    "Paper: 1/2 (128)": {"precision": 0.69, "recall": 0.92, "f1": 0.79},
    "Paper: 1/4 (64)": {"precision": 0.71, "recall": 0.84, "f1": 0.77},
    "Paper: 1/8 (32)": {"precision": 0.69, "recall": 0.88, "f1": 0.77},
    "Paper: 1/16 (16)": {"precision": 0.68, "recall": 0.79, "f1": 0.73},
}

# This repo's own HuggingFace-downloaded reference checkpoints (hooktheory-only,
# full 256-frame context; no chord_names_path override so unaffected by the
# shared-vocab drift documented in journal/CONFIGS.md).
DEFAULT_HF_REFERENCE = {
    "contrastive": [
        ("HF: contrastive_reward", "logs/huggingface/contrastive_reward/step=8000.ckpt"),
        ("HF: contrastive_reward_2", "logs/huggingface/contrastive_reward_2/step=8000.ckpt"),
    ],
    "discriminative": [
        (
            "HF: discriminative_reward_128_bs",
            "logs/huggingface/discriminative_reward_128_bs/step=1500.ckpt",
        ),
        (
            "HF: discriminative_reward_128_bs_2",
            "logs/huggingface/discriminative_reward_128_bs_2/step=1500.ckpt",
        ),
    ],
}

# Dataset-construction kwargs to carry over from the checkpoint's own args.yml,
# so the test set matches whatever the checkpoint was actually trained on.
_DATASET_KWARGS = (
    "datasets",
    "weights",
    "alpha",
    "frame_counts_path",
    "chord_names_path",
    "data_augmentation",
    "max_len",
    "model_type",
    "model_part",
    "seed",
    "data_path",
    "frame_per_beat",
    "load_augmented_chord_names",
    "train_samples_multiplier",
    "max_train_samples",
    "sampler_chunk_size",
)


def parse_model_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    return label, path


def build_test_dataloader(
    args: dict,
    batch_size: int,
    num_workers: int,
    eval_datasets: list = None,
    eval_data_augmentation: bool = None,
):
    """One deterministic crop per test song, at the checkpoint's own max_len.

    `eval_datasets`/`eval_data_augmentation`, when given, override the
    checkpoint's own training-time values -- e.g. to match the paper's
    eval setup (hooktheory-only gallery, no augmentation) regardless of
    what the checkpoint itself was trained on.
    """
    kwargs = {k: args[k] for k in _DATASET_KWARGS if k in args}
    kwargs["split"] = "test"
    if eval_datasets is not None:
        kwargs["datasets"] = eval_datasets
        kwargs["weights"] = None
    if eval_data_augmentation is not None:
        # Which chord vocabulary file gets loaded depends on data_augmentation
        # (see HooktheoryDataset.cache_file_exists: "_augmented" suffix iff
        # load_augmented_chord_names or data_augmentation) whenever
        # chord_names_path itself isn't pinned in args. Overriding
        # data_augmentation for eval must NOT silently swap in a different
        # (wrong-size, differently-ordered) vocabulary than the checkpoint
        # was actually trained with -- pin load_augmented_chord_names to
        # whatever the checkpoint's own training config used, independent of
        # this override, unless chord_names_path is itself explicit.
        if "chord_names_path" not in kwargs:
            kwargs["load_augmented_chord_names"] = bool(
                args.get("data_augmentation", True)
            )
        kwargs["data_augmentation"] = eval_data_augmentation
    dataset = create_weighted_joint_dataset(**kwargs)
    return get_dataloader(
        dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )


@torch.no_grad()
def encode_contrastive(lit_module, dataloader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model = lit_module.model.to(device).eval()
    melody_embeds, chord_embeds = [], []
    for batch in dataloader:
        melody_tokens, chord_tokens, melody_mask, chord_mask = lit_module.get_inputs(
            batch
        )
        chord_embed, melody_embed, _ = model(
            chord=chord_tokens.to(device),
            melody=melody_tokens.to(device),
            chord_mask=chord_mask.to(device),
            melody_mask=melody_mask.to(device),
        )
        melody_embeds.append(melody_embed.float().cpu())
        chord_embeds.append(chord_embed.float().cpu())
    return torch.cat(melody_embeds, dim=0), torch.cat(chord_embeds, dim=0)


def retrieval_metrics(
    queries: torch.Tensor, gallery: torch.Tensor, ks=(1, 5, 10)
) -> Dict[str, float]:
    """Corpus-level retrieval: query i's ground-truth match is gallery i."""
    sims = queries @ gallery.T
    n = sims.shape[0]
    ranks = sims.argsort(dim=1, descending=True)
    targets = torch.arange(n).unsqueeze(1)
    correct_rank = (ranks == targets).float().argmax(dim=1)

    metrics = {f"R@{k}": (correct_rank < k).float().mean().item() for k in ks}
    ap = torch.where(
        correct_rank < 10,
        1.0 / (correct_rank.float() + 1.0),
        torch.zeros_like(correct_rank, dtype=torch.float32),
    )
    metrics["mAP@10"] = ap.mean().item()
    return metrics


@torch.no_grad()
def classify_discriminative(lit_module, dataloader, device) -> Dict[str, float]:
    model = lit_module.model.to(device).eval()
    tp = fp = fn = tn = 0
    for batch in dataloader:
        inputs, input_mask, labels = lit_module.get_inputs(batch)
        logits = model(inputs.to(device), input_mask.to(device))
        preds = logits.argmax(dim=1).cpu()
        tp += int(((preds == 1) & (labels == 1)).sum())
        fp += int(((preds == 1) & (labels == 0)).sum())
        fn += int(((preds == 0) & (labels == 1)).sum())
        tn += int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_checkpoint(
    model_path: str,
    reward_type: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    eval_datasets: list = None,
    eval_data_augmentation: bool = None,
) -> Dict:
    args = argbind.load_args(Path(model_path).parent / "args.yml")
    lit_module_cls = LIT_MODULE_CLASSES[reward_type]

    lit_module = load_lit_model(
        model_path,
        lit_module_cls=lit_module_cls,
        compile=False,
        return_lit_module=True,
    )
    dataloader = build_test_dataloader(
        args, batch_size, num_workers, eval_datasets, eval_data_augmentation
    )

    if reward_type == "contrastive":
        melody_embeds, chord_embeds = encode_contrastive(lit_module, dataloader, device)
        return {
            "note_to_chord": retrieval_metrics(melody_embeds, chord_embeds),
            "chord_to_note": retrieval_metrics(chord_embeds, melody_embeds),
        }
    return classify_discriminative(lit_module, dataloader, device)


def print_contrastive_table(results: Dict[str, Dict]):
    header = (
        f"{'label':30s} | {'N2C R@1':>7s} {'N2C R@5':>7s} {'N2C R@10':>8s} {'N2C mAP@10':>10s}"
        f" | {'C2N R@1':>7s} {'C2N R@5':>7s} {'C2N R@10':>8s} {'C2N mAP@10':>10s}"
    )
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        n2c, c2n = m["note_to_chord"], m["chord_to_note"]
        print(
            f"{label:30s} | {n2c['R@1']:7.3f} {n2c['R@5']:7.3f} {n2c['R@10']:8.3f} {n2c['mAP@10']:10.3f}"
            f" | {c2n['R@1']:7.3f} {c2n['R@5']:7.3f} {c2n['R@10']:8.3f} {c2n['mAP@10']:10.3f}"
        )


def print_discriminative_table(results: Dict[str, Dict]):
    header = f"{'label':30s} | {'precision':>9s} {'recall':>7s} {'f1':>7s}"
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        print(f"{label:30s} | {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:7.3f}")


def write_contrastive_csv(results: Dict[str, Dict], model_paths: Dict[str, str], path: Path):
    fieldnames = [
        "label",
        "model_path",
        "n2c_R@1", "n2c_R@5", "n2c_R@10", "n2c_mAP@10",
        "c2n_R@1", "c2n_R@5", "c2n_R@10", "c2n_mAP@10",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, m in results.items():
            n2c, c2n = m["note_to_chord"], m["chord_to_note"]
            writer.writerow(
                {
                    "label": label,
                    "model_path": model_paths[label],
                    "n2c_R@1": n2c["R@1"], "n2c_R@5": n2c["R@5"],
                    "n2c_R@10": n2c["R@10"], "n2c_mAP@10": n2c["mAP@10"],
                    "c2n_R@1": c2n["R@1"], "c2n_R@5": c2n["R@5"],
                    "c2n_R@10": c2n["R@10"], "c2n_mAP@10": c2n["mAP@10"],
                }
            )


def write_discriminative_csv(results: Dict[str, Dict], model_paths: Dict[str, str], path: Path):
    fieldnames = ["label", "model_path", "precision", "recall", "f1"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, m in results.items():
            writer.writerow(
                {
                    "label": label,
                    "model_path": model_paths[label],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reward_type", choices=["contrastive", "discriminative"], required=True
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        dest="models",
        default=[],
        help="LABEL=PATH to a checkpoint (.ckpt). Repeatable.",
    )
    parser.add_argument(
        "--no_paper_reference",
        action="store_true",
        help="Skip the fixed ReaLchords paper Table 4/5 reference rows (included by default).",
    )
    parser.add_argument(
        "--no_hf_reference",
        action="store_true",
        help="Skip auto-evaluating this repo's HuggingFace-downloaded reference "
        "checkpoint(s) (included by default).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=None,
        help="Override the eval gallery's dataset list (e.g. --eval_datasets hooktheory), "
        "regardless of what each checkpoint itself trained on. Default: use each "
        "checkpoint's own training datasets.",
    )
    parser.add_argument(
        "--eval_augmentation",
        choices=["checkpoint", "on", "off"],
        default="checkpoint",
        help="Force data_augmentation on/off for the eval gallery, or 'checkpoint' "
        "(default) to use whatever each checkpoint was itself trained with.",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/eval/recall",
        help="Directory to write the results table as a CSV (default: logs/eval/recall). "
        "Filename is <reward_type>_<timestamp>.csv.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_data_augmentation = {"checkpoint": None, "on": True, "off": False}[
        args.eval_augmentation
    ]

    results = {}
    model_paths = {}

    if not args.no_paper_reference:
        paper_ref = (
            PAPER_CONTRASTIVE_REFERENCE
            if args.reward_type == "contrastive"
            else PAPER_DISCRIMINATIVE_REFERENCE
        )
        for label, metrics in paper_ref.items():
            results[label] = metrics
            model_paths[label] = "(ReaLchords paper, Appendix G)"

    models_to_run = list(args.models)
    if not args.no_hf_reference:
        existing_paths = {path for _, path in models_to_run}
        for label, path in DEFAULT_HF_REFERENCE[args.reward_type]:
            if path not in existing_paths and Path(path).exists():
                models_to_run.insert(0, (label, path))

    for label, model_path in models_to_run:
        print(f"Evaluating {label} ({model_path})...")
        model_paths[label] = model_path
        results[label] = evaluate_checkpoint(
            model_path,
            args.reward_type,
            args.batch_size,
            args.num_workers,
            device,
            eval_datasets=args.eval_datasets,
            eval_data_augmentation=eval_data_augmentation,
        )

    print()
    if args.reward_type == "contrastive":
        print_contrastive_table(results)
    else:
        print_discriminative_table(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{args.reward_type}_{timestamp}.csv"
    if args.reward_type == "contrastive":
        write_contrastive_csv(results, model_paths, csv_path)
    else:
        write_discriminative_csv(results, model_paths, csv_path)
    print(f"\nWrote results to {csv_path}")


if __name__ == "__main__":
    main()
