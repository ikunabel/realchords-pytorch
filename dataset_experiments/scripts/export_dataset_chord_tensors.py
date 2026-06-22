"""
Export ground-truth chord sequences from each dataset cache as .pt tensors,
in the same format expected by scripts/plot_chord_embedding_tsne.py.

Format: rank-2 LongTensor (N, seq_len), chord-first interleaved with BOS prepended:
    [BOS, c0, m0, c1, m1, ..., EOS, PAD, ...]

Output: dataset_experiments/output/tsne_tensors/{dataset}.pt

Usage:
    python dataset_experiments/scripts/export_dataset_chord_tensors.py
    python dataset_experiments/scripts/export_dataset_chord_tensors.py --split train
    python dataset_experiments/scripts/export_dataset_chord_tensors.py --max_per_dataset 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[2]
CACHE_DIR = ROOT / "data" / "cache"
CHORD_NAMES = ROOT / "data" / "cache" / "chord_names.json"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "tsne_tensors"

DATASETS = ["hooktheory", "pop909", "nottingham", "wikifonia"]


def load_tokenizer():
    import sys
    sys.path.insert(0, str(ROOT))
    from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
    with open(CHORD_NAMES) as f:
        chord_names = json.load(f)
    return HooktheoryTokenizer(chord_names=chord_names)


def interleave_chord_first(chord: np.ndarray, melody: np.ndarray) -> np.ndarray:
    out = np.zeros(len(chord) + len(melody), dtype=np.int64)
    out[0::2] = chord
    out[1::2] = melody
    return out


def encode_example(example: dict, tokenizer) -> np.ndarray | None:
    try:
        encoded = tokenizer.encode(example)
    except Exception:
        return None
    chord = encoded["chord"]
    melody = encoded["melody"]
    if len(chord) == 0 or len(melody) == 0:
        return None
    interleaved = interleave_chord_first(chord, melody)
    bos = np.array([tokenizer.bos_token], dtype=np.int64)
    eos = np.array([tokenizer.eos_token], dtype=np.int64)
    return np.concatenate([bos, interleaved, eos])


def load_cache(dataset: str, split: str) -> list[dict]:
    path = CACHE_DIR / dataset / f"{split}.jsonl"
    if not path.exists():
        print(f"  {path} not found, skipping")
        return []
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def export_dataset(dataset: str, split: str, tokenizer, max_per_dataset: int | None) -> int:
    examples = load_cache(dataset, split)
    if not examples:
        return 0

    if max_per_dataset is not None:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(examples), size=min(max_per_dataset, len(examples)), replace=False)
        examples = [examples[i] for i in sorted(idx)]

    sequences = []
    for ex in examples:
        seq = encode_example(ex, tokenizer)
        if seq is not None:
            sequences.append(seq)

    if not sequences:
        print(f"  {dataset}: no valid sequences")
        return 0

    max_len = max(len(s) for s in sequences)
    pad = tokenizer.pad_token
    padded = np.full((len(sequences), max_len), pad, dtype=np.int64)
    for i, s in enumerate(sequences):
        padded[i, : len(s)] = s

    tensor = torch.from_numpy(padded).long()
    out_path = OUTPUT_DIR / f"{dataset}_{split}.pt"
    torch.save(tensor, out_path)
    print(f"  {dataset:12s}  {tensor.shape[0]:>6,} sequences  shape {tuple(tensor.shape)}  → {out_path.name}")
    return tensor.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Cap sequences per dataset (sampled with seed 42). "
                             "Recommended: 2000 to balance dataset sizes for t-SNE.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    print(f"Exporting {args.split} split  (max_per_dataset={args.max_per_dataset})\n")
    total = 0
    for ds in DATASETS:
        total += export_dataset(ds, args.split, tokenizer, args.max_per_dataset)

    print(f"\nTotal sequences exported: {total:,}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    print("Run the t-SNE plot with:")
    cmd_parts = " \\\n    ".join(
        f'--group "{ds.capitalize()}={OUTPUT_DIR}/{ds}_{args.split}.pt"'
        for ds in DATASETS
        if (OUTPUT_DIR / f"{ds}_{args.split}.pt").exists()
    )
    print(f"python scripts/plot_chord_embedding_tsne.py \\\n    {cmd_parts} \\\n    --output_plot logs/tsne_datasets.png \\\n    --max_sequences_per_group {args.max_per_dataset or 2000}")


if __name__ == "__main__":
    main()
