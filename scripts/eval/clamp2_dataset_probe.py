#!/usr/bin/env python3
"""Probe whether CLaMP2 can differentiate our datasets: zero-shot genre voting
plus a t-SNE projection of the embeddings, colored by source dataset.

Two things this checks:
    1. Zero-shot genre classification (CLIP-style): embed a handful of
       candidate genre-description strings with CLaMP2's text encoder, embed
       each dataset's GT MIDIs with its music encoder, and report which genre
       text each song's embedding is closest to (cosine similarity). This
       exercises a capability the ``frechet_music_distance`` package's
       CLaMP2Extractor doesn't expose out of the box -- its bundled CLaMP2
       model loads the text encoder's weights but never calls them. The
       ``get_text_features`` recipe below (tokenize -> text_model ->
       avg_pooling -> text_proj) is copied from the official CLaMP2 repo
       (sanderwood/clamp2), not guessed -- it reuses the ``avg_pooling``
       method and ``text_model``/``text_proj`` submodules already present and
       correctly loaded on the wrapper's model instance.
    2. A 2D t-SNE projection of all songs' music embeddings across datasets,
       to see visually whether datasets form distinct clusters.

Expects GT MIDI already exported per dataset via
``realchords/utils/custom_evaluation.py``'s ``--gt_only`` mode (i.e. run
``paired_gt_all`` from ``scripts/eval/custom_eval.sh`` first) -- this script
doesn't generate anything, only reads
``logs/paired_eval/gt/<dataset>_<split>/full_songs/midi/*.mid``.

Usage::

    python scripts/eval/clamp2_dataset_probe.py
    python scripts/eval/clamp2_dataset_probe.py --max_per_dataset 20
    python scripts/eval/clamp2_dataset_probe.py --genres "a pop song" "a jazz song"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

DEFAULT_GENRE_PROMPTS = [
    "This is a pop song.",
    "This is a rock song.",
    "This is a jazz song.",
    "This is a folk song.",
    "This is a classical piece.",
]


def embed_text(extractor, text: str, max_length: int = 128) -> np.ndarray:
    """Embed a text string with CLaMP2's text encoder.

    Reconstructs the official ``get_text_features(..., get_normalized=True)``
    recipe (tokenize -> text_model -> avg_pooling -> text_proj) against the
    ``frechet_music_distance`` package's stripped-down CLaMP2 model, which
    has the text encoder's weights loaded but no method to run them.
    """
    tokenized = extractor._tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = tokenized["input_ids"].to(extractor._device)
    attention_mask = tokenized["attention_mask"].to(extractor._device)

    with torch.no_grad():
        last_hidden = extractor._model.text_model(
            input_ids, attention_mask=attention_mask
        )["last_hidden_state"]
        pooled = extractor._model.avg_pooling(last_hidden, attention_mask)
        projected = extractor._model.text_proj(pooled)

    return projected.detach().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between all rows of ``a`` and all rows of ``b``."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def discover_dataset_midi_dirs(gt_root: Path, split_mode: str) -> Dict[str, Path]:
    """Find ``<dataset>/<split_mode>/midi`` directories under ``gt_root``."""
    dirs: Dict[str, Path] = {}
    for dataset_dir in sorted(gt_root.iterdir()):
        midi_dir = dataset_dir / split_mode / "midi"
        if midi_dir.is_dir() and any(midi_dir.glob("*.mid")):
            dirs[dataset_dir.name] = midi_dir
    return dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt_root",
        type=str,
        default="logs/paired_eval/gt",
        help="Root directory containing <dataset>_<split>/{cropped_songs,full_songs}/midi/",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="full_songs",
        choices=("full_songs", "cropped_songs"),
    )
    parser.add_argument(
        "--max_per_dataset",
        type=int,
        default=None,
        help="Cap on number of MIDI files embedded per dataset (default: all available).",
    )
    parser.add_argument(
        "--genres",
        type=str,
        nargs="+",
        default=None,
        help="Candidate genre-description strings for zero-shot voting "
        "(default: a small built-in pop/rock/jazz/folk/classical list).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/paired_eval/clamp2_probe",
        help="Where to save the embeddings (.npz) and t-SNE plot (.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genre_prompts = args.genres or DEFAULT_GENRE_PROMPTS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_midi_dirs = discover_dataset_midi_dirs(Path(args.gt_root), args.split_mode)
    if not dataset_midi_dirs:
        raise SystemExit(
            f"No <dataset>/{args.split_mode}/midi/*.mid found under {args.gt_root} "
            "-- run paired_gt_all first."
        )
    print(f"Found {len(dataset_midi_dirs)} datasets with exported GT MIDI:")
    for name, midi_dir in dataset_midi_dirs.items():
        n = len(list(midi_dir.glob("*.mid")))
        print(f"  {name}: {n} files ({midi_dir})")

    from frechet_music_distance.models import CLaMP2Extractor

    extractor = CLaMP2Extractor(verbose=True)

    # ---- Embed music (per dataset, using the package's own directory-level
    # embedder, which caches results via joblib so re-runs are instant) ----
    all_embeddings: List[np.ndarray] = []
    all_dataset_labels: List[str] = []
    dataset_embeddings: Dict[str, np.ndarray] = {}

    for name, midi_dir in dataset_midi_dirs.items():
        print(f"\nEmbedding {name} ...")
        if args.max_per_dataset is not None:
            files = sorted(midi_dir.glob("*.mid"))[: args.max_per_dataset]
            feats = np.vstack([extractor.extract_feature(str(f)) for f in files])
        else:
            feats = extractor.extract_features(str(midi_dir))
        dataset_embeddings[name] = feats
        all_embeddings.append(feats)
        all_dataset_labels.extend([name] * feats.shape[0])

    music_embeddings = np.vstack(all_embeddings)
    dataset_labels = np.array(all_dataset_labels)

    # ---- Embed candidate genre text prompts ----
    print(f"\nEmbedding {len(genre_prompts)} genre prompts ...")
    genre_embeddings = np.vstack([embed_text(extractor, g) for g in genre_prompts])

    # ---- Zero-shot genre voting, per dataset ----
    similarities = cosine_similarity(music_embeddings, genre_embeddings)
    predicted_genre_idx = similarities.argmax(axis=1)
    predicted_genres = np.array([genre_prompts[i] for i in predicted_genre_idx])

    print("\n=== Zero-shot genre vote distribution per dataset ===")
    for name in dataset_midi_dirs:
        mask = dataset_labels == name
        n = int(mask.sum())
        votes = predicted_genres[mask]
        print(f"\n{name} (n={n}):")
        for genre in genre_prompts:
            count = int((votes == genre).sum())
            if count > 0:
                print(f"    {genre:30s} {count:3d}  ({count / n:.0%})")

    # ---- Save raw embeddings + labels for further analysis ----
    npz_path = out_dir / "clamp2_embeddings.npz"
    np.savez(
        npz_path,
        music_embeddings=music_embeddings,
        dataset_labels=dataset_labels,
        genre_prompts=np.array(genre_prompts),
        genre_embeddings=genre_embeddings,
        predicted_genres=predicted_genres,
    )
    print(f"\nSaved embeddings to {npz_path}")

    # ---- t-SNE projection, colored by dataset ----
    print("\nRunning t-SNE ...")
    from sklearn.manifold import TSNE

    n_samples = music_embeddings.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    projected = tsne.fit_transform(music_embeddings)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    unique_datasets = sorted(dataset_midi_dirs.keys())
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(unique_datasets):
        mask = dataset_labels == name
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            label=name,
            color=cmap(i % 10),
            s=25,
            alpha=0.75,
        )
    ax.set_title("t-SNE of CLaMP2 music embeddings, colored by dataset")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    plot_path = out_dir / "tsne_by_dataset.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved t-SNE plot to {plot_path}")


if __name__ == "__main__":
    main()
