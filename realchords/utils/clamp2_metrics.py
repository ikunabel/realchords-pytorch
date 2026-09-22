"""CLaMP 2 distance metrics for generated accompaniment (used by custom_evaluation.py).

These measure closeness to the reference *as judged by CLaMP 2's embedding space*, not
musical quality or realism as such.

For a sample of test sequences, GT and every model are rendered to MIDI with the
plain export voicing (block chords + bass note, ``midi_export._write_one_midi``)
in two views:
  - "full":   melody + chords
  - "chords": chord track only (isolates the accompaniment; melody is identical
              across sources and would otherwise dominate the embedding)
Each file is embedded with CLaMP 2 (``frechet_music_distance``'s CLaMP2Extractor),
and per model and view:
  - fmd:     Frechet Music Distance to GT's embeddings (Gaussian fit, the package's
             own formula). Lower = closer to GT in CLaMP 2's embedding space. Depends on sample size, so only
             compare values computed with the same number of sequences.
  - cos:     mean cosine similarity between the model's and GT's embedding of the
             same sequence (paired). Higher = closer to the original piece.
  - split-half anchor (GT vs GT is trivially 0): the used sequences are split once
    at random (fixed seed) into halves A and B. ``clamp2_fmd_{view}_split_half`` =
    FMD(GT on A, GT on B), i.e. how far two samples of real pieces already are
    from each other; reported in the GT row (key "__gt__"). For each model,
    ``clamp2_fmd_{view}_vs_gt_half`` = FMD(GT on B, model on A) -- same sample
    size and the same "different pieces" situation, so directly comparable to
    the split-half value (a model at that value is as close to GT as GT is to
    itself).
Sequences for which any source fails to render/embed (e.g. an empty chord track)
are dropped for all sources, so the comparison stays paired. Embeddings are saved
to ``out_dir/embeddings_{view}.npz`` (one array per source + ``seq_indices``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from realchords.utils.midi_export import _write_one_midi

VIEWS = {"full": True, "chords": False}  # view -> include_melody
SPLIT_SEED = 0
GT_KEY = "__gt__"


def _embed_source(extractor, tensor, tokenizer, indices, out_dir: Path,
                  include_melody: bool, include_chord_bass: bool) -> List[np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feats: List[np.ndarray] = []
    for k, i in enumerate(indices):
        path = out_dir / f"{k:05d}_seq{i:05d}.mid"
        try:
            _write_one_midi(tensor[i], tokenizer, path, strict_chords=True,
                            include_chord_bass=include_chord_bass,
                            include_melody=include_melody)
            feats.append(np.asarray(extractor.extract_feature(str(path)), dtype=np.float64).ravel())
        except Exception as exc:  # noqa: BLE001 -- one bad file shouldn't kill the eval
            print(f"    clamp2: skipping {path.name} ({type(exc).__name__}: {exc})")
            feats.append(None)
    return feats


def compute_clamp2_metrics(
    gt_tensor: torch.Tensor,
    model_tensors: Dict[str, torch.Tensor],
    gt_tokenizer,
    model_tokenizer,
    indices: List[int],
    out_dir: Path,
    include_chord_bass: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Returns {model_key: {clamp2_fmd_{view}_vs_gt, clamp2_fmd_{view}_vs_gt_half,
    clamp2_cos_per_song_{view}_vs_gt, clamp2_num_sequences_{view}}, "__gt__":
    {clamp2_fmd_{view}_split_half}}. MIDI files are kept under out_dir/<view>/<source>/."""
    from frechet_music_distance import FrechetMusicDistance
    from frechet_music_distance.models import CLaMP2Extractor

    extractor = CLaMP2Extractor(verbose=False)
    fmd = FrechetMusicDistance(feature_extractor=extractor, gaussian_estimator="mle", verbose=False)
    results: Dict[str, Dict[str, float]] = {key: {} for key in model_tensors}
    results[GT_KEY] = {}

    def fmd_between(a: np.ndarray, b: np.ndarray) -> float:
        mu_a, cov_a = fmd._gaussian_estimator.estimate_parameters(a)
        mu_b, cov_b = fmd._gaussian_estimator.estimate_parameters(b)
        return float(fmd._compute_fmd(mu_a, mu_b, cov_a, cov_b))

    for view, include_melody in VIEWS.items():
        print(f"  clamp2 [{view}]: embedding {len(indices)} sequences x {1 + len(model_tensors)} sources")
        feats = {"gt": _embed_source(extractor, gt_tensor, gt_tokenizer, indices,
                                     out_dir / view / "gt", include_melody, include_chord_bass)}
        for key, tensor in model_tensors.items():
            feats[key] = _embed_source(extractor, tensor, model_tokenizer, indices,
                                       out_dir / view / key, include_melody, include_chord_bass)
        keep = [k for k in range(len(indices)) if all(f[k] is not None for f in feats.values())]
        if len(keep) < 2:
            print(f"  clamp2 [{view}]: fewer than 2 usable sequences, skipped")
            continue
        stacked = {src: np.stack([f[k] for k in keep]) for src, f in feats.items()}
        np.savez(out_dir / f"embeddings_{view}.npz",
                 seq_indices=np.asarray([indices[k] for k in keep]), **stacked)
        gt = stacked["gt"]
        gt_unit = gt / np.linalg.norm(gt, axis=1, keepdims=True)
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(keep))
        half_a, half_b = perm[: len(keep) // 2], perm[len(keep) // 2 : 2 * (len(keep) // 2)]
        results[GT_KEY][f"clamp2_fmd_{view}_split_half"] = fmd_between(gt[half_a], gt[half_b])
        for key in model_tensors:
            test = stacked[key]
            test_unit = test / np.linalg.norm(test, axis=1, keepdims=True)
            results[key][f"clamp2_fmd_{view}_vs_gt"] = fmd_between(gt, test)
            results[key][f"clamp2_fmd_{view}_vs_gt_half"] = fmd_between(gt[half_b], test[half_a])
            results[key][f"clamp2_cos_per_song_{view}_vs_gt"] = float((gt_unit * test_unit).sum(1).mean())
            results[key][f"clamp2_num_sequences_{view}"] = len(keep)
    return results
