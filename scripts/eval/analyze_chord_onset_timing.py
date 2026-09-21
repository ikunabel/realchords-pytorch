#!/usr/bin/env python3
"""Measure *when* the chord model expects chord changes, and how confident it
is about *which* chord when forced to change at a moment it didn't choose.

Motivation
----------
ReaLJam's manual chord-timing mode (see journal/REALJAM_CUSTOM_VOICINGS.md)
lets the performer decide when each chord lands: sampling is restricted to
chord *onset* tokens, so the model picks which chord but never when. That
restriction is not free. At any frame the model's probability mass is split
between "hold the current chord", "silence", and the ~2.8k chord onsets;
masking away the first two and renormalising means manual mode samples from
whatever onset mass happens to be left. Where that leftover mass is large the
forced choice matches what the model would have done anyway, but where it is
vanishingly small the renormalised distribution is close to uniform noise
over thousands of chords -- which is audible as bizarre chord choices.

This script quantifies that over real data: it teacher-forces ground-truth
Hooktheory melody/chord sequences and records, at every frame, the mass on
chord onsets and the entropy of the onset distribution *after* renormalising
-- i.e. exactly the distribution manual mode samples from. Results are
aggregated by metrical position (where in the bar/beat the frame sits) and by
how long the current chord has been held.

Measured over 512 test songs, metrical position dominates: on a downbeat the
model puts ~0.81 on some chord onset at ~1.6 nats of entropy, while on a
16th-note offbeat it puts ~0.01 on onsets at ~6.0 nats -- against a 7.94-nat
maximum, i.e. close to uniform across all 2.8k chords. Forcing a chord change
off the grid therefore samples something close to noise.

Method notes
------------
One forward pass per song, not per frame. The sequence is interleaved as
``[BOS, chord_0, note_0, chord_1, note_1, ...]``, so the next-token
distribution for ``chord_k`` is read off the logits at position ``2k`` -- a
single teacher-forced pass yields every frame's distribution at once, which
is what makes running over the whole dataset feasible.

Crops are forced onto bar lines. ``HooktheoryDataset.random_crop`` only aligns
crop starts to beat boundaries, so a crop may begin on beat 3 of a bar; frame
index modulo the bar length would then not be the real position in the bar,
silently scrambling the single thing this script is trying to measure. The
crop is overridden here to start on multiples of ``--frames_per_bar``, keeping
``frame % frames_per_bar`` equal to true bar position while still sampling
windows from across each song.

Usage::

    # quick look
    python scripts/eval/analyze_chord_onset_timing.py --num_songs 200

    # the full dataset, every live model
    python scripts/eval/analyze_chord_onset_timing.py \\
        --num_songs -1 --models GAPT ReaLchords "Online MLE"
"""

from __future__ import annotations

import argparse
import json
import math
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from realchords.constants import FRAME_PER_BEAT, REALJAM_CHECKPOINT_DIR
from realchords.dataset.hooktheory_dataloader import (
    HooktheoryDataset,
    get_dataloader,
)
from realchords.realjam import agent_interface
from realchords.utils.experiment_utils import DATASET_CACHE_DIRS


# ---------------------------------------------------------------------------
# Dataloader setup
# ---------------------------------------------------------------------------

def _crop_on_bar_line(
    self,
    melody: torch.Tensor,
    chord: torch.Tensor,
    idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like ``HooktheoryDataset.random_crop``, but starts only on bar lines.

    The stock crop aligns to beats, which is enough for training but not for
    measuring metrical position: a crop starting on beat 3 shifts every
    frame's apparent position in the bar. Restricting starts to whole bars
    keeps ``frame_index % frames_per_bar`` meaningful. Deterministic in
    ``idx`` so repeated runs measure the same windows.
    """
    assert melody.shape[0] == chord.shape[0]
    if melody.shape[0] <= self.max_len_per_part:
        return melody, chord

    max_len = min(self.max_len_per_part, melody.shape[0])
    frames_per_bar = getattr(self, "_frames_per_bar", 16)
    starts = list(range(0, melody.shape[0] - max_len + 1, frames_per_bar))
    if not starts:
        starts = [0]
    start = starts[idx % len(starts)]
    return melody[start:start + max_len], chord[start:start + max_len]


def _check_tokenizers_agree(model_tokenizer, dataset_tokenizer) -> None:
    """Abort unless the model and the dataset share a chord vocabulary.

    The dataset defaults to its own chord_names file while the ReaLJam models
    use the copy in the checkpoint directory, and the two differ: the cache
    vocabulary carries ~150 exotic chords the shipped checkpoints were never
    trained on, which shifts every chord token id between them. Tokenising the
    data with the model's own vocabulary (see build_loader) keeps them lined
    up; this asserts it actually happened, because the failure is otherwise
    silent -- the measurement would just describe the wrong chords.
    """
    mismatches = []
    if model_tokenizer.num_tokens != dataset_tokenizer.num_tokens:
        mismatches.append(
            f"vocab size {model_tokenizer.num_tokens} vs "
            f"{dataset_tokenizer.num_tokens}")
    for attr in ("chord_on_token_range", "chord_hold_token_range",
                 "note_token_range"):
        a = getattr(model_tokenizer, attr)
        b = getattr(dataset_tokenizer, attr)
        if tuple(a) != tuple(b):
            mismatches.append(f"{attr} {tuple(a)} vs {tuple(b)}")
    if mismatches:
        raise SystemExit(
            "Model and dataset tokenizers disagree, so token ids would not "
            "line up:\n  " + "\n  ".join(mismatches))


def build_loader(args, agent):
    """Dataloader over ground-truth songs, tokenised with the *model's* chord
    vocabulary and cropped on bar lines.

    Not using ``create_dataset_dataloaders`` because it resolves the chord
    vocabulary from the data cache, which doesn't match what the ReaLJam
    checkpoints were trained on (see _check_tokenizers_agree).
    """
    dataset = HooktheoryDataset(
        cache_dir=DATASET_CACHE_DIRS[args.dataset_name.lower()],
        split=args.dataset_split.lower(),
        model_type="decoder_only",
        model_part="chord",
        max_len=args.max_len,
        data_augmentation=False,
        load_augmented_chord_names=True,
        chord_names_path=args.chord_names_path,
    )
    _check_tokenizers_agree(agent.tokenizer, dataset.tokenizer)

    dataset._frames_per_bar = args.frames_per_bar
    dataset.random_crop = types.MethodType(_crop_on_bar_line, dataset)

    return get_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def chord_frame_distributions(
    model: torch.nn.Module,
    seq: torch.Tensor,
    tokenizer,
) -> Dict[str, torch.Tensor]:
    """Next-chord-token distribution at every frame, in one forward pass.

    Args:
      model: decoder model to probe.
      seq: ``[B, L]`` interleaved ground-truth sequence starting with BOS.
      tokenizer: supplies the chord/note token ranges.

    Returns dict of ``[B, F]`` tensors (F = frames covered by the sequence):
      p_onset   -- probability mass on all chord-onset tokens
      p_hold    -- probability mass on all chord-hold tokens
      p_silence -- probability of the silence token
      entropy   -- entropy (nats) of the onset distribution after
                   renormalising, i.e. what manual mode samples from
      top_onset -- token id of the most likely chord onset
    """
    on_lo, on_hi = tokenizer.chord_on_token_range
    hold_lo, hold_hi = tokenizer.chord_hold_token_range

    with torch.no_grad():
        logits, _ = model.net(seq[:, :-1], return_intermediates=True)
        # Position 2k predicts token 2k+1, which is chord_k
        chord_logits = logits[:, 0::2, :].float()
        probs = F.softmax(chord_logits, dim=-1)

        onset = probs[:, :, on_lo:on_hi + 1]
        p_onset = onset.sum(-1)
        renormalised = onset / p_onset.clamp_min(1e-30).unsqueeze(-1)
        entropy = -(renormalised * (renormalised + 1e-30).log()).sum(-1)

        return {
            "p_onset": p_onset,
            "p_hold": probs[:, :, hold_lo:hold_hi + 1].sum(-1),
            "p_silence": probs[:, :, tokenizer.silence_token],
            "entropy": entropy,
            "top_onset": onset.argmax(-1) + on_lo,
        }


def frames_since_previous_onset(
    chord_tokens: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """For each frame, how many frames since the last chord onset (-1 if none
    yet). ``chord_tokens`` is ``[B, F]`` of ground-truth chord-lane tokens."""
    on_lo, on_hi = tokenizer.chord_on_token_range
    is_onset = (chord_tokens >= on_lo) & (chord_tokens <= on_hi)
    out = torch.full_like(chord_tokens, -1)
    for b in range(chord_tokens.size(0)):
        last = -1
        for k in range(chord_tokens.size(1)):
            out[b, k] = -1 if last < 0 else k - last
            if is_onset[b, k]:
                last = k
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class Accumulator:
    """Running sums per bucket, so nothing has to be held in memory per frame."""

    def __init__(self) -> None:
        self.sums: Dict[str, Dict[int, List[float]]] = {}

    def add(self, group: str, bucket: int, p_onset: float, entropy: float,
            gt_onset: bool) -> None:
        g = self.sums.setdefault(group, {})
        row = g.setdefault(bucket, [0.0, 0.0, 0.0, 0.0])
        row[0] += p_onset
        row[1] += entropy
        row[2] += 1.0
        row[3] += 1.0 if gt_onset else 0.0

    def add_many(self, group: str, buckets, p_onset, entropy, gt_onset) -> None:
        for bucket, p, e, g in zip(buckets, p_onset, entropy, gt_onset):
            self.add(group, int(bucket), float(p), float(e), bool(g))

    def table(self, group: str) -> List[dict]:
        rows = []
        for bucket, (p_sum, e_sum, n, gt) in sorted(self.sums.get(group, {}).items()):
            rows.append({
                "bucket": bucket,
                "n_frames": int(n),
                "mean_p_onset": p_sum / n,
                "mean_entropy": e_sum / n,
                "gt_onset_rate": gt / n,
            })
        return rows


def print_table(title: str, rows: List[dict], bucket_label: str,
                bucket_fmt=lambda b: str(b)) -> None:
    print(f"\n{title}")
    print(f"  {bucket_label:>14} {'frames':>9} {'P(onset)':>10} "
          f"{'entropy':>9} {'GT changes':>11}")
    print("  " + "-" * 58)
    for row in rows:
        print(f"  {bucket_fmt(row['bucket']):>14} {row['n_frames']:>9,} "
              f"{row['mean_p_onset']:>10.4f} {row['mean_entropy']:>9.2f} "
              f"{row['gt_onset_rate']:>10.1%}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_real(args, agent, model, model_label: str) -> dict:
    """Teacher-force ground-truth songs and bucket every frame."""
    loader = build_loader(args, agent)
    tokenizer = agent.tokenizer
    acc = Accumulator()
    songs_done = 0
    n_batches = (len(loader) if args.num_songs < 0
                 else math.ceil(args.num_songs / args.batch_size))

    for batch_idx, batch in enumerate(
            tqdm(loader, total=n_batches, desc=f"{model_label}")):
        if batch_idx >= n_batches:
            break

        seq = batch["targets"]
        dist = chord_frame_distributions(model, seq, tokenizer)

        # Ground-truth chord lane: chord_k sits at sequence position 2k+1
        chord_tokens = seq[:, 1::2][:, :dist["p_onset"].size(1)]
        on_lo, on_hi = tokenizer.chord_on_token_range
        gt_onset = (chord_tokens >= on_lo) & (chord_tokens <= on_hi)
        since = frames_since_previous_onset(chord_tokens, tokenizer)

        # Frames past the end of the song are padding, not data
        valid = (chord_tokens != tokenizer.pad_token) & \
                (chord_tokens != tokenizer.eos_token)

        for b in range(seq.size(0)):
            keep = valid[b]
            if not keep.any():
                continue
            frames = torch.arange(chord_tokens.size(1))[keep]
            p = dist["p_onset"][b][keep]
            e = dist["entropy"][b][keep]
            g = gt_onset[b][keep]

            acc.add_many("bar_position",
                         (frames % args.frames_per_bar).tolist(),
                         p.tolist(), e.tolist(), g.tolist())
            acc.add_many("beat_position",
                         (frames % FRAME_PER_BEAT).tolist(),
                         p.tolist(), e.tolist(), g.tolist())

            s = since[b][keep]
            held = s >= 0
            if held.any():
                acc.add_many("frames_since_change",
                             s[held].tolist(),
                             p[held].tolist(), e[held].tolist(),
                             g[held].tolist())
            songs_done += 1

    print(f"\n{'=' * 64}\n{model_label}: {songs_done} songs\n{'=' * 64}")
    bar_rows = acc.table("bar_position")
    beat_rows = acc.table("beat_position")
    since_rows = [r for r in acc.table("frames_since_change")
                  if r["bucket"] <= args.max_since_frames]

    print_table(
        "Position in bar (frame 0 = downbeat)", bar_rows, "frame in bar",
        lambda b: f"{b} ({b / FRAME_PER_BEAT:.2f} bt)")
    print_table("Position in beat", beat_rows, "frame in beat")
    print_table(
        "Frames since the previous chord change", since_rows, "since",
        lambda b: f"{b} ({b / FRAME_PER_BEAT:.2f} bt)")

    return {
        "mode": "real",
        "model": model_label,
        "songs": songs_done,
        "bar_position": bar_rows,
        "beat_position": beat_rows,
        "frames_since_change": since_rows,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["ReaLchords"],
                   help="Which ReaLJam models to probe (see agent_interface.MODEL_PATHS).")
    p.add_argument("--dataset_name", default="hooktheory")
    p.add_argument("--dataset_split", default="test",
                   help="train/valid/test/all (default test).")
    p.add_argument("--num_songs", type=int, default=512,
                   help="Songs to probe; -1 for the whole split.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=512,
                   help="Interleaved sequence length, so frames = max_len/2.")
    p.add_argument("--frames_per_bar", type=int, default=16,
                   help="4 beats x 4 frames. Crops are aligned to this.")
    p.add_argument("--max_since_frames", type=int, default=64,
                   help="Largest 'frames since last change' bucket to print.")
    p.add_argument(
        "--chord_names_path",
        default=str(Path(REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"),
        help=("Chord vocabulary used to tokenise the data. Defaults to the "
              "checkpoints' own copy, which is what the ReaLJam models were "
              "trained on -- the data cache's copy has ~150 extra chords and "
              "would shift every chord token id."))
    p.add_argument("--output_dir", default="logs/eval/chord_onset_timing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading ReaLJam models (this is the same stack the live app uses) …")
    agent = agent_interface.Agent(compile=False, voicings_path=None)

    results = []
    for label in args.models:
        if label not in agent.models:
            raise SystemExit(
                f"Unknown model '{label}'. Available: {list(agent.models)}")
        model = agent.models[label]
        results.append(run_real(args, agent, model, label))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"chord_onset_timing_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
