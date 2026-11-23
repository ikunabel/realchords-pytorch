#!/usr/bin/env python3
"""Measure *when* the chord model expects chord changes, and how confident it
is about *which* chord when forced to change at a moment it didn't choose.

Motivation
----------
ReaLJam's manual chord-timing mode (see journal/reference/REALJAM_CUSTOM_VOICINGS.md)
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


def build_loader(args, agent, model_label):
    """Dataloader over ground-truth songs, tokenised with the *model's* chord
    vocabulary and cropped on bar lines. Models differ (2026-11-19): the old
    ones use the checkpoints' list (--chord_names_path, 2,821 chords), the
    7-dataset ones the data cache's (2,969 chords); the Agent knows which.

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
        chord_names_path=(args.chord_names_path
                          if agent.tokenizer_for(model_label) is agent.tokenizer
                          else agent_interface.GROWN_CHORD_NAMES_PATH),
    )
    _check_tokenizers_agree(agent.tokenizer_for(model_label), dataset.tokenizer)

    # Optional song filters (2026-11-19, for the other corpora): only songs
    # entirely in 4/4 (the bar positions assume 16 frames per bar), and only
    # songs whose chords all exist in the old vocabulary -- so models in either
    # vocabulary are measured on exactly the same songs
    if args.only_4_4 or args.common_vocab:
        from realchords.utils.data_utils import to_chord_name
        old_names = set(agent.chord_names)
        kept, dropped = [], {"not_4_4": 0, "chord_not_in_old_vocab": 0}
        for i in range(len(dataset.data)):
            item = dataset.data[i]
            ann = item["annotations"]
            meters = {(m.get("beats_per_bar"), m.get("beat_unit")) for m in ann.get("meters") or []}
            if args.only_4_4 and meters != {(4, 4)}:
                dropped["not_4_4"] += 1
                continue
            if args.common_vocab and any(
                    to_chord_name(h["root_pitch_class"], h["root_position_intervals"]) not in old_names
                    for h in ann.get("harmony") or []):
                dropped["chord_not_in_old_vocab"] += 1
                continue
            kept.append(item)
        dataset.data = kept
        args.songs_dropped = dropped
        print(f"Song filters: kept {len(kept)}, dropped {dropped}")

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
    keep_onset_dist: bool = False,
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

        out = {"onset_dist": renormalised} if keep_onset_dist else {}
        return {
            **out,
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
# Quality of the forced choice (2026-11-19)
# ---------------------------------------------------------------------------

FORCED_FIELDS = ("surprisal", "m_current", "m_next", "m_other", "note_in_chord", "in_key", "n_key")


class ForcedQuality:
    """Is the chord forced at a frame musically reasonable? Per frame, for one
    draw from the renormalised onset distribution at temperature T (as ReaLJam
    plays a forced chord):

      surprisal     -log P(change): how much forcing a change now goes against
                    the model's own timing (nats)
      m_current     chance the draw re-strikes the chord sounding at the frame before
      m_next        chance it is the song's next real chord (the next onset at or
                    after this frame, if different from the current one): the
                    model just changes earlier than the song
      m_other       anything else
      note_in_chord expected share of the melody frames of the next beat (frames
                    k..k+3 with a note) whose pitch class is in the drawn chord
      in_key        chance all its pitch classes are in the song's key (single-key
                    songs with a key annotation only; Hooktheory has real keys)
    """

    def __init__(self, tokenizer, onset_info: Dict[int, dict], temperature: float):
        on_lo, on_hi = tokenizer.chord_on_token_range
        hold_lo, _ = tokenizer.chord_hold_token_range
        self.on_lo, self.on_hi, self.hold_lo = on_lo, on_hi, hold_lo
        self.offset = on_lo - hold_lo          # hold id + offset = onset id
        self.T = temperature
        n_on = on_hi - on_lo + 1
        self.pc_mask = torch.zeros(n_on, 12)   # chord onset -> its pitch classes
        for tok, info in onset_info.items():
            self.pc_mask[tok - on_lo, info["pitchClasses"]] = 1.0
        self.n_pcs = self.pc_mask.sum(-1)
        self.parsed = self.n_pcs > 0           # unparsable chords: never "fit"
        self.melody_pc = torch.full((tokenizer.num_tokens,), -1, dtype=torch.long)
        for tid, name in tokenizer.id_to_name.items():
            if name.startswith("NOTE_ON_") or (name.startswith("NOTE_") and name[5:].isdigit()):
                self.melody_pc[tid] = int(name.rsplit("_", 1)[1]) % 12
        self.sums: Dict[str, Dict[int, List[float]]] = {}

    def chord_onset_id(self, token: int) -> int:
        """The onset id of the chord a chord token (onset or hold) belongs to, else -1."""
        if self.on_lo <= token <= self.on_hi:
            return token
        if self.hold_lo <= token < self.on_lo:
            return token + self.offset
        return -1

    def add_song(self, onset_dist, p_onset, chord_tokens, melody_tokens, frames_keep,
                 bar_pos, gt_onset, scale_pcs) -> None:
        """One song's frames. onset_dist [F, n_on], p_onset [F], chord_tokens /
        melody_tokens [F] (ground truth), frames_keep [F] bool, bar_pos [F],
        gt_onset [F] bool, scale_pcs: set of the key's pitch classes or None."""
        F_ = chord_tokens.size(0)
        ids = [self.chord_onset_id(int(t)) for t in chord_tokens]
        nxt, upcoming = [-1] * F_, -1
        for k in range(F_ - 1, -1, -1):        # next onset at or after k
            if self.on_lo <= int(chord_tokens[k]) <= self.on_hi:
                upcoming = int(chord_tokens[k])
            nxt[k] = upcoming
        w = onset_dist.clamp_min(1e-30) ** (1.0 / self.T)
        w = w / w.sum(-1, keepdim=True)
        mel = self.melody_pc[melody_tokens]                     # [F] pc or -1
        hist = torch.zeros(F_, 12)
        L = mel.size(0)                        # the melody lane can be one frame shorter
        for k in range(F_):
            for j in range(k, min(k + 4, F_, L)):
                if mel[j] >= 0 and frames_keep[j]:
                    hist[k, mel[j]] += 1
        fit = (w @ self.pc_mask)                                 # [F, 12] P(pc in the drawn chord)
        n_mel = hist.sum(-1)
        nic = (fit * hist).sum(-1) / n_mel.clamp_min(1)
        if scale_pcs is not None:
            outside = torch.ones(12)
            outside[list(scale_pcs)] = 0
            chord_in_key = ((self.pc_mask @ outside) == 0) & self.parsed
            in_key = (w * chord_in_key.float()).sum(-1)
        for k in range(F_):
            if not frames_keep[k]:
                continue
            cur = ids[k - 1] if k > 0 else -1
            m_cur = float(w[k, cur - self.on_lo]) if cur >= 0 else 0.0
            n = nxt[k]
            m_next = float(w[k, n - self.on_lo]) if (n >= 0 and n != cur) else 0.0
            vals = [-math.log(max(float(p_onset[k]), 1e-30)), m_cur, m_next, 1.0 - m_cur - m_next,
                    float(nic[k]) if n_mel[k] > 0 else float("nan"),
                    float(in_key[k]) if scale_pcs is not None else float("nan")]
            for group, bucket in (("bar_position", int(bar_pos[k])),) + (
                    (("real_change", int(bar_pos[k])),) if gt_onset[k] else ()):
                row = self.sums.setdefault(group, {}).setdefault(bucket, [0.0] * 9)
                row[0] += vals[0]; row[1] += vals[1]; row[2] += vals[2]; row[3] += vals[3]
                if not math.isnan(vals[4]):
                    row[4] += vals[4]; row[7] += 1
                if not math.isnan(vals[5]):
                    row[5] += vals[5]; row[8] += 1
                row[6] += 1

    def table(self, group: str) -> List[dict]:
        out = []
        for bucket, r in sorted(self.sums.get(group, {}).items()):
            n = r[6]
            out.append({"bucket": bucket, "n_frames": int(n), "surprisal": r[0] / n,
                        "m_current": r[1] / n, "m_next": r[2] / n, "m_other": r[3] / n,
                        "note_in_chord": r[4] / r[7] if r[7] else None,
                        "in_key": r[5] / r[8] if r[8] else None, "n_key_frames": int(r[8])})
        return out


def song_scale_pcs(item: dict):
    """The key's pitch classes of a single-key song, or None (no key / key changes)."""
    keys = item["annotations"].get("keys") or []
    if len(keys) != 1 or "scale_degree_intervals" not in keys[0]:
        return None
    pcs, pc = {keys[0]["tonic_pitch_class"] % 12}, keys[0]["tonic_pitch_class"]
    for step in keys[0]["scale_degree_intervals"]:
        pc += step
        pcs.add(pc % 12)
    return pcs


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
    loader = build_loader(args, agent, model_label)
    tokenizer = agent.tokenizer_for(model_label)
    acc = Accumulator()
    forced = (ForcedQuality(tokenizer, agent._onset_chord_info(tokenizer), args.forced_temperature)
              if args.forced_quality else None)
    use_keys = args.dataset_name.lower() == "hooktheory"   # the others have placeholder keys
    songs_done = 0
    n_batches = (len(loader) if args.num_songs < 0
                 else math.ceil(args.num_songs / args.batch_size))

    for batch_idx, batch in enumerate(
            tqdm(loader, total=n_batches, desc=f"{model_label}")):
        if batch_idx >= n_batches:
            break

        seq = batch["targets"]
        dist = chord_frame_distributions(model, seq, tokenizer, keep_onset_dist=forced is not None)

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
            if forced is not None:
                n_frames = dist["p_onset"].size(1)
                melody_tokens = seq[b, 2::2][:n_frames]
                item = loader.dataset.data[batch_idx * args.batch_size + b]
                forced.add_song(
                    dist["onset_dist"][b], dist["p_onset"][b], chord_tokens[b], melody_tokens,
                    keep, torch.arange(n_frames) % args.frames_per_bar, gt_onset[b],
                    song_scale_pcs(item) if use_keys else None)
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

    extra = {}
    if forced is not None:
        extra["forced_quality"] = {"temperature": args.forced_temperature,
                                   "bar_position": forced.table("bar_position"),
                                   "real_change": forced.table("real_change")}
        print("\nForced-choice quality (one draw at T %.2f), by position in the bar" % args.forced_temperature)
        print("  pos  frames  -logP  current   next   other  note-in-chord  in-key")
        for r in forced.table("bar_position"):
            nic = "-" if r["note_in_chord"] is None else f"{r['note_in_chord']:.3f}"
            ik = "-" if r["in_key"] is None else f"{r['in_key']:.3f}"
            print(f"  {r['bucket']:>3} {r['n_frames']:>7,} {r['surprisal']:6.2f} {r['m_current']:8.3f} "
                  f"{r['m_next']:6.3f} {r['m_other']:7.3f} {nic:>14} {ik:>7}")
    return {
        **extra,
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
    p.add_argument("--only_4_4", action="store_true",
                   help="Only songs entirely in 4/4 (e.g. Nottingham has 6/8 and 3/4 tunes).")
    p.add_argument("--common_vocab", action="store_true",
                   help="Only songs whose chords all exist in the old vocabulary, so old- and "
                        "new-vocabulary models are measured on the same songs.")
    p.add_argument("--forced_quality", action="store_true",
                   help="Also measure the forced choice's quality (ForcedQuality): surprisal, "
                        "re-strike / next real chord / other, note-in-chord, in-key.")
    p.add_argument("--forced_temperature", type=float, default=0.8,
                   help="Temperature of the forced draw (ReaLJam's default 0.8).")
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
