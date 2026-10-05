#!/usr/bin/env python3
"""Does metrical position matter for chord completion?

ReaLJam's "complete" mode lets the performer play a few notes and asks the
model which chord to complete them into: the model's onset-only distribution
at the current frame, restricted to chords containing every pitch class held.
Its melody input is silent (the performer's notes are constraints, not
melody), so it relies on the chord history alone.

The off-beat analysis (analyze_chord_onset_timing.py) showed that forcing a
chord onset off the beat samples from a near-flat distribution over all
2,821 chords. This measures how much of that survives the completion
constraint: whether the model still has a real preference *among the
compatible chords* off the beat, in exactly the complete-mode setting.

Method: teacher-force ground-truth Hooktheory songs (bar-aligned crops, the
checkpoints' own chord vocabulary -- both via analyze_chord_onset_timing's
build_loader). At every frame where a chord is sounding, the simulated
performer holds that chord's root and third. For each frame, from the onset
distribution renormalised over the compatible chords:

  entropy      -- nats; exp(entropy) = effective number of choices
  n_compatible -- how many vocabulary chords contain the held notes
  p_true       -- probability the true chord is drawn (temperature 1)
  p_true_t08   -- the same at temperature 0.8 (the app's default)
  top1_true    -- the most probable compatible chord is the true one
  in_top200    -- share of compatible mass inside the onset top-200 the
                  app actually fetches

Two conditions per song: "silent" (melody replaced by silence, as in
complete mode) and "melody" (the real melody, for reference).

Usage::

    python scripts/eval/analyze_completion_timing.py --num_songs 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_chord_onset_timing import build_loader  # noqa: E402

from realchords.constants import REALJAM_CHECKPOINT_DIR  # noqa: E402
from realchords.realjam import agent_interface  # noqa: E402

POSITIONS = {  # frame % 16 -> label
    0: "downbeat", 8: "beat 3", 4: "beats 2/4", 12: "beats 2/4",
    2: "8th offbeat", 6: "8th offbeat", 10: "8th offbeat", 14: "8th offbeat",
}
ORDER = ["downbeat", "beat 3", "beats 2/4", "8th offbeat", "16th offset"]
METRICS = ["entropy", "n_compatible", "p_true", "p_true_t08", "top1_true", "in_top200"]


def onset_tables(agent, device):
    """Per onset token: pitch-class membership [V, 12] and root/third pcs [V]."""
    tk = agent.tokenizer
    lo, hi = tk.chord_on_token_range
    info = agent._onset_chord_info()
    n = hi - lo + 1
    pcs = torch.zeros(n, 12, dtype=torch.bool)
    root = torch.full((n,), -1, dtype=torch.long)
    third = torch.full((n,), -1, dtype=torch.long)
    for token, d in info.items():
        i = token - lo
        pcs[i, d["pitchClasses"]] = True
        r = d["root"]
        root[i] = r
        thirds = [pc for pc in ((r + 4) % 12, (r + 3) % 12) if pc in d["pitchClasses"]]
        others = [pc for pc in d["pitchClasses"] if pc != r]
        third[i] = thirds[0] if thirds else (others[0] if others else r)
    return pcs.to(device), root.to(device), third.to(device)


def current_onset_ids(chord_tokens, tk):
    """Map each frame's chord-lane token (onset or hold) to its onset id; -1
    for silence, padding and special tokens."""
    out = torch.full_like(chord_tokens, -1)
    hold_to_onset = {}
    lo, hi = tk.chord_hold_token_range
    for t in range(lo, hi + 1):
        name = tk.id_to_name[t]
        onset_name = "CHORD_ON_" + name[len("CHORD_"):]
        if onset_name in tk.name_to_id:
            hold_to_onset[t] = tk.name_to_id[onset_name]
    on_lo, on_hi = tk.chord_on_token_range
    for b in range(chord_tokens.size(0)):
        for k in range(chord_tokens.size(1)):
            t = int(chord_tokens[b, k])
            if on_lo <= t <= on_hi:
                out[b, k] = t
            elif t in hold_to_onset:
                out[b, k] = hold_to_onset[t]
    return out


def measure(model, seq, current, tables, tk, top_k):
    """Per-frame completion metrics for one batch. Returns dict of [B, F]."""
    pcs, root, third = tables
    on_lo, on_hi = tk.chord_on_token_range
    with torch.no_grad():
        logits, _ = model.net(seq[:, :-1], return_intermediates=True)
    chord_logits = logits[:, 0::2, on_lo:on_hi + 1].float()        # [B, F, V]
    F_ = min(chord_logits.size(1), current.size(1))
    chord_logits, current = chord_logits[:, :F_], current[:, :F_]
    onset = F.softmax(chord_logits, dim=-1)                          # renormalised over onsets

    valid = current >= 0
    cur = (current - on_lo).clamp_min(0)                             # [B, F] vocab index
    r, t = root[cur], third[cur]                                     # held pitch classes
    pcs_t = pcs.t()                                                  # [12, V]
    compat = pcs_t[r] & pcs_t[t]                                     # [B, F, V]

    def renorm(p):
        p = p * compat
        return p / p.sum(-1, keepdim=True).clamp_min(1e-30)

    pc = renorm(onset)
    entropy = -(pc * (pc + 1e-30).log()).sum(-1)
    p_true = pc.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
    pc08 = renorm(onset.pow(1 / 0.8))
    p_true_t08 = pc08.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
    top1_true = (pc.argmax(-1) == cur).float()

    kth = onset.topk(top_k, dim=-1).values[..., -1:]                 # top-k threshold
    in_top = ((onset >= kth) & compat).float()
    in_top200 = (onset * in_top).sum(-1) / (onset * compat).sum(-1).clamp_min(1e-30)

    return {
        "valid": valid, "entropy": entropy, "n_compatible": compat.sum(-1).float(),
        "p_true": p_true, "p_true_t08": p_true_t08, "top1_true": top1_true,
        "in_top200": in_top200,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ReaLchords")
    p.add_argument("--dataset_name", default="hooktheory")
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--num_songs", type=int, default=512, help="-1 for the whole split")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--frames_per_bar", type=int, default=16)
    p.add_argument("--top_k", type=int, default=200, help="what the app fetches")
    p.add_argument("--chord_names_path",
                   default=str(Path(REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))
    p.add_argument("--output_dir", default="logs/eval/completion_timing")
    args = p.parse_args()

    agent = agent_interface.Agent(compile=False, voicings_path=None)
    model, tk = agent.models[args.model], agent.tokenizer
    tables = onset_tables(agent, agent.device)
    silent_note = int(agent.melody_to_frame_tokens([], 1)[0])  # what complete mode sends

    loader = build_loader(args, agent)
    n_batches = len(loader) if args.num_songs < 0 else math.ceil(args.num_songs / args.batch_size)
    sums: Dict[str, Dict[str, Dict[str, float]]] = {"silent": {}, "melody": {}}

    for i, batch in enumerate(tqdm(loader, total=n_batches, desc=args.model)):
        if i >= n_batches:
            break
        seq = batch["targets"].to(agent.device)
        current = current_onset_ids(seq[:, 1::2], tk)
        silent = seq.clone()
        mel = silent[:, 2::2]                                         # melody lane
        real = (mel != tk.pad_token) & (mel != tk.eos_token)
        mel[real] = silent_note
        for cond, s in (("melody", seq), ("silent", silent)):
            m = measure(model, s, current, tables, tk, args.top_k)
            frames = torch.arange(m["valid"].size(1), device=seq.device).expand_as(m["valid"])
            for b, k in m["valid"].nonzero().tolist():
                pos = POSITIONS.get(int(frames[b, k]) % args.frames_per_bar, "16th offset")
                row = sums[cond].setdefault(pos, {x: 0.0 for x in METRICS + ["n"]})
                row["n"] += 1
                for x in METRICS:
                    row[x] += float(m[x][b, k])

    results = {}
    for cond in ("silent", "melody"):
        print(f"\n{args.model}, melody {cond}: held = true chord's root + third")
        print(f"  {'position':<12} {'frames':>8} {'compatible':>10} {'eff.choices':>11} "
              f"{'P(true)':>8} {'P(true)@0.8':>11} {'top1=true':>9} {'in top200':>9}")
        results[cond] = {}
        for pos in ORDER:
            row = sums[cond].get(pos)
            if not row:
                continue
            n = row["n"]
            mean = {x: row[x] / n for x in METRICS}
            results[cond][pos] = {"n_frames": int(n), **mean}
            print(f"  {pos:<12} {int(n):>8,} {mean['n_compatible']:>10.0f} "
                  f"{math.exp(mean['entropy']):>11.1f} {mean['p_true']:>8.2f} "
                  f"{mean['p_true_t08']:>11.2f} {mean['top1_true']:>9.1%} {mean['in_top200']:>9.1%}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"completion_timing_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
