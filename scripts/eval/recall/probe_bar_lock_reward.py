"""Bar-lock probe: do the frozen RL reward models prefer one-chord-per-bar accompaniment?

Question: ReaLchords-M (multiscale reward) collapses to exactly one chord per bar, while
full-context ReaLchords does not (journal/RL_MULTISCALE_RHYTHM_COLLAPSE.md). Is that because the
reward models themselves score bar-locked chords higher than real ones (reward hacking), or did
the policy just stop exploring there?

Method (no RL): Hooktheory test pieces and manipulated copies of their chord track (melody
unchanged) are scored with every reward model of an RL run, computed exactly as the RL reward
functions compute them (same classes, same sliding windows):

  gt              ground-truth chords
  bar_lock        per 16-frame bar (from frame 0), the chord sounding on the downbeat (or the
                  first chord in the bar if the downbeat is silent) held for the whole bar --
                  what ReaLchords-M learned
  bar_lock_beat   same, but bars shifted by one beat (4 frames), so chord changes never coincide
                  with a sliding-window boundary (window starts are multiples of 8 frames)
  half_bar_lock   same with 8-frame blocks (two chords per bar)
  <model>         the policies' own outputs from a benchmark run (--eval_dir/models/<slug>)
  <model>+bar_lock  the same outputs bar-locked -- the relevant RL comparison: does locking the
                  policy's current behaviour raise its reward (GT is not what RL compares against)

Reading the result:
  - short-window scores bar_lock > gt, full-context not -> the multiscale reward pays for
    bar-locking (reward hacking)
  - bar_lock > gt but bar_lock_beat not -> preference for chord changes on window boundaries
    (a windowing artifact)
  - no score prefers bar_lock -> the reward does not favour it; exploration collapse instead
  - rhythm scores: whether the (GAPT-style) rhythm-only rewards favour bar-locking

Scores per source: contrastive/discriminative full-context ("full"), per window scale ("w16" ..)
and the multiscale combination ("combined"), the rhythm models (mean over seeds), their sum as the
RL run weights them ("total_model_reward"), note-in-chord ratio, and the paired fraction of pieces
scored above GT. Output: <out_dir>/bar_lock_probe.json (+ printed table).

Input: gt.pt / models/<slug>/preds.pt from a custom_evaluation.py run on the hooktheory test set
(BOS + interleaved chord/melody tokens, chord first).

Usage (GPU; --limit 16 for a quick check):
    python scripts/eval/recall/probe_bar_lock_reward.py \
        --eval_dir logs/custom_eval/rl_2x2_hooktheory \
        --rl_args /hpcwork/thes2192/realchords/logs/my_logs/rl/realchords_multiscale/args.yml \
        --out_dir logs/custom_eval/rl_2x2_hooktheory/bar_lock_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import argbind
import numpy as np
import torch
import yaml

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.contrastive_reward_rhythm import LitContrastiveRewardRhythm
from realchords.lit_module.contrastive_reward_segment import LitContrastiveRewardSegment
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.lit_module.discriminative_reward_rhythm import LitDiscriminativeRewardRhythm
from realchords.lit_module.discriminative_reward_segment import LitDiscriminativeRewardSegment
from realchords.rl.reward.model_based_rewards import (
    ContrastiveRewardRhythmFn,
    DiscriminativeRewardRhythmFn,
)
from realchords.rl.reward.multiscale_contrastive_rewards import MultiscaleContrastiveRewardFn
from realchords.rl.reward.multiscale_discriminative_rewards import (
    MultiscaleDiscriminativeRewardFn,
)
from realchords.utils.eval_utils import evaluate_note_in_chord_ratio
from realchords.utils.sequence_penalty_analysis import strip_bos

FRAMES_PER_BAR = 16


def lock_chords(seq: torch.Tensor, tok: HooktheoryTokenizer, block: int, offset: int) -> torch.Tensor:
    """Hold one chord per block of ``block`` frames; blocks start at ``offset`` (mod block)."""
    out = seq.clone()
    chord = seq[:, 1::2]  # chord lane (drop BOS; chord first)
    n_chords = len(tok.chord_names)
    on_lo, _ = tok.chord_on_token_range
    hold_lo, hold_hi = tok.chord_hold_token_range
    silence, pad = tok.silence_token, tok.pad_token
    new = chord.clone()
    num_frames = chord.shape[1]
    starts = sorted({0, *range(offset % block, num_frames, block)})
    for b in range(chord.shape[0]):
        row = chord[b]
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else num_frames
            seg = row[s:e]
            valid = seg != pad
            if not valid.any():
                continue
            hold_ids = torch.where(seg >= on_lo, seg - n_chords, seg)
            is_chord = (hold_ids >= hold_lo) & (hold_ids <= hold_hi)
            if not is_chord.any():
                new[b, s:e] = torch.where(valid, torch.full_like(seg, silence), seg)
                continue
            hold = hold_ids[is_chord.nonzero()[0, 0]]
            filled = torch.where(valid, hold, seg)
            filled[0] = hold + n_chords  # onset on the block's first frame
            new[b, s:e] = filled
    out[:, 1::2] = new
    return out


def note_in_chord(seq: torch.Tensor, tok: HooktheoryTokenizer) -> float:
    """Pooled note-in-chord ratio, as the benchmark reports it."""
    _, valid, correct = evaluate_note_in_chord_ratio(
        strip_bos(seq, tok), tok, model_part="chord", return_count=True)
    return float(correct.sum().item()) / max(1, int(valid.sum().item()))


def load(paths, cls, device):
    """load_lit_model without building the training dataloaders (slow for segment models)."""
    models = []
    for p in paths:
        args = argbind.load_args(Path(p).parent / "args.yml")
        args["compile"] = False
        with argbind.scope(args):
            lit = cls()
        sd = torch.load(p, weights_only=True, map_location="cpu")["state_dict"]
        lit.load_state_dict({k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in sd.items()})
        m = lit.model.to(device).eval()
        if not hasattr(m, "device"):
            m.device = device
        models.append(m)
    return models


def fns_vocab_size(ckpt: str) -> int:
    """Token-embedding size of a (non-rhythm) reward checkpoint."""
    sd = torch.load(ckpt, weights_only=True, map_location="cpu")["state_dict"]
    return next(v.shape[0] for k, v in sd.items() if k.endswith("token_emb.emb.weight"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="logs/custom_eval/rl_2x2_hooktheory")
    ap.add_argument("--rl_args", default="/hpcwork/thes2192/realchords/logs/my_logs/rl/realchords_multiscale/args.yml")
    ap.add_argument("--models", nargs="*", default=["mle", "realchords", "realchords_m"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="first N sequences only (quick check)")
    ap.add_argument("--out_dir", default=None, help="default: <eval_dir>/bar_lock_probe")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "bar_lock_probe"
    rl = yaml.safe_load(open(args.rl_args))
    # the model vocabulary the sequences are encoded in (chord_names_augmented.json is the
    # dataset's larger one and gives wrong onset/hold offsets)
    tok = HooktheoryTokenizer(chord_names=json.load(open(eval_dir / "_model_chord_vocab.json")))
    num_reward_tokens = fns_vocab_size(rl["contrastive_reward_model_path"][0])
    assert tok.num_tokens == num_reward_tokens, (tok.num_tokens, num_reward_tokens)

    gt = torch.load(eval_dir / "gt" / "gt.pt")[: args.limit]
    assert gt.shape[1] % 2 == 1 and tok.num_tokens > int(gt.max()), "unexpected gt.pt format"
    for name, fn in (("bar_lock", lambda x: lock_chords(x, tok, FRAMES_PER_BAR, 0)),):
        assert int(fn(gt[:4]).max()) < tok.num_tokens, name
    sources = {
        "gt": gt,
        "bar_lock": lock_chords(gt, tok, FRAMES_PER_BAR, 0),
        "bar_lock_beat": lock_chords(gt, tok, FRAMES_PER_BAR, 4),
        "half_bar_lock": lock_chords(gt, tok, FRAMES_PER_BAR // 2, 0),
    }
    for slug in args.models:
        p = eval_dir / "models" / slug / "preds.pt"
        if p.exists():
            sources[slug] = torch.load(p)[: args.limit]
            # the policy's own output, bar-locked: does locking raise *its* reward?
            sources[f"{slug}+bar_lock"] = lock_chords(sources[slug], tok, FRAMES_PER_BAR, 0)
        else:
            print(f"skipping {slug}: {p} missing")

    common = dict(pad_token_id=tok.pad_token, bos_token_id=tok.bos_token,
                  eos_token_id=tok.eos_token, model_part="chord")
    fns = {
        "contrastive": MultiscaleContrastiveRewardFn(
            legacy_models=load(rl["contrastive_reward_model_path"], LitContrastiveReward, device),
            multiscale_models=load(rl["multiscale_contrastive_reward_model_path"], LitContrastiveRewardSegment, device),
            window_lens=rl["multiscale_contrastive_window_lens"], **common),
        "discriminative": MultiscaleDiscriminativeRewardFn(
            legacy_models=load(rl["discriminative_reward_model_path"], LitDiscriminativeReward, device),
            multiscale_models=load(rl["multiscale_discriminative_reward_model_path"], LitDiscriminativeRewardSegment, device),
            window_lens=rl["multiscale_discriminative_window_lens"], **common),
    }
    rhythm = [
        ("contrastive_rhythm", ContrastiveRewardRhythmFn(model=m, tokenizer=tok, model_part="chord"))
        for m in load(rl["contrastive_reward_rhythm_model_path"], LitContrastiveRewardRhythm, device)
    ] + [
        ("discriminative_rhythm", DiscriminativeRewardRhythmFn(model=m, tokenizer=tok, model_part="chord"))
        for m in load(rl["discriminative_reward_rhythm_model_path"], LitDiscriminativeRewardRhythm, device)
    ]

    per_seq = {}
    for name, seqs in sources.items():
        cols: dict[str, list] = {}
        for i in range(0, seqs.shape[0], args.batch_size):
            batch = seqs[i : i + args.batch_size].to(device)
            samples = SimpleNamespace(
                sequences=batch, action_mask=torch.ones_like(batch[:, 1:], dtype=torch.bool))
            with torch.no_grad():
                for kind, fn in fns.items():
                    for k, v in fn(samples).items():
                        if k.startswith("multiscale_"):
                            short = k.replace(f"multiscale_{kind}_", "")
                            short = "full" if short == "w256" else short
                            cols.setdefault(f"{kind}_{short}", []).append(v.float().cpu())
                for j, (kind, fn) in enumerate(rhythm):
                    cols.setdefault(f"{kind}#{j}", []).append(fn(samples)["reward"].sum(1).float().cpu())
        r = {k: torch.cat(v).numpy() for k, v in cols.items()}
        for kind in ("contrastive_rhythm", "discriminative_rhythm"):
            seeds = [r.pop(k) for k in list(r) if k.startswith(f"{kind}#")]
            r[kind] = np.mean(seeds, 0)
            r[f"{kind}_sum"] = np.sum(seeds, 0)
        # model-based part of the RL reward (weights 1, one term per rhythm model)
        r["total_model_reward"] = (r["contrastive_combined"] + r["discriminative_combined"]
                                   + r.pop("contrastive_rhythm_sum") + r.pop("discriminative_rhythm_sum"))
        per_seq[name] = r
        print(f"scored {name}")

    metrics = list(per_seq["gt"].keys())
    summary = {name: {m: float(v[m].mean()) for m in metrics} for name, v in per_seq.items()}
    for name in sources:
        summary[name]["note_in_chord"] = note_in_chord(sources[name], tok)
        for m in metrics:
            summary[name][f"frac_above_gt/{m}"] = float((per_seq[name][m] > per_seq["gt"][m]).mean())
    summary["_meta"] = {"eval_dir": str(eval_dir), "rl_args": args.rl_args,
                        "num_sequences": int(gt.shape[0])}

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "bar_lock_probe.json", "w"), indent=2)

    names = list(sources)
    print(f"\n{'mean score':28s}" + "".join(f"{n:>15s}" for n in names))
    for m in metrics + ["note_in_chord"]:
        print(f"{m:28s}" + "".join(f"{summary[n][m]:15.4f}" for n in names))
    print("\nfraction of pieces scored above GT (paired)")
    for m in metrics:
        print(f"{m:28s}" + "".join(f"{summary[n][f'frac_above_gt/{m}']:15.3f}" for n in names))
    print(f"\nwrote {out_dir / 'bar_lock_probe.json'}")


if __name__ == "__main__":
    main()
