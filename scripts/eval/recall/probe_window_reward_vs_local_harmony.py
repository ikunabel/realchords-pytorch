"""Do the per-scale (multi-scale) reward models actually track *local* harmony?

Motivation: multi-scale rewards do not improve note-in-chord ratio in this implementation, whether
the window scores are aggregated onto the last token or placed at each window's own position
(journal/RL_MULTISCALE_RHYTHM_COLLAPSE.md). Before tuning how the reward is placed or discounted,
check the premise: within one sliding window, does a window model's score move with how well the
chords there fit the melody?

Method (no RL): for each window scale, every sliding window (50% overlap, the same grid used during
RL) of every evaluated sequence is scored with that scale's contrastive and discriminative model,
and paired with the note-in-chord ratio computed over exactly those frames. Reported per scale:
Pearson and Spearman correlation between window score and window note-in-chord ratio, plus the
score gap between the best and worst note-in-chord deciles. The full-context ("legacy") models are
included as a reference at sequence level.

Sources: ground truth plus, optionally, model predictions from the same eval dir -- GT alone has a
narrow harmony range, model outputs widen it, which is what a reward model must separate during RL.

A near-zero correlation means no placement scheme or discount factor can turn these models into a
local harmony signal; a clear correlation means the signal exists and the problem is in how it
reaches the policy.

Output: <out_dir>/window_reward_vs_local_harmony.json (+ printed table).

Usage (GPU):
    python scripts/eval/recall/probe_window_reward_vs_local_harmony.py \
        --eval_dir logs/custom_eval/rl_2x2_hooktheory \
        --rl_args /hpcwork/thes2192/realchords/logs/my_logs/rl/realchords_multiscale/args.yml \
        --out_dir logs/custom_eval/rl_2x2_hooktheory/window_reward_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from scipy import stats

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.contrastive_reward_segment import LitContrastiveRewardSegment
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.lit_module.discriminative_reward_segment import LitDiscriminativeRewardSegment
from realchords.rl.reward.multiscale_contrastive_rewards import (
    MULTISCALE_OVERLAP_FRACTION,
    MultiscaleContrastiveRewardFn,
    split_interleaved_lanes,
    valid_frame_lengths,
    window_starts,
)
from realchords.rl.reward.multiscale_discriminative_rewards import MultiscaleDiscriminativeRewardFn
from realchords.utils.eval_utils import evaluate_note_in_chord_per_frame
from realchords.utils.sequence_penalty_analysis import strip_bos


def window_frames(valid_lens: torch.Tensor, window_len: int) -> List[tuple]:
    """(sequence index, first frame, last frame+1) for every window, in the RL window order."""
    stride = max(1, int(window_len * MULTISCALE_OVERLAP_FRACTION))
    spans = []
    for i, n in enumerate(valid_lens.tolist()):
        for start in window_starts(n, window_len, stride):
            spans.append((i, start, start + min(window_len, n - start)))
    return spans


def local_nicr(tensor: torch.Tensor, tok: HooktheoryTokenizer, spans: List[tuple]) -> np.ndarray:
    """Note-in-chord ratio inside each window (NaN when the window has no valid melody frame)."""
    per_frame = evaluate_note_in_chord_per_frame(strip_bos(tensor, tok), tok)
    matches, valid = per_frame["matches"], per_frame["valid"]
    out = np.full(len(spans), np.nan)
    for k, (i, lo, hi) in enumerate(spans):
        v = valid[i, lo:hi]
        if bool(v.any()):
            out[k] = float(matches[i, lo:hi][v].mean())
    return out


@torch.no_grad()
def window_scores(fn, model, model_tokens, context_tokens, valid_lens, window_len) -> np.ndarray:
    _, _, scores = fn._score_sliding_average(
        model, model_tokens, context_tokens, valid_lens, window_len=window_len)
    return scores.float().cpu().numpy()


def summarize(score: np.ndarray, nicr: np.ndarray, seq_idx: np.ndarray = None) -> Dict[str, float]:
    keep = ~(np.isnan(score) | np.isnan(nicr))
    s, n = score[keep], nicr[keep]
    within = {}
    if seq_idx is not None and len(s) > 10:
        # subtract each sequence's own mean: does the score track harmony *within* a piece,
        # rather than merely separating well- from badly-harmonised pieces?
        idx = seq_idx[keep]
        order = np.argsort(idx, kind="stable")
        s_c, n_c = s[order].copy(), n[order].copy()
        ids = idx[order]
        bounds = np.flatnonzero(np.diff(ids)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(ids)]):
            if hi - lo > 1:
                s_c[lo:hi] -= s_c[lo:hi].mean()
                n_c[lo:hi] -= n_c[lo:hi].mean()
            else:
                s_c[lo:hi] = n_c[lo:hi] = 0.0
        nz = (s_c != 0) | (n_c != 0)
        if nz.sum() > 10 and s_c[nz].std() > 0 and n_c[nz].std() > 0:
            within = {"pearson_within_sequence": float(stats.pearsonr(s_c[nz], n_c[nz])[0])}
    if len(s) < 10 or s.std() == 0:
        return {"n": int(len(s)), **within}
    lo, hi = np.percentile(n, 10), np.percentile(n, 90)
    return {
        "n": int(len(s)),
        **within,
        "pearson": float(stats.pearsonr(s, n)[0]),
        "spearman": float(stats.spearmanr(s, n)[0]),
        "score_mean": float(s.mean()),
        "score_sd": float(s.std()),
        # score gap between the worst and best note-in-chord deciles, in score sds
        "decile_gap": float((s[n >= hi].mean() - s[n <= lo].mean())),
        "decile_gap_in_sd": float((s[n >= hi].mean() - s[n <= lo].mean()) / s.std()),
    }


def report(key: str, name: str, r: Dict[str, float]) -> None:
    if "pearson" not in r:
        print(f"{key:24s} {name:14s} n={r['n']:7d} (too few windows)")
        return
    within = r.get("pearson_within_sequence")
    print(f"{key:24s} {name:14s} n={r['n']:7d} pearson={r['pearson']:+.3f} "
          f"spearman={r['spearman']:+.3f} "
          + (f"within-seq={within:+.3f} " if within is not None else "")
          + f"decile_gap={r['decile_gap']:+.3f} ({r['decile_gap_in_sd']:+.2f} sd)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="logs/custom_eval/rl_2x2_hooktheory")
    ap.add_argument("--rl_args", default="/hpcwork/thes2192/realchords/logs/my_logs/rl/realchords_multiscale/args.yml")
    ap.add_argument("--models", nargs="*", default=[],
                    help="model slugs from <eval_dir>/models/ to score alongside GT")
    ap.add_argument("--limit", type=int, default=None, help="first N sequences only")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out_dir", default=None, help="default: <eval_dir>/window_reward_probe")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "window_reward_probe"
    rl = yaml.safe_load(open(args.rl_args))
    tok = HooktheoryTokenizer(chord_names=json.load(open(eval_dir / "_model_chord_vocab.json")))

    sources = {"gt": torch.load(eval_dir / "gt" / "gt.pt")[: args.limit]}
    for slug in args.models:
        p = eval_dir / "models" / slug / "preds.pt"
        if p.exists():
            sources[slug] = torch.load(p)[: args.limit]
        else:
            print(f"skipping {slug}: {p} missing")

    legacy = {"contrastive": (rl["contrastive_reward_model_path"][0], LitContrastiveReward,
                              MultiscaleContrastiveRewardFn),
              "discriminative": (rl["discriminative_reward_model_path"][0], LitDiscriminativeReward,
                                 MultiscaleDiscriminativeRewardFn)}
    kinds = {
        "contrastive": (rl["multiscale_contrastive_reward_model_path"],
                        rl["multiscale_contrastive_window_lens"],
                        LitContrastiveRewardSegment, MultiscaleContrastiveRewardFn),
        "discriminative": (rl["multiscale_discriminative_reward_model_path"],
                           rl["multiscale_discriminative_window_lens"],
                           LitDiscriminativeRewardSegment, MultiscaleDiscriminativeRewardFn),
    }
    common = dict(pad_token_id=tok.pad_token, bos_token_id=tok.bos_token,
                  eos_token_id=tok.eos_token, model_part="chord")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kind, (paths, lens, lit_cls, fn_cls) in kinds.items():
        for path, window_len in zip(paths, lens):
            model = load_segment_model(path, lit_cls, device)
            fn = fn_cls(legacy_models=[model], multiscale_models=[model], window_lens=[window_len],
                        **common)
            for name, seqs in sources.items():
                scores, nicrs, seq_ids = [], [], []
                for start in range(0, seqs.shape[0], args.batch_size):
                    batch = seqs[start : start + args.batch_size].to(device)
                    model_tokens, context_tokens = split_interleaved_lanes(
                        batch, tok.pad_token, tok.bos_token)
                    valid_lens = valid_frame_lengths(model_tokens, tok.pad_token)
                    scores.append(window_scores(fn, model, model_tokens, context_tokens,
                                                valid_lens, window_len))
                    spans = window_frames(valid_lens.cpu(), window_len)
                    nicrs.append(local_nicr(batch.cpu(), tok, spans))
                    seq_ids.append(np.array([start + i for i, _, _ in spans]))
                key = f"{kind}_w{window_len}"
                results.setdefault(key, {})[name] = summarize(
                    np.concatenate(scores), np.concatenate(nicrs), np.concatenate(seq_ids))
                report(key, name, results[key][name])
            del model, fn
            torch.cuda.empty_cache()

    # full-context models, for comparison: one score per sequence vs that sequence's note-in-chord
    for kind, (path, lit_cls, fn_cls) in legacy.items():
        model = load_segment_model(path, lit_cls, device)
        fn = fn_cls(legacy_models=[model], multiscale_models=[model], window_lens=[16], **common)
        for name, seqs in sources.items():
            scores, nicrs = [], []
            for start in range(0, seqs.shape[0], args.batch_size):
                batch = seqs[start : start + args.batch_size].to(device)
                model_tokens, context_tokens = split_interleaved_lanes(
                    batch, tok.pad_token, tok.bos_token)
                valid_lens = valid_frame_lengths(model_tokens, tok.pad_token)
                with torch.no_grad():
                    scores.append(fn._score_full_legacy(
                        model, model_tokens, context_tokens, valid_lens).float().cpu().numpy())
                spans = [(i, 0, int(n)) for i, n in enumerate(valid_lens.cpu().tolist())]
                nicrs.append(local_nicr(batch.cpu(), tok, spans))
            key = f"{kind}_full"
            results.setdefault(key, {})[name] = summarize(
                np.concatenate(scores), np.concatenate(nicrs))
            report(key, name, results[key][name])
        del model, fn
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"eval_dir": str(eval_dir), "rl_args": args.rl_args,
               "sources": {k: int(v.shape[0]) for k, v in sources.items()}, "results": results}
    (out_dir / "window_reward_vs_local_harmony.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'window_reward_vs_local_harmony.json'}")


def load_segment_model(path: str, lit_cls, device):
    """Segment reward checkpoint without building its training dataloaders."""
    import argbind
    cfg = argbind.load_args(Path(path).parent / "args.yml")
    cfg["compile"] = False
    with argbind.scope(cfg):
        lit = lit_cls()
    state = torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    lit.load_state_dict({k.replace("_orig_mod.", "").replace("._orig_mod", ""): v
                         for k, v in state.items()})
    model = lit.model.to(device).eval()
    if not hasattr(model, "device"):
        model.device = device
    return model


if __name__ == "__main__":
    main()
