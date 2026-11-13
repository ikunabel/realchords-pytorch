#!/usr/bin/env python3
"""Has RL changed what the chord model predicts the *melody* will do?

Under commit-ahead (commit_replay.py) the chord is chosen against a melody the model imagines
itself, and GAPT's imagined melody has 24-40% fewer note onsets than the real one
(journal/experiments/COMMIT_AHEAD.md). Candidate cause: RL never trains the melody
predictions -- advantages and the KL reward are read at chord positions only -- yet they share
every weight with the chord predictions, so they can drift.

This tests that directly, without sampling. One teacher-forced pass over REAL test windows
(real chords, real melody); at every melody position, read the model's predicted distribution
for the next melody token and record:

    p_onset    probability of a note onset (NOTE_ON_*), under the distribution the sampler
               actually draws from: ReaLJam's melody validity filter, temperature 0.8
    real       whether the real next token is an onset (the rate p_onset should match)
    nll        negative log-likelihood of the real next token, raw logits (no filter, T = 1)

Pairs compared, each an RL model against the exact checkpoint it was initialised from:

    7sets       MLE decoder seed S  vs  GAPT seed S (gapt_loo_7sets, entropy bonus 0.01)
    realchords  decoder_only_online_chord  vs  realchords_rerun (entropy bonus 0)

The second pair separates the two possible causes: if its melody predictions also lost onsets,
plain drift is enough; if only GAPT's did, the entropy bonus -- the one RL term that reaches
melody positions -- is implicated. The realchords pair uses the OLD chord vocabulary (5,902
tokens), so its test windows are remapped from the current one by name and windows using a chord
it does not know are dropped (journal/reference/TOKENIZERS.md, section 0). Every model's tokenizer is
checked against its data before anything runs.

Usage:
    python scripts/eval/custom_eval/melody_probe.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from commit_replay import load_models  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.decoder_only import LitDecoder
from realchords.model.sampling import filter_invalid_tokens
from realchords.utils.inference_utils import load_lit_model, load_rl_model
from realchords.utils.reward_scoring import build_token_remap

RUNS = Path("/hpcwork/thes2192/realchords/logs/my_logs")
RC_BASE = RUNS / "decoder_only/decoder_only_online_chord/step=11000.ckpt"
RC_RL = RUNS / "rl/realchords_rerun/actor.pth"


def _tok_from(path: Path) -> HooktheoryTokenizer:
    names = json.loads(path.read_text())
    return HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names))


@torch.no_grad()
def probe(model, tok, seqs: torch.Tensor, device, temperature: float, batch: int = 64) -> dict:
    """seqs: [N, 513] = BOS + interleaved (chord, melody), in the model's own vocabulary."""
    on_lo, on_hi = tok.note_on_token_range
    pad = tok.pad_token
    p_sum = real_on = nll_sum = n = 0.0
    for start in range(0, seqs.shape[0], batch):
        x = seqs[start:start + batch].to(device)
        logits, _ = model.net(x[:, :-1], return_intermediates=True)
        frames = (x.shape[1] - 1) // 2
        for t in range(1, frames):                     # melody frame t sits at x[:, 2 + 2t]
            tgt, prev = x[:, 2 + 2 * t], x[:, 2 * t]
            valid = (tgt != pad) & (prev != pad)
            if not valid.any():
                continue
            lg = logits[:, 1 + 2 * t].float()[valid]     # predicts x[:, 2 + 2t]
            tgt_v, prev_v = tgt[valid], prev[valid]
            nll_sum += float(-F.log_softmax(lg, -1).gather(1, tgt_v[:, None]).sum())
            filt = filter_invalid_tokens(lg, "melody", prev_v, tok)
            p = F.softmax(filt / temperature, dim=-1)
            p_sum += float(p[:, on_lo:on_hi + 1].sum())
            real_on += float(((tgt_v >= on_lo) & (tgt_v <= on_hi)).sum())
            n += int(valid.sum())
    return {"p_onset": p_sum / n, "real_onset": real_on / n, "nll": nll_sum / n, "positions": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["hooktheory", "pop909"])
    ap.add_argument("--n_windows", type=int, default=-1, help="-1 = every test window")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--gt_root", type=Path, default=Path("logs/custom_eval/dataset_loo_new"))
    ap.add_argument("--out", type=Path,
                    default=Path("logs/custom_eval/commit_replay_imagined/melody_probe.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lim = None if args.n_windows < 0 else args.n_windows
    results = {}
    print(f"{'pair':<11}{'corpus':<12}{'model':<12}{'windows':>8}{'real onset':>12}"
          f"{'P(onset)':>10}{'ratio':>7}{'NLL':>8}")

    def report(pair, corpus, label, r, n_win):
        results[f"{pair}/{corpus}/{label}"] = {**r, "windows": n_win}
        print(f"{pair:<11}{corpus:<12}{label:<12}{n_win:>8}{r['real_onset']:>12.3f}"
              f"{r['p_onset']:>10.3f}{r['p_onset'] / r['real_onset']:>7.2f}{r['nll']:>8.3f}",
              flush=True)

    # --- 7sets: current vocabulary on both sides ------------------------------------------
    stages = {"mle": load_models("mle", device), "gapt": load_models("gapt", device)}
    for corpus in args.corpora:
        data_tok = _tok_from(args.gt_root / corpus / "chord_names_augmented.json")
        gt = torch.load(args.gt_root / corpus / "gt" / "gt.pt", map_location="cpu",
                        weights_only=False)[:lim]
        for seed in ("s42", "s43", "s44"):
            for stage, models in stages.items():
                model, tok = models[seed]
                assert tok.name_to_id == data_tok.name_to_id, f"{stage}_{seed}: vocab != data"
                report("7sets", corpus, f"{stage}_{seed}",
                       probe(model, tok, gt, device, args.temperature), gt.shape[0])
    del stages
    torch.cuda.empty_cache()

    # --- realchords pair: OLD vocabulary, so remap the Hooktheory windows by name -----------
    base, rc_tok, _ = load_lit_model(model_path=str(RC_BASE), lit_module_cls=LitDecoder,
                                     batch_size=1, compile=False)
    rl = load_rl_model(model_path=str(RC_RL), model=copy.deepcopy(base), compile=False)
    base, rl = base.eval().to(device), rl.eval().to(device)
    assert rc_tok.num_tokens == 5902, f"expected the old vocabulary, got {rc_tok.num_tokens}"
    data_tok = _tok_from(args.gt_root / "hooktheory" / "chord_names_augmented.json")
    gt = torch.load(args.gt_root / "hooktheory" / "gt" / "gt.pt", map_location="cpu",
                    weights_only=False)[:lim]
    table = build_token_remap(data_tok, rc_tok, allow_missing=True)
    mapped = table[gt]
    keep = ~(mapped < 0).any(dim=1)
    mapped = mapped[keep]
    print(f"  [realchords] {int(keep.sum())}/{len(keep)} Hooktheory windows use only chords the "
          f"old vocabulary knows; the rest are dropped", flush=True)
    for label, model in (("anchor", base), ("rl", rl)):
        report("realchords", "hooktheory", label,
               probe(model, rc_tok, mapped, device, args.temperature), int(keep.sum()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
