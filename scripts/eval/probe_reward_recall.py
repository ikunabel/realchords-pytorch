"""Retrieval recall of a reward model: given a melody, does it rank the true chord track first?

This is the reward models' own training objective measured directly. The contrastive model is
trained with InfoNCE, i.e. "pick the matching chord track out of the batch", so recall@k *is* that
task on held-out data. The discriminative model is trained to call a true pair real and a mismatched
pair fake, so ranking candidates by its P(real) is the same question asked of it.

Why recall rather than a pairwise score: it needs no invented corruption and no calibration. Every
query has exactly one correct answer among k candidates, chance is k^-1 by construction, and two
models are comparable as long as k is the same -- unlike the training losses, which are not
comparable at all here (masking cross-corpus candidates lowers InfoNCE's chance level from log 196
to about log 129, so the masked model's loss falls for a reason unrelated to what it learned).

Candidates are drawn **within one corpus**, which is the case that matters: cross-corpus candidates
can be rejected on style alone. Pairs are built by taking a real sequence and replacing its chord
lane with the candidate's, then scored through the same reward functions RL uses.

Output: <out> with recall@1 and recall@5 per corpus per model.

Usage:
    python scripts/eval/probe_reward_recall.py --families_json fams.json --k 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.reward_scoring import _FN_CLS, build_token_remap, load_reward_model


def corpus_dirs(root: Path, view: str) -> dict:
    return {d.parent.name.replace("_all", ""): d
            for d in sorted(root.glob(f"*_all/{view}")) if (d / "gt.pt").exists()}


@torch.no_grad()
def score_pairs(model, kind, rtok, seqs, table, device, batch_size=128):
    fn = _FN_CLS[kind](model=model, pad_token_id=rtok.pad_token, bos_token_id=rtok.bos_token,
                       eos_token_id=rtok.eos_token, model_part="chord")
    out = []
    for start in range(0, seqs.size(0), batch_size):
        batch = seqs[start : start + batch_size]
        if table is not None:
            batch = table[batch]
        samples = SimpleNamespace(
            sequences=batch.to(device),
            action_mask=torch.ones_like(batch[:, 1:], dtype=torch.bool, device=device))
        out.append(fn(samples)["reward"].sum(dim=1).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs/custom_eval/gt")
    ap.add_argument("--view", default="cropped_songs")
    ap.add_argument("--families_json", required=True)
    ap.add_argument("--k", type=int, default=16, help="candidates per query (chance = 1/k)")
    ap.add_argument("--pools", type=int, default=16, help="candidate pools per corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/custom_eval/gt/reward_recall.json")
    args = ap.parse_args()

    families = json.load(open(args.families_json))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    results: dict = {}

    for corpus, d in corpus_dirs(Path(args.root), args.view).items():
        tensor = torch.load(d / "gt.pt")
        tok = HooktheoryTokenizer(chord_names=json.load(open(d / "chord_names_augmented.json")))
        need = args.k * args.pools
        if tensor.size(0) < args.k:
            continue
        idx = rng.permutation(tensor.size(0))[:need]
        pools = [idx[i * args.k : (i + 1) * args.k] for i in range(len(idx) // args.k)]
        # every (query melody, candidate chords) combination, per pool
        combos, truth = [], []
        for pool in pools:
            base = tensor[sorted(pool.tolist())]
            for qi in range(base.size(0)):
                block = base[qi : qi + 1].repeat(base.size(0), 1)   # melody of the query
                block[:, 1::2] = base[:, 1::2]                      # each candidate's chords
                combos.append(block)
                truth.append(qi)
        combos = torch.cat(combos)
        entry = {"queries": len(truth), "k": args.k, "models": {}}
        for family, kinds in families.items():
            for kind, ckpts in kinds.items():
                model, rtok = load_reward_model(ckpts[0], kind, device, tok)
                table = None if rtok.name_to_id == tok.name_to_id else build_token_remap(tok, rtok)
                scores = score_pairs(model, kind, rtok, combos, table, device)
                del model
                torch.cuda.empty_cache()
                scores = scores.view(len(truth), args.k)
                order = scores.argsort(dim=1, descending=True)
                rank = (order == torch.tensor(truth).unsqueeze(1)).float().argmax(dim=1)
                entry["models"][f"{family}/{kind}"] = {
                    "recall@1": float((rank == 0).float().mean()),
                    "recall@5": float((rank < 5).float().mean()),
                    "mean_rank": float(rank.float().mean()) + 1.0,
                }
        results[corpus] = entry
        print(f"  scored {corpus}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    names = sorted({m for e in results.values() for m in e["models"]})
    for metric in ("recall@1", "recall@5"):
        print(f"\n=== {metric} (chance = {1/args.k:.3f} / {min(5, args.k)/args.k:.3f})\n")
        print(f"{'corpus':22s}" + "".join(f"{n[:22]:>24s}" for n in names))
        for corpus, e in results.items():
            print(f"{corpus:22s}" + "".join(f"{e['models'][n][metric]:24.3f}" for n in names))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
