"""Do Hooktheory-only reward models generalise to the other corpora, or must they be retrained?

The LOO ablations fine-tune each dataset mixture with RL, and RL needs reward models. Training a
contrastive+discriminative pair (x2 seeds, x2 rhythm variants = 8 runs) for each of the 9 mixtures is
expensive, so the question is whether the existing reward models transfer. This probe answers it
without training anything, on three levels:

  1. **Vocabulary coverage.** A reward model's chord embedding table is fixed to its training
     vocabulary. The Hooktheory-era models know 2,821 chords; the current corpora use 2,969. Chords
     outside the table cannot be scored at all -- sequences containing one are counted, not scored.
  2. **Discrimination (the decision metric).** For each corpus, real (melody, chord) pairs are
     scored against corrupted ones -- the same melodies with another song's chord track -- and the
     separability is reported as AUC. A reward model that cannot tell real from corrupted on a
     corpus supplies no usable learning signal there, whatever its absolute scores look like. AUC is
     scale-free, so the two families are directly comparable.
  3. **Agreement.** Spearman correlation between the two families' per-sequence scores on real
     pairs. High agreement means the extra data changed the scale but not the ranking, i.e. retraining
     buys little; low agreement on the corpora that a mixture keeps means it buys a lot.

Reward scores are sums over the sequence, as in RL and in custom_evaluation.py's reward metrics.
Checkpoints are the seed replicates named in configs/rl/gapt_hooktheory.yml (hooktheory) and
configs/rl/gapt_7sets.yml (7sets); scores are averaged over them.

Usage (GPU):
    python scripts/eval/probe_reward_model_domain.py --n 256 \
        --out logs/custom_eval/gt/reward_model_domain.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.reward_scoring import _FN_CLS, load_reward_model

HPC = "/hpcwork/thes2192/realchords/logs/my_logs"

FAMILIES = {
    "hooktheory": {
        "contrastive": [f"{HPC}/contrastive_reward/contrastive_reward/step=8000.ckpt",
                        f"{HPC}/contrastive_reward/contrastive_reward_2/step=8000.ckpt"],
        "discriminative": [f"{HPC}/discriminative_reward/discriminative_reward_128_bs/step=1500.ckpt",
                           f"{HPC}/discriminative_reward/discriminative_reward_128_bs_2/step=1500.ckpt"],
    },
    "7sets": {
        "contrastive": [f"{HPC}/contrastive_reward/contrastive_reward_7sets_jeti8h2l/step=4000.ckpt",
                        f"{HPC}/contrastive_reward/contrastive_reward_2_7sets_qsnz0vp2/step=6000.ckpt"],
        "discriminative": [f"{HPC}/discriminative_reward/discriminative_reward_128_bs_7sets_ylftoy2r/step=2000.ckpt",
                           f"{HPC}/discriminative_reward/discriminative_reward_128_bs_2_7sets_d2xc1z1b/step=3000.ckpt"],
    },
}
# the Hooktheory-era reward runs predate the chord_names_path arg; their vocabulary is this file
LEGACY_VOCAB = {"hooktheory": "data/cache/old_chord_names_augmented.json"}

# the 7 datasets the 7sets reward models were trained on; filobass/wjd are out of domain for both
SEVEN = ("hooktheory", "pop909", "nottingham", "wikifonia", "chord_melody_dataset", "jazzmus",
         "emopia_plus")


def family_tokenizer(family: str, legacy: HooktheoryTokenizer) -> HooktheoryTokenizer:
    """The vocabulary a family's reward models were trained on (read once, not per checkpoint)."""
    if family == "hooktheory":
        return legacy
    import argbind
    ckpt = Path(next(iter(FAMILIES[family].values()))[0])
    path = argbind.load_args(ckpt.parent / "args.yml").get("chord_names_path")
    return HooktheoryTokenizer(chord_names=json.load(open(path)))


def corpus_dirs(root: Path, view: str) -> Dict[str, Path]:
    return {d.parent.name.replace("_all", ""): d
            for d in sorted(root.glob(f"*_all/{view}")) if (d / "gt.pt").exists()}


def scorable_mask(tensors: Sequence[torch.Tensor], src: HooktheoryTokenizer,
                  dst: HooktheoryTokenizer) -> tuple:
    """(remap table with -1 for unknown names, mask of rows every tensor can express).

    All tensors must be maskable together: a corrupted sequence borrows another row's chord track,
    so a row is scorable only if both its own and its donor's tokens exist in the reward model's
    vocabulary.
    """
    table = torch.full((src.num_tokens,), -1, dtype=torch.long)
    for name, src_id in src.name_to_id.items():
        if name in dst.name_to_id:
            table[src_id] = dst.name_to_id[name]
    ok = torch.ones(tensors[0].size(0), dtype=torch.bool)
    for t in tensors:
        ok &= (table[t] >= 0).all(dim=1)
    return table, ok


def corrupt(tensor: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Same melodies, another song's chord track (a derangement, so no sequence keeps its own)."""
    n = tensor.size(0)
    perm = rng.permutation(n)
    fixed = np.nonzero(perm == np.arange(n))[0]
    for i in fixed:                                   # break fixed points
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    out = tensor.clone()
    out[:, 1::2] = tensor[perm, 1::2]
    return out


@torch.no_grad()
def per_sequence(model, fn_cls, rtok, tensor, table, device, batch_size=64) -> np.ndarray:
    if table is not None:
        assert int(table[tensor].min()) >= 0, "unmapped token reached scoring"
    assert int((table[tensor] if table is not None else tensor).max()) < rtok.num_tokens, \
        "token id outside the reward model's embedding table"
    fn = fn_cls(model=model, pad_token_id=rtok.pad_token, bos_token_id=rtok.bos_token,
                eos_token_id=rtok.eos_token, model_part="chord")
    out = []
    for start in range(0, tensor.size(0), batch_size):
        batch = tensor[start : start + batch_size]
        if table is not None:
            batch = table[batch]
        samples = SimpleNamespace(
            sequences=batch.to(device),
            action_mask=torch.ones_like(batch[:, 1:], dtype=torch.bool, device=device))
        out.append(fn(samples)["reward"].sum(dim=1).float().cpu().numpy())
    return np.concatenate(out)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(real scores above corrupted), ties counted as half."""
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks within ties
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n1, n2 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs/custom_eval/gt")
    ap.add_argument("--view", default="cropped_songs", help="cropped_songs: uniform 128-frame crops")
    ap.add_argument("--n", type=int, default=256, help="sequences sampled per corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/custom_eval/gt/reward_model_domain.json")
    ap.add_argument("--families_json", default=None,
                    help="override FAMILIES: {'name': {'contrastive': [ckpt...], ...}}")
    ap.add_argument("--legacy_vocab", default=LEGACY_VOCAB["hooktheory"],
                    help="vocabulary of the Hooktheory-era reward runs, whose args.yml omits it")
    args = ap.parse_args()

    if args.families_json:
        FAMILIES.clear()
        FAMILIES.update(json.load(open(args.families_json)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy_tok = HooktheoryTokenizer(chord_names=json.load(open(args.legacy_vocab)))
    fam_tok = {f: family_tokenizer(f, legacy_tok) for f in FAMILIES}
    for f, t in fam_tok.items():
        print(f"{f} reward vocabulary: {t.num_tokens} tokens")
    rng = np.random.default_rng(args.seed)
    corpora = corpus_dirs(Path(args.root), args.view)
    if not corpora:
        raise SystemExit(f"no <corpus>_all/{args.view}/gt.pt under {args.root}")

    results: Dict[str, dict] = {}
    for name, d in corpora.items():
        tensor = torch.load(d / "gt.pt")
        tok = HooktheoryTokenizer(chord_names=json.load(open(d / "chord_names_augmented.json")))
        idx = rng.permutation(tensor.shape[0])[: args.n]
        real = tensor[sorted(idx.tolist())]
        fake = corrupt(real, rng)
        # score every family on identical rows: those expressible in *all* reward vocabularies
        tables, common = {}, torch.ones(real.size(0), dtype=torch.bool)
        coverage = {}
        for family in FAMILIES:
            table, ok = scorable_mask([real, fake], tok, fam_tok[family])
            tables[family] = None if fam_tok[family].name_to_id == tok.name_to_id else table
            coverage[family] = float(ok.float().mean())
            common &= ok
        entry = {"sequences": int(real.size(0)), "sequences_scored": int(common.sum()),
                 "in_7sets": name in SEVEN, "coverage": coverage, "families": {}}
        if not common.any():
            results[name] = entry
            print(f"  {name}: no sequence expressible in every reward vocabulary", flush=True)
            continue
        r_ok, f_ok = real[common], fake[common]
        for family, kinds in FAMILIES.items():
            for kind, ckpts in kinds.items():
                aucs, gaps, reals = [], [], []
                for ckpt in ckpts:
                    model, rtok = load_reward_model(ckpt, kind, device, fam_tok[family])
                    rs = per_sequence(model, _FN_CLS[kind], rtok, r_ok, tables[family], device)
                    fs = per_sequence(model, _FN_CLS[kind], rtok, f_ok, tables[family], device)
                    aucs.append(auc(rs, fs)); gaps.append(float(rs.mean() - fs.mean()))
                    reals.append(rs)
                    del model; torch.cuda.empty_cache()
                entry["families"][f"{family}/{kind}"] = {
                    "coverage": coverage[family], "auc": float(np.mean(aucs)),
                    "real_minus_corrupted": float(np.mean(gaps)),
                    "real_mean": float(np.mean([r.mean() for r in reals])),
                    "_real_scores": np.mean(reals, axis=0).tolist(),
                }
        for kind in ("contrastive", "discriminative"):
            a = entry["families"].get(f"hooktheory/{kind}", {}).get("_real_scores")
            b = entry["families"].get(f"7sets/{kind}", {}).get("_real_scores")
            if a and b and len(a) == len(b):
                entry[f"spearman_{kind}"] = spearman(np.array(a), np.array(b))
        for f in entry["families"].values():
            f.pop("_real_scores", None)
        results[name] = entry
        print(f"  scored {name}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))

    for kind in ("contrastive", "discriminative"):
        print(f"\n=== {kind}: AUC real vs corrupted (0.5 = no signal), coverage, rank agreement\n")
        print(f"{'corpus':22s}{'in 7sets':>9s}{'hooktheory AUC':>16s}{'7sets AUC':>11s}"
              f"{'ht coverage':>13s}{'spearman':>10s}{'n':>8s}")
        for name, r in results.items():
            h = r["families"].get(f"hooktheory/{kind}", {})
            s = r["families"].get(f"7sets/{kind}", {})
            fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "n/a"
            print(f"{name:22s}{str(r['in_7sets']):>9s}{fmt(h.get('auc')):>16s}"
                  f"{fmt(s.get('auc')):>11s}{fmt(h.get('coverage')):>13s}"
                  f"{fmt(r.get(f'spearman_{kind}')):>10s}{r['sequences_scored']:>8d}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
