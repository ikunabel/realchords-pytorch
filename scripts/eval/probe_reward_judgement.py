"""Do reward models trained on different data score sequences differently, beyond what a new seed does?

Every reward model scores the same sequences: per test window, the real chords and the chords generated
by MLE and GAPT (7sets_s42/43/44). Scores are standardised per model within a corpus, because absolute
scores differ between models. Then per window:

    d = mean |z_a - z_b| over model pairs across families - mean |z_a - z_b| over same-family seed pairs

7sets (checkpoints the GAPT LOO runs used) is compared with every family trained without some corpus,
in the current vocabulary, on the corpora that family has not seen but 7sets has: 7sets_no_<C> on C (one
per LOO variant), hooktheory_only on the other six. Per reward type: contrastive and discriminative,
harmony and rhythm.

Standardising removes each model's scale, and a reward's scale relative to the others matters in RL
(advantages are normalised over the summed reward, not per term). So the score spread (sd of raw scores
over a corpus's sequences) is reported too: family over 7sets.

No significance tests. Each quantity comes with a 95% confidence interval from a bootstrap over windows
(resampled within each corpus): mean d, the ratio of across-family to same-family disagreement, and the
spread ratio. For scale, the largest seed-to-seed spread ratio within a family is reported next to it.
Windows are the replicates, so the intervals are about these trained models on new windows, not about
retraining in general (2 seeds per reward type). Raw scores: reward_judgement_scores.npz.

Usage (GPU): sbatch scripts/jobscripts/eval/run_reward_judgement.sh
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import argbind
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "scripts/data")
from probe_reward_model_domain import HPC  # noqa: E402
from select_loo_checkpoints import read_val_history  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.contrastive_reward_rhythm import LitContrastiveRewardRhythm
from realchords.lit_module.discriminative_reward_rhythm import LitDiscriminativeRewardRhythm
from realchords.rl.reward.model_based_rewards import (ContrastiveRewardRhythmFn,
                                                      DiscriminativeRewardRhythmFn)
from realchords.utils.reward_scoring import _FN_CLS, load_reward_model

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham", "wikifonia",
           "chord_melody_dataset", "jazzmus"]
MLE, GAPT = Path("logs/custom_eval/dataset_loo_new"), Path("logs/custom_eval/gapt_loo")
SEEDS = ["s42", "s43", "s44"]
CONFIG = "configs/rl/gapt_loo_7sets.alpha=0.5.seed=42.yml"
SOURCES = ["real", "gen_mle", "gen_gapt"]
FRAMES = 128
KINDS = ["contrastive", "discriminative", "contrastive_rhythm", "discriminative_rhythm"]
_RHYTHM = {"contrastive_rhythm": (LitContrastiveRewardRhythm, ContrastiveRewardRhythmFn),
           "discriminative_rhythm": (LitDiscriminativeRewardRhythm, DiscriminativeRewardRhythmFn)}


def load_model(ckpt: str, kind: str, device, tok):
    """(model, reward function). Rhythm checkpoints have their own 6-token model class."""
    if kind not in _RHYTHM:
        model, rtok = load_reward_model(ckpt, kind, device, tok)
        assert rtok.name_to_id == tok.name_to_id, f"{ckpt}: vocabulary differs"
        return model, _FN_CLS[kind](model=model, pad_token_id=rtok.pad_token, bos_token_id=rtok.bos_token,
                                    eos_token_id=rtok.eos_token, model_part="chord")
    lit_cls, fn_cls = _RHYTHM[kind]
    path = Path(ckpt)
    args = argbind.load_args(path.parent / "args.yml")
    args["compile"] = False
    with argbind.scope(args):
        lit = lit_cls()
    state = torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    lit.load_state_dict({k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state.items()})
    model = lit.model.to(device).eval()
    if not hasattr(model, "device"):
        model.device = device
    return model, fn_cls(model=model, tokenizer=tok, model_part="chord")


@torch.no_grad()
def score(fn, tensor, device, batch_size=64) -> np.ndarray:
    """Per-sequence reward, summed as in RL."""
    out = []
    for start in range(0, tensor.size(0), batch_size):
        batch = tensor[start : start + batch_size]
        samples = SimpleNamespace(sequences=batch.to(device),
                                  action_mask=torch.ones_like(batch[:, 1:], dtype=torch.bool, device=device))
        out.append(fn(samples)["reward"].sum(dim=1).float().cpu().numpy())
    return np.concatenate(out)


def best_ckpt(folder: str, name: str) -> str:
    runs = [d for d in Path(HPC, folder).glob(f"{name}*") if d.is_dir() and list(d.glob("step=*.ckpt"))]
    assert len(runs) == 1, f"{name}: expected one run with checkpoints, found {runs}"
    ckpts = {int(c.stem.split("=")[1]): c for c in runs[0].glob("step=*.ckpt")}
    usable = {s: v for s, v in read_val_history(runs[0]).items() if s in ckpts}
    assert usable, f"{name}: no validation history for its checkpoints"
    return str(ckpts[min(usable, key=usable.get)])


def trained_family(suffix: str) -> dict:
    c, d = "contrastive_reward", "discriminative_reward"
    r, b = "no_augmentation_rhythm", "128_bs"
    return {"contrastive": [best_ckpt(c, f"{c}_{suffix}"), best_ckpt(c, f"{c}_2_{suffix}")],
            "discriminative": [best_ckpt(d, f"{d}_{b}_{suffix}"), best_ckpt(d, f"{d}_{b}_2_{suffix}")],
            "contrastive_rhythm": [best_ckpt(c, f"{c}_{r}_{suffix}"), best_ckpt(c, f"{c}_{r}_2_{suffix}")],
            "discriminative_rhythm": [best_ckpt(d, f"{d}_{r}_{suffix}"), best_ckpt(d, f"{d}_{r}_2_{suffix}")]}


def families() -> dict:
    cfg = yaml.safe_load(open(CONFIG))
    return {"hooktheory_only": trained_family("hooktheory_only"),
            **{f"no_{c}": trained_family(f"7sets_no_{c}") for c in CORPORA},
            "7sets": {"contrastive": cfg["contrastive_reward_model_path"],
                      "discriminative": cfg["discriminative_reward_model_path"],
                      "contrastive_rhythm": cfg["contrastive_reward_rhythm_model_path"],
                      "discriminative_rhythm": cfg["discriminative_reward_rhythm_model_path"]}}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare(n: int, rng) -> tuple:
    """Per corpus: real chords and three seeds each of MLE and GAPT on shared full-length rows."""
    L = 2 * FRAMES + 1
    out, tok = {}, None
    for c in CORPORA:
        tok = HooktheoryTokenizer(chord_names=json.load(open(MLE / c / "chord_names_augmented.json")))
        gt_m, gt_g = load(MLE / c / "gt" / "gt.pt"), load(GAPT / c / "gt" / "gt.pt")
        m = min(len(gt_m), len(gt_g))
        assert torch.equal(gt_m[:m], gt_g[:m]), f"{c}: ground truth differs"
        full = (gt_m[:m, :L] != tok.pad_token).all(dim=1).nonzero().squeeze(1).numpy()
        rows = np.sort(rng.permutation(full)[:n])
        out[c] = {"real": gt_m[rows][:, :L],
                  "gen_mle": torch.cat([load(MLE / c / "models" / f"7sets_{s}" / "preds.pt")[rows][:, :L] for s in SEEDS]),
                  "gen_gapt": torch.cat([load(GAPT / c / "models" / f"7sets_{s}" / "preds.pt")[rows][:, :L] for s in SEEDS])}
    return out, tok


def bootstrap(sizes: list, stat, reps: int, rng) -> list:
    """stat(per-corpus window-index arrays) -> float. [estimate, 2.5%, 97.5%], windows resampled
    within each corpus."""
    est = stat([np.arange(n) for n in sizes])
    draws = [stat([rng.integers(0, n, n) for n in sizes]) for _ in range(reps)]
    return [est, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/reward_judgement.json"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fams = families()
    data, tok = prepare(args.n, rng)
    scores = {}
    for family, kinds in fams.items():
        for kind, ckpts in kinds.items():
            for i, ckpt in enumerate(ckpts):
                model, fn = load_model(ckpt, kind, device, tok)
                for c, d in data.items():
                    for s in SOURCES:
                        scores[(family, kind, i, c, s)] = score(fn, d[s], device)
                del model
                torch.cuda.empty_cache()
                print(f"  scored {family}/{kind}/{i}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out.with_name("reward_judgement_scores.npz"),
             **{"|".join(map(str, k)): v for k, v in scores.items()})

    def z_scores(f, kind, i, c):
        """[sequences per window, windows], standardised over the corpus."""
        n = len(data[c]["real"])
        v = np.concatenate([scores[(f, kind, i, c, s)] for s in SOURCES]).reshape(-1, n)
        return (v - v.mean()) / (v.std() or 1.0)

    def disagreement(pairs):
        return np.mean([np.abs(a - b).mean(axis=0) for a, b in pairs], axis=0)      # per window

    unseen = {"hooktheory_only": [c for c in CORPORA if c != "hooktheory"], **{f"no_{c}": [c] for c in CORPORA}}

    def raw(f, kind, i, c):
        """[sequences per window, windows], raw scores."""
        n = len(data[c]["real"])
        return np.concatenate([scores[(f, kind, i, c, s)] for s in SOURCES]).reshape(-1, n)

    results = {"rows": {c: int(len(data[c]["real"])) for c in CORPORA}, "checkpoints": fams, "comparisons": {}}
    for kind in KINDS:
        for f, corp in unseen.items():
            across, same, raw_f, raw_7 = [], [], [], []
            for c in corp:
                za, zb = [z_scores(f, kind, i, c) for i in (0, 1)], [z_scores("7sets", kind, i, c) for i in (0, 1)]
                across.append(disagreement(list(itertools.product(za, zb))))
                same.append(disagreement([tuple(za), tuple(zb)]))
                raw_f.append([raw(f, kind, i, c) for i in (0, 1)])
                raw_7.append([raw("7sets", kind, i, c) for i in (0, 1)])

            def diff(ix):
                return float(np.mean([(a[j] - b[j]).mean() for a, b, j in zip(across, same, ix)]))

            def ratio(ix):
                return float(np.mean([a[j].mean() for a, j in zip(across, ix)])
                             / np.mean([b[j].mean() for b, j in zip(same, ix)]))

            def spread(ix):
                return float(np.mean([np.mean([m[:, j].std() for m in fm]) / np.mean([m[:, j].std() for m in sm])
                                      for fm, sm, j in zip(raw_f, raw_7, ix)]))

            sizes = [len(a) for a in across]
            results["comparisons"][f"{f}/{kind}"] = {
                "corpora": corp,
                "across_minus_same": bootstrap(sizes, diff, args.boot, rng),
                "across_over_same": bootstrap(sizes, ratio, args.boot, rng),
                "spread_family_over_7sets": bootstrap(sizes, spread, args.boot, rng),
                "spread_seed_over_seed_max": float(max(max(m.std() for m in ms) / min(m.std() for m in ms)
                                                       for ms in raw_f + raw_7)),
            }
    args.out.write_text(json.dumps(results, indent=2))

    print(f"rows per corpus: {results['rows']}\n")
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    print(f"{'family vs 7sets':<26}{'reward':<23}{'across - same':>22}{'across / same':>22}"
          f"{'spread family/7sets':>22}{'seed spread max':>17}")
    for k, r in results["comparisons"].items():
        f, kind = k.split("/")
        print(f"{f:<26}{kind:<23}{fmt(r['across_minus_same']):>22}{fmt(r['across_over_same']):>22}"
              f"{fmt(r['spread_family_over_7sets']):>22}{r['spread_seed_over_seed_max']:>17.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
