#!/usr/bin/env python3
"""Two follow-ups to the LOO test, from the existing evaluation outputs.

Q3, specificity: does holding out corpus C increase the distance to real music more on C than on
the other corpora? For each metric and stage, build a matrix with one row per training mix
(C held out) and one column per test corpus D:

    M[C, D] = mean over seeds of (dist(D, no_C_seed) - mean dist(D, full models)) / sd_D

sd_D = sd of the distances of all full and held-out models on D, so it does not depend on any
labelling. Specificity of C = M[C, C] - mean over D != C of M[C, D]; the subtraction removes that
mix's general change. T = mean specificity over corpora. Null: the diagonal is not special, so
which test corpus belongs to which mix is arbitrary. Exact test over all permutations of the
corpora (5,040 for 7, 720 for 6); two-sided. Holm across the 12 metric x stage tests.

Q2, direct effect: are models trained on all seven closer to real music than Hooktheory-only?
  (b) on the other corpora: the pooled stratified permutation test of loo_significance.py,
      Holm across 12.
  (a) on Hooktheory's own test set: descriptive, mean difference with a Welch 95% CI; a single
      corpus with 3 vs 3 models cannot go below p = 0.1 in a permutation test.

Distances and exclusions as in loo_significance.py. Per-model distances are cached in --values.

Usage:
    python scripts/eval/custom_eval/loo_specificity.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from loo_significance import (CORPORA, METRICS, SEEDS, SHORT, STAGES, _computed_values, holm,  # noqa: E402
                              pooled_permutation)


def all_labels():
    out = [f"7sets_{s}" for s in SEEDS] + [f"hooktheory_only_{s}" for s in SEEDS]
    out += [f"7sets_no_{c}_{s}" for c in CORPORA for s in SEEDS]
    return out


def load_distances(cache: Path) -> dict:
    """{stage: {metric: {test corpus: {label: distance to GT}}}}."""
    if cache.exists():
        return json.loads(cache.read_text())
    out = {stage: {m[0]: {} for m in METRICS} for stage in STAGES}
    for stage, stage_dir in STAGES.items():
        for corpus in CORPORA:
            means = json.loads((stage_dir / corpus / "means.json").read_text())
            comp = _computed_values(stage_dir, corpus, all_labels())
            for name, source, key, kind, excluded in METRICS:
                if corpus in excluded:
                    continue
                if source == "means":
                    gt = means["gt"].get(key)
                    vals = {l: means["models"][l][key] for l in all_labels()}
                else:
                    gt = comp["gt"].get(name)
                    vals = {l: comp["models"][l][name] for l in all_labels()}
                out[stage][name][corpus] = {l: abs(v - gt) if kind == "abs" else v
                                            for l, v in vals.items()}
            print(f"  [{stage}] {SHORT[corpus]} done", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return out


def specificity(dist: dict) -> dict:
    corp = list(dist)
    full = {d: np.array([dist[d][f"7sets_{s}"] for s in SEEDS]) for d in corp}
    held = {d: {c: np.array([dist[d][f"7sets_no_{c}_{s}"] for s in SEEDS]) for c in corp} for d in corp}
    sd = {d: np.concatenate([full[d]] + [held[d][c] for c in corp]).std(ddof=1) or 1.0 for d in corp}
    M = np.array([[(held[d][c].mean() - full[d].mean()) / sd[d] for d in corp] for c in corp])
    n = len(corp)

    def stat(col):       # col[i] = the test corpus treated as row i's own
        return np.mean([M[i, col[i]] - (M[i].sum() - M[i, col[i]]) / (n - 1) for i in range(n)])

    T = stat(list(range(n)))
    null = np.array([stat(p) for p in itertools.permutations(range(n))])
    per = {c: float(M[i, i] - (M[i].sum() - M[i, i]) / (n - 1)) for i, c in enumerate(corp)}
    return {"T": float(T), "p": float((np.abs(null) >= abs(T) - 1e-12).mean()),
            "n_permutations": len(null), "per_corpus": per,
            "corpora_positive": sum(v > 0 for v in per.values()), "n_corpora": n}


def direct_effect(dist: dict, draws: int, rng) -> dict:
    blocks = []
    for d in dist:
        if d == "hooktheory":
            continue
        v = np.array([dist[d][f"7sets_{s}"] for s in SEEDS]
                     + [dist[d][f"hooktheory_only_{s}"] for s in SEEDS], dtype=float)
        blocks.append((v, v.std(ddof=1) or 1.0))
    T, p = pooled_permutation(blocks, draws, rng)
    worse = sum(v[3:].mean() > v[:3].mean() for v, _ in blocks)
    out = {"T": T, "p": p, "corpora_hk_only_worse": int(worse), "n_corpora": len(blocks)}
    if "hooktheory" in dist:
        f = np.array([dist["hooktheory"][f"7sets_{s}"] for s in SEEDS])
        h = np.array([dist["hooktheory"][f"hooktheory_only_{s}"] for s in SEEDS])
        diff = float(h.mean() - f.mean())       # + = Hooktheory-only further from real music
        va, vb = f.var(ddof=1) / 3, h.var(ddof=1) / 3
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2) if se > 0 else 2
        half = float(stats.t.ppf(0.975, df) * se)
        out["on_hooktheory"] = {"diff": diff, "ci95": [diff - half, diff + half]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=Path, default=Path("logs/custom_eval/loo_all_distances.json"))
    ap.add_argument("--out", type=Path, default=Path("logs/custom_eval/loo_specificity.json"))
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    D = load_distances(args.values)

    q3, q2 = {}, {}
    for stage in STAGES:
        for m in METRICS:
            q3[f"{stage}/{m[0]}"] = specificity(D[stage][m[0]])
            q2[f"{stage}/{m[0]}"] = direct_effect(D[stage][m[0]], args.draws, rng)
    for fam in (q3, q2):
        for k, a in zip(fam, holm([fam[k]["p"] for k in fam])):
            fam[k]["p_holm"] = a

    print("\nQ3 specificity (+ = held-out corpus hurt more than the other corpora)\n")
    print(f"{'stage':<6}{'metric':<24}{'T':>7}{'corpora +':>11}{'p':>9}{'Holm':>8}   per corpus")
    for k, r in q3.items():
        st, name = k.split("/", 1)
        per = " ".join(f"{SHORT[c]} {v:+.2f}" for c, v in r["per_corpus"].items())
        print(f"{st:<6}{name:<24}{r['T']:>+7.2f}{r['corpora_positive']:>8}/{r['n_corpora']:<2}"
              f"{r['p']:>9.4f}{r['p_holm']:>8.4f}   {per}")
    print("\nQ2 direct effect (+ = Hooktheory-only further from real music than all seven)\n")
    print(f"{'stage':<6}{'metric':<24}{'other corpora':>14}{'p':>9}{'Holm':>8}   on Hooktheory: diff [95% CI]")
    for k, r in q2.items():
        st, name = k.split("/", 1)
        h = r.get("on_hooktheory")
        txt = f"{h['diff']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}]" if h else "-"
        print(f"{st:<6}{name:<24}{r['corpora_hk_only_worse']:>11}/{r['n_corpora']:<2}"
              f"{r['p']:>9.4f}{r['p_holm']:>8.4f}   {txt}")
    args.out.write_text(json.dumps({"specificity": q3, "direct_effect": q2}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
