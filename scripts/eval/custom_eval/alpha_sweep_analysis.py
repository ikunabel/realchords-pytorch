#!/usr/bin/env python3
"""Alpha sweep: every reported metric, both stages, and the evaluation-protocol checks
(journal/experiments/ALPHA_SWEEP.md).

alpha sets the corpus sampling weights of the seven-corpus mixture, w_i ∝ frames_i ** alpha
(0 = uniform over corpora, 1 = proportional to size). Cells: alpha in {0, 0.25, 0.5, 1} x seeds
42/43/44, MLE decoders and the GAPT models fine-tuned from them. Descriptive only, no tests.

1. Final protocol (16-bar windows, all windows of every song), both stages, the six reported metrics
   as distances to ground truth (as in loo_significance.py): per corpus and alpha the 3 seed values,
   mean, min, max; the best and worst alpha per corpus; mean rank of each alpha over the 7 corpora.
2. GAPT's note-in-chord gain over its MLE initialisation as a share of the MLE-to-GT gap.
3. Variance decomposition, MLE note-in-chord, final protocol: per cell the sd over seeds against the
   standard error of the pooled ratio from resampling windows (bootstrap) within one model.
4. Protocol comparison, MLE note-in-chord level: 8 bars / one random window per song, 16 bars / one
   window, 16 bars / all windows; and the 8-bar protocol under evaluation seeds 1 and 7 (JAZZMUS).

Note-in-chord is the pooled ratio (`note_in_chord_ratio_mean`, the reported metric), not the per-song
average the first alpha analysis (2026-09-30) used; the two rank the alphas differently on some corpora.

Usage:
    python scripts/eval/custom_eval/alpha_sweep_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from loo_significance import CORPORA, METRICS, SEEDS, _computed_values  # noqa: E402

ALPHAS = ["a0", "a0p25", "a0p5", "a1"]
ALPHA_VALUE = {"a0": 0.0, "a0p25": 0.25, "a0p5": 0.5, "a1": 1.0}
L = Path("logs/custom_eval")
STAGES = {"MLE": L / "alpha_sweep_allwin", "GAPT": L / "gapt_alpha_allwin"}
PROTOCOLS = {"8 bars, 1 window": L / "alpha_sweep", "16 bars, 1 window": L / "alpha_sweep_16bar",
             "16 bars, all windows": L / "alpha_sweep_allwin"}
EVALSEED = {"seed 1": L / "alpha_sweep_evalseed" / "jazzmus_seed1",
            "seed 7": L / "alpha_sweep_evalseed" / "jazzmus_seed7"}


def labels(alpha):
    return [f"{alpha}_{s}" for s in SEEDS]


def distances(stage_dir: Path, corpus: str) -> dict:
    """{metric: {label: distance to GT}} for the six reported metrics."""
    means = json.loads((stage_dir / corpus / "means.json").read_text())
    all_labels = [l for a in ALPHAS for l in labels(a)]
    computed = _computed_values(stage_dir, corpus, all_labels)
    out = {}
    for name, source, key, kind, excluded in METRICS:
        if corpus in excluded:
            continue
        vals = {}
        for lab in all_labels:
            if source == "means":
                v, g = means["models"][lab][key], means["gt"][key]
                vals[lab] = None if v is None else (abs(v - g) if kind == "abs" else v)
            else:
                vals[lab] = computed["models"][lab][name]
        out[name] = vals
    return out


def ranks(per_alpha_means: dict) -> dict:
    """1 = closest to GT."""
    order = sorted(per_alpha_means, key=per_alpha_means.get)
    return {a: order.index(a) + 1 for a in per_alpha_means}


def nicr_level(run_dir: Path, corpus: str) -> dict:
    m = json.loads((run_dir / corpus / "means.json").read_text())
    return {"gt": m["gt"]["note_in_chord_ratio_mean"],
            **{lab: m["models"][lab]["note_in_chord_ratio_mean"] for lab in m["models"]}}


def window_se(model_dir: Path, reps: int, rng) -> float:
    """Bootstrap SE of the pooled note-in-chord ratio over windows, one fixed model."""
    d = torch.load(model_dir / "nicr_per_frame.pt", map_location="cpu", weights_only=False)
    m = np.nansum(d["matches"].float().numpy(), axis=1)      # NaN = frame without a melody note
    v = d["valid"].float().sum(1).numpy()
    n = len(m)
    idx = rng.integers(0, n, size=(reps, n))
    return float(np.std(m[idx].sum(1) / v[idx].sum(1), ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=L / "alpha_sweep_analysis.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    res = {"final_protocol": {}, "mean_rank": {}, "gapt_gap_closed": {}, "variance": {}, "protocols": {}}

    # 1. final protocol, all metrics, both stages
    for stage, sdir in STAGES.items():
        res["final_protocol"][stage] = {}
        rank_acc = {}
        for c in CORPORA:
            dist = distances(sdir, c)
            res["final_protocol"][stage][c] = {}
            for metric, vals in dist.items():
                cell = {}
                for a in ALPHAS:
                    xs = [vals[l] for l in labels(a)]
                    cell[a] = {"seeds": xs, "mean": float(np.mean(xs)), "min": min(xs), "max": max(xs)}
                means = {a: cell[a]["mean"] for a in ALPHAS}
                r = ranks(means)
                cell["best"], cell["worst"] = min(means, key=means.get), max(means, key=means.get)
                cell["rank"] = r
                res["final_protocol"][stage][c][metric] = cell
                rank_acc.setdefault(metric, []).append(r)
        res["mean_rank"][stage] = {m: {a: float(np.mean([r[a] for r in rs])) for a in ALPHAS}
                                   for m, rs in rank_acc.items()}
        # on how many corpora alpha = 1 is further from GT than each other alpha (means over seeds)
        fp = res["final_protocol"][stage]
        res.setdefault("a1_further", {})[stage] = {
            m: {a: sum(fp[c][m]["a1"]["mean"] > fp[c][m][a]["mean"] for c in CORPORA if m in fp[c])
                for a in ALPHAS[:-1]}
            for m in rank_acc}

    # 2. GAPT note-in-chord gain as share of the MLE-to-GT gap (levels, mean over alphas and seeds)
    for c in CORPORA:
        mle, gapt = nicr_level(STAGES["MLE"], c), nicr_level(STAGES["GAPT"], c)
        labs = [l for a in ALPHAS for l in labels(a)]
        share = [(gapt[l] - mle[l]) / (mle["gt"] - mle[l]) for l in labs]
        res["gapt_gap_closed"][c] = {"mean": float(np.mean(share)), "min": min(share), "max": max(share),
                                     "gt": mle["gt"]}

    # 3. variance decomposition, MLE note-in-chord, final protocol
    for c in CORPORA:
        lev = nicr_level(STAGES["MLE"], c)
        sd_seed = [float(np.std([lev[l] for l in labels(a)], ddof=1)) for a in ALPHAS]
        se_win = [window_se(STAGES["MLE"] / c / "models" / l, args.boot, rng)
                  for a in ALPHAS for l in labels(a)]
        n_win = torch.load(STAGES["MLE"] / c / "gt" / "gt.pt", map_location="cpu", weights_only=False).shape[0]
        alpha_means = [float(np.mean([lev[l] for l in labels(a)])) for a in ALPHAS]
        res["variance"][c] = {"n_windows": int(n_win), "sd_seed": float(np.mean(sd_seed)),
                              "pooled_sd": float(np.sqrt(np.mean(np.square(sd_seed)))),
                              "se_window": float(np.mean(se_win)),
                              "alpha_spread": float(max(alpha_means) - min(alpha_means))}

    # 4. protocol comparison (MLE, note-in-chord level, higher = closer to GT for every model here)
    runs = {**{p: (d, CORPORA) for p, d in PROTOCOLS.items()}}
    for p, (d, corp) in runs.items():
        res["protocols"][p] = {}
        for c in corp:
            if not (d / c / "means.json").exists():
                continue
            lev = nicr_level(d, c)
            means = {a: float(np.mean([lev[l] for l in labels(a)])) for a in ALPHAS}
            res["protocols"][p][c] = {"n_windows": sum(1 for _ in open(d / c / "metadata.jsonl")),
                                      "mean": means, "best": max(means, key=means.get),
                                      "sd_seed": float(np.mean([np.std([lev[l] for l in labels(a)], ddof=1)
                                                                for a in ALPHAS]))}
    res["protocols"]["evalseed_jazzmus"] = {}
    for p, d in EVALSEED.items():
        m = json.loads((d / "means.json").read_text())
        lev = {lab: m["models"][lab]["note_in_chord_ratio_mean"] for lab in m["models"]}
        key = lambda a, s: next(k for k in lev if k.replace(".", "_") == f"{a}_{s}")
        means = {a: float(np.mean([lev[key(a, s)] for s in SEEDS])) for a in ALPHAS}
        res["protocols"]["evalseed_jazzmus"][p] = {"mean": means, "best": max(means, key=means.get)}

    args.out.write_text(json.dumps(res, indent=2))

    for stage in STAGES:
        print(f"\n{stage}, mean rank over corpora (1 = closest to real music)")
        for m, r in res["mean_rank"][stage].items():
            print(f"  {m:<24}" + "".join(f"{a:>8}{r[a]:5.2f}" for a in ALPHAS))
        print(f"{stage}, corpora where alpha = 1 is further from real music than alpha = 0 / 0.25 / 0.5")
        for m, v in res["a1_further"][stage].items():
            print(f"  {m:<24}" + "".join(f"{v[a]:>6}" for a in ALPHAS[:-1]))
        print(f"{stage}, corpora where alpha = 1 is the worst / the best")
        for m in res["mean_rank"][stage]:
            cs = [c for c in CORPORA if m in res["final_protocol"][stage][c]]
            w = sum(res["final_protocol"][stage][c][m]["worst"] == "a1" for c in cs)
            b = sum(res["final_protocol"][stage][c][m]["best"] == "a1" for c in cs)
            print(f"  {m:<24} worst {w}/{len(cs)}  best {b}/{len(cs)}")
    print("\nGAPT note-in-chord gain / MLE-to-GT gap:",
          {c: round(v["mean"], 2) for c, v in res["gapt_gap_closed"].items()})
    print("\nvariance (MLE note-in-chord, final protocol):")
    for c, v in res["variance"].items():
        print(f"  {c:<22} n {v['n_windows']:>5}  sd_seed {v['sd_seed']:.4f}  SE_window {v['se_window']:.4f}"
              f"  pooled sd {v['pooled_sd']:.4f}  alpha spread {v['alpha_spread']:.3f}")
    print("\nprotocols, best alpha per corpus:")
    for p, d in res["protocols"].items():
        print(f"  {p:<22}", {c: v["best"] for c, v in d.items()})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
