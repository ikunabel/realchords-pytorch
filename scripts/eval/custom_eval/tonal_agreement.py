#!/usr/bin/env python3
"""Tonal agreement: do the chords sit in the melody's key?

Per window, the melody and the chord track each get a 24-key profile -- the Krumhansl-Schmuckler
correlation with every major and minor key (realchords/utils/key_estimation.py; music21's
analyser, computed by an identical numpy path that is re-checked against music21 every run).
Tonal agreement is the Pearson correlation between the two profiles: high when the chords favour
the same keys as the melody. It compares profile shape, not level, and takes no hard key decision,
so a melody ambiguous between relative keys is not penalised for an arbitrary pick. Melody-
conditional, label-free, usable on every corpus; read against the GT row like the other metrics.

  global  one profile pair per 16-bar window
  local   one pair per 4-bar segment (64 frames), averaged per window: does the harmony follow the
          melody phrase by phrase?

Modes:
  (default)          the metric for --eval_dirs (and --cells), GT from the first dir
  --validate         controls on GT, where the answer is known:
                       transposition: chords transposed by each of the 12 intervals (exact, by
                       rotating the profile -- KS is transposition-equivariant), reported by
                       circle-of-fifths distance; agreement must fall with distance
                       mismatch: each melody paired with GT chords of a window from a DIFFERENT
                       song (fixed seed); agreement must fall, and the true pairing must win
  --validate_labels  Hooktheory only, against annotated keys: where the label ranks in the
                       melody and chord profiles, and whether this label-free metric ranks models
                       like a label-based measure (the chord profile's correlation with the
                       annotated key) across every model in --eval_dirs
  --by_segment       agreement per 4-bar segment (bars 1-4 .. 13-16), full windows only, for GT
                       and the --cells models of each run (mean over them): does the model
                       recover from a cold start within the window?

Usage:
    python scripts/eval/custom_eval/tonal_agreement.py --validate \\
        --eval_dirs GT=logs/custom_eval/dataset_loo_new
    python scripts/eval/custom_eval/tonal_agreement.py --validate_labels \\
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new GAPT=logs/custom_eval/gapt_loo
    python scripts/eval/custom_eval/tonal_agreement.py \\
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new GAPT=logs/custom_eval/gapt_loo \\
        --out logs/custom_eval/tonal_agreement_loo.json
    python scripts/eval/custom_eval/tonal_agreement.py --by_segment --cells 7sets_s \\
        --corpora hooktheory pop909 nottingham wikifonia \\
        --eval_dirs MLE=logs/custom_eval/dataset_loo_new GAPT=logs/custom_eval/gapt_loo \\
        --out logs/custom_eval/tonal_agreement_by_segment.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from key_estimator_validation import section_keys  # noqa: E402

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.key_estimation import (chord_histogram, chord_stream, circle_of_fifths_distance,
                                             fast_profile, key_profile, melody_histogram,
                                             melody_stream, tonal_agreement, transpose_profile)

CORPORA = ["hooktheory", "pop909", "emopia_plus", "nottingham",
           "wikifonia", "chord_melody_dataset", "jazzmus"]
SHORT = {"hooktheory": "Hooktheory", "pop909": "POP909", "emopia_plus": "EMOPIA+",
         "nottingham": "Nottingham", "wikifonia": "Wikifonia",
         "chord_melody_dataset": "CMD", "jazzmus": "JAZZMUS"}
PAD = 0
SEGMENT = 64          # frames = 4 bars


def _tok(corpus_dir: Path):
    names = json.loads((corpus_dir / "chord_names_augmented.json").read_text())
    return HooktheoryTokenizer(chord_names=names if isinstance(names, list) else list(names)).id_to_name


def _segments(row, local: bool):
    """(melody lane, chord lane) for the whole valid window, or per 4-bar segment."""
    mel, cho = row[2::2], row[1::2]
    valid = sum(1 for t in mel if t != PAD)
    mel, cho = mel[:valid], cho[:valid]
    if not local:
        return [(mel, cho)]
    return [(mel[s:s + SEGMENT], cho[s:s + SEGMENT]) for s in range(0, valid, SEGMENT)
            if valid - s >= SEGMENT // 2]


def profiles(rows, i2n, local: bool):
    """Per window, a list of (melody profile, chord profile) pairs (one, or one per segment)."""
    return [[(fast_profile(melody_histogram(m)), fast_profile(chord_histogram(c, i2n)))
             for m, c in _segments(r, local)] for r in rows]


def window_agreement(pairs):
    vals = [a for a in (tonal_agreement(m, c) for m, c in pairs) if a is not None]
    return float(np.mean(vals)) if vals else None


def check_against_music21(rows, i2n, n: int = 20, tol: float = 1e-9):
    """The numpy path must equal music21's analyser on a sample, or the run stops."""
    idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    for i in idx:
        for (m, c) in _segments(rows[i], local=False):
            for fast, ref in ((fast_profile(melody_histogram(m)), key_profile(melody_stream(m))),
                              (fast_profile(chord_histogram(c, i2n)), key_profile(chord_stream(c, i2n)))):
                if (fast is None) != (ref is None) or (fast is not None and np.abs(fast - ref).max() > tol):
                    raise RuntimeError(f"numpy profile differs from music21's on window {i}")


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False).tolist()


def _model_dirs(run_dir: Path, corpus: str, cells):
    d = run_dir / corpus / "models"
    return [s for s in sorted(d.iterdir()) if (s / "preds.pt").exists()
            and (cells is None or any(s.name.startswith(c) for c in cells))] if d.exists() else []


# ---------------------------------------------------------------------------- the metric
def run_metric(runs, cells, out_path):
    results = {}
    print(f"\ntonal agreement (Krumhansl-Schmuckler profiles, melody vs chords)")
    print(f"{'corpus':<12}{'run':<8}{'global':>9}{'|-GT|':>8}{'local':>9}{'|-GT|':>8}{'models':>8}")
    for corpus in CORPORA:
        base = runs[0][1] / corpus
        if not (base / "gt" / "gt.pt").exists():
            continue
        i2n = _tok(base)
        gt_rows = _load(base / "gt" / "gt.pt")
        check_against_music21(gt_rows, i2n)
        gt = {loc: [window_agreement(p) for p in profiles(gt_rows, i2n, loc)] for loc in (False, True)}
        row = {"gt": {"global": float(np.nanmean([v for v in gt[False] if v is not None])),
                      "local": float(np.nanmean([v for v in gt[True] if v is not None]))}}
        print(f"{SHORT[corpus]:<12}{'GT':<8}{row['gt']['global']:>9.3f}{'':>8}{row['gt']['local']:>9.3f}")
        for label, run_dir in runs:
            subs = _model_dirs(run_dir, corpus, cells)
            if not subs:
                continue
            g_rows = _load(run_dir / corpus / "gt" / "gt.pt")
            g_ref = {loc: float(np.nanmean([v for v in (window_agreement(p) for p in profiles(g_rows, _tok(run_dir / corpus), loc)) if v is not None]))
                     for loc in (False, True)}
            vals = {False: [], True: []}
            for s in subs:
                rows = _load(s / "preds.pt")
                for loc in (False, True):
                    w = [v for v in (window_agreement(p) for p in profiles(rows, i2n, loc)) if v is not None]
                    vals[loc].append(float(np.mean(w)))
            g, l = float(np.mean(vals[False])), float(np.mean(vals[True]))
            row[label] = {"global": g, "local": l, "gt_global": g_ref[False], "gt_local": g_ref[True],
                          "per_model_global": vals[False], "per_model_local": vals[True],
                          "models": [s.name for s in subs]}
            print(f"{'':<12}{label:<8}{g:>9.3f}{abs(g - g_ref[False]):>8.3f}{l:>9.3f}"
                  f"{abs(l - g_ref[True]):>8.3f}{len(subs):>8}")
        results[corpus] = row
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out_path}")


# ------------------------------------------------------------------------- validation
def run_validate(runs, out_path, seed: int = 0):
    rng = np.random.default_rng(seed)
    results = {}
    print("\nvalidation on GT: agreement with the true chords, transposed chords, and chords of "
          "another song")
    head = "".join(f"{f'cof {d}':>8}" for d in range(7))
    print(f"{'corpus':<12}{'scope':<7}{head}{'mismatch':>10}{'true wins':>11}")
    for corpus in CORPORA:
        base = runs[0][1] / corpus
        if not (base / "gt" / "gt.pt").exists():
            continue
        i2n = _tok(base)
        rows = _load(base / "gt" / "gt.pt")
        check_against_music21(rows, i2n)
        songs = [json.loads(l).get("song_id") for l in (base / "metadata.jsonl").read_text().splitlines() if l.strip()]
        # partner: a window of a different song
        partner = []
        for i in range(len(rows)):
            cand = [j for j in rng.permutation(len(rows))[:50] if songs[j] != songs[i]]
            partner.append(int(cand[0]) if cand else None)
        results[corpus] = {}
        for local in (False, True):
            P = profiles(rows, i2n, local)
            by_dist = {d: [] for d in range(7)}
            for t in range(12):
                d = circle_of_fifths_distance(0, t)
                for pairs in P:
                    a = window_agreement([(m, None if c is None else transpose_profile(c, t)) for m, c in pairs])
                    if a is not None:
                        by_dist[d].append(a)
            true_vals, mis_vals, wins = [], [], []
            for i, pairs in enumerate(P):
                j = partner[i]
                if j is None:
                    continue
                own = window_agreement(pairs)
                # melody of window i against chords of window j, segment by segment where both exist
                other = window_agreement([(m, P[j][s][1]) for s, (m, _) in enumerate(pairs) if s < len(P[j])])
                if own is None or other is None:
                    continue
                true_vals.append(own); mis_vals.append(other); wins.append(own > other)
            scope = "local" if local else "global"
            curve = [float(np.mean(by_dist[d])) for d in range(7)]
            results[corpus][scope] = {"by_cof_distance": curve, "true": float(np.mean(true_vals)),
                                      "mismatch": float(np.mean(mis_vals)), "true_wins": float(np.mean(wins)),
                                      "n": len(wins)}
            print(f"{SHORT[corpus]:<12}{scope:<7}" + "".join(f"{v:>8.3f}" for v in curve)
                  + f"{np.mean(mis_vals):>10.3f}{np.mean(wins):>11.1%}")
    print("\ncof d = chords transposed by an interval d steps away on the circle of fifths "
          "(cof 0 = the true chords)")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


def run_validate_labels(runs, out_path, labels_path: Path):
    """Hooktheory: rank of the annotated key in each profile, and model-level agreement between
    this label-free metric and a label-based one."""
    labels = section_keys(labels_path)
    results = {"ranks": {}, "models": []}
    base = runs[0][1] / "hooktheory"
    i2n = _tok(base)

    def labelled(run_dir):
        meta = [json.loads(l) for l in (run_dir / "hooktheory" / "metadata.jsonl").read_text().splitlines() if l.strip()]
        return [(i, labels[m["song_id"]]) for i, m in enumerate(meta) if m.get("song_id") in labels]

    # 1. where does the annotated key rank in GT melody / chord profiles?
    rows = _load(base / "gt" / "gt.pt")
    check_against_music21(rows, i2n)
    print("\nrank of the annotated key in each GT profile (1 = the profile's best key)")
    for name, fn in (("melody", lambda r: fast_profile(melody_histogram(r[2::2]))),
                     ("chords", lambda r: fast_profile(chord_histogram(r[1::2], i2n)))):
        ranks = []
        for i, key in labelled(runs[0][1]):
            v = fn(rows[i])
            if v is not None:
                ranks.append(int((v > v[key]).sum()) + 1)
        ranks = np.array(ranks)
        results["ranks"][name] = {"top1": float((ranks == 1).mean()), "top3": float((ranks <= 3).mean()),
                                  "median": float(np.median(ranks)), "n": len(ranks)}
        print(f"  {name:<7} top-1 {np.mean(ranks == 1):.1%}   top-3 {np.mean(ranks <= 3):.1%}   "
              f"median rank {np.median(ranks):.0f} of 24   (n = {len(ranks)})")

    # 2. across models: label-free agreement vs label-based key fit of the chords
    print("\nper model on Hooktheory: tonal agreement (label-free) vs the chord profile's "
          "correlation with the annotated key (label-based)")
    for label, run_dir in runs:
        lab = labelled(run_dir)
        sources = [("GT", run_dir / "hooktheory" / "gt" / "gt.pt")] + \
                  [(s.name, s / "preds.pt") for s in _model_dirs(run_dir, "hooktheory", None)]
        for name, path in sources:
            if label != runs[0][0] and name == "GT":
                continue
            rows = _load(path)
            ag, fit = [], []
            for i, key in lab:
                pairs = profiles([rows[i]], i2n, local=False)[0]
                a = window_agreement(pairs)
                c = pairs[0][1]
                if a is None or c is None:
                    continue
                ag.append(a); fit.append(float(c[key]))
            results["models"].append({"run": label, "model": name, "agreement": float(np.mean(ag)),
                                      "label_fit": float(np.mean(fit)), "n": len(ag)})
    ms = [m for m in results["models"] if m["model"] != "GT"]
    rho, p = spearmanr([m["agreement"] for m in ms], [m["label_fit"] for m in ms])
    results["spearman"] = {"rho": float(rho), "p": float(p), "n_models": len(ms)}
    for m in results["models"]:
        print(f"  {m['run']:<6}{m['model']:<36}agreement {m['agreement']:.3f}   label fit {m['label_fit']:.3f}")
    print(f"\n  Spearman across {len(ms)} models: rho = {rho:.3f}, p = {p:.2g}")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")


def run_by_segment(runs, cells, corpora, out_path):
    out = {}
    print(f"{'corpus':<12}{'source':<8}{'bars 1-4':>10}{'5-8':>8}{'9-12':>8}{'13-16':>8}")
    for corpus in corpora:
        i2n = _tok(runs[0][1] / corpus)
        srcs = [("GT", [runs[0][1] / corpus / "gt" / "gt.pt"])]
        srcs += [(label, [m / "preds.pt" for m in _model_dirs(d, corpus, cells)]) for label, d in runs]
        for label, paths in srcs:
            assert paths, f"{corpus}/{label}: no models match --cells {cells}"
            per_model = []
            for p in paths:
                P = profiles(_load(p), i2n, local=True)
                full = [pairs for pairs in P if len(pairs) == 4]
                per_model.append([np.nanmean([v for v in (tonal_agreement(*pairs[s]) for pairs in full)
                                              if v is not None]) for s in range(4)])
            out[f"{corpus}/{label}"] = np.mean(per_model, axis=0).tolist()
            print(f"{SHORT[corpus]:<12}{label:<8}" + "".join(f"{v:>8.3f}" for v in out[f"{corpus}/{label}"]))
    if out_path:
        out_path.write_text(json.dumps(out, indent=2))
        print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dirs", nargs="+", required=True, metavar="LABEL=DIR")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate_labels", action="store_true")
    ap.add_argument("--by_segment", action="store_true")
    ap.add_argument("--corpora", nargs="+", default=CORPORA, help="--by_segment only")
    ap.add_argument("--labels", type=Path, default=Path("data/cache/hooktheory/test.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = [(i.split("=", 1)[0], Path(i.split("=", 1)[1])) for i in args.eval_dirs]
    if args.by_segment:
        run_by_segment(runs, args.cells, args.corpora, args.out)
    elif args.validate:
        run_validate(runs, args.out)
    elif args.validate_labels:
        run_validate_labels(runs, args.out, args.labels)
    else:
        run_metric(runs, args.cells, args.out)


if __name__ == "__main__":
    main()
