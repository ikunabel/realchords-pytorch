#!/usr/bin/env python3
"""Concatenate commit_replay.py runs over different windows into one replay folder.

The commit-ahead replay ran twice, on rows 0-127 and on rows 128-255 of each corpus's evaluation
gt.pt (the second run split over one root per stage). This writes one folder in the same layout,
<out>/k<k>/<corpus>/{gt/gt.pt, metadata.jsonl, chord_names_augmented.json,
models/<label>/{preds.pt, nicr_per_frame.pt}}, so every metric script runs on all windows
unchanged.

--sets: one argument per window range, in row order; within an argument, comma-separated roots
that cover the SAME windows (their gt.pt must be identical). A (k, corpus, model) is written only
if every set has it. Vocabularies must be identical across sets.

Usage:
    python scripts/eval/custom_eval/merge_commit_replays.py --out logs/custom_eval/commit_replay_all \\
        --sets logs/custom_eval/commit_replay \\
               logs/custom_eval/commit_replay_confirm_mle,logs/custom_eval/commit_replay_confirm_gapt
    python scripts/eval/custom_eval/merge_commit_replays.py --out logs/custom_eval/commit_replay_all_mleimag \\
        --sets logs/custom_eval/commit_replay_mleimag logs/custom_eval/commit_replay_confirm_mleimag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sets = [[Path(r) for r in s.split(",")] for s in args.sets]

    ks = set.intersection(*[{d.name for r in roots for d in r.glob("k*")} for roots in sets])
    for k in sorted(ks, key=lambda n: int(n[1:])):
        corpora = set.intersection(*[{c.name for r in roots for c in (r / k).glob("*") if c.is_dir()}
                                     for roots in sets])
        for corpus in sorted(corpora):
            gts, metas, names, models = [], [], None, []
            for roots in sets:
                dirs = [r / k / corpus for r in roots if (r / k / corpus / "gt" / "gt.pt").exists()]
                gt = _load(dirs[0] / "gt" / "gt.pt")
                for d in dirs[1:]:
                    assert torch.equal(gt, _load(d / "gt" / "gt.pt")), f"{d}: gt differs within a set"
                n = json.loads((dirs[0] / "chord_names_augmented.json").read_text())
                assert names is None or n == names, f"{dirs[0]}: chord vocabulary differs between sets"
                names = n
                gts.append(gt)
                metas.append((dirs[0] / "metadata.jsonl").read_text().strip())
                models.append({m.name: m for d in dirs for m in (d / "models").iterdir()
                               if (m / "preds.pt").exists()})
            out = args.out / k / corpus
            (out / "gt").mkdir(parents=True, exist_ok=True)
            torch.save(torch.cat(gts), out / "gt" / "gt.pt")
            (out / "metadata.jsonl").write_text("\n".join(metas) + "\n")
            (out / "chord_names_augmented.json").write_text(json.dumps(names))
            common = sorted(set.intersection(*[set(m) for m in models]))
            for label in common:
                (out / "models" / label).mkdir(parents=True, exist_ok=True)
                torch.save(torch.cat([_load(m[label] / "preds.pt") for m in models]),
                           out / "models" / label / "preds.pt")
                nicr = [_load(m[label] / "nicr_per_frame.pt") for m in models]
                torch.save({key: torch.cat([x[key] for x in nicr]) for key in nicr[0]},
                           out / "models" / label / "nicr_per_frame.pt")
            print(f"{k}/{corpus}: {sum(g.shape[0] for g in gts)} windows, models {common}")


if __name__ == "__main__":
    main()
