#!/usr/bin/env python3
"""Pick one checkpoint per LOO run and write the per-corpus custom_eval configs.

Selection rule: minimum **aggregate** `val/loss`, i.e. the validation loss over the run's own
training corpora. This is deliberate and differs from the alpha sweep, which selected per test
corpus: in a leave-one-out run the held-out corpus is by definition absent from validation, so
a per-corpus rule cannot be applied to it. Using the aggregate everywhere keeps selection
identical across variants and guarantees the held-out corpus never influences which checkpoint
is chosen -- which is the whole point of the design.

The runs train with `checkpoint_top_k: -1`, so every 1000-step checkpoint is on disk and the
choice is made here rather than by the trainer.

The runs were never synced to wandb, so history is read from the local `.wandb` datastore.
Note for anyone extending this: wandb >= 0.25 stores history items under repeated `nested_key`
fields, not the older single `key`, and reading `item.key` silently yields nothing.

Usage:
    python scripts/data/select_loo_checkpoints.py --dry_run
    python scripts/data/select_loo_checkpoints.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("/hpcwork/thes2192/realchords/logs/my_logs")
SEVEN = ["hooktheory", "pop909", "emopia_plus", "nottingham",
         "wikifonia", "chord_melody_dataset", "jazzmus"]
VARIANTS = ["7sets", "hooktheory_only"] + [f"7sets_no_{d}" for d in SEVEN]
SEEDS = [42, 43, 44]


def read_val_history(run_dir: Path) -> dict[int, float]:
    """{global_step: aggregate val/loss} from the local wandb datastore."""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    files = list(run_dir.glob("wandb/latest-run/*.wandb")) or list(run_dir.glob("wandb/*/*.wandb"))
    if not files:
        return {}
    ds = datastore.DataStore()
    ds.open_for_scan(str(files[0]))
    out, step = {}, None
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        if r.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in r.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            row[key] = item.value_json
        if "trainer/global_step" in row:
            try:
                step = int(float(json.loads(row["trainer/global_step"])))
            except (ValueError, TypeError):
                pass
        if "val/loss" in row and step is not None:
            try:
                # trainer/global_step is 0-indexed while checkpoints are named by the
                # 1-indexed step, so the validation at global_step 999 is step=1000.ckpt.
                # Matching without this silently finds no step and falls back to the LAST
                # checkpoint for every run -- i.e. the most overfit one.
                out[step + 1] = float(json.loads(row["val/loss"]))
            except (ValueError, TypeError):
                pass
    return out


def pick(run_dir: Path) -> tuple[Path | None, float | None, int]:
    """(best checkpoint, its val/loss, how many checkpoints were available)."""
    ckpts = {int(c.stem.split("=")[1]): c for c in run_dir.glob("step=*.ckpt")}
    if not ckpts:
        return None, None, 0
    hist = read_val_history(run_dir)
    usable = {s: v for s, v in hist.items() if s in ckpts}
    if not usable:
        # no history: fall back to the last checkpoint, and say so
        last = max(ckpts)
        return ckpts[last], None, len(ckpts)
    best = min(usable, key=usable.get)
    return ckpts[best], usable[best], len(ckpts)


def run_dirs(family: str, prefix: str, variant: str, seed: int) -> list[Path]:
    """Run dirs for one cell, newest first. Names may carry a wandb hash suffix."""
    stem = f"{prefix}{variant}.seed={seed}.alpha=0.5"
    found = [d for d in (RUNS / family).glob(stem + "*")
             if d.is_dir() and list(d.glob("step=*.ckpt"))]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path("configs/custom_eval/dataset_loo_new"))
    args = ap.parse_args()

    chosen: dict[str, Path] = {}
    print(f"{'cell':<46}{'ckpts':>6}{'picked':>9}{'val/loss':>10}")
    missing = []
    for variant in VARIANTS:
        for seed in SEEDS:
            dirs = run_dirs("decoder_only", "decoder.chord.", variant, seed)
            if not dirs and variant == "hooktheory_only":
                dirs = [d for d in (RUNS / "decoder_only").glob(
                    f"decoder.chord.{variant}.seed={seed}*") if list(d.glob("step=*.ckpt"))]
            if not dirs:
                missing.append(f"{variant} seed={seed}")
                continue
            ck, loss, n = pick(dirs[0])
            if ck is None:
                missing.append(f"{variant} seed={seed}")
                continue
            label = f"{variant}_s{seed}" if variant != "7sets" else f"7sets_s{seed}"
            chosen[label] = ck
            step = ck.stem.split("=")[1]
            print(f"{variant + ' seed=' + str(seed):<46}{n:>6}{step:>9}"
                  f"{(f'{loss:.4f}' if loss is not None else 'LAST'):>10}")
    if missing:
        print(f"\nWARNING: no checkpoint for {len(missing)} cell(s): {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = next(iter(chosen.values()))
    header = ('# Leave-one-out grid, retrained 2026-10-05 on the corrected caches, scored under\n'
              '# the 16-bar all-windows protocol (cropped_max_len 512, all_windows true), the same\n'
              '# protocol as configs/custom_eval/alpha_sweep/.\n'
              '#\n'
              '# One checkpoint per run, chosen by minimum AGGREGATE val/loss over that run\'s own\n'
              '# training corpora. Not per-test-corpus as in the alpha sweep: the held-out corpus\n'
              '# has no validation loss in its own LOO run, so a per-corpus rule is undefined there.\n'
              '# The aggregate rule is identical across variants and keeps the held-out corpus out\n'
              '# of model selection entirely. See scripts/data/select_loo_checkpoints.py.\n')
    for corpus in SEVEN:
        lines = [header,
                 f'# Submit: sbatch scripts/jobscripts/eval/run_custom_eval.sh '
                 f'{args.out_dir}/{corpus}.yml',
                 f'base_model: "{base}"', 'model:']
        for label, ck in chosen.items():
            lines.append(f'  - "{label}={ck}"')
        lines += ['seed: 42', '', f'dataset_name: {corpus}', 'dataset_split: test',
                  f'save_dir: logs/custom_eval/dataset_loo_new/{corpus}',
                  'batch_size: 64', 'num_batches: -1', 'midi_samples: 0', 'clamp2_samples: 0',
                  'cropped_max_len: 512', 'all_windows: true', '']
        path = args.out_dir / f"{corpus}.yml"
        if not args.dry_run:
            path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{len(chosen)} models per config; "
          f"{'would write' if args.dry_run else 'wrote'} {len(SEVEN)} configs in {args.out_dir}")


if __name__ == "__main__":
    main()
