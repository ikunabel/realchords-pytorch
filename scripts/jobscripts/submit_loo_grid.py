"""Submit the full dataset-mixture (LOO) training grid: reward models, enc-dec anchors, decoders.

One job per config, through the existing per-family `submit_*.sh` wrappers, which derive the SLURM
job name from the config stem and therefore the run directory under $RUNS_ROOT. Configs come from
scripts/data/generate_loo_configs.py.

Scope (see journal/DATASET_MIX_LOO.md): the JAZZMUS duplicate cache and the Chord Melody Dataset
octave defect invalidated every model trained on a mixture containing either corpus, so all nine
variants are retrained -- except the `hooktheory_only` decoders, which contain neither corpus, are
already trained at seeds 42/43/44 on the 2,969-chord vocabulary, and are reused.

    27 enc-dec  = 9 variants x seeds 42/43/44
    24 decoder  = 8 variants x seeds 42/43/44   (hooktheory_only reused)
    -- 51 jobs by default.

Per-variant reward models are NOT submitted unless --with_rewards is passed. The transfer
probe (journal/DATASET_MIX_LOO.md, "Do the LOO variants need their own reward models?") found
the 7-set reward models separate genuine from corrupted accompaniment at 0.87-0.98 AUC on
corpora they never saw, so a per-variant set buys nothing and the 7-set ensemble is used for
every variant. That is 72 of the 123 jobs this script used to plan.

They are also not needed for the MLE-stage grid at all: rewards only enter at the RL stage,
and that stage uses the 7-set ensemble regardless.

Usage:
    python scripts/jobscripts/submit_loo_grid.py --dry_run
    python scripts/jobscripts/submit_loo_grid.py --only decoder
    python scripts/jobscripts/submit_loo_grid.py --with_rewards   # only if that changes
    python scripts/jobscripts/submit_loo_grid.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts/data")
from generate_loo_configs import ENC_DEC_SEEDS, REWARD_TEMPLATES, VARIANTS  # noqa: E402

JOBS = Path("scripts/jobscripts/train_variants")
SUBMIT = {
    "contrastive": JOBS / "contrastive_reward/submit_contrastive_reward.sh",
    "contrastive_rhythm": JOBS / "contrastive_reward/submit_contrastive_reward_rhythm.sh",
    "discriminative": JOBS / "discriminative_reward/submit_discriminative_reward.sh",
    "discriminative_rhythm": JOBS / "discriminative_reward/submit_discriminative_reward_rhythm.sh",
    "enc_dec": JOBS / "enc_dec/submit_enc_dec.sh",
    "decoder": JOBS / "decoder_only/submit_decoder_only.sh",
}
# enc-dec trains to early stopping (patience 15 at val_interval 1000). Measured from sacct over
# every completed run since 2026-08: enc-dec n=13, mean 83 min, **max 1:54:02**; decoder n=41,
# mean 56 min, max 1:03:59; zero timeouts. The previous 8h here rested on a note that "the 7sets
# run came close to 5h", which the accounting data contradicts. 3h leaves ~1.6x headroom over
# the worst case ever seen and queues faster on a busy partition.
EXTRA_SBATCH = {"enc_dec": ["--time=03:00:00"]}
RUNS_ROOT = Path("/hpcwork/thes2192/realchords/logs/my_logs")
RUN_FAMILY = {"contrastive": "contrastive_reward", "contrastive_rhythm": "contrastive_reward",
              "discriminative": "discriminative_reward", "discriminative_rhythm": "discriminative_reward",
              "enc_dec": "enc_dec", "decoder": "decoder_only"}
# decoders already trained on data neither defect touched
REUSE_DECODERS = {"hooktheory_only"}


def kind_of(folder: str, stem: str) -> str:
    rhythm = "no_augmentation_rhythm" in stem
    if folder == "contrastive_reward":
        return "contrastive_rhythm" if rhythm else "contrastive"
    return "discriminative_rhythm" if rhythm else "discriminative"


def plan(with_rewards: bool = False) -> list:
    out = []
    for variant in VARIANTS:
        if with_rewards:
            for folder, _template, out_stem in REWARD_TEMPLATES:
                stem = out_stem.format(v=variant)
                out.append((kind_of(folder, stem), Path("configs") / folder / f"{stem}.yml", variant))
        for seed in ENC_DEC_SEEDS:
            out.append(("enc_dec",
                        Path(f"configs/enc_dec/enc_dec.chord.{variant}.seed={seed}.alpha=0.5.yml"),
                        variant))
        if variant not in REUSE_DECODERS:
            for seed in ENC_DEC_SEEDS:
                out.append(("decoder",
                            Path(f"configs/decoder_only/decoder.chord.{variant}.seed={seed}.alpha=0.5.yml"),
                            variant))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only", choices=["reward", "enc_dec", "decoder"], default=None)
    ap.add_argument("--with_rewards", action="store_true",
                    help="also submit per-variant reward models (72 jobs). Off by default: the "
                         "7-set reward models were shown to transfer, so these buy nothing.")
    args = ap.parse_args()

    jobs = plan(with_rewards=args.with_rewards or args.only == "reward")
    if args.only == "reward":
        jobs = [j for j in jobs if j[0] not in ("enc_dec", "decoder")]
    elif args.only:
        jobs = [j for j in jobs if j[0] == args.only]

    missing = [c for _, c, _ in jobs if not c.exists()]
    occupied = []
    for kind, cfg, _ in jobs:
        run_dir = RUNS_ROOT / RUN_FAMILY[kind] / cfg.stem
        if run_dir.exists() and list(run_dir.glob("*.ckpt")):
            occupied.append(run_dir)
    if missing:
        print(f"ERROR: {len(missing)} config(s) missing, e.g. {missing[:3]}")
        return 1
    if occupied:
        print(f"ERROR: {len(occupied)} target run dir(s) already hold checkpoints and would be "
              f"mixed with new ones, e.g. {occupied[:3]}")
        return 1

    counts: dict = {}
    for kind, _, _ in jobs:
        counts[kind] = counts.get(kind, 0) + 1
    print(f"{len(jobs)} job(s): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    submitted = 0
    for kind, cfg, variant in jobs:
        cmd = [str(SUBMIT[kind]), str(cfg)] + EXTRA_SBATCH.get(kind, [])
        if args.dry_run:
            print(f"  (dry) {' '.join(cmd)}")
            continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        ok = res.returncode == 0
        print(f"  [{'ok' if ok else 'FAIL'}] {cfg.stem:64s} {res.stdout.strip() or res.stderr.strip()[:80]}")
        if ok:
            submitted += 1
    if not args.dry_run:
        print(f"\n{submitted}/{len(jobs)} submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
