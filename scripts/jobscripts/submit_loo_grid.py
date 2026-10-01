"""Submit the full dataset-mixture (LOO) training grid: reward models, enc-dec anchors, decoders.

One job per config, through the existing per-family `submit_*.sh` wrappers, which derive the SLURM
job name from the config stem and therefore the run directory under $RUNS_ROOT. Configs come from
scripts/data/generate_loo_configs.py.

Scope (see journal/DATASET_MIX_LOO.md): the JAZZMUS duplicate cache and the Chord Melody Dataset
octave defect invalidated every model trained on a mixture containing either corpus, so all nine
variants are retrained -- except the `hooktheory_only` decoders, which contain neither corpus, are
already trained at seeds 42/43/44 on the 2,969-chord vocabulary, and are reused.

    72 reward   = 9 variants x (2 contrastive, 2 discriminative, 2 rhythm-contrastive, 2 rhythm-disc)
    27 enc-dec  = 9 variants x seeds 42/43/44
    24 decoder  = 8 variants x seeds 42/43/44   (hooktheory_only reused)

Usage:
    python scripts/jobscripts/submit_loo_grid.py --dry_run
    python scripts/jobscripts/submit_loo_grid.py --only reward
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
# enc-dec trains to early stopping (patience 15 at val_interval 1000); its runner defaults to 5h,
# which the 7sets run came close to, so ask for more rather than lose a run to the wall clock
EXTRA_SBATCH = {"enc_dec": ["--time=08:00:00"]}
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


def plan() -> list:
    out = []
    for variant in VARIANTS:
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
    args = ap.parse_args()

    jobs = plan()
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
