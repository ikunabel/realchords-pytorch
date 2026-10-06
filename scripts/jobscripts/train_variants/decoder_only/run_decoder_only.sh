#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=decoder_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# 2h from sacct over every completed decoder run since 2026-08 (n=41): mean 56 min, max
# 1:03:59, zero timeouts. Runtime barely depends on mixture size -- 7sets_no_X max 1:00
# (n=21), full 7sets max 1:03 (n=15), hooktheory_only max 1:03 -- because training is
# step-bounded (fixed train_steps / early stopping on steps), not epoch-bounded.
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config_yml>"
  exit 2
fi

CONFIG_YML="$1"

RUN_DIR="${RUNS_ROOT}/decoder_only/${SLURM_JOB_NAME}"
mkdir -p "${RUN_DIR}"

srun python scripts/train/train_decoder_only.py \
  --args.load "${CONFIG_YML}" \
  --save_dir "${RUN_DIR}"

