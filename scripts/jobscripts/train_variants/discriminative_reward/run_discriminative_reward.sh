#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=discriminative_reward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config_yml>"
  exit 2
fi

CONFIG_YML="$1"

RUN_DIR="${RUNS_ROOT}/${SLURM_JOB_NAME}"
mkdir -p "${RUN_DIR}"

srun python scripts/train/train_discriminative_reward.py \
  --args.load "${CONFIG_YML}" \
  --save_dir "${RUN_DIR}"

