#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
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

module load CUDA/12.8.0

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

srun python scripts/train/train_rl_ensemble_rhythm_reward_offline_anchor_gail.py \
  --args.load "${CONFIG_YML}" \
  --save_dir "${RUN_DIR}"