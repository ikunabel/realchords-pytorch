#!/usr/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yml> [sbatch args...]"
  echo "Example: $0 configs/reward_models/discriminative_reward_128_bs_2_4_datasets.yml"
  echo "Example: $0 configs/reward_models/discriminative_reward_128_bs_2_4_datasets.yml --time=02:00:00"
  echo "Example: $0 configs/reward_models/discriminative_reward_128_bs_2_4_datasets.yml --time=02:00:00 --gres=gpu:1"
  exit 2
fi

CONFIG_YML="$1"
JOB_NAME="${${CONFIG_YML:t}%.yml}"
shift

mkdir -p "jobscripts/slurm_logs/${JOB_NAME}"
sbatch --job-name="${JOB_NAME}" "$@" jobscripts/train_variants/discriminative_reward/run_discriminative_reward.sh "${CONFIG_YML}"

