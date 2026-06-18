#!/usr/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yml> [sbatch args...]"
  echo "Example: $0 configs/reward_models/discriminative_reward_no_augmentation_rhythm_2_4_datasets.yml"
  echo "Example: $0 configs/reward_models/discriminative_reward_no_augmentation_rhythm_2_4_datasets.yml --time=02:00:00"
  echo "Example: $0 configs/reward_models/discriminative_reward_no_augmentation_rhythm_2_4_datasets.yml --time=02:00:00 --gres=gpu:1"
  exit 2
fi

CONFIG_YML="$1"
JOB_NAME="${${CONFIG_YML:t}%.yml}"
shift

mkdir -p "jobscripts/slurm_logs/${JOB_NAME}"
sbatch --job-name="${JOB_NAME}" "$@" jobscripts/train_variants/discriminative_reward/run_discriminative_reward_rhythm.sh "${CONFIG_YML}"

