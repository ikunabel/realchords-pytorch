#!/usr/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yml> [sbatch args...]"
  echo "Example: $0 configs/reward_models/multiscale/contrastive_reward_w64_3_datasets.yml"
  echo "Example: $0 configs/reward_models/multiscale/contrastive_reward_w64_sliding_3_datasets.yml --time=02:00:00"
  echo "Example: $0 configs/reward_models/multiscale/contrastive_reward_w64_3_datasets.yml --time=02:00:00 --gres=gpu:1"
  exit 2
fi

CONFIG_YML="$1"
JOB_NAME="${${CONFIG_YML:t}%.yml}"
shift

mkdir -p "scripts/jobscripts/slurm_logs/${JOB_NAME}"
sbatch --job-name="${JOB_NAME}" "$@" scripts/jobscripts/train_variants/contrastive_reward/run_contrastive_reward_segment.sh "${CONFIG_YML}"
