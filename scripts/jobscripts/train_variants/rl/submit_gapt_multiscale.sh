#!/usr/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yml> [sbatch args...]"
  echo "Example: $0 configs/thesis/gapt_multiscale.yml"
  echo "Example: $0 configs/thesis/gapt_multiscale.yml --time=02:00:00"
  echo "Example: $0 configs/thesis/gapt_multiscale.yml --time=02:00:00 --gres=gpu:1 --cpus-per-gpu=16"
  exit 2
fi

CONFIG_YML="$1"
JOB_NAME="${${CONFIG_YML:t}%.yml}"
shift

mkdir -p "scripts/jobscripts/slurm_logs/${JOB_NAME}"
sbatch --job-name="${JOB_NAME}" "$@" scripts/jobscripts/train_variants/rl/run_gapt_multiscale.sh "${CONFIG_YML}"
