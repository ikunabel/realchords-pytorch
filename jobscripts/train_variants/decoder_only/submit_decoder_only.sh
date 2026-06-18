#!/usr/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_yml> [sbatch args...]"
  echo "Example: $0 configs/gen_models/decoder_only_online_chord_3_datasets.yml"
  echo "Example: $0 configs/gen_models/decoder_only_online_chord_3_datasets.yml --time=02:00:00"
  echo "Example: $0 configs/gen_models/decoder_only_online_chord_3_datasets.yml --time=02:00:00 --gres=gpu:1"
  exit 2
fi

CONFIG_YML="$1"
JOB_NAME="${${CONFIG_YML:t}%.yml}"
shift

mkdir -p "jobscripts/slurm_logs/${JOB_NAME}"
sbatch --job-name="${JOB_NAME}" "$@" jobscripts/train_variants/decoder_only/run_decoder_only.sh "${CONFIG_YML}"

