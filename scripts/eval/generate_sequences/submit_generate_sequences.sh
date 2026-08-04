#!/usr/bin/zsh
#
# Submit:
#   sbatch scripts/eval/generate_sequences/submit_generate_sequences.sh gt_vs_gapt model_vs_model/gapt_melody_vs_gapt_chord_free_generation
#
# Single SLURM job, one GPU. Every positional arg is a preset name
# (category/name) or a bare category name (gt, gt_vs_mle, gt_vs_realchords,
# gt_vs_gapt, model_vs_model -- expands to every preset in that folder). See
# `python scripts/eval/generate_sequences/run_generate_sequences.py --list` for all presets.
#
# Override parallelism at submit time, e.g.:
#   sbatch scripts/eval/generate_sequences/submit_generate_sequences.sh model_vs_model --max_parallel 2
#
#SBATCH --partition=c23g
#SBATCH --job-name=generate_sequences
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

set -euo pipefail

source scripts/jobscripts/_common_env.sh

LOG_DIR="scripts/jobscripts/slurm_logs/${SLURM_JOB_NAME}"
mkdir -p "${LOG_DIR}"

if [[ $# -eq 0 ]]; then
  echo "ERROR: pass at least one preset or category name, e.g.:"
  echo "  sbatch scripts/eval/generate_sequences/submit_generate_sequences.sh model_vs_model"
  python scripts/eval/generate_sequences/run_generate_sequences.py --list
  exit 2
fi

python -u scripts/eval/generate_sequences/run_generate_sequences.py \
  --max_parallel "${MAX_PARALLEL:-4}" \
  --log_dir "${LOG_DIR}" \
  "$@"
