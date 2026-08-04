#!/usr/bin/zsh
#
# Submit:
#   sbatch scripts/eval/evaluate_sequences/submit_evaluate_sequences.sh gt_vs_gapt gt_vs_mle
#
# Every positional arg is a preset name (category/name), a bare category
# name (gt, gt_vs_mle, gt_vs_realchords, gt_vs_gapt, model_vs_model), or a
# glob pattern (e.g. 'model_vs_model/*gapt_multiscale*'). See
# `python scripts/eval/evaluate_sequences/run_evaluate_sequences.py --list` for on-disk status
# of every known system.
#
# Examples:
#   sbatch scripts/eval/evaluate_sequences/submit_evaluate_sequences.sh model_vs_model
#   sbatch scripts/eval/evaluate_sequences/submit_evaluate_sequences.sh 'model_vs_model/*gapt_multiscale*'
#
#SBATCH --partition=c23g
#SBATCH --job-name=evaluate_sequences
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

set -euo pipefail

source scripts/jobscripts/_common_env.sh

mkdir -p "scripts/jobscripts/slurm_logs/${SLURM_JOB_NAME}"

if [[ $# -eq 0 ]]; then
  echo "ERROR: pass at least one preset, category, or glob pattern, e.g.:"
  echo "  sbatch scripts/eval/evaluate_sequences/submit_evaluate_sequences.sh model_vs_model"
  python scripts/eval/evaluate_sequences/run_evaluate_sequences.py --list
  exit 2
fi

python -u scripts/eval/evaluate_sequences/run_evaluate_sequences.py "$@"

echo "=== evaluation finished ==="
