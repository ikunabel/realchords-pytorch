#!/usr/bin/zsh
#
# Submit:
#   sbatch scripts/eval/submit_evaluate_sequences.sh
#
# Pick systems via ONE of:
#   EVAL_GROUP=gapt_multiscale_batch   (default below)
#   EVAL_SYSTEMS='hooktheory_gt wikifonia_gt'
#   edit eval_fn to call eval_systems with explicit args
#
# Examples:
#   EVAL_GROUP=melody_vs_mle sbatch scripts/eval/submit_evaluate_sequences.sh
#   EVAL_GROUP=all sbatch scripts/eval/submit_evaluate_sequences.sh
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

source scripts/eval/evaluate_sequences_functions.sh

: "${EVAL_GROUP:=gapt_multiscale_batch}"

echo "=== evaluate_sequences (EVAL_GROUP=${EVAL_GROUP:-unset}, EVAL_SYSTEMS=${EVAL_SYSTEMS:-unset}) ==="

if [[ -n "${EVAL_SYSTEMS:-}" ]]; then
  eval_systems
elif [[ -n "${EVAL_GROUP:-}" ]]; then
  eval_group "${EVAL_GROUP}"
else
  eval_all
fi

echo "=== evaluation finished ==="
