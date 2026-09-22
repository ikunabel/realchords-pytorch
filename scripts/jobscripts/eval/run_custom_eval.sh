#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=custom_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

# Runs realchords/utils/custom_evaluation.py on one argbind config as a SLURM job
# (login-node GPU processes are killed after 30 min; multi-model full-test-set
# evals can take longer).
# Usage (from the repo root): sbatch scripts/jobscripts/eval/run_custom_eval.sh <config.yml>

source scripts/jobscripts/_common_env.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <custom_eval config yml>"
  exit 2
fi

srun python realchords/utils/custom_evaluation.py --args.load "$1"
