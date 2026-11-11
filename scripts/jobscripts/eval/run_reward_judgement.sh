#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=reward_judgement
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

# Runs scripts/eval/probe_reward_judgement.py: 24 reward models x 7 corpora x 5 chord sources.
# Usage (from the repo root): sbatch scripts/jobscripts/eval/run_reward_judgement.sh

source scripts/jobscripts/_common_env.sh

srun python scripts/eval/probe_reward_judgement.py "$@"
