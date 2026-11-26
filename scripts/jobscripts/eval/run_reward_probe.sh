#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=reward_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

# Runs scripts/eval/reward_probe.py (journal/experiments/REWARD_PROBE.md) for both reward-model families:
# the released Hooktheory ones (logs/huggingface/, vocabulary v1) and the 7-set ones gapt_7sets trained with
# (configs/rl/gapt_7sets.yml, vocabulary v2). Extra arguments are passed to both runs.
# Usage (from the repo root): sbatch scripts/jobscripts/eval/run_reward_probe.sh [--n_songs 200]

source scripts/jobscripts/_common_env.sh

srun python scripts/eval/reward_probe.py --family hooktheory "$@"
srun python scripts/eval/reward_probe.py --family 7sets "$@"
