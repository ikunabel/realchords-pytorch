#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=commit_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

# Runs scripts/eval/custom_eval/commit_replay.py as a SLURM job, for when the shared login GPU is
# out of memory. All arguments are passed through. About 3 min per model and corpus at k = 16 on a
# dedicated GPU, so 30 min covers one stage, one seed, both corpora, k = 8 and 16.
# Usage (from the repo root):
#   sbatch scripts/jobscripts/eval/run_commit_replay.sh --stage gapt --imagine_with mle \
#       --seeds s42 --k 8 16 --out_root logs/custom_eval/commit_replay_mleimag

source scripts/jobscripts/_common_env.sh

srun python scripts/eval/custom_eval/commit_replay.py "$@"
