#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=render_cmd_octave_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh
srun python scripts/data/render_cmd_octave_ab.py --cache data/cache/chord_melody_dataset__rebuild2 --out data/cmd_octave_ab/after_fix
