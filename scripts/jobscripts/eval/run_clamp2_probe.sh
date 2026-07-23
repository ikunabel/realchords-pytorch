#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=clamp2_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

# Zero-shot genre voting + t-SNE of CLaMP2 embeddings across ALL songs in
# every dataset's full-songs GT MIDI export (logs/paired_eval/gt/<dataset>_all/full_songs/midi).
# Run scripts/eval/custom_eval.sh's `paired_gt_all` first (with NUM_MIDIS=-1,
# GT_NUM_BATCHES=-1, already the defaults) to populate those MIDI directories.

srun python scripts/eval/clamp2_dataset_probe.py \
  --gt_root logs/paired_eval/gt \
  --split_mode full_songs \
  --out_dir logs/paired_eval/clamp2_probe_full \
  "$@"
