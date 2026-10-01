#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=gt_refresh_cmd_jazzmus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

# Regenerate the ground-truth tensors, metrics and MIDI exports for the two corpora whose caches
# changed: chord_melody_dataset (octave fix + one representative key per song, 4,252 rows -> 473)
# and jazzmus (duplicate removal, 292 -> 163). The previous outputs are kept alongside as
# *__stale_* directories; they contain MIDI for songs that no longer exist, which would otherwise be
# picked up by the CLaMP 2 probe. See journal/DATASET_SUMMARY.md.

srun python scripts/eval/custom_eval/run_custom_eval.py \
  gt_chord_melody_dataset gt_jazzmus \
  midi_gt_chord_melody_dataset midi_gt_jazzmus
