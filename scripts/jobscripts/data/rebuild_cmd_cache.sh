#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=rebuild_cmd_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

# Rebuild the Chord Melody Dataset cache with the current converter.
#
# The live cache (data/cache/chord_melody_dataset, built 2026-07-14) predates commit bca52f1
# (2026-07-25), which changed two things:
#   1. split/augmentation: the corpus ships one MusicXML per transposed key, and the old converter
#      used that 12-key duplication as the augmentation -- making TRAIN ~12x its song count while
#      VALID/TEST stayed 1:1 (80/10/10 by song but ~98/1/1 by row). The converter now takes one
#      representative key per song and applies the standard [-6,+6] augmentation to TRAIN only.
#   2. `score.toSoundingPitch()`: about half the files are written an octave above sounding pitch.
#      Without it those melodies land an octave too high -- verified in the live cache, whose
#      median melody pitch is bimodal with the clusters exactly 12 semitones apart (key 'c' only,
#      so transposition cannot explain it). See journal/DATASET_SUMMARY.md.
#
# Writes to a NEW directory; the live cache is left untouched until the rebuild is verified and
# swapped in deliberately.

OUT="data/cache/chord_melody_dataset"
echo "=== rebuilding into ${OUT} (live cache untouched) ==="

srun python scripts/convert_data_to_cache/convert_chord_melody_dataset_to_cache.py \
  --output_dir "${OUT}" \
  --augmentation

echo "=== row counts ==="
wc -l "${OUT}"/*.jsonl
