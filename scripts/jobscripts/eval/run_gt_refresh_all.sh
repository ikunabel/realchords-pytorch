#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=gt_refresh_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source scripts/jobscripts/_common_env.sh

# Regenerate every corpus's ground-truth tensors and metrics from the current caches, so the whole
# GT table comes from one consistent pass (chord_melody_dataset and jazzmus changed; the rest are
# unchanged but are re-run so no row predates a metric or converter change).
#
# --run_full_songs True: the configs do not set it and the effective default is False, which would
#   silently leave full_songs/ missing -- the view the corpus tables and the CLaMP 2 probe read.
# --midi_samples 0: write no MIDI. The MIDI exports are already complete and de-duplicated; a
#   default run would drop 10 sampled files into midi/gt/, which the CLaMP 2 probe prefers over the
#   full export sitting in midi/, silently reducing it to 10 songs per corpus.

for c in chord_melody_dataset emopia_plus filobass hooktheory jazzmus nottingham pop909 wikifonia wjd; do
  echo "=== ${c} ==="
  srun python realchords/utils/custom_evaluation.py \
    --args.load "configs/custom_eval/gt/${c}.yml" \
    --run_full_songs True \
    --midi_samples 0 || echo "FAILED: ${c}"
done

echo "=== experiment summary ==="
srun python scripts/eval/custom_eval/summarize_experiment.py logs/custom_eval/gt || true
