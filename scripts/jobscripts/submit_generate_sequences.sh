#!/usr/bin/zsh
#
# Submit:
#   sbatch scripts/jobscripts/submit_generate_sequences.sh
#
# Single SLURM job, one GPU. Add any functions to gen_fns below.
# Up to MAX_PARALLEL of them run concurrently on that same GPU.
#
# Override parallelism at submit time, e.g.:
#   MAX_PARALLEL=2 sbatch scripts/jobscripts/submit_generate_sequences.sh
#
#SBATCH --partition=c23g
#SBATCH --job-name=generate_sequences
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=scripts/jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=scripts/jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

set -euo pipefail

MAX_PARALLEL="${MAX_PARALLEL:-4}"

source scripts/jobscripts/_common_env.sh

mkdir -p "scripts/jobscripts/slurm_logs/${SLURM_JOB_NAME}"
LOG_DIR="scripts/jobscripts/slurm_logs/${SLURM_JOB_NAME}"

source scripts/jobscripts/generate_sequences_functions.sh

gen_fns=(
  hooktheory_melody_vs_gapt_multiscale_chord
  wikifonia_melody_vs_gapt_multiscale_chord
  pop909_melody_vs_gapt_multiscale_chord
  nottingham_melody_vs_gapt_multiscale_chord
  mle_melody_vs_gapt_multiscale_chord_free_generation
  mle_melody_vs_gapt_multiscale_chord_with_prompt
  realchords_melody_vs_gapt_multiscale_chord_free_generation
  realchords_melody_vs_gapt_multiscale_chord_with_prompt
  gapt_multiscale_melody_vs_mle_chord_free_generation
  gapt_multiscale_melody_vs_mle_chord_with_prompt
  gapt_multiscale_melody_vs_realchords_chord_free_generation
  gapt_multiscale_melody_vs_realchords_chord_with_prompt
  gapt_multiscale_melody_vs_gapt_multiscale_chord_free_generation
  gapt_multiscale_melody_vs_gapt_multiscale_chord_with_prompt
)

if (( ${#gen_fns[@]} == 0 )); then
  echo "ERROR: gen_fns is empty. Add function names to scripts/jobscripts/submit_generate_sequences.sh"
  exit 2
fi

wait_for_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    sleep 2
  done
}

echo "=== ${#gen_fns[@]} generations, max ${MAX_PARALLEL} parallel on 1 GPU ==="

for fn in "${gen_fns[@]}"; do
  wait_for_slot
  echo "=== starting: ${fn} ==="
  (
    echo "=== running: ${fn} ==="
    "${fn}"
    echo "=== finished: ${fn} ==="
  ) > "${LOG_DIR}/${fn}.out" 2>&1 &
done

wait

echo "=== all generations finished ==="
