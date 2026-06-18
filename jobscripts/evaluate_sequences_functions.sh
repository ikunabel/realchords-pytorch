#!/usr/bin/env zsh
# Evaluate generated sequence folders under `logs/generated/`.
#
# Usage:
#   source jobscripts/evaluate_sequences_functions.sh
#   eval_hooktheory_melody_vs_decoder_only_online_chord

typeset -g RPROMPT="${RPROMPT-}"

# -----------------------------------------------------------------------------
# Choose from systems:

#     hooktheory_gt
#     wikifonia_gt
#     nottingham_gt
#     pop909_gt
#     hooktheory_melody_vs_mle_chord
#     hooktheory_melody_vs_mle_chord_3_datasets
#     wikifonia_melody_vs_mle_chord
#     wikifonia_melody_vs_mle_chord_3_datasets
#     pop909_melody_vs_mle_chord
#     pop909_melody_vs_mle_chord_3_datasets
#     nottingham_melody_vs_mle_chord
#     nottingham_melody_vs_mle_chord_3_datasets
#     hooktheory_melody_vs_realchords_chord
#     wikifonia_melody_vs_realchords_chord
#     nottingham_melody_vs_realchords_chord
#     pop909_melody_vs_realchords_chord
#     hooktheory_melody_vs_gapt_chord
#     wikifonia_melody_vs_gapt_chord
#     nottingham_melody_vs_gapt_chord
#     pop909_melody_vs_gapt_chord
#     mle_melody_vs_mle_chord_free_generation
#     mle_melody_vs_mle_chord_with_prompt
#     mle_melody_vs_realchords_chord_free_generation
#     mle_melody_vs_realchords_chord_with_prompt
#     mle_melody_vs_gapt_chord_free_generation
#     mle_melody_vs_gapt_chord_with_prompt
#     realchords_melody_vs_realchords_chord_free_generation
#     realchords_melody_vs_realchords_chord_with_prompt
#     realchords_melody_vs_gapt_chord_free_generation
#     realchords_melody_vs_gapt_chord_with_prompt
#     realchords_melody_vs_mle_chord_free_generation
#     realchords_melody_vs_mle_chord_with_prompt
#     gapt_melody_vs_gapt_chord_free_generation
#     gapt_melody_vs_gapt_chord_with_prompt
#     gapt_melody_vs_realchords_chord_free_generation
#     gapt_melody_vs_realchords_chord_with_prompt
#     gapt_melody_vs_mle_chord_free_generation
#     gapt_melody_vs_mle_chord_with_prompt
#     hooktheory_melody_vs_gapt_multiscale_chord
#     wikifonia_melody_vs_gapt_multiscale_chord
#     pop909_melody_vs_gapt_multiscale_chord
#     nottingham_melody_vs_gapt_multiscale_chord
#     mle_melody_vs_gapt_multiscale_chord_free_generation
#     mle_melody_vs_gapt_multiscale_chord_with_prompt
#     realchords_melody_vs_gapt_multiscale_chord_free_generation
#     realchords_melody_vs_gapt_multiscale_chord_with_prompt
#     gapt_multiscale_melody_vs_mle_chord_free_generation
#     gapt_multiscale_melody_vs_mle_chord_with_prompt
#     gapt_multiscale_melody_vs_realchords_chord_free_generation
#     gapt_multiscale_melody_vs_realchords_chord_with_prompt
#     gapt_multiscale_melody_vs_gapt_multiscale_chord_free_generation
#     gapt_multiscale_melody_vs_gapt_multiscale_chord_with_prompt

# -----------------------------------------------------------------------------

eval_all() {
  # List system folder names once; each expands to:
  #   --system "<name>=${LOGS_ROOT}/<name>"
  local -a systems
  systems=(
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

  local -a cmd
  cmd=(python scripts/evaluate_generated_sequences.py)
  local name dir
  local -i skipped=0 included=0
  for name in "${systems[@]}"; do
    dir="logs/generated/${name}"
    if [[ ! -d "${dir}" ]]; then
      echo "WARNING: skipping missing system directory: ${dir}" >&2
      (( skipped++ )) || true
      continue
    fi
    cmd+=(--system "${name}=${dir}")
    (( included++ )) || true
  done

  if (( included == 0 )); then
    echo "ERROR: no system directories found under logs/generated/" >&2
    exit 1
  fi

  echo "Evaluating ${included} system(s) (${skipped} skipped)"

  cmd+=(
    --analysis_root "logs/eval"
    --summary_path "logs/eval/summary.json"
    --config configs/thesis/realchords.yml
  )

  "${cmd[@]}"
}
