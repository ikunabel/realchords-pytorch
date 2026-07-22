#!/usr/bin/env zsh
# Evaluate generated sequence folders under logs/generated/.
#
# Usage:
#   source scripts/jobscripts/evaluate_sequences_functions.sh
#
#   eval_all                              # all known systems that exist on disk
#   eval_group melody_vs_mle melody_vs_gapt   # one or more named groups
#   eval_systems hooktheory_gt '*mle*'    # explicit names or globs
#   eval_discover                         # any logs/generated/* dir with .pt files
#   list_eval_systems                     # show registry + on-disk status
#
# Slurm:
#   EVAL_GROUP=gapt_multiscale_batch sbatch scripts/jobscripts/submit_evaluate_sequences.sh
#   EVAL_SYSTEMS='hooktheory_gt wikifonia_gt' sbatch scripts/jobscripts/submit_evaluate_sequences.sh

typeset -g RPROMPT="${RPROMPT-}"

setopt extendedglob

: "${GENERATED_ROOT:=logs/generated}"
: "${EVAL_ANALYSIS_ROOT:=logs/eval}"
: "${EVAL_SUMMARY_PATH:=logs/eval/summary.json}"
: "${EVAL_CONFIG:=configs/thesis/realchords.yml}"

source "${0:A:h}/generated_systems.sh"

_generated_system_dir() {
  print -r -- "${GENERATED_ROOT}/$1"
}

_generated_system_has_outputs() {
  local dir
  dir="$(_generated_system_dir "$1")"
  [[ -d "${dir}" ]] && print -l "${dir}"/**/*.pt(N) 2>/dev/null | grep -q .
}

_resolve_eval_patterns() {
  local -a patterns=("$@")
  local -a resolved=()
  local pattern name

  if (( ${#patterns[@]} == 0 )); then
    return 1
  fi

  for pattern in "${patterns[@]}"; do
    if [[ "${pattern}" == *('*'|'?'|'[')* ]]; then
      local -a matches=(${ (M)GENERATED_SYSTEMS:#${~pattern} })
      if (( ${#matches[@]} == 0 )); then
        echo "WARNING: pattern matched no registered systems: ${pattern}" >&2
      else
        resolved+=("${matches[@]}")
      fi
    elif [[ " ${GENERATED_SYSTEMS[*]} " == *" ${pattern} "* ]]; then
      resolved+=("${pattern}")
    else
      echo "WARNING: unknown system (not in registry): ${pattern}" >&2
      resolved+=("${pattern}")
    fi
  done

  if (( ${#resolved[@]} == 0 )); then
    return 1
  fi

  # Deduplicate while preserving order.
  local -a unique=(${(ou)resolved})
  REPLY=("${unique[@]}")
}

_systems_with_outputs() {
  local -a names=("$@")
  local -a existing=()
  local name

  for name in "${names[@]}"; do
    if _generated_system_has_outputs "${name}"; then
      existing+=("${name}")
    fi
  done

  REPLY=("${existing[@]}")
}

list_eval_groups() {
  local group
  for group in ${(ok)GENERATED_SYSTEM_GROUPS}; do
    local -a members=(${=GENERATED_SYSTEM_GROUPS[$group]})
    print -r -- "${group} (${#members[@]} systems)"
  done
}

list_eval_systems() {
  local name
  for name in "${GENERATED_SYSTEMS[@]}"; do
    if _generated_system_has_outputs "${name}"; then
      print -r -- "[ready]  ${name}"
    elif [[ -d "$(_generated_system_dir "${name}")" ]]; then
      print -r -- "[empty]  ${name}"
    else
      print -r -- "[missing] ${name}"
    fi
  done
}

_discover_generated_systems() {
  local -a discovered=()
  local dir name

  for dir in "${GENERATED_ROOT}"/*(/N); do
    name="${dir:t}"
    if _generated_system_has_outputs "${name}"; then
      discovered+=("${name}")
    fi
  done

  REPLY=(${(o)discovered})
}

eval_systems() {
  local -a requested=()
  if (( $# > 0 )); then
    requested=("$@")
  elif [[ -n "${EVAL_SYSTEMS:-}" ]]; then
    requested=(${=EVAL_SYSTEMS})
  else
    echo "ERROR: eval_systems needs system names, EVAL_SYSTEMS, or use eval_all / eval_group" >&2
    return 2
  fi

  _resolve_eval_patterns "${requested[@]}" || return 2
  local -a systems=("${REPLY[@]}")
  _systems_with_outputs "${systems[@]}"
  local -a ready=("${REPLY[@]}")
  local -i skipped=$(( ${#systems[@]} - ${#ready[@]} ))

  if (( ${#ready[@]} == 0 )); then
    echo "ERROR: no system directories with .pt files found" >&2
    return 1
  fi

  local -a cmd=(python scripts/evaluate_generated_sequences.py)
  local name dir
  for name in "${ready[@]}"; do
    dir="$(_generated_system_dir "${name}")"
    cmd+=(--system "${name}=${dir}")
  done

  echo "Evaluating ${#ready[@]} system(s) (${skipped} skipped, no outputs)"

  cmd+=(
    --analysis_root "${EVAL_ANALYSIS_ROOT}"
    --summary_path "${EVAL_SUMMARY_PATH}"
    --config "${EVAL_CONFIG}"
  )

  "${cmd[@]}"
}

eval_group() {
  if (( $# == 0 )); then
    echo "ERROR: eval_group needs at least one group name. Available:" >&2
    list_eval_groups >&2
    return 2
  fi

  local -a systems=()
  local group
  for group in "$@"; do
    if [[ -z "${GENERATED_SYSTEM_GROUPS[$group]:-}" ]]; then
      echo "ERROR: unknown group '${group}'. Available:" >&2
      list_eval_groups >&2
      return 2
    fi
    systems+=(${=GENERATED_SYSTEM_GROUPS[$group]})
  done

  local -a unique=(${(ou)systems})
  eval_systems "${unique[@]}"
}

eval_discover() {
  _discover_generated_systems
  local -a systems=("${REPLY[@]}")
  if (( ${#systems[@]} == 0 )); then
    echo "ERROR: no directories with .pt files under ${GENERATED_ROOT}/" >&2
    return 1
  fi
  eval_systems "${systems[@]}"
}

eval_all() {
  if (( $# > 0 )); then
    eval_systems "$@"
    return $?
  fi
  if [[ -n "${EVAL_GROUP:-}" ]]; then
    eval_group "${EVAL_GROUP}"
    return $?
  fi
  if [[ -n "${EVAL_SYSTEMS:-}" ]]; then
    eval_systems
    return $?
  fi

  # Default: every registered system that already has generated .pt outputs.
  _systems_with_outputs "${GENERATED_SYSTEMS[@]}"
  local -a ready=("${REPLY[@]}")
  if (( ${#ready[@]} == 0 )); then
    echo "ERROR: no registered systems with outputs under ${GENERATED_ROOT}/" >&2
    echo "Run list_eval_systems to inspect, or eval_discover for unregistered folders." >&2
    return 1
  fi
  eval_systems "${ready[@]}"
}

