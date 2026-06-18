#!/usr/bin/zsh
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure repo root is importable when running scripts/...
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# User-built fluidsynth (for W&B MIDI audio logging)
export PATH="${HOME}/.local/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${HOME}/.local/lib:${HOME}/.local/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Define directory for torch extensions (use job ID as subdir if locks occur)
export TORCH_EXTENSIONS_DIR="/hpcwork/thes2192/realchords/.cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

# Set master port to avoid jobs on same node from conflicting
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-0} % 20000)))

# Modules / venv
module load Python/3.12.3
module load FFmpeg/7.0.2
source .venv/bin/activate

echo "=== toolchain check ==="
echo "PATH=$PATH"
echo "ffmpeg: $(command -v ffmpeg || echo NOT_FOUND)"
echo "fluidsynth: $(command -v fluidsynth || echo NOT_FOUND)"
ffmpeg -version | head -n 1 || true
fluidsynth --version | head -n 1 || true
python -c "import audioread; print('audioread backends:', audioread.available_backends())" || true
echo "=== end toolchain check ==="

echo "=== nvidia-smi (job ${SLURM_JOB_ID:-local}) ==="
nvidia-smi
echo "=== end nvidia-smi ==="

# W&B
export WANDB_ENTITY="${WANDB_ENTITY:-ikunabel}"
export WANDB_PROJECT="${WANDB_PROJECT:-realchords}"

# Store checkpoints on HPCWORK
export RUNS_ROOT="${RUNS_ROOT:-/hpcwork/thes2192/realchords/logs/my_logs}"
mkdir -p "${RUNS_ROOT}"