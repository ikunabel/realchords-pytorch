#!/usr/bin/zsh
#SBATCH --partition=c23g
#SBATCH --job-name=enc_dec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=24
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=jobscripts/slurm_logs/%x/%x_%j.out
#SBATCH --error=jobscripts/slurm_logs/%x/%x_%j.err
#SBATCH --account=thes2192

source jobscripts/_common_env.sh

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config_yml>"
  exit 2
fi

CONFIG_YML="$1"

RUN_DIR="${RUNS_ROOT}/${SLURM_JOB_NAME}"
mkdir -p "${RUN_DIR}"

srun python scripts/train_enc_dec.py \
  --args.load "${CONFIG_YML}" \
  --save_dir "${RUN_DIR}"

