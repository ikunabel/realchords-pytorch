#!/usr/bin/zsh
# Runs scripts/eval/recall/eval_reward_recall.py on all 8 hooktheory-only multiscale
# contrastive reward checkpoints (w16/w32/w64/w128, fixed + sliding), plus the
# ReaLchords paper's own Table 4 numbers and this repo's HuggingFace reference
# checkpoints (both included by default -- see eval_reward_recall.py).
#
# Usage: run from the repo root.
#   ./scripts/eval/recall/run_recall_eval_contrastive.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."

module load Python/3.12.3
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

R="/hpcwork/thes2192/realchords/logs/my_logs/contrastive_reward"

python -u scripts/eval/recall/eval_reward_recall.py --reward_type contrastive \
  --eval_augmentation off \
  --model "our Full (256)=$R/contrastive_reward/step=8000.ckpt" \
  --model "our Full (256) 2=$R/contrastive_reward_2/step=8000.ckpt" \
  --model "w16 fixed=$R/contrastive_reward_w16_bswzdhj9/step=26500.ckpt" \
  --model "w16 sliding=$R/contrastive_reward_w16_sliding_2o1vixac/step=18500.ckpt" \
  --model "w32 fixed=$R/contrastive_reward_w32_mfzn3d5c/step=22500.ckpt" \
  --model "w32 sliding=$R/contrastive_reward_w32_sliding_57e9e6hx/step=28000.ckpt" \
  --model "w64 fixed=$R/contrastive_reward_w64_2v18qctj/step=12000.ckpt" \
  --model "w64 sliding=$R/contrastive_reward_w64_sliding_etmmo7ww/step=21500.ckpt" \
  --model "w128 fixed=$R/contrastive_reward_w128_xyvrn0ah/step=10500.ckpt" \
  --model "w128 sliding=$R/contrastive_reward_w128_sliding_zrdzoy4s/step=13500.ckpt"
