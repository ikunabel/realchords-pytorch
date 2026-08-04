#!/usr/bin/zsh
# Runs scripts/eval/recall/eval_reward_recall.py on our own hooktheory-only
# full-context (256-frame) and multiscale (w16/32/64/128, fixed+sliding)
# discriminative reward checkpoints, plus the ReaLchords paper's own Table 5
# numbers and this repo's HuggingFace reference checkpoints (both included by
# default -- see eval_reward_recall.py). --model paths point at run
# directories, not specific .ckpt files -- eval_reward_recall.py resolves
# each to that run's actual lowest-val-loss checkpoint automatically.
#
# Usage: run from the repo root.
#   ./scripts/eval/recall/run_recall_eval_discriminative.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."

module load Python/3.12.3
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

R="/hpcwork/thes2192/realchords/logs/my_logs/discriminative_reward"

python -u scripts/eval/recall/eval_reward_recall.py --reward_type discriminative \
  --eval_augmentation off \
  --model "our Full (256)=$R/discriminative_reward_128_bs/step=3000.ckpt" \
  --model "our Full (256) 2=$R/discriminative_reward_128_bs_2/step=3000.ckpt" \
  --model "w16 fixed=$R/discriminative_reward_w16_7zou96cj" \
  --model "w16 sliding=$R/discriminative_reward_w16_sliding_koofktu6" \
  --model "w32 fixed=$R/discriminative_reward_w32_8dn4xzrr" \
  --model "w32 sliding=$R/discriminative_reward_w32_sliding_g9vebcbo" \
  --model "w64 fixed=$R/discriminative_reward_w64_zu8l6z5o" \
  --model "w64 sliding=$R/discriminative_reward_w64_sliding_5yvynlt3" \
  --model "w128 fixed=$R/discriminative_reward_w128_w86dgv56" \
  --model "w128 sliding=$R/discriminative_reward_w128_sliding_0in6e5z8"
