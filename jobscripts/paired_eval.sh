#!/usr/bin/env zsh
# Paired chord evaluation: generate chord predictions from multiple models
# conditioned on the same GT melody crops so every row in gt.pt / *.pt
# corresponds to the same song and can be compared directly.
#
# Usage:
#   source jobscripts/paired_eval.sh
#   paired_hooktheory
#   paired_wikifonia
#   paired_pop909
#   paired_nottingham

# --base_model: the MLE chord Lightning checkpoint (.ckpt).
#   Serves two purposes:
#     1. Provides the model architecture (reads args.yml from the same folder).
#     2. Its weights are deep-copied as the starting point before RL actor
#        weights (.pth) are loaded on top. Must match the architecture that
#        RealJam / GAPT were fine-tuned from.
_BASE="logs/decoder_only_online_chord/step=11000.ckpt"
_REALCHORDS="logs/realchords/actor.pth"
_GAPT="logs/gapt/actor.pth"

paired_hooktheory() {
  python scripts/paired_chord_evaluation.py \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  hooktheory \
    --dataset_split test \
    --save_dir      logs/paired_eval/hooktheory \
    --batch_size    8 \
    --num_batches   1 \
    --seed          42
}

paired_wikifonia() {
  python scripts/paired_chord_evaluation.py \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  wikifonia \
    --dataset_split test \
    --save_dir      logs/paired_eval/wikifonia \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_pop909() {
  python scripts/paired_chord_evaluation.py \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  pop909 \
    --dataset_split test \
    --save_dir      logs/paired_eval/pop909 \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_nottingham() {
  python scripts/paired_chord_evaluation.py \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  nottingham \
    --dataset_split test \
    --save_dir      logs/paired_eval/nottingham \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}
