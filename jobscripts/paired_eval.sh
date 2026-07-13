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
#
# GT-only (no models) — for cross-dataset chord distribution analysis:
#   paired_gt_all
#   paired_gt_hooktheory

_PAIRED_GT_DIR="logs/paired_eval/gt"

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

# ---------------------------------------------------------------------------
# GT-only collection (no model loading) — chord distributions per dataset
# Outputs: logs/paired_eval/gt/<dataset>/gt_chord_distribution.json
# ---------------------------------------------------------------------------

paired_gt_hooktheory() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  hooktheory \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/hooktheory" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_wikifonia() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  wikifonia \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/wikifonia" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_pop909() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  pop909 \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/pop909" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_nottingham() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  nottingham \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/nottingham" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_jazzmus() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  jazzmus \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/jazzmus" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_wjazzd() {
  python scripts/paired_chord_evaluation.py \
    --gt_only \
    --dataset_name  wjazzd \
    --dataset_split test \
    --save_dir      "$_PAIRED_GT_DIR/wjazzd" \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_gt_all() {
  paired_gt_hooktheory
  paired_gt_wikifonia
  paired_gt_pop909
  paired_gt_nottingham
  paired_gt_jazzmus
  paired_gt_wjazzd
}
