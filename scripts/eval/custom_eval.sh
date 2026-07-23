#!/usr/bin/env zsh
# Generate chord predictions from multiple models
# conditioned on the same GT melody crops so every row in gt.pt / *.pt
# corresponds to the same song and can be compared directly.
#
# Usage:
#   source scripts/eval/custom_eval.sh
#   paired_hooktheory
#   paired_wikifonia
#   paired_pop909
#   paired_nottingham
#
# GT-only (no models) — for cross-dataset chord distribution analysis:
#   paired_gt_all
#   paired_gt_hooktheory
#
# --save_dir is suffixed with the dataset split (e.g. "wjd_test", "wjd_all")
# so cropped_songs/ and full_songs/ from different splits never collide:
#   gt/wjd_test/cropped_songs, gt/wjd_test/full_songs
#   gt/wjd_all/cropped_songs,  gt/wjd_all/full_songs
#
_PAIRED_GT_DIR="logs/paired_eval/gt"
NUM_MIDIS=-1      # -1 = export all sequences
MELODY_OCTAVE=0   # offset added to stored melody octaves in MIDI export
CHORD_OCTAVE=4    # octave for naive chord voicings

# Each call below writes two variants under --save_dir:
#   cropped_songs/  legacy 256-frame (8-bar melody + 8-bar chord) crop
#   full_songs/     whole songs, uncropped
_paired_eval() {
  python realchords/utils/custom_evaluation.py \
    --midi_samples "$NUM_MIDIS" \
    --melody_octave "$MELODY_OCTAVE" \
    --chord_octave "$CHORD_OCTAVE" \
    "$@"
}

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
  _paired_eval \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  hooktheory \
    --dataset_split test \
    --save_dir      logs/paired_eval/hooktheory_test \
    --batch_size    8 \
    --num_batches   1 \
    --seed          42
}

paired_wikifonia() {
  _paired_eval \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  wikifonia \
    --dataset_split test \
    --save_dir      logs/paired_eval/wikifonia_test \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_pop909() {
  _paired_eval \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  pop909 \
    --dataset_split test \
    --save_dir      logs/paired_eval/pop909_test \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

paired_nottingham() {
  _paired_eval \
    --base_model    "$_BASE" \
    --model         "MLE=base" \
    --model         "RealJam=$_REALCHORDS" \
    --model         "GAPT=$_GAPT" \
    --dataset_name  nottingham \
    --dataset_split test \
    --save_dir      logs/paired_eval/nottingham_test \
    --batch_size    64 \
    --num_batches   -1 \
    --seed          42
}

# ---------------------------------------------------------------------------
# GT-only collection (no model loading) — chord distributions per dataset
# Outputs: logs/paired_eval/gt/<dataset>_<split>/gt_chord_distribution.json
# ---------------------------------------------------------------------------

GT_BATCH_SIZE=64
GT_NUM_BATCHES=-1  # -1 = iterate the full dataloader (all songs)

paired_gt_hooktheory() {
  _paired_eval \
    --gt_only \
    --dataset_name  hooktheory \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/hooktheory_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_wikifonia() {
  _paired_eval \
    --gt_only \
    --dataset_name  wikifonia \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/wikifonia_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_pop909() {
  _paired_eval \
    --gt_only \
    --dataset_name  pop909 \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/pop909_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_nottingham() {
  _paired_eval \
    --gt_only \
    --dataset_name  nottingham \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/nottingham_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_jazzmus() {
  _paired_eval \
    --gt_only \
    --dataset_name  jazzmus \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/jazzmus_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_wjd() {
  CHORD_OCTAVE=5 _paired_eval \
    --gt_only \
    --dataset_name  wjd \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/wjd_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_chord_melody_dataset() {
  _paired_eval \
    --gt_only \
    --dataset_name  chord_melody_dataset \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/chord_melody_dataset_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_emopia_plus() {
  _paired_eval \
    --gt_only \
    --dataset_name  emopia_plus \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/emopia_plus_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_filobass() {
  CHORD_OCTAVE=5 _paired_eval \
    --gt_only \
    --dataset_name  filobass \
    --dataset_split all \
    --save_dir      "$_PAIRED_GT_DIR/filobass_all" \
    --batch_size    $GT_BATCH_SIZE \
    --num_batches   $GT_NUM_BATCHES \
    --seed          42
}

paired_gt_all() {
  paired_gt_hooktheory
  paired_gt_wikifonia
  paired_gt_pop909
  paired_gt_nottingham
  paired_gt_jazzmus
  paired_gt_wjd
  paired_gt_chord_melody_dataset
  paired_gt_emopia_plus
  paired_gt_filobass
}
