#!/usr/bin/env zsh
# Source this file to register short commands:
#   source scripts/run_generate_sequences.sh
#
# This file should only declare aliases/functions

export MY_CHECKPOINTS="${MY_CHECKPOINTS:-/hpcwork/thes2192/realchords/logs/my_logs}"


# -----------------------------------------------------------------------------
# GT Data
# -----------------------------------------------------------------------------

hooktheory_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_gt
}

wikifonia_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_gt
}

nottingham_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_gt
}

pop909_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_gt
}

jazzmus_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name jazzmus \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/jazzmus_gt
}

wjd_gt() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_chord_data \
    --dataset_name wjd \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wjd_gt
}

# -----------------------------------------------------------------------------
# Online MLE
# -----------------------------------------------------------------------------

hooktheory_melody_vs_mle_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_melody_vs_mle_chord \
}


hooktheory_melody_vs_mle_chord_3_datasets() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord_3_datasets/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_melody_vs_mle_chord_3_datasets \
}


wikifonia_melody_vs_mle_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_melody_vs_mle_chord \
}


wikifonia_melody_vs_mle_chord_3_datasets() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord_3_datasets/step=11000.ckpt" \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_melody_vs_mle_chord_3_datasets \
}

pop909_melody_vs_mle_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_melody_vs_mle_chord \
}

pop909_melody_vs_mle_chord_3_datasets() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord_3_datasets/step=11000.ckpt" \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_melody_vs_mle_chord_3_datasets \
}

nottingham_melody_vs_mle_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_melody_vs_mle_chord \
}

nottingham_melody_vs_mle_chord_3_datasets() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord_3_datasets/step=11000.ckpt" \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_melody_vs_mle_chord_3_datasets \
}

# -----------------------------------------------------------------------------
# Realchords
# -----------------------------------------------------------------------------

hooktheory_melody_vs_realchords_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_melody_vs_realchords_chord \
}

wikifonia_melody_vs_realchords_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_melody_vs_realchords_chord \
}


nottingham_melody_vs_realchords_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_melody_vs_realchords_chord \
}


pop909_melody_vs_realchords_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_melody_vs_realchords_chord \
}

# -----------------------------------------------------------------------------
# GAPT
# -----------------------------------------------------------------------------

hooktheory_melody_vs_gapt_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_melody_vs_gapt_chord \
}


wikifonia_melody_vs_gapt_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_melody_vs_gapt_chord \
}

pop909_melody_vs_gapt_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_melody_vs_gapt_chord \
}

nottingham_melody_vs_gapt_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_melody_vs_gapt_chord \
}


# -----------------------------------------------------------------------------
# Model vs Model
# -----------------------------------------------------------------------------

# Decoder melody vs all
# -----------------------------------------------------------------------------

mle_melody_vs_mle_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/mle_melody_vs_mle_chord_free_generation \
}

mle_melody_vs_mle_chord_with_prompt() {
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/mle_melody_vs_mle_chord_with_prompt \
}

mle_melody_vs_realchords_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --num_batches 64 \
    --save_dir logs/generated/mle_melody_vs_realchords_chord_free_generation \
}

mle_melody_vs_realchords_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/mle_melody_vs_realchords_chord_with_prompt \
}

mle_melody_vs_gapt_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/mle_melody_vs_gapt_chord_free_generation \
}

mle_melody_vs_gapt_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/mle_melody_vs_gapt_chord_with_prompt \
}

# Realchords melody vs all
# -----------------------------------------------------------------------------

realchords_melody_vs_mle_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/realchords_melody_vs_mle_chord_free_generation \
}

realchords_melody_vs_mle_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/realchords_melody_vs_mle_chord_with_prompt \
}

realchords_melody_vs_realchords_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/realchords_melody_vs_realchords_chord_free_generation \
}

realchords_melody_vs_realchords_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/realchords_melody_vs_realchords_chord_with_prompt \
}

realchords_melody_vs_gapt_chord_free_generation(){  
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/realchords_melody_vs_gapt_chord_free_generation \
}

realchords_melody_vs_gapt_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/realchords_melody_vs_gapt_chord_with_prompt \
}


# GAPT melody vs all
# -----------------------------------------------------------------------------

gapt_melody_vs_mle_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_melody_vs_mle_chord_free_generation \
}

gapt_melody_vs_mle_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_melody_vs_mle_chord_with_prompt \
}

gapt_melody_vs_realchords_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_melody_vs_realchords_chord_free_generation \
}

gapt_melody_vs_realchords_chord_with_prompt(){    
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_melody_vs_realchords_chord_with_prompt \
}

gapt_melody_vs_gapt_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_melody_vs_gapt_chord_free_generation \
}

gapt_melody_vs_gapt_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_melody_vs_gapt_chord_with_prompt \
}


# -----------------------------------------------------------------------------
# GAPT Multiscale
# -----------------------------------------------------------------------------

hooktheory_melody_vs_gapt_multiscale_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/hooktheory_melody_vs_gapt_multiscale_chord \
}


wikifonia_melody_vs_gapt_multiscale_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name wikifonia \
    --dataset_split all \
    --num_batches -1 \
    --save_dir logs/generated/wikifonia_melody_vs_gapt_multiscale_chord \
}

pop909_melody_vs_gapt_multiscale_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name pop909 \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/pop909_melody_vs_gapt_multiscale_chord \
}

nottingham_melody_vs_gapt_multiscale_chord() {
  python scripts/generate_sequences.py \
    --mode melody_data_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name nottingham \
    --dataset_split test \
    --num_batches -1 \
    --save_dir logs/generated/nottingham_melody_vs_gapt_multiscale_chord \
}

mle_melody_vs_gapt_multiscale_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/mle_melody_vs_gapt_multiscale_chord_free_generation \
}

mle_melody_vs_gapt_multiscale_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode mle_melody_vs_rl_chord \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/mle_melody_vs_gapt_multiscale_chord_with_prompt \
}

realchords_melody_vs_gapt_multiscale_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/realchords_melody_vs_gapt_multiscale_chord_free_generation \
}

realchords_melody_vs_gapt_multiscale_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/realchords_melody/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/realchords_melody_vs_gapt_multiscale_chord_with_prompt \
}

gapt_multiscale_melody_vs_mle_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_mle_chord_free_generation \
}

gapt_multiscale_melody_vs_mle_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_mle_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_mle_chord_with_prompt \
}

gapt_multiscale_melody_vs_realchords_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_realchords_chord_free_generation \
}

gapt_multiscale_melody_vs_realchords_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/realchords/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_realchords_chord_with_prompt \
}

gapt_multiscale_melody_vs_gapt_multiscale_chord_free_generation(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --num_batches 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_gapt_multiscale_chord_free_generation \
}

gapt_multiscale_melody_vs_gapt_multiscale_chord_with_prompt(){
  python scripts/generate_sequences.py \
    --mode rl_melody_vs_rl_chord \
    --rl_melody_model_path "$MY_CHECKPOINTS/rl_melody_gail_4_datasets/actor.pth" \
    --rl_chord_model_path "$MY_CHECKPOINTS/gapt_multiscale/actor.pth" \
    --mle_melody_model_path "$MY_CHECKPOINTS/decoder_only_online_melody_4_datasets/step=11000.ckpt" \
    --mle_chord_model_path "$MY_CHECKPOINTS/decoder_only_online_chord/step=11000.ckpt" \
    --dataset_name hooktheory \
    --dataset_split test \
    --num_batches 64 \
    --prompt_frames 64 \
    --save_dir logs/generated/gapt_multiscale_melody_vs_gapt_multiscale_chord_with_prompt \
}