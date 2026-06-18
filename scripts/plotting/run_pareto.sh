#!/usr/bin/env zsh

plot_hooktheory_test_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=hooktheory_melody_vs_mle_chord" \
        --variant "Online MLE (3 datasets)=hooktheory_melody_vs_mle_chord_3_datasets" \
        --variant "RLPT=hooktheory_melody_vs_realchords_chord" \
        --variant "GAPT=hooktheory_melody_vs_gapt_chord" \
        --variant "GAPT-M=hooktheory_melody_vs_gapt_multiscale_chord" \
        --variant "GT=hooktheory_gt" \
        --title "Melody condition from Hooktheory test set" \
        --out scripts/plotting/hooktheory_test_set.pdf
}

plot_wikifonia_full_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=wikifonia_melody_vs_mle_chord" \
        --variant "Online MLE (3 datasets)=wikifonia_melody_vs_mle_chord_3_datasets" \
        --variant "RLPT=wikifonia_melody_vs_realchords_chord" \
        --variant "GAPT=wikifonia_melody_vs_gapt_chord" \
        --variant "GAPT-M=wikifonia_melody_vs_gapt_multiscale_chord" \
        --variant "GT=wikifonia_gt" \
        --title "Melody condition from wikifonia full set" \
        --out scripts/plotting/wikifonia_full_set.pdf
}

plot_nottingham_test_set(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=nottingham_melody_vs_mle_chord" \
        --variant "Online MLE (3 datasets)=nottingham_melody_vs_mle_chord_3_datasets" \
        --variant "RLPT=nottingham_melody_vs_realchords_chord" \
        --variant "GAPT=nottingham_melody_vs_gapt_chord" \
        --variant "GAPT-M=nottingham_melody_vs_gapt_multiscale_chord" \
        --variant "GT=nottingham_gt" \
        --title "Melody condition from nottingham test set" \
        --out scripts/plotting/nottingham_test_set.pdf
}

plot_pop909_test_set(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=pop909_melody_vs_mle_chord" \
        --variant "Online MLE (3 datasets)=pop909_melody_vs_mle_chord_3_datasets" \
        --variant "RLPT=pop909_melody_vs_realchords_chord" \
        --variant "GAPT=pop909_melody_vs_gapt_chord" \
        --variant "GAPT-M=pop909_melody_vs_gapt_multiscale_chord" \
        --variant "GT=pop909_gt" \
        --title "Melody condition from pop909 test set" \
        --out scripts/plotting/pop909_test_set.pdf
}

plot_model_vs_model_free_generation(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "RLPT_vs_MLE=realchords_melody_vs_mle_chord_free_generation" \
        --variant "RLPT_vs_RLPT=realchords_melody_vs_realchords_chord_free_generation" \
        --variant "RLPT_vs_GAPT=realchords_melody_vs_gapt_chord_free_generation" \
        --variant "GAPT_vs_MLE=gapt_melody_vs_mle_chord_free_generation" \
        --variant "GAPT_vs_RLPT=gapt_melody_vs_realchords_chord_free_generation" \
        --variant "GAPT_vs_GAPT=gapt_melody_vs_gapt_chord_free_generation" \
        --variant "MLE_vs_MLE=mle_melody_vs_mle_chord_free_generation" \
        --variant "MLE_vs_RLPT=mle_melody_vs_realchords_chord_free_generation" \
        --variant "MLE_vs_GAPT=mle_melody_vs_gapt_chord_free_generation" \
        --title "Melody model vs chord model unconditioned" \
        --out scripts/plotting/melody_model_vs_chord_model_unconditioned.pdf
}

plot_model_vs_model_hooktheory_prompt(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "RLPT_vs_MLE=realchords_melody_vs_mle_chord_with_prompt" \
        --variant "RLPT_vs_RLPT=realchords_melody_vs_realchords_chord_with_prompt" \
        --variant "RLPT_vs_GAPT=realchords_melody_vs_gapt_chord_with_prompt" \
        --variant "GAPT_vs_MLE=gapt_melody_vs_mle_chord_with_prompt" \
        --variant "GAPT_vs_RLPT=gapt_melody_vs_realchords_chord_with_prompt" \
        --variant "GAPT_vs_GAPT=gapt_melody_vs_gapt_chord_with_prompt" \
        --variant "MLE_vs_MLE=mle_melody_vs_mle_chord_with_prompt" \
        --variant "MLE_vs_RLPT=mle_melody_vs_realchords_chord_with_prompt" \
        --variant "MLE_vs_GAPT=mle_melody_vs_gapt_chord_with_prompt" \
        --variant "GT=hooktheory_gt" \
        --title "Melody model vs chord model with prompt" \
        --out scripts/plotting/melody_model_vs_chord_model_with_prompt.pdf
}

plot_gapt_melody_vs_chord_models_unconditioned(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "MLE=gapt_melody_vs_mle_chord_free_generation" \
        --variant "RLPT=gapt_melody_vs_realchords_chord_free_generation" \
        --variant "GAPT=gapt_melody_vs_gapt_chord_free_generation" \
        --variant "GAPT-M=gapt_melody_vs_gapt_multiscale_chord_free_generation" \
        --title "GAPT melody vs chord models unconditioned" \
        --out scripts/plotting/gapt_melody_vs_chord_models_unconditioned.pdf
}

plot_gapt_melody_vs_chord_models_with_prompt(){
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "MLE=gapt_melody_vs_mle_chord_with_prompt" \
        --variant "RLPT=gapt_melody_vs_realchords_chord_with_prompt" \
        --variant "GAPT=gapt_melody_vs_gapt_chord_with_prompt" \
        --variant "GAPT-M=gapt_melody_vs_gapt_multiscale_chord_with_prompt" \
        --variant "GT=hooktheory_gt" \
        --title "GAPT melody vs chord models with prompt" \
        --out scripts/plotting/gapt_melody_vs_chord_models_with_prompt.pdf
}