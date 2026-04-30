#!/usr/bin/env zsh

plot_test_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=hooktheory_melody_vs_decoder_only_online_chord" \
        --variant "Online MLE (3 datasets)=hooktheory_melody_vs_decoder_only_online_chord_3_datasets" \
        --variant "ReaLchords=hooktheory_melody_vs_realchords" \
        --variant "GAPT=hooktheory_melody_vs_gapt" \
        --out scripts/plotting/pareto_test_set.pdf
}

plot_held_out_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=wikifonia_melody_vs_decoder_only_online_chord" \
        --variant "Online MLE (3 datasets)=wikifonia_melody_vs_decoder_only_online_chord_3_datasets" \
        --variant "ReaLchords=wikifonia_melody_vs_realchords" \
        --variant "GAPT=wikifonia_melody_vs_gapt" \
        --out scripts/plotting/pareto_held_out_set.pdf
}

plot_test_set_paper_authentic() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=hooktheory_melody_vs_decoder_only_online_chord" \
        --variant "Online MLE (3 datasets)=hooktheory_melody_vs_decoder_only_online_chord_3_datasets" \
        --variant "ReaLchords=hooktheory_melody_vs_realchords" \
        --variant "GAPT=hooktheory_melody_vs_gapt_paper_authentic" \
        --out scripts/plotting/pareto_test_set_paper_authentic.pdf
}

plot_held_out_set_paper_authentic() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant "Online MLE=wikifonia_melody_vs_decoder_only_online_chord" \
        --variant "Online MLE (3 datasets)=wikifonia_melody_vs_decoder_only_online_chord_3_datasets" \
        --variant "ReaLchords=wikifonia_melody_vs_realchords" \
        --variant "GAPT=wikifonia_melody_vs_gapt_paper_authentic" \
        --out scripts/plotting/pareto_held_out_set_paper_authentic.pdf
}