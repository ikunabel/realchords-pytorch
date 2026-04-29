#!/usr/bin/env zsh

plot_test_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant hooktheory_melody_vs_decoder_only_online_chord \
        --variant hooktheory_melody_vs_decoder_only_online_chord_3_datasets \
        --variant hooktheory_melody_vs_realchords \
        --variant hooktheory_melody_vs_gapt \
        --out scripts/plotting/pareto_test_set.pdf
}

plot_held_out_set() {
    python scripts/plotting/pareto.py \
        --summary logs/eval/summary.json \
        --variant wikifonia_melody_vs_decoder_only_online_chord \
        --variant wikifonia_melody_vs_decoder_only_online_chord_3_datasets \
        --variant wikifonia_melody_vs_realchords \
        --variant wikifonia_melody_vs_gapt \
        --out scripts/plotting/pareto_held_out_set.pdf
}

