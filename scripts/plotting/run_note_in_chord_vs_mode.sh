#!/usr/bin/env zsh
# Plot note-in-chord vs strict note-in-mode from eval summary.

plot_note_in_chord_vs_mode_dataset_melody() {
  python scripts/plotting/note_in_chord_vs_mode.py \
    --preset dataset_melody \
    --summary logs/eval/summary.json \
    --group gt \
    --group melody_vs_mle \
    --group melody_vs_realchords \
    --group melody_vs_gapt \
    --title "GT melody + generated chord: note-in-chord vs note-in-mode" \
    --out scripts/plotting/note_in_chord_vs_mode_dataset_melody.pdf
}

plot_note_in_chord_vs_mode_free_generation() {
  python scripts/plotting/note_in_chord_vs_mode.py \
    --preset free_generation \
    --summary logs/eval/summary.json \
    --title "Free generation: note-in-chord vs note-in-mode" \
    --out scripts/plotting/note_in_chord_vs_mode_free_generation.pdf
}
