"""Free generation: the online model writes melody and chords together, from nothing, in one pass.

Samples the interleaved sequence (BOS, c1, m1, c2, m2, ...) from BOS with the live models (ReaLJam's
checkpoints, loaded by agent_interface.Agent), with the live server's validity filter for generating both
lanes (filter_invalid_tokens_generate_unconditional), for the model's full context (256 frames = 16 bars of
4/4). Unlike ReaLJam's live loop, nothing is resampled or spliced: every token conditions on everything
generated before it. Writes a MIDI file per song (tokenizer's decode_to_midi: melody, chords and a bass note an
octave below) and, with fluidsynth, a WAV.

    python scripts/generate_free_songs.py [--models GAPT "Online MLE" ReaLchords] [--songs 4] [--temperature 1.0]
"""
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
from realchords.realjam import agent_interface  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", nargs="+", default=["GAPT", "Online MLE", "ReaLchords"])
ap.add_argument("--songs", type=int, default=4, help="songs per model (seeds 0..N-1)")
ap.add_argument("--frames", type=int, default=256, help="16th-note frames (max: the model's context, 256)")
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--bpm", type=int, default=100)
ap.add_argument("--out", default="logs/generated/free_songs")
ap.add_argument("--soundfont", default="/usr/share/soundfonts/FluidR3_GM.sf2")
args = ap.parse_args()

logging.disable(logging.INFO)
agent = agent_interface.Agent(compile=False, voicings_path=None)
assert args.frames <= agent.max_frames
fluidsynth = shutil.which("fluidsynth") if Path(args.soundfont).exists() else None

for model_name in args.models:
    tk = agent.tokenizer_for(model_name)
    out_dir = Path(args.out) / model_name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.songs):
        torch.manual_seed(seed)
        prompt = torch.tensor([[tk.bos_token]], device=agent.device)
        gen = agent.gen_online_model(agent.models[model_name], prompt, 2 * args.frames, args.temperature,
                                     filter_fn=agent._filters(tk)["online"], filter_kwargs={})
        seq = gen[0].cpu().numpy()
        if seq[0] == tk.bos_token:  # generate() may or may not return the prompt
            seq = seq[1:]
        chords, melody = seq[0::2][:args.frames], seq[1::2][:args.frames]
        stem = out_dir / f"seed{seed}"
        tk.decode_to_midi(chord_frames=chords, melody_frames=melody, bpm=args.bpm).write(f"{stem}.mid")
        names = [tk.id_to_name.get(int(t), "?") for t in chords]
        changes = [n[len("CHORD_ON_"):] for n in names if n.startswith("CHORD_ON_")]
        n_notes = sum(tk.id_to_name.get(int(t), "").startswith("NOTE_ON_") for t in melody)
        silent = sum(n == "SILENCE" for n in names)
        if fluidsynth:
            subprocess.run([fluidsynth, "-ni", "-q", "-F", f"{stem}.wav", "-r", "44100", args.soundfont, f"{stem}.mid"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{model_name} seed {seed}: {n_notes} melody notes, {len(changes)} chord onsets, "
              f"{silent} silent chord frames: {' '.join(changes[:24])}", flush=True)
print(f"written to {args.out}/")
