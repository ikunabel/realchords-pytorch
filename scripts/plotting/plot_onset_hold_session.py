"""Chord onset / hold / silence probability over one song, frame by frame (teacher-forced).

For one Hooktheory test song, the real melody and real chords are fed as history (teacher forcing, as in
journal/experiments/ONSET_TIMING.md) and at every frame the model's next-chord distribution is split into
P(chord onset), P(chord hold) and P(silence). Plotted as stacked areas per model, with beats, bar lines and the
song's real chord changes (labelled with the chord). The per-frame values are logged as JSON.

    python scripts/plotting/plot_onset_hold_session.py [--song_index N] [--min_frames 128]
           [--models "GAPT 7 datasets" "Online MLE"] [--out_dir logs/eval/onset_hold_session]
"""
import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval")]
import analyze_chord_onset_timing as A  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--song_index", type=int, default=None, help="test-split index; default: first with >= min_frames")
ap.add_argument("--min_frames", type=int, default=128)
ap.add_argument("--models", nargs="+", default=["GAPT 7 datasets", "Online MLE"])
ap.add_argument("--out_dir", default=str(REPO / "logs/eval/onset_hold_session"))
args = ap.parse_args()

agent = A.agent_interface.Agent(compile=False, voicings_path=None)
largs = types.SimpleNamespace(dataset_name="hooktheory", dataset_split="test", max_len=512, batch_size=1,
                              frames_per_bar=16, only_4_4=True, common_vocab=False,
                              chord_names_path=str(Path(A.REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))


def from_start(self, melody, chord, idx=0):  # no random crop: the song from its first frame
    L = self.max_len_per_part
    return melody[:L], chord[:L]


runs, song = [], None
for label in args.models:
    model, tk = agent.models[label], agent.tokenizer_for(label)
    ds = A.build_loader(largs, agent, label).dataset
    ds.random_crop = types.MethodType(from_start, ds)
    if song is None:  # pick the song once, by the first model's dataset
        idxs = [args.song_index] if args.song_index is not None else range(len(ds))
        for i in idxs:
            seq = ds[i]["targets"]
            chords = seq[1::2]
            n = int(((chords != tk.pad_token) & (chords != tk.eos_token)).sum())
            if n >= args.min_frames or args.song_index is not None:
                song = i
                break
    item = ds[song]
    seq = item["targets"].unsqueeze(0)
    d = A.chord_frame_distributions(model, seq, tk)
    chords = seq[0, 1::2][: d["p_onset"].size(1)]
    valid = (chords != tk.pad_token) & (chords != tk.eos_token)
    F = int(valid.sum())
    lo, hi = tk.chord_on_token_range
    onsets = [(k, tk.id_to_name[int(chords[k])].replace("CHORD_ON_", ""))
              for k in range(F) if lo <= int(chords[k]) <= hi]
    runs.append({"model": label,
                 "p_onset": d["p_onset"][0, :F].tolist(), "p_hold": d["p_hold"][0, :F].tolist(),
                 "p_silence": d["p_silence"][0, :F].tolist(), "real_onsets": onsets,
                 "song_url": item.get("song_url")})
    print(f"{label}: song {song} ({item.get('song_url')}), {F} frames, {len(onsets)} real chord changes", flush=True)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({"song_index": song, "runs": runs}, open(out / f"onset_hold_song{song}_{stamp}.json", "w"))

fig, axes = plt.subplots(len(runs), 1, figsize=(16, 3.2 * len(runs)), sharex=True)
axes = [axes] if len(runs) == 1 else axes
for ax, r in zip(axes, runs):
    x = range(len(r["p_onset"]))
    ax.stackplot(x, r["p_onset"], r["p_hold"], r["p_silence"],
                 labels=["chord onset", "chord hold", "silence"],
                 colors=["#d62728", "#1f77b4", "#bbbbbb"], step="post", alpha=0.85)
    for k in range(0, len(r["p_onset"]) + 1, 4):
        ax.axvline(k, color="k", lw=1.2 if k % 16 == 0 else 0.3, alpha=0.6 if k % 16 == 0 else 0.25)
    for k, name in r["real_onsets"]:
        ax.plot([k, k], [1.0, 1.06], color="k", lw=1.5, clip_on=False)
        ax.text(k, 1.08, name, fontsize=7, rotation=60, ha="left", va="bottom")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(r["p_onset"]))
    ax.set_ylabel(f"{r['model']}\nprobability")
axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, -0.02), ncol=3, fontsize=8)
axes[-1].set_xlabel("frame (16th note); thick lines = bar lines, thin = beats; ticks on top = real chord changes")
fig.suptitle(f"Next-chord token type per frame, teacher-forced, Hooktheory test song {song}", y=1.02)
fig.tight_layout()
png = out / f"onset_hold_song{song}_{stamp}.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", png)
