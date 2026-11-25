"""Chord onset / hold / silence probability over songs when the model plays its own chords (melody forced).

Like plot_onset_hold_session.py, but free-running on the chord side: the song's melody is fed frame by frame, and
the chords in the history are the model's own samples (a simulated jam on a fixed melody). At every frame the
model's next-chord distribution is split into P(onset), P(hold) and P(silence); then a chord token is drawn at
temperature T (ReaLJam's default 0.8) among the onsets, silence and the hold of the chord sounding (holds of other
chords are not valid continuations and are dropped before drawing). Plotted per song and model, with beats, bar
lines, the song's real chord changes (black, top) and the model's own changes (red, bottom). Values logged as JSON.

    python scripts/plotting/plot_onset_hold_free_running.py [--n_songs 4] [--min_frames 128]
           [--models "GAPT 7 datasets" "Online MLE"] [--temperature 0.8] [--seed 0]
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
import torch.nn.functional as F  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval")]
import analyze_chord_onset_timing as A  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n_songs", type=int, default=4)
ap.add_argument("--song_indices", type=int, nargs="*", default=None)
ap.add_argument("--min_frames", type=int, default=128)
ap.add_argument("--models", nargs="+", default=["GAPT 7 datasets", "Online MLE"])
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_cache", action="store_true", help="recompute the whole prefix every frame (slow; for checking)")
ap.add_argument("--prime_bars", type=int, default=0,
                help="feed the song's real chords for the first N bars (16 frames each), then sample")
ap.add_argument("--out_dir", default=str(REPO / "logs/eval/onset_hold_session"))
args = ap.parse_args()

agent = A.agent_interface.Agent(compile=False, voicings_path=None)
largs = types.SimpleNamespace(dataset_name="hooktheory", dataset_split="test", max_len=512, batch_size=1,
                              frames_per_bar=16, only_4_4=True, common_vocab=False,
                              chord_names_path=str(Path(A.REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))


def from_start(self, melody, chord, idx=0):
    L = self.max_len_per_part
    return melody[:L], chord[:L]


def free_run(model, tk, real_seq, n_frames, gen):
    """Melody from real_seq, chords sampled. Returns per-frame masses and the sampled chord ids."""
    on_lo, on_hi = tk.chord_on_token_range
    hold_lo, hold_hi = tk.chord_hold_token_range
    shift = on_lo - hold_lo
    seq = [int(real_seq[0])]  # BOS
    p_on, p_hold, p_sil, chords = [], [], [], []
    current = None  # onset id of the chord sounding
    # KV cache: x-transformers treats only the last token of a cached call as new, so the cache is fed one token
    # at a time: the chord drawn (cache update only), then the melody note (whose logits give the next chord).
    cache = None
    with torch.no_grad():
        logits, cache = model.net(torch.tensor([seq]), return_intermediates=True)  # BOS
        for k in range(n_frames):
            probs = F.softmax(logits[0, -1].float(), dim=-1)
            p_on.append(float(probs[on_lo:on_hi + 1].sum()))
            p_hold.append(float(probs[hold_lo:hold_hi + 1].sum()))
            p_sil.append(float(probs[tk.silence_token]))
            allowed = torch.zeros_like(probs, dtype=torch.bool)
            allowed[on_lo:on_hi + 1] = True
            allowed[tk.silence_token] = True
            if current is not None:
                allowed[current - shift] = True
            lg = logits[0, -1].float() / args.temperature
            lg[~allowed] = float("-inf")
            tok = int(torch.multinomial(F.softmax(lg, dim=-1), 1, generator=gen))
            if k < 16 * args.prime_bars:  # priming: the song's own chord instead of the draw
                tok = int(real_seq[1 + 2 * k])
            if on_lo <= tok <= on_hi:
                current = tok
            elif tok == tk.silence_token:
                current = None
            chords.append(tok)
            seq += [tok, int(real_seq[2 + 2 * k])]  # the chord drawn, then the song's melody token
            if args.no_cache:
                logits, _ = model.net(torch.tensor([seq]), return_intermediates=True)
            else:
                _, cache = model.net(torch.tensor([seq[:-1]]), return_intermediates=True, cache=cache)
                logits, cache = model.net(torch.tensor([seq]), return_intermediates=True, cache=cache)
    return p_on, p_hold, p_sil, chords


runs, songs = [], None
for label in args.models:
    model, tk = agent.models[label], agent.tokenizer_for(label)
    ds = A.build_loader(largs, agent, label).dataset
    ds.random_crop = types.MethodType(from_start, ds)
    on_lo, on_hi = tk.chord_on_token_range
    if songs is None:
        if args.song_indices:
            songs = args.song_indices
        else:
            songs = []
            for i in range(len(ds)):
                c = ds[i]["targets"][1::2]
                if int(((c != tk.pad_token) & (c != tk.eos_token)).sum()) >= args.min_frames:
                    songs.append(i)
                if len(songs) == args.n_songs:
                    break
    for s in songs:
        item = ds[s]
        real = item["targets"]
        rc = real[1::2]
        n = int(((rc != tk.pad_token) & (rc != tk.eos_token)).sum())
        gen = torch.Generator().manual_seed(args.seed)
        p_on, p_hold, p_sil, chords = free_run(model, tk, real, n, gen)
        name = lambda t: tk.id_to_name[int(t)].replace("CHORD_ON_", "")
        runs.append({"model": label, "song": s, "song_url": item.get("song_url"),
                     "p_onset": p_on, "p_hold": p_hold, "p_silence": p_sil,
                     "real_onsets": [(k, name(rc[k])) for k in range(n) if on_lo <= int(rc[k]) <= on_hi],
                     "model_onsets": [(k, name(t)) for k, t in enumerate(chords) if on_lo <= t <= on_hi]})
        r = runs[-1]
        print(f"{label}: song {s} ({r['song_url']}), {n} frames; real changes {len(r['real_onsets'])}, "
              f"model changes {len(r['model_onsets'])}", flush=True)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tag = f"_prime{args.prime_bars}" if args.prime_bars else ""
json.dump({"temperature": args.temperature, "seed": args.seed, "prime_bars": args.prime_bars, "runs": runs},
          open(out / f"free_running{tag}_{stamp}.json", "w"))

for s in songs:
    rs = [r for r in runs if r["song"] == s]
    fig, axes = plt.subplots(len(rs), 1, figsize=(16, 3.6 * len(rs)), sharex=True)
    axes = [axes] if len(rs) == 1 else axes
    for ax, r in zip(axes, rs):
        x = range(len(r["p_onset"]))
        ax.stackplot(x, r["p_onset"], r["p_hold"], r["p_silence"], labels=["chord onset", "chord hold", "silence"],
                     colors=["#d62728", "#1f77b4", "#bbbbbb"], step="post", alpha=0.85)
        for k in range(0, len(r["p_onset"]) + 1, 4):
            ax.axvline(k, color="k", lw=1.2 if k % 16 == 0 else 0.3, alpha=0.6 if k % 16 == 0 else 0.25)
        for k, nm in r["real_onsets"]:
            ax.plot([k, k], [1.0, 1.06], color="k", lw=1.5, clip_on=False)
            ax.text(k, 1.08, nm, fontsize=7, rotation=60, ha="left", va="bottom")
        for k, nm in r["model_onsets"]:
            ax.plot([k, k], [0.0, -0.06], color="#d62728", lw=1.5, clip_on=False)
            ax.text(k, -0.08, nm, fontsize=7, rotation=-60, ha="left", va="top", color="#d62728")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(r["p_onset"]))
        ax.set_ylabel(f"{r['model']}\nprobability")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.32), ncol=3, fontsize=8)
    axes[-1].set_xlabel("frame (16th note); bar lines thick, beats thin; black ticks top = real song changes, "
                        "red ticks bottom = the model's own changes", labelpad=40)
    fig.suptitle(f"Free-running chords on the song's melody (T {args.temperature}"
                 f"{', primed with the real chords for ' + str(args.prime_bars) + ' bar(s)' if args.prime_bars else ''}), "
                 f"Hooktheory test song {s}: "
                 f"{rs[0]['song_url'].rsplit('/', 2)[-2]} / {rs[0]['song_url'].rsplit('/', 1)[-1]}", y=1.04)
    fig.tight_layout()
    png = out / f"free_running{tag}_song{s}_{stamp}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    print("wrote", png)
