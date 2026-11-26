"""Reproduce ReaLchords Fig. 4 (paper §5.4): accompaniment quality over beats in three scenarios.

Note-in-chord per beat, songs' beats 1-32, Hooktheory test melodies, chords free-running (the model's own samples
in the history, the song's melody fed frame by frame), three scenarios as in the paper:

  primed      the song's real chords for the first 8 beats (32 frames), then the model's own
  cold        the model's own chords from frame 0
  perturbed   cold, and the melody transposed up a tritone (+6) from beat 17 (frame 64) to the end

plus the test set (real chords with the real, unperturbed melody) as reference. Design (2026-11-25, with the user):
the four local models, the paper's perturbation, 200 songs x 3 seeds, temperature 1.0.

Songs: Hooktheory test, 4/4 throughout, >= 128 frames, all chords in both vocabularies (so the old- and
new-vocabulary models see the same songs), 200 drawn with --song_seed; segment = the song's first 128 frames.
A song whose transposed melody would leave the tokenizer's pitch range is transposed down a tritone instead
(same pitch classes; counted in the output). Sampling: at every frame the model's next-chord distribution at
temperature T, restricted to valid continuations (any onset, silence, the hold of the chord sounding), as
scripts/plotting/plot_onset_hold_free_running.py; songs run batched through the KV cache.

Note-in-chord: realchords.utils.eval_utils (_note_in_chord_pair): a frame counts if it has a melody note and a
chord; per run and beat the mean over its counted frames; per beat the mean over runs, with a 95% interval
(1.96 x standard error over runs).

    python scripts/eval/fig4_recovery.py [--models ...] [--n_songs 200] [--seeds 0 1 2] [--temperature 1.0]
    python scripts/eval/fig4_recovery.py --plot_only logs/eval/fig4_recovery/<run>.json
"""
import argparse
import json
import random
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1].parent
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "eval")]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", nargs="+", default=["Online MLE", "ReaLchords", "GAPT", "GAPT 7 datasets"])
ap.add_argument("--scenarios", nargs="+", default=["primed", "cold", "perturbed"])
ap.add_argument("--n_songs", type=int, default=200)
ap.add_argument("--song_seed", type=int, default=0)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--frames", type=int, default=128, help="32 beats")
ap.add_argument("--prime_frames", type=int, default=32, help="8 beats, as Fig. 4a")
ap.add_argument("--perturb_frame", type=int, default=64, help="beat 17, as Fig. 4c")
ap.add_argument("--perturb_semitones", type=int, default=6)
ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--out_dir", default=str(REPO / "logs/eval/fig4_recovery"))
ap.add_argument("--plot_only", default=None)
args = ap.parse_args()

FPB = 4


def per_beat(matches):
    """[runs, frames] of 1 / 0 / nan -> [runs, beats] mean over counted frames (nan if none)."""
    m = np.asarray(matches, dtype=float).reshape(len(matches), -1, FPB)
    with np.errstate(invalid="ignore"):
        return np.nanmean(m, axis=2)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = {"primed": f"(a) primed with real chords for {res['prime_frames'] // FPB} beats",
              "cold": "(b) cold start",
              "perturbed": f"(c) melody +{res['perturb_semitones']} from beat {res['perturb_frame'] // FPB + 1}"}
    scen = [s for s in ("primed", "cold", "perturbed") if s in res["scenarios"]]
    fig, axes = plt.subplots(1, len(scen), figsize=(6 * len(scen), 4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(1, res["frames"] // FPB + 1)
    with np.errstate(invalid="ignore"):
        def band(ax, mat, label, **kw):
            b = per_beat(mat)
            mean = np.nanmean(b, axis=0)
            se = np.nanstd(b, axis=0) / np.sqrt(np.sum(~np.isnan(b), axis=0))
            line, = ax.plot(x, mean, marker=".", label=label, **kw)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=line.get_color(), alpha=0.15)
        for ax, s in zip(axes, scen):
            band(ax, res["test_set"], "Test set", color="#6b8fd6")
            for m in res["models"]:
                band(ax, res["runs"][m][s], m)
            if s == "primed" and "cold" in res["scenarios"]:
                b = per_beat(res["runs"]["Online MLE"]["cold"]) if "Online MLE" in res["models"] else None
                if b is not None:
                    ax.plot(x, np.nanmean(b, axis=0), "--", color="grey", label="Online MLE cold")
            ax.set_title(titles[s])
            ax.set_xlabel("beat")
            ax.set_xticks(range(4, x[-1] + 1, 4))
            ax.grid(alpha=0.3)
    axes[0].set_ylabel("note-in-chord")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"ReaLchords Fig. 4 reproduction: {res['n_songs']} Hooktheory test songs x {len(res['seeds'])} seeds, "
                 f"T {res['temperature']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


if args.plot_only:
    res = json.load(open(args.plot_only))
    plot(res, Path(args.plot_only).with_suffix(".png"))
    raise SystemExit

import analyze_chord_onset_timing as A  # noqa: E402
from realchords.utils.eval_utils import _note_in_chord_pair  # noqa: E402

agent = A.agent_interface.Agent(compile=False, voicings_path=None)
largs = types.SimpleNamespace(dataset_name="hooktheory", dataset_split="test", max_len=512, batch_size=1,
                              frames_per_bar=16, only_4_4=True, common_vocab=True,
                              chord_names_path=str(Path(A.REALJAM_CHECKPOINT_DIR) / "chord_names_augmented.json"))


def from_start(self, melody, chord, idx=0):
    L = self.max_len_per_part
    return melody[:L], chord[:L]


def songs_for(label):
    """{song_url: interleaved targets (BOS, c0, m0, ...)} with >= args.frames frames, in the model's vocabulary."""
    tk = agent.tokenizer_for(label)
    ds = A.build_loader(largs, agent, label).dataset
    ds.random_crop = types.MethodType(from_start, ds)
    out = {}
    for i in range(len(ds)):
        item = ds[i]
        t = item["targets"]
        c = t[1::2]
        if int(((c != tk.pad_token) & (c != tk.eos_token)).sum()) >= args.frames:
            out[item["song_url"]] = t[:1 + 2 * args.frames].clone()
    return out


def transpose_melody(t, tk, frm, semis):
    """Melody tokens from frame `frm` on, +semis (or -semis if a pitch would leave the vocabulary)."""
    out = t.clone()
    idx = [2 + 2 * k for k in range(frm, args.frames)]
    names = [tk.id_to_name[int(out[j])] for j in idx]
    for shift in (semis, -semis):
        new = []
        for n in names:
            if n.startswith("NOTE_ON_") or n.startswith("NOTE_"):
                pre = "NOTE_ON_" if n.startswith("NOTE_ON_") else "NOTE_"
                nm = f"{pre}{int(n[len(pre):]) + shift}"
                if nm not in tk.name_to_id:
                    break
                new.append(tk.name_to_id[nm])
            else:
                new.append(tk.name_to_id[n])
        else:
            for j, v in zip(idx, new):
                out[j] = v
            return out, shift
    raise ValueError("melody cannot be transposed either way")


def nic(chords, melody, tk):
    """Per-frame note-in-chord: 1 / 0 / nan (no melody note or no chord)."""
    res = []
    for c, m in zip(chords, melody):
        valid, ok = _note_in_chord_pair(int(m), int(c), tk)
        res.append(float(ok) if valid else float("nan"))
    return res


def free_run_batch(model, tk, real, prime, gen):
    """real: [B, 1 + 2 * frames] interleaved; chords sampled (the first `prime` frames forced to the real ones).
    Returns [B, frames] chord ids."""
    on_lo, on_hi = tk.chord_on_token_range
    hold_lo = tk.chord_hold_token_range[0]
    shift = on_lo - hold_lo
    B = real.shape[0]
    seq = real[:, :1].clone()  # BOS
    current = torch.full((B,), -1, dtype=torch.long)
    chords = torch.zeros((B, args.frames), dtype=torch.long)
    with torch.no_grad():
        logits, cache = model.net(seq, return_intermediates=True)
        for k in range(args.frames):
            lg = logits[:, -1].float() / args.temperature
            allowed = torch.zeros_like(lg, dtype=torch.bool)
            allowed[:, on_lo:on_hi + 1] = True
            allowed[:, tk.silence_token] = True
            rows = (current >= 0).nonzero().flatten()
            allowed[rows, current[rows] - shift] = True
            lg[~allowed] = float("-inf")
            tok = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=gen).flatten()
            if k < prime:
                tok = real[:, 1 + 2 * k].clone()
            is_on = (tok >= on_lo) & (tok <= on_hi)
            current = torch.where(is_on, tok, torch.where(tok == tk.silence_token, torch.full_like(tok, -1), current))
            chords[:, k] = tok
            seq = torch.cat([seq, tok[:, None], real[:, 2 + 2 * k:3 + 2 * k]], dim=1)
            _, cache = model.net(seq[:, :-1], return_intermediates=True, cache=cache)
            logits, cache = model.net(seq, return_intermediates=True, cache=cache)
    return chords


per_model_songs = {m: songs_for(m) for m in args.models}
common = sorted(set.intersection(*(set(s) for s in per_model_songs.values())))
random.Random(args.song_seed).shuffle(common)
urls = sorted(common[:args.n_songs])
print(f"{len(common)} eligible songs in all models' datasets; using {len(urls)}", flush=True)

ref_label = args.models[0]
ref_tk = agent.tokenizer_for(ref_label)
test_set = [nic(per_model_songs[ref_label][u][1::2], per_model_songs[ref_label][u][2::2], ref_tk) for u in urls]

runs, flips = {}, {}
for label in args.models:
    model, tk = agent.models[label], agent.tokenizer_for(label)
    model.eval()
    runs[label] = {}
    for scen in args.scenarios:
        real, n_down = [], 0
        for u in urls:
            t = per_model_songs[label][u]
            if scen == "perturbed":
                t, used = transpose_melody(t, tk, args.perturb_frame, args.perturb_semitones)
                n_down += used < 0
            real.append(t)
        real = torch.stack(real)
        flips[f"{label}/{scen}"] = n_down
        prime = args.prime_frames if scen == "primed" else 0
        mats = []
        for seed in args.seeds:
            gen = torch.Generator().manual_seed(seed)
            for b in range(0, len(urls), args.batch):
                chunk = real[b:b + args.batch]
                ch = free_run_batch(model, tk, chunk, prime, gen)
                mats += [nic(ch[i], chunk[i, 2::2], tk) for i in range(len(chunk))]
            print(f"{label} {scen} seed {seed}: done", flush=True)
        runs[label][scen] = mats
        b = per_beat(mats)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(b, axis=0)
        print(f"{label} {scen}: note-in-chord beats 1-4 {np.nanmean(m[:4]):.3f}, 13-16 {np.nanmean(m[12:16]):.3f}, "
              f"17-20 {np.nanmean(m[16:20]):.3f}, 29-32 {np.nanmean(m[28:32]):.3f}"
              + (f" ({n_down} songs transposed down)" if scen == "perturbed" else ""), flush=True)

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
res = {k: v for k, v in vars(args).items() if k != "plot_only"}
res.update({"songs": urls, "test_set": test_set, "runs": runs, "transposed_down": flips})
path = out / f"fig4_{stamp}.json"
json.dump(res, open(path, "w"))
print("wrote", path)
plot(res, path.with_suffix(".png"))
