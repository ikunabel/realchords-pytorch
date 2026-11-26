"""Behavioural probe of the frozen reward models: do they reward frequent, on-beat chord changes that follow the melody?

Hypothesis (2026-11-26, journal/reference/RESEARCH_QUESTIONS.md, H5): the coherence reward models do not penalise,
and may reward, GAPT's harmonic rhythm (too many changes, on the beat). Test: edit real songs and score original and
edited chords with the reward models RL trains against; every comparison is paired per song.

Models (--family): contrastive, discriminative, rhythm-only contrastive, rhythm-only discriminative, 2 seeds each.
  hooktheory  the eight models the released ReaLchords was trained with (logs/huggingface/realchords/args.yml),
              vocabulary v1
  7sets       the eight models gapt_7sets was trained with (configs/rl/gapt_7sets.yml, cluster paths), vocabulary v2
Score = exactly what RL receives (contrastive: cosine similarity of chord and melody embeddings; discriminative:
P(real)). Vocabularies: journal/reference/TOKENIZERS.md, section 0.

Songs: Hooktheory test, 4/4 throughout, >= --frames frames, the first --frames frames (16th-note frames, 16 per bar),
--n_songs drawn with --song_seed, tokenised in the family's vocabulary.

Sanity check, per model: how often the real pair scores above (a) the same chords with the next song's melody and
(b) the chords delayed by 2 frames (an eighth note; the rhythm models' training negatives). Near 0.5 on (a) = no
signal (the released rhythm-discriminative models, 2026-11-27).

Part A, whole-song edits at candidate frames. A candidate is a frame on a beat (not frame 0) where the chord lane
holds a chord, the melody has a note onset that is not a chord tone, and the song has another chord that contains
the note (the dose response uses all of these); the single-insert variants also need a song chord that does not. Design (with the user, 2026-11-27): the inserted chord is one of the song's
own chords (keeps key and vocabulary), the most frequent qualifying one. An insert lasts until the next original
chord change (+1 change).
  follow          insert at the candidate a song chord containing the melody note
  follow_offbeat  the same chord one 16th later
  mismatch        insert a song chord not containing the note
  restrike        re-attack the sounding chord (+1 onset, harmony unchanged)
  merge           remove an original change: the previous chord is held over it (-1 change); its own frames
Dose response: follow with k = 1, 2, 4, 8 inserts (songs with >= k candidates).

Part B, position profile: one extra change at each of the 16 positions of a bar (one random bar per song and
position, among bars where that frame holds a chord), as a re-strike and as a change to the song chord containing
the sounding melody note (if any). Mean score change per position.

    python scripts/eval/reward_probe.py [--family hooktheory|7sets] [--n_songs 200] [--device cuda]
"""
import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import argbind
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from realchords.dataset.hooktheory_dataloader import HooktheoryDataset  # noqa: E402
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer  # noqa: E402
from realchords.lit_module.contrastive_reward import LitContrastiveReward  # noqa: E402
from realchords.lit_module.contrastive_reward_rhythm import LitContrastiveRewardRhythm  # noqa: E402
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward  # noqa: E402
from realchords.lit_module.discriminative_reward_rhythm import LitDiscriminativeRewardRhythm  # noqa: E402
from realchords.rl.reward.model_based_rewards import (  # noqa: E402
    ContrastiveRewardFn, ContrastiveRewardRhythmFn, DiscriminativeRewardFn, DiscriminativeRewardRhythmFn)
from realchords.utils.eval_utils import _note_in_chord_pair  # noqa: E402
from realchords.utils.experiment_utils import DATASET_CACHE_DIRS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n_songs", type=int, default=200)
ap.add_argument("--song_seed", type=int, default=0)
ap.add_argument("--frames", type=int, default=128)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--parts", nargs="+", default=["A", "B"])
ap.add_argument("--b_songs", type=int, default=100, help="part B on the first n songs (cost)")
ap.add_argument("--family", choices=["hooktheory", "7sets"], default="hooktheory")
ap.add_argument("--kinds", nargs="+", default=["contrastive", "discriminative", "contrastive_rhythm",
                                                "discriminative_rhythm"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out_dir", default=str(REPO / "logs/eval/reward_probe"))
args = ap.parse_args()
torch.set_num_threads(args.threads)

HF = REPO / "logs/huggingface"
if args.family == "hooktheory":
    VOCAB = REPO / "data/cache/old_chord_names_augmented.json"
    _m = {"contrastive": [("contrastive_reward", 8000), ("contrastive_reward_2", 8000)],
          "discriminative": [("discriminative_reward_128_bs", 1500), ("discriminative_reward_128_bs_2", 1500)],
          "contrastive_rhythm": [("contrastive_reward_no_augmentation_rhythm", 2000),
                                 ("contrastive_reward_no_augmentation_rhythm_2", 2000)],
          "discriminative_rhythm": [("discriminative_reward_no_augmentation_rhythm", 3000),
                                    ("discriminative_reward_no_augmentation_rhythm_2", 3000)]}
    MODELS = {k: [(run, HF / run / f"step={st}.ckpt") for run, st in v] for k, v in _m.items()}
else:
    import yaml
    VOCAB = REPO / "data/cache/chord_names_augmented.json"
    cfg = yaml.safe_load(open(REPO / "configs/rl/gapt_7sets.yml"))
    keys = {"contrastive": "contrastive_reward_model_path", "discriminative": "discriminative_reward_model_path",
            "contrastive_rhythm": "contrastive_reward_rhythm_model_path",
            "discriminative_rhythm": "discriminative_reward_rhythm_model_path"}
    MODELS = {k: [(Path(p).parent.name, Path(p)) for p in cfg[key]] for k, key in keys.items()}
MODELS = {k: v for k, v in MODELS.items() if k in args.kinds}
DEV = torch.device(args.device)
LIT = {"contrastive": LitContrastiveReward, "discriminative": LitDiscriminativeReward,
       "contrastive_rhythm": LitContrastiveRewardRhythm, "discriminative_rhythm": LitDiscriminativeRewardRhythm}
FPB, FPBAR = 4, 16

tk = HooktheoryTokenizer(chord_names=json.load(open(VOCAB)))
ON_LO, ON_HI = tk.chord_on_token_range
HOLD_LO, HOLD_HI = tk.chord_hold_token_range
SHIFT = ON_LO - HOLD_LO


# --------------------------------------------------------------------------- models

def load(kind, run, ckpt):
    a = argbind.load_args(Path(ckpt).parent / "args.yml")
    a["compile"] = False
    with argbind.scope(a):
        lit = LIT[kind]()
    state = torch.load(ckpt, weights_only=True, map_location="cpu")["state_dict"]
    lit.load_state_dict({k.replace("_orig_mod.", "").replace("._orig_mod", ""): v for k, v in state.items()})
    model = lit.model.to(DEV).eval()
    model.device = DEV
    rows = next(v.shape[0] for k, v in state.items() if k.endswith("token_emb.emb.weight"))
    if kind in ("contrastive", "discriminative"):
        assert rows == tk.num_tokens, f"{run}: {rows} embedding rows, the vocabulary has {tk.num_tokens}"
        fn = (ContrastiveRewardFn if kind == "contrastive" else DiscriminativeRewardFn)(
            model=model, pad_token_id=tk.pad_token, bos_token_id=tk.bos_token, eos_token_id=tk.eos_token,
            model_part="chord")
    else:
        fn = (ContrastiveRewardRhythmFn if kind == "contrastive_rhythm" else DiscriminativeRewardRhythmFn)(
            model=model, tokenizer=tk, model_part="chord")
    return fn


@torch.no_grad()
def score(fn, seqs):
    """seqs: [N, 1 + 2F] (BOS, c0, m0, ...) -> [N] scores."""
    out = []
    for s in range(0, len(seqs), args.batch):
        b = seqs[s:s + args.batch].to(DEV)
        samples = SimpleNamespace(sequences=b, action_mask=torch.ones_like(b[:, 1:], dtype=torch.bool))
        out.append(fn(samples)["reward"].sum(dim=1).float().cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- songs

def load_songs():
    ds = HooktheoryDataset(cache_dir=DATASET_CACHE_DIRS["hooktheory"], split="test", model_type="decoder_only",
                           model_part="chord", max_len=512, data_augmentation=False, load_augmented_chord_names=True,
                           chord_names_path=str(VOCAB))
    assert ds.tokenizer.name_to_id == tk.name_to_id
    keep = []
    for item in ds.data:
        meters = {(m.get("beats_per_bar"), m.get("beat_unit")) for m in item["annotations"].get("meters") or []}
        if meters == {(4, 4)}:
            keep.append(item)
    ds.data = keep
    ds.random_crop = lambda melody, chord, idx=0: (melody[:ds.max_len_per_part], chord[:ds.max_len_per_part])
    songs = {}
    for i in range(len(ds)):
        try:
            it = ds[i]
        except KeyError:  # a chord missing from the vocabulary
            continue
        t = it["targets"]
        c = t[1::2]
        if int(((c != tk.pad_token) & (c != tk.eos_token)).sum()) >= args.frames:
            songs[it["song_url"]] = t[:1 + 2 * args.frames].clone()
    urls = sorted(songs)
    random.Random(args.song_seed).shuffle(urls)
    urls = sorted(urls[:args.n_songs])
    print(f"{len(songs)} eligible 4/4 test songs, using {len(urls)}", flush=True)
    return urls, [songs[u] for u in urls]


def lanes(t):
    return t[1::2].clone(), t[2::2].clone()


def interleave(c, m):
    out = torch.empty(1 + 2 * len(c), dtype=torch.long)
    out[0] = tk.bos_token
    out[1::2], out[2::2] = c, m
    return out


def is_on(x):
    return ON_LO <= x <= ON_HI


def is_hold(x):
    return HOLD_LO <= x <= HOLD_HI


def chord_id(x):
    """onset or hold token -> onset token of that chord; -1 otherwise."""
    return x if is_on(x) else (x + SHIFT if is_hold(x) else -1)


def note_on(x):
    return tk.id_to_name.get(int(x), "").startswith("NOTE_ON_")


def sounding_note(x):
    n = tk.id_to_name.get(int(x), "")
    return n.startswith("NOTE_")


def fits(note_tok, chord_on_tok):
    valid, ok = _note_in_chord_pair(int(note_tok), int(chord_on_tok), tk)
    return valid and ok


def song_chords(c):
    """Song's chords (onset tokens) by frequency of onsets, most frequent first."""
    cnt = Counter(int(x) for x in c if is_on(int(x)))
    return [k for k, _ in cnt.most_common()]


def insert(c, f, new_on):
    """Chord new_on from frame f until the next original onset / silence (+1 change)."""
    c = c.clone()
    c[f] = new_on
    g = f + 1
    while g < len(c) and is_hold(int(c[g])):
        c[g] = new_on - SHIFT
        g += 1
    return c


def candidates(c, m, chords):
    out = []
    for f in range(FPB, len(c), FPB):
        cur = int(c[f])
        if not is_hold(cur) or not note_on(m[f]):
            continue
        cur_on = cur + SHIFT
        if fits(m[f], cur_on):
            continue
        fit = [x for x in chords if x != cur_on and fits(m[f], x)]
        mis = [x for x in chords if x != cur_on and not fits(m[f], x)]
        if fit and f + 1 < len(c) and is_hold(int(c[f + 1])):
            out.append((f, fit[0], mis[0] if mis else None, cur_on))
    return out


# --------------------------------------------------------------------------- variants

def part_a(urls, songs, rng):
    variants = {k: [] for k in ("original", "follow", "follow_offbeat", "mismatch", "restrike")}
    idx_a, merge_orig, merge_var, idx_m = [], [], [], []
    dose = {k: ([], [], []) for k in (1, 2, 4, 8)}  # (song index, original, variant)
    n_cand = []
    for i, t in enumerate(songs):
        c, m = lanes(t)
        chords = song_chords(c)
        cand_all = candidates(c, m, chords)  # dose response: a fitting song chord exists
        cand = [x for x in cand_all if x[2] is not None]  # part A: also a non-fitting one
        n_cand.append(len(cand_all))
        if cand:
            f, fit, mis, cur = cand[rng.randrange(len(cand))]
            variants["original"].append(t)
            variants["follow"].append(interleave(insert(c, f, fit), m))
            variants["follow_offbeat"].append(interleave(insert(c, f + 1, fit), m))
            variants["mismatch"].append(interleave(insert(c, f, mis), m))
            variants["restrike"].append(interleave(insert(c, f, cur), m))
            idx_a.append(i)
        for k in dose:
            if len(cand_all) >= k:
                cc = c
                for f, fit, _, _ in sorted(rng.sample(cand_all, k)):
                    cc = insert(cc, f, fit)
                dose[k][0].append(i)
                dose[k][1].append(t)
                dose[k][2].append(interleave(cc, m))
        onsets = [g for g in range(1, len(c)) if is_on(int(c[g])) and chord_id(int(c[g - 1])) >= 0
                  and chord_id(int(c[g - 1])) != int(c[g])]
        if onsets:
            g = onsets[rng.randrange(len(onsets))]
            prev = chord_id(int(c[g - 1]))
            cc = c.clone()
            h = g
            while h < len(cc) and (h == g or is_hold(int(cc[h]))):
                cc[h] = prev - SHIFT
                h += 1
            merge_orig.append(t)
            merge_var.append(interleave(cc, m))
            idx_m.append(i)
    return variants, idx_a, (merge_orig, merge_var, idx_m), dose, n_cand


def part_b(songs, rng):
    """For each bar position q: (originals, restrike variants, change variants, song idx)."""
    out = {q: ([], [], [], []) for q in range(FPBAR)}
    for i, t in enumerate(songs):
        c, m = lanes(t)
        chords = song_chords(c)
        nbars = len(c) // FPBAR
        for q in range(FPBAR):
            bars = [b for b in range(nbars) if b * FPBAR + q > 0 and is_hold(int(c[b * FPBAR + q]))]
            if not bars:
                continue
            f = rng.choice(bars) * FPBAR + q
            cur = int(c[f]) + SHIFT
            # melody note sounding at f (onset or hold)
            note = int(m[f]) if sounding_note(m[f]) else None
            fit = [x for x in chords if x != cur and note is not None and fits(note, x)]
            if not fit:
                continue
            out[q][0].append(t)
            out[q][1].append(interleave(insert(c, f, cur), m))
            out[q][2].append(interleave(insert(c, f, fit[0]), m))
            out[q][3].append(i)
    return out


def summarise(diff):
    d = np.asarray(diff)
    rng = np.random.default_rng(0)
    boots = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "share_up": float((d > 0).mean()), "n": int(len(d))}


def main():
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    urls, songs = load_songs()
    rng = random.Random(args.song_seed)
    res = {"args": vars(args), "songs": urls, "models": {k: [r for r, _ in v] for k, v in MODELS.items()}}

    jobs = {}  # name -> stacked tensor
    if "A" in args.parts:
        variants, idx_a, merge, dose, n_cand = part_a(urls, songs, rng)
        res["A"] = {"n_candidates_per_song": n_cand, "song_idx": idx_a, "merge_song_idx": merge[2],
                    "dose_song_idx": {k: v[0] for k, v in dose.items()}}
        print(f"Part A: {len(idx_a)} songs with a candidate frame (median {np.median(n_cand):.0f} per song); "
              f"merge {len(merge[2])}; dose k=1/2/4/8: {[len(dose[k][0]) for k in dose]}", flush=True)
        for k, v in variants.items():
            jobs[f"A/{k}"] = torch.stack(v)
        jobs["A/merge_original"], jobs["A/merge"] = torch.stack(merge[0]), torch.stack(merge[1])
        for k, (_, o, v) in dose.items():
            if o:
                jobs[f"dose/{k}/original"], jobs[f"dose/{k}/follow"] = torch.stack(o), torch.stack(v)
    if "B" in args.parts:
        pb = part_b(songs[:args.b_songs], rng)
        res["B"] = {"n_per_position": [len(pb[q][3]) for q in range(FPBAR)]}
        print(f"Part B: songs per position {res['B']['n_per_position']}", flush=True)
        for q in range(FPBAR):
            if pb[q][3]:
                jobs[f"B/{q}/original"] = torch.stack(pb[q][0])
                jobs[f"B/{q}/restrike"] = torch.stack(pb[q][1])
                jobs[f"B/{q}/change"] = torch.stack(pb[q][2])

    # sanity check: other song's melody, chords delayed by 2 frames
    def delayed(t, s=2):
        c, m = lanes(t)
        c2 = torch.cat([c[:1].repeat(s), c[:-s]])
        if is_on(int(c[0])):
            c2[1:s + 1] = int(c[0]) - SHIFT
        return interleave(c2, m)
    jobs["S/real"] = torch.stack(songs)
    jobs["S/other_melody"] = torch.stack([interleave(lanes(songs[i])[0], lanes(songs[(i + 1) % len(songs)])[1])
                                          for i in range(len(songs))])
    jobs["S/delayed"] = torch.stack([delayed(t) for t in songs])

    n_seq = sum(len(v) for v in jobs.values())
    print(f"{n_seq} sequences x {sum(len(v) for v in MODELS.values())} models", flush=True)
    scores = {}
    for kind, runs in MODELS.items():
        for run, ckpt in runs:
            fn = load(kind, run, ckpt)  # ckpt: full path
            scores[run] = {name: score(fn, x) for name, x in jobs.items()}
            print(f"  scored {run}", flush=True)
            del fn

    # paired differences, per model and per kind (mean of the two seeds' differences)
    def diffs(run, var, base):
        return scores[run][var] - scores[run][base]

    table = {}
    pairs = []
    if "A" in args.parts:
        pairs += [(f"A/{k}", "A/original") for k in ("follow", "follow_offbeat", "mismatch", "restrike")]
        pairs += [("A/merge", "A/merge_original")]
        pairs += [(f"dose/{k}/follow", f"dose/{k}/original") for k in (1, 2, 4, 8) if f"dose/{k}/original" in jobs]
    if "B" in args.parts:
        pairs += [(f"B/{q}/{v}", f"B/{q}/original") for q in range(FPBAR) for v in ("restrike", "change")
                  if f"B/{q}/original" in jobs]
    for var, base in pairs:
        table[var] = {}
        for kind, runs in MODELS.items():
            per_run = [diffs(r, var, base) for r, _ in runs]
            table[var][kind] = summarise(np.mean(per_run, axis=0))
            for r, _ in runs:
                table[var][r] = summarise(diffs(r, var, base))
    res["sanity"] = {r: {"real_above_other_melody": float(np.mean(scores[r]["S/real"] > scores[r]["S/other_melody"])),
                         "real_above_delayed": float(np.mean(scores[r]["S/real"] > scores[r]["S/delayed"])),
                         "mean_real": float(np.mean(scores[r]["S/real"]))} for r in scores}
    print("\nsanity: share of songs where real > other melody / real > delayed 2 frames; mean score on real")
    for r, v in res["sanity"].items():
        print(f"  {r:<60} {v['real_above_other_melody']:.2f} / {v['real_above_delayed']:.2f}   {v['mean_real']:.4f}")
    res["levels"] = {r: {"A/original": float(np.mean(scores[r]["A/original"]))} for r in scores
                     if "A/original" in scores[r]}
    res["table"] = table
    res["raw"] = {r: {k: v.tolist() for k, v in s.items()} for r, s in scores.items()}
    path = out_dir / f"reward_probe_{args.family}_{stamp}.json"
    json.dump(res, open(path, "w"))
    print("wrote", path)

    kinds = list(MODELS)
    print("\nmean score change (edited - original), mean over the 2 seeds; 95% bootstrap CI; share of songs up")
    print(f"{'variant':<22}" + "".join(f"{k:>34}" for k in kinds))
    for var, _ in pairs:
        if var.startswith("B/"):
            continue
        row = "".join(f"{table[var][k]['mean']:>+12.4f} [{table[var][k]['lo']:+.4f},{table[var][k]['hi']:+.4f}]"
                      f" {table[var][k]['share_up']:.2f}" for k in kinds)
        print(f"{var:<22}{row}")
    if "B" in args.parts:
        print("\nPart B, mean change per bar position (16ths; 0 = downbeat)")
        for v in ("restrike", "change"):
            for k in kinds:
                vals = [table.get(f"B/{q}/{v}", {}).get(k, {}).get("mean", float('nan')) for q in range(FPBAR)]
                print(f"{v:<9}{k:<22}" + " ".join(f"{x:+.3f}" for x in vals))
    print("\nlevels on the unedited songs:", {r: round(v["A/original"], 4) for r, v in res["levels"].items()})


if __name__ == "__main__":
    main()
