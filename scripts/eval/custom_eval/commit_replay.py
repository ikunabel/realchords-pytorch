#!/usr/bin/env python3
"""Replay ReaLJam's commit-ahead over test windows, offline and batched.

ReaLJam (realchords/realjam/agent_interface.py, generate_live) freezes chords a fixed number
of frames before they sound -- the commit-ahead. In steady state commit_frames settles at
k - 1 (journal/REALJAM_CHORD_TIMING.md), so the chord that SOUNDS at frame u is decided once,
at frame f = u - k + 1, from:

    real melody      frames  < f
    played chords    frames  < u   (the k - 1 from f on were committed by earlier calls)
    imagined melody  frames  f .. u-1, sampled by the model under the committed chords

That is exactly what this replays, one frame at a time, for a whole batch of windows:

    1. prompt = [BOS, c_0, m_0, ..., c_{f-1}, m_{f-1}]       (played chords, real melody)
    2. generate_online(conditions = c_f .. c_{u-1}, reverse_condition_order=True)
         forces each committed chord and samples the melody after it
         (gen_commit_online_model_teacher_forcing, the variant generate_live uses)
    3. generate(seq_len=1) samples the new chord c_u        (gen_online_model)

with ReaLJam's own validity filters. Frames generated further ahead are overwritten before
they sound and generation is causal, so with no response delay LOOKAHEAD DOES NOT AFFECT
WHAT IS PLAYED -- only commit-ahead does. k = 1 frame is commit-ahead 0: f = u, nothing is
imagined, and the replay reduces to one ordinary generate_online pass with the real melody
as conditions, which is what the existing evaluation does. --check runs that equivalence.

Start of a window: f = max(0, u - k + 1), so the first k chords are decided from imagined
melody only (cold start, as in the evaluation; ReaLJam's offline-model intro is not
replayed). Rows shorter than the window are masked to PAD exactly as the evaluation masks
them. Response delay (and so splicing between plans) is not modelled.

Output mirrors the evaluation layout, one directory per k, so progression_match.py,
fifths_motion.py and chord_placement.py run on it unchanged:

    <out_root>/k<k>/<corpus>/{gt/gt.pt, metadata.jsonl, chord_names_augmented.json,
                              models/<stage>_<seed>/{preds.pt, nicr_per_frame.pt}}

Windows are the first --n_windows rows of the existing evaluation's gt.pt, so they are the
same windows the evaluation scored.

Usage:
    python scripts/eval/custom_eval/commit_replay.py --stage mle --k 1 4 8 16
    python scripts/eval/custom_eval/commit_replay.py --check      # k=1 equivalence, greedy
    # keep the melody each call imagined, then compare it with the real melody
    python scripts/eval/custom_eval/commit_replay.py --stage gapt --k 8 16 --seeds s42 \
        --n_windows 32 --keep_imagined --out_root logs/custom_eval/commit_replay_imagined
    python scripts/eval/custom_eval/commit_replay.py --stage gapt --k 8 16 --seeds s42 \
        --analyze_imagined --out_root logs/custom_eval/commit_replay_imagined
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import yaml

from realchords.lit_module.decoder_only import LitDecoder
from realchords.model.sampling import (
    filter_invalid_tokens_generate_online,
    filter_invalid_tokens_generate_unconditional,
)
from realchords.utils.custom_evaluation import _save_nicr_per_frame
from realchords.utils.inference_utils import load_lit_model, load_rl_model

MLE_CONFIG = "configs/custom_eval/dataset_loo_new/hooktheory.yml"
GAPT_CONFIG = "configs/custom_eval/gapt_loo/hooktheory.yml"
SEEDS = ["s42", "s43", "s44"]


def _model_paths(config: str):
    """label -> path for the 7sets cells, plus the config's base_model."""
    cfg = yaml.safe_load(open(config))
    paths = {}
    for entry in cfg["model"]:
        label, path = entry.split("=", 1)
        if label in {f"7sets_{s}" for s in SEEDS}:
            paths[label.replace("7sets_", "")] = path
    return paths, cfg["base_model"]


def load_models(stage: str, device):
    """seed -> (model, tokenizer). MLE: Lightning .ckpt. GAPT: RL actor into the MLE anchor."""
    out = {}
    if stage == "mle":
        paths, _ = _model_paths(MLE_CONFIG)
        for seed, path in paths.items():
            m, tok, _ = load_lit_model(model_path=path, lit_module_cls=LitDecoder,
                                       batch_size=1, compile=False)
            out[seed] = (m.eval().to(device), tok)
    else:
        paths, base = _model_paths(GAPT_CONFIG)
        base_model, tok, _ = load_lit_model(model_path=base, lit_module_cls=LitDecoder,
                                            batch_size=1, compile=False)
        for seed, path in paths.items():
            m = load_rl_model(model_path=path, model=copy.deepcopy(base_model), compile=False)
            out[seed] = (m.eval().to(device), tok)
    return out


def _filters(tok):
    chord = partial(filter_invalid_tokens_generate_unconditional, model_part="chord",
                    tokenizer=tok, filter_opposite_part=True)
    melody = partial(filter_invalid_tokens_generate_online, model_part="melody", tokenizer=tok)
    chord_online = partial(filter_invalid_tokens_generate_online, model_part="chord",
                           tokenizer=tok)
    return chord, melody, chord_online


@torch.no_grad()
def replay(model, tok, gt: torch.Tensor, k: int, temperature: float,
           keep_imagined: bool = False, imag_model=None):
    """gt: [B, 513] = BOS + interleaved (chord, melody). Returns preds of the same shape.

    With keep_imagined, also returns imagined [B, F, k-1]: for the call that decides the chord
    at frame u, the melody the model imagined for frames u-k+1 .. u-1 under the committed
    chords (-1 where the call imagined fewer frames, i.e. the first k-1 frames of a window).

    imag_model: if given, it imagines the melody in the commit step while `model` still picks
    every chord -- e.g. the MLE checkpoint GAPT was initialised from, whose melody predictions RL
    has not drifted (melody_probe.py). Must share `tok`'s vocabulary.
    """
    chord_f, melody_f, _ = _filters(tok)
    melody = gt[:, 2::2].clone()                 # real melody lane, [B, F]
    # Rows shorter than the window end in PAD, and ReaLJam's melody filter refuses to imagine
    # a note after PAD. Those frames are masked back to PAD below and generation is causal,
    # so feeding SILENCE there cannot change any valid frame.
    melody[melody == tok.pad_token] = tok.name_to_id["SILENCE"]
    n_frames = melody.shape[1]
    played = torch.empty_like(melody)            # chords decided so far
    imagined = (torch.full((gt.shape[0], n_frames, max(k - 1, 1)), -1, dtype=gt.dtype,
                           device=gt.device) if keep_imagined else None)
    bos = gt[:, :1]
    for u in range(n_frames):
        f = max(0, u - k + 1)
        hist = torch.stack([played[:, :f], melody[:, :f]], dim=2).reshape(gt.shape[0], -1)
        prompt = torch.cat([bos, hist], dim=1)
        n_commit = u - f
        if n_commit > 0:
            committed = (imag_model if imag_model is not None else model).generate_online(
                prompts=prompt, conditions=played[:, f:u], seq_len=2 * n_commit,
                temperature=temperature, cache_kv=True, filter_logits_fn=melody_f,
                filter_kwargs=dict(), reverse_condition_order=True)
            # the forced chord positions must come back unchanged
            assert torch.equal(committed[:, 0::2], played[:, f:u])
            if keep_imagined:
                imagined[:, u, k - 1 - n_commit:] = committed[:, 1::2]
            prompt = torch.cat([prompt, committed], dim=1)
        new = model.generate(prompts=prompt, seq_len=1, temperature=temperature,
                             cache_kv=True, filter_logits_fn=chord_f, filter_kwargs=dict())
        played[:, u] = new[:, 0]
    preds = gt.clone()
    preds[:, 1::2] = played
    preds[gt == tok.pad_token] = tok.pad_token   # same row masking as the evaluation
    return (preds, imagined) if keep_imagined else preds


def analyze_imagined(corpus_dir: Path, label: str, k: int, tok) -> dict:
    """Imagined vs real melody, over the same frames, for every steady-state call.

    For the call deciding the chord at frame u (u >= k-1, inside the row's valid length), the
    window is frames u-k+1 .. u-1, under chords that were already committed. Compares the
    model's imagined melody with the user's real melody on that window:
      fit      note-in-chord of the committed chords against each melody (pooled)
      onsets   melody note onsets (NOTE_ON_*) per frame
      offbeat  share of those onsets off the quarter-note grid (frame mod 4 != 0)
    Tail windows are dropped first (end-aligned, so their bar phase is off), as in
    chord_placement.py.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from chord_placement import keep_rows
    from realchords.utils.eval_utils import evaluate_note_in_chord_ratio

    gt = torch.load(corpus_dir / "gt" / "gt.pt", map_location="cpu", weights_only=False)
    d = corpus_dir / "models" / label
    preds = torch.load(d / "preds.pt", map_location="cpu", weights_only=False)
    imag = torch.load(d / "imagined.pt", map_location="cpu", weights_only=False)
    rows, _ = keep_rows(corpus_dir / "metadata.jsonl", gt.shape[0])
    chords, real = preds[:, 1::2], gt[:, 2::2]
    on_lo, on_hi = tok.note_on_token_range
    w = k - 1
    seq_i, seq_r = [], []
    on = {"imagined": [0, 0, 0], "real": [0, 0, 0]}          # onsets, offbeat onsets, frames
    for i in rows:
        valid = int((real[i] != tok.pad_token).sum())
        for u in range(w, valid):
            f = u - w
            c, mi, mr = chords[i, f:u], imag[i, u], real[i, f:u]
            seq_i.append(torch.stack([c, mi], 1).reshape(-1))
            seq_r.append(torch.stack([c, mr], 1).reshape(-1))
            pos = torch.arange(f, u)
            for name, m in (("imagined", mi), ("real", mr)):
                is_on = (m >= on_lo) & (m <= on_hi)
                on[name][0] += int(is_on.sum())
                on[name][1] += int((is_on & (pos % 4 != 0)).sum())
                on[name][2] += w
    out = {}
    for name, seqs in (("imagined", seq_i), ("real", seq_r)):
        _, valid_c, correct_c = evaluate_note_in_chord_ratio(
            torch.stack(seqs), tok, model_part="chord", return_count=True)
        n_on, n_off, n_fr = on[name]
        out[name] = {"fit": float(correct_c.sum() / valid_c.sum()),
                     "onsets_per_bar": 16 * n_on / n_fr,
                     "offbeat_share": n_off / n_on if n_on else None}
    out["n_calls"] = len(seq_i)
    return out


@torch.no_grad()
def single_pass(model, tok, gt: torch.Tensor, temperature: float) -> torch.Tensor:
    """One generate_online pass with the real melody as conditions (the evaluation's way)."""
    _, _, chord_online = _filters(tok)
    gen = model.generate_online(prompts=gt[:, :1], conditions=gt[:, 2::2],
                                seq_len=gt.shape[1] - 1, temperature=temperature,
                                cache_kv=True, filter_logits_fn=chord_online,
                                filter_kwargs=dict())
    preds = torch.cat([gt[:, :1], gen], dim=1)
    preds[gt == tok.pad_token] = tok.pad_token
    return preds


def _prepare_corpus_dir(out: Path, gt_root: Path, corpus: str, n: int) -> torch.Tensor:
    src = gt_root / corpus
    gt = torch.load(src / "gt" / "gt.pt", map_location="cpu", weights_only=False)[:n]
    (out / "gt").mkdir(parents=True, exist_ok=True)
    torch.save(gt, out / "gt" / "gt.pt")
    lines = [l for l in (src / "metadata.jsonl").read_text().splitlines() if l.strip()][:n]
    (out / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    shutil.copy(src / "chord_names_augmented.json", out / "chord_names_augmented.json")
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mle", "gapt"], default="mle")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 4, 8, 16],
                    help="commit-ahead in FRAMES (16th notes): 1 = commit 0, 16 = 4 beats")
    ap.add_argument("--corpora", nargs="+", default=["hooktheory", "pop909"])
    ap.add_argument("--n_windows", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8, help="ReaLJam's default")
    ap.add_argument("--gt_root", type=Path, default=Path("logs/custom_eval/dataset_loo_new"))
    ap.add_argument("--out_root", type=Path, default=Path("logs/custom_eval/commit_replay"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", default=SEEDS)
    ap.add_argument("--imagine_with", choices=["self", "mle"], default="self",
                    help="mle: the MLE checkpoint of the same seed imagines the melody, the "
                         "--stage model still picks the chords")
    ap.add_argument("--keep_imagined", action="store_true",
                    help="also save the melody each call imagined, as imagined.pt")
    ap.add_argument("--analyze_imagined", action="store_true",
                    help="no generation: compare saved imagined melody with the real one")
    ap.add_argument("--check", action="store_true",
                    help="greedy k=1 replay vs greedy single pass on 16 windows: must be "
                         "token-identical (an implementation check, not a condition)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.analyze_imagined:
        from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
        results = {}
        print(f"{'corpus':<12}{'model':<10}{'k':>4}  {'melody':<9}{'fit':>7}"
              f"{'onsets/bar':>12}{'offbeat':>9}")
        for k in args.k:
            for corpus in args.corpora:
                cdir = args.out_root / f"k{k}" / corpus
                names = json.loads((cdir / "chord_names_augmented.json").read_text())
                tok = HooktheoryTokenizer(chord_names=names)
                for seed in args.seeds:
                    label = f"{args.stage}_{seed}"
                    if not (cdir / "models" / label / "imagined.pt").exists():
                        continue
                    r = analyze_imagined(cdir, label, k, tok)
                    results[f"k{k}/{corpus}/{label}"] = r
                    for name in ("real", "imagined"):
                        x = r[name]
                        print(f"{corpus:<12}{label:<10}{k:>4}  {name:<9}{x['fit']:>7.3f}"
                              f"{x['onsets_per_bar']:>12.2f}{x['offbeat_share']:>9.1%}")
        out = args.out_root / f"imagined_{args.stage}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"wrote {out}")
        return

    models = {s: v for s, v in load_models(args.stage, device).items() if s in args.seeds}
    imag = {}
    if args.imagine_with == "mle":
        imag = {s: v for s, v in load_models("mle", device).items() if s in args.seeds}
        for s_, (_, itok) in imag.items():
            assert itok.name_to_id == models[s_][1].name_to_id, f"{s_}: vocabularies differ"
    if args.check:
        gt = torch.load(args.gt_root / args.corpora[0] / "gt" / "gt.pt",
                        map_location="cpu", weights_only=False)[:16].to(device)
        for seed, (m, tok) in models.items():
            a = replay(m, tok, gt, k=1, temperature=0.0)
            b = single_pass(m, tok, gt, temperature=0.0)
            same = (a == b).all(dim=1)
            print(f"  {args.stage}_{seed}: greedy k=1 replay == single pass on "
                  f"{int(same.sum())}/{len(same)} windows")
        return

    for k in args.k:
        for corpus in args.corpora:
            out = args.out_root / f"k{k}" / corpus
            gt = _prepare_corpus_dir(out, args.gt_root, corpus, args.n_windows).to(device)
            for seed, (m, tok) in models.items():
                torch.manual_seed(args.seed)
                t0 = time.time()
                res = replay(m, tok, gt, k=k, temperature=args.temperature,
                             keep_imagined=args.keep_imagined,
                             imag_model=imag[seed][0] if seed in imag else None)
                preds, imagined = res if args.keep_imagined else (res, None)
                preds = preds.cpu()
                d = out / "models" / f"{args.stage}_{seed}"
                d.mkdir(parents=True, exist_ok=True)
                torch.save(preds, d / "preds.pt")
                if imagined is not None:
                    torch.save(imagined.cpu(), d / "imagined.pt")
                _, nicr = _save_nicr_per_frame(preds, tok, d / "nicr_per_frame.pt")
                print(f"  k={k:<3}{corpus:<12}{args.stage}_{seed}  NiCR {nicr:.3f}  "
                      f"{time.time() - t0:5.1f}s", flush=True)


if __name__ == "__main__":
    main()
