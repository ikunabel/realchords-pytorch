#!/usr/bin/env python3
"""Check that ReaLJam wires each model to its own chord vocabulary.

Takes the same model twice: once in the new vocabulary (2,969 chords, 6,198
tokens) and once converted to ReaLJam's old one (2,821 chords, 5,902 tokens,
scripts/convert_checkpoint_for_realjam.py). Both are loaded through the
server's Agent, which picks each one's tokenizer from its token count
(journal/reference/TOKENIZERS.md, section 0). On logged session histories (in
the old vocabulary; remapped by name for the new model), the two must give the
same logits for every token both vocabularies share, and the new model's
generate_live must decode to valid chord names.

Usage (repo root, the .venv python)::

    # the 7-dataset GAPT: the actor as trained vs a conversion of it
    python scripts/check_realjam_vocab.py \\
        --new "checkpoints/gapt_loo_7sets.alpha=0.5.seed=42/actor.pth" --old <converted>/actor.ckpt
    # the WJD+FiloBass chord model (training original, stripped into a scratch dir)
    python scripts/check_realjam_vocab.py --new <dir>/step=2700.ckpt \\
        --old checkpoints/wjd_filobass_online_chord/step=2700.ckpt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from realchords.realjam import agent_interface  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, help="checkpoint in the new vocabulary (args.yml next to it)")
    p.add_argument("--old", required=True, help="the same model converted to the old vocabulary")
    p.add_argument("--sessions", type=int, default=6, help="logged sessions to take histories from")
    p.add_argument("--frames", type=int, default=8, help="frames per session")
    args = p.parse_args()

    # Load only these two (plus the always-loaded MLE / offline models)
    agent_interface.MODEL_PATH_CANDIDATES.clear()
    agent_interface.MODEL_PATH_CANDIDATES.update({"NEW": [args.new], "OLD": [args.old]})
    agent = agent_interface.Agent()
    tk_new, tk_old = agent.tokenizer_for("NEW"), agent.tokenizer_for("OLD")
    print(f"NEW: {tk_new.num_tokens} tokens, OLD: {tk_old.num_tokens} tokens")
    assert tk_new.num_tokens != tk_old.num_tokens, "both models in the same vocabulary?"
    shared = [n for n in tk_old.name_to_id if n in tk_new.name_to_id]
    idx_old = torch.tensor([tk_old.name_to_id[n] for n in shared])
    idx_new = torch.tensor([tk_new.name_to_id[n] for n in shared])
    print(f"{len(shared)} shared tokens; {tk_new.num_tokens - len(shared)} only in NEW")

    files = sorted(glob.glob("logs/realjam/*"), key=os.path.getmtime)[-40:]
    worst, n = 0.0, 0
    for f in files:
        if n >= args.sessions * args.frames:
            break
        d = json.load(open(f))
        tokens, notes = d.get("chordTokens") or [], d.get("noteHistory") or []
        if len(tokens) < 40 or not notes or not str(d.get("description", "")).endswith("GAPT"):
            continue  # old-vocabulary sessions only (the GAPT models)
        for fr in range(16, min(len(tokens), 16 + 16 * args.frames), 16):
            hist_old = [t if t is not None else -1 for t in tokens[:fr]]
            hist_new = agent.remap_chord_tokens(hist_old, tk_old, tk_new)
            note_hist = agent.melody_to_frame_tokens([e for e in notes if e["frame"] < fr], fr).tolist()
            logits = {}
            for name, hist, tk in (("OLD", hist_old, tk_old), ("NEW", hist_new, tk_new)):
                prompt, _, _ = agent._build_interleaved_prompt(
                    list(note_hist), list(hist), fr, agent.max_frames - 1, tokenizer=tk)
                with torch.no_grad():
                    logits[name] = agent.models[name].net(prompt, return_intermediates=True)[0][0, -1].float()
            diff = (logits["OLD"][idx_old] - logits["NEW"][idx_new]).abs().max().item()
            worst = max(worst, diff)
            n += 1
    print(f"{n} prompts: largest logit difference on shared tokens {worst:.2e}")
    assert n > 0, "no logged sessions found"
    assert worst < 1e-3, "logits differ: the vocabulary wiring (or the conversion) is wrong"

    # generate_live end to end with the new model: tokens in its vocabulary, valid names
    d = json.load(open(files[-1]))
    fr = min(64, len(d["chordTokens"]))
    hist = agent.remap_chord_tokens([t if t is not None else -1 for t in d["chordTokens"][:fr]], tk_old, tk_new)
    chords, toks, _ = agent.generate_live("NEW", [e for e in d["noteHistory"] if e["frame"] < fr],
                                          hist, fr, 8, 0, 0.8, 0, True)
    assert all(0 <= t < tk_new.num_tokens for t in toks)
    names = [c[0] for c in chords]
    print("generate_live (NEW):", names)
    entries = agent.chord_vocab_entries("NEW")
    print(f"chord_vocab_entries: NEW {len(entries)}, OLD {len(agent.chord_vocab_entries('OLD'))}")
    print("OK")


if __name__ == "__main__":
    main()
