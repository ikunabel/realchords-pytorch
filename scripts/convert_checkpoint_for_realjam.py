#!/usr/bin/env python3
"""Turn a decoder-only training checkpoint into one the ReaLJam server loads.

Two differences have to be bridged:

* Format: training saves a full Lightning checkpoint (weights under
  ``state_dict`` with a ``model._orig_mod.`` prefix, plus optimizer state);
  ReaLJam loads a plain state dict with an ``args.yml`` next to it.
* Vocabulary: newer models are tokenised with the data cache's chord list
  (``data/cache/chord_names_augmented.json``, which grew to 2,969 chords as
  datasets were added), while ReaLJam's tokenizer uses the checkpoints' list
  (``checkpoints/chord_names_augmented.json``, 2,821 chords) -- in a different
  order, so the same token id means different chords.

The vocabulary is bridged by token *name*: for every token in ReaLJam's
vocabulary, the matching rows of the input embedding and the output
projection are copied over. ReaLJam's chord list is a subset of the larger one,
so every token is found; chords only the larger list has are dropped, i.e. the
converted model can't output them (ReaLJam couldn't represent them anyway).
All other weights are identical.

Usage::

    python scripts/convert_checkpoint_for_realjam.py \\
        --ckpt "checkpoints/decoder_only/<run>/step=2700.ckpt" \\
        --config "configs/decoder_only/<run>.yml" \\
        --out checkpoints/<name>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer

PREFIX = "model._orig_mod."
VOCAB_KEYS = ("decoder.token_emb.emb.weight", "decoder.to_logits.weight")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Lightning checkpoint from training.")
    p.add_argument("--config", required=True,
                   help="The training config (.yml); copied as args.yml for the model "
                        "hyperparameters.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--train_chord_names", default="data/cache/chord_names_augmented.json",
                   help="Chord list the model was trained with.")
    p.add_argument("--realjam_chord_names", default="checkpoints/chord_names_augmented.json",
                   help="Chord list of ReaLJam's tokenizer.")
    args = p.parse_args()

    train_tok = HooktheoryTokenizer(chord_names=json.load(open(args.train_chord_names)))
    jam_tok = HooktheoryTokenizer(chord_names=json.load(open(args.realjam_chord_names)))
    missing = [n for n in jam_tok.name_to_id if n not in train_tok.name_to_id]
    if missing:
        raise SystemExit(f"{len(missing)} ReaLJam tokens unknown to the training vocabulary, "
                         f"e.g. {missing[:5]}")
    index = torch.tensor([train_tok.name_to_id[jam_tok.id_to_name[i]]
                          for i in range(jam_tok.num_tokens)])

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v for k, v in state.items()}
    for key in VOCAB_KEYS:
        if state[key].shape[0] != train_tok.num_tokens:
            raise SystemExit(f"{key} has {state[key].shape[0]} rows, the training vocabulary "
                             f"{train_tok.num_tokens} tokens -- wrong --train_chord_names?")
        state[key] = state[key][index].clone()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / Path(args.ckpt).name
    torch.save(state, weights)
    shutil.copy(args.config, out / "args.yml")
    print(f"{train_tok.num_tokens} -> {jam_tok.num_tokens} tokens "
          f"({train_tok.num_tokens - jam_tok.num_tokens} dropped); wrote {weights} and "
          f"{out / 'args.yml'}")


if __name__ == "__main__":
    main()
