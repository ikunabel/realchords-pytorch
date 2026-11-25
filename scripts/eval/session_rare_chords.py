"""ReaLJam complete-mode sessions: how rare are the played chords' tokens, and does a rare one confuse the model?

For the post-fix complete-mode sessions (as session_timing_replay.py), every played chord token is classed by how
common its chord *type* is in the training data (key-free: root removed, slash bass as interval; counts from the
chord conversion database, Hooktheory training split, the corpus the model was trained on):
  common   one of the 10 most frequent types (maj, m, 7, m7, ... ~94% of chords),
  medium   rank 11-50,
  rare     rank > 50 or never seen.
Measured, from the history as the server builds it:
  at the gesture:      P(played chord | change now), rank;
  one beat later:      entropy of the onset distribution and P(onset) (is the model unsettled by the chord?);
  at the next gesture: P(next played chord), entropy (does it recover / follow?).

    python scripts/eval/session_rare_chords.py [--model GAPT] [--after "2026-11-18 16:20"]
"""
import argparse
import collections
import datetime
import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from note_seq import chord_symbols_lib as csl

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from realchords.realjam import agent_interface  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="GAPT")
ap.add_argument("--after", default="2026-11-18 16:20")
args = ap.parse_args()


def ctype(name):
    try:
        r, k, m, b = csl._split_chord_symbol(name)
        t = (k or "maj") + m
        if b:
            t += "/" + str((csl.chord_symbol_bass(name) - csl.chord_symbol_root(name)) % 12)
        return t
    except Exception:  # noqa: BLE001
        return None


db = sqlite3.connect(str(REPO / "data/chord_conversion/chord_conversion.sqlite"))
tcount = collections.Counter()
for n, k in db.execute("select token_name, count(*) from events where dataset='hooktheory' and split='train' "
                       "group by token_name"):
    tcount[ctype(n)] += k
type_rank = {t: i + 1 for i, (t, _) in enumerate(tcount.most_common())}
cls = lambda name: ("common" if type_rank.get(ctype(name), 10 ** 6) <= 10
                    else "medium" if type_rank.get(ctype(name), 10 ** 6) <= 50 else "rare")

agent = agent_interface.Agent(compile=False, voicings_path=None)
model, tk = agent.models[args.model], agent.tokenizer_for(args.model)
on_lo, on_hi = tk.chord_on_token_range
SIL = tk.name_to_id["SILENCE"]
after = datetime.datetime.strptime(args.after, "%Y-%m-%d %H:%M")


def dist(notes, chords, k):
    past = [e for e in notes if e["frame"] < k]
    nh = agent.melody_to_frame_tokens(past, k).tolist()
    prompt, _, _ = agent._build_interleaved_prompt(nh, list(chords), k, agent.max_frames - 1, tokenizer=tk)
    with torch.no_grad():
        logits, _ = model.net(prompt, return_intermediates=True)
    p = torch.softmax(logits[0, -1].float(), dim=-1)
    p_on = float(p[on_lo:on_hi + 1].sum())
    r = (p[on_lo:on_hi + 1] / p_on).numpy()
    return p_on, r, float(-(r * np.log(r + 1e-30)).sum())


rows = []
for f in sorted(glob.glob(str(REPO / "logs/realjam/GenJam Session*"))):
    if "timingcomplete" not in f or f"_{args.model}" not in f.split("timingcomplete")[-1]:
        continue
    d = json.load(open(f))
    if datetime.datetime.fromtimestamp(d["startTime"] / 1000) < after:
        continue
    chords = [SIL if c < 0 else int(c) for c in d["chordTokens"]]
    notes = d["noteHistory"]
    gest = [c["frame"] for c in d.get("completionHistory") or []
            if c.get("symbol") and c["frame"] < len(chords) and on_lo <= chords[c["frame"]] <= on_hi]
    gest = sorted(set(g for g in gest if g >= 8))
    for i, k in enumerate(gest):
        tok = chords[k]
        name = tk.id_to_name[tok].replace("CHORD_ON_", "")
        p_on, r, ent = dist(notes, chords, k)
        row = {"session": Path(f).name[16:29], "frame": k, "chord": name, "class": cls(name),
               "p_played": float(r[tok - on_lo]), "rank": int((r > r[tok - on_lo]).sum()) + 1, "entropy": ent}
        k1 = k + 4  # one beat later, if the chord still sounds
        nxt = gest[i + 1] if i + 1 < len(gest) else None
        if k1 < len(chords) and (nxt is None or k1 < nxt):
            p1, _, e1 = dist(notes, chords, k1)
            row.update({"p_onset_beat_later": p1, "entropy_beat_later": e1})
        if nxt is not None:
            tn = chords[nxt]
            _, rn, en = dist(notes, chords, nxt)
            row.update({"p_next_played": float(rn[tn - on_lo]), "entropy_next": en,
                        "next_class": cls(tk.id_to_name[tn].replace("CHORD_ON_", ""))})
        rows.append(row)
    print(Path(f).name[16:29], len(gest), "gestures", flush=True)

out = REPO / "logs/eval/session_timing_replay"
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({"model": args.model, "type_rank_top": tcount.most_common(60), "rows": rows},
          open(out / f"rare_chords_{args.model}_{stamp}.json", "w"))

med = lambda xs: float(np.median(xs)) if xs else float("nan")
mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
print(f"\n{len(rows)} gestures. Type ranks from Hooktheory training: top 10 = {[t for t, _ in tcount.most_common(10)]}")
print("| class | gestures | median P(played) | median rank | entropy at gesture | 1 beat later: P(onset), entropy "
      "| next gesture: median P(played), entropy |")
for c in ("common", "medium", "rare"):
    R = [x for x in rows if x["class"] == c]
    if not R:
        continue
    bl = [x for x in R if "entropy_beat_later" in x]
    nx = [x for x in R if "p_next_played" in x]
    nx_common = [x for x in nx if x["next_class"] == "common"]
    print(f"| {c} | {len(R)} ({100 * len(R) / len(rows):.0f}%) | {med([x['p_played'] for x in R]):.4f} | "
          f"{med([x['rank'] for x in R]):.0f} | {mean([x['entropy'] for x in R]):.2f} | "
          f"{mean([x['p_onset_beat_later'] for x in bl]):.3f}, {mean([x['entropy_beat_later'] for x in bl]):.2f} | "
          f"{med([x['p_next_played'] for x in nx]):.4f}, {mean([x['entropy_next'] for x in nx]):.2f} "
          f"(next is common: {med([x['p_next_played'] for x in nx_common]):.4f}, n={len(nx_common)}) |")
print("most frequent rare chords played:", collections.Counter(x["chord"] for x in rows if x["class"] == "rare").most_common(12))
