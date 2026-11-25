"""Replay ReaLJam complete-mode sessions: do chords written slightly off the beat blunt the model?

For each logged session (after the 2026-11-18 first-key-frame fix), the model input is rebuilt exactly as the
server builds it (agent.melody_to_frame_tokens + _build_interleaved_prompt over the logged chordTokens). Two
variants of the chord history:
  logged   the chords where the session wrote them;
  snapped  every chord onset within one 16th of a beat moved onto that beat (holds follow; "and"s untouched).
Measured:
  at each chord gesture (the frame the performer's chord starts, snapped in the snapped variant):
    P(onset) at that frame, entropy of the renormalised onset distribution (nats), P and rank of the played chord;
  over the session (frames from 64 on, teacher-forced through the sliding context):
    mean P(onset) on beats vs. off the beat ("beat sharpness").

    python scripts/eval/session_timing_replay.py [--model GAPT] [--after "2026-11-18 16:20"]
"""
import argparse
import datetime
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from realchords.realjam import agent_interface  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="GAPT")
ap.add_argument("--after", default="2026-11-18 16:20")
ap.add_argument("--logs", default=str(REPO / "logs/realjam"))
args = ap.parse_args()

agent = agent_interface.Agent(compile=False, voicings_path=None)
model, tk = agent.models[args.model], agent.tokenizer_for(args.model)
on_lo, on_hi = tk.chord_on_token_range
hold_lo, hold_hi = tk.chord_hold_token_range
SIL = tk.name_to_id["SILENCE"]
after = datetime.datetime.strptime(args.after, "%Y-%m-%d %H:%M")


def full_dist(notes, chords, frame):
    """The model's next-chord distribution at `frame` (all tokens), as the server's prompt gives it."""
    past = [e for e in notes if e["frame"] < frame]  # what the server had received by then
    note_hist = agent.melody_to_frame_tokens(past, frame).tolist()
    prompt, _, _ = agent._build_interleaved_prompt(note_hist, list(chords), frame, agent.max_frames - 1, tokenizer=tk)
    with torch.no_grad():
        logits, _ = model.net(prompt, return_intermediates=True)
    return torch.softmax(logits[0, -1].float(), dim=-1)


def snap(chords, gestures):
    """Move onsets within one 16th of a beat onto the beat; holds follow. Returns new chords, new gesture frames."""
    n = len(chords)
    onsets = [(k, t) for k, t in enumerate(chords) if on_lo <= t <= on_hi]
    moved = {}
    for k, t in onsets:
        o = ((k + 2) % 4) - 2
        moved[k] = k - o if o in (-1, 1) else k
    out = [SIL if (c is None or c < 0) else c for c in chords]
    # rebuild from the (possibly moved) onsets: each onset sounds until the next one or the original end of its holds
    ends = {}
    for i, (k, t) in enumerate(onsets):
        e = k + 1
        while e < n and chords[e] == t - (on_lo - hold_lo):
            e += 1
        ends[k] = e
    new = [SIL] * n
    for k, t in onsets:
        s, e = moved[k], ends[k] + (moved[k] - k)
        for f in range(max(s, 0), min(e, n)):
            new[f] = t if f == s else t - (on_lo - hold_lo)
    return new, {g: moved.get(g, g) for g in gestures}


rows = {"logged": [], "snapped": []}
sharp = {"logged": [[], []], "snapped": [[], []]}
files = sorted(glob.glob(str(Path(args.logs) / "GenJam Session*")))
used = 0
for f in files:
    if "timingcomplete" not in f or f"_{args.model}" not in f.split("timingcomplete")[-1]:
        continue
    d = json.load(open(f))
    if datetime.datetime.fromtimestamp(d["startTime"] / 1000) < after:
        continue
    notes = d["noteHistory"]
    chords = [SIL if (c is None or c < 0) else int(c) for c in d["chordTokens"]]
    gest = [(c["frame"], c["symbol"]) for c in d.get("completionHistory") or []
            if c.get("symbol") and c["frame"] < len(chords) and on_lo <= chords[c["frame"]] <= on_hi]
    if not gest:
        continue
    used += 1
    snapped, gmap = snap(chords, [g for g, _ in gest])
    for variant, ch in (("logged", chords), ("snapped", snapped)):
        for g, sym in gest:
            k = gmap[g] if variant == "snapped" else g
            if k < 8:
                continue
            p = full_dist(notes, ch, k)
            p_on = float(p[on_lo:on_hi + 1].sum())
            r = p[on_lo:on_hi + 1] / p_on
            ent = float(-(r * (r + 1e-30).log()).sum())
            tok = ch[k]
            pc = float(r[tok - on_lo]) if on_lo <= tok <= on_hi else float("nan")
            rank = int((r > r[tok - on_lo]).sum()) + 1 if on_lo <= tok <= on_hi else -1
            rows[variant].append({"session": Path(f).name[:30], "frame": k, "pos": k % 16, "p_onset": p_on,
                                  "entropy": ent, "p_played": pc, "rank": rank, "moved": k != g})
        # beat sharpness over the session (from frame 64)
        for k in list(range(64, len(ch) - 1, 3))[:250]:  # every 3rd frame (all beat positions over time), capped
            p = full_dist(notes, ch, k)
            sharp[variant][0 if k % 4 == 0 else 1].append(float(p[on_lo:on_hi + 1].sum()))
    print(f"{Path(f).name[16:40]}: {len(gest)} gestures", flush=True)

out = REPO / "logs/eval/session_timing_replay"
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json.dump({"model": args.model, "after": args.after, "rows": rows, "sharpness": sharp},
          open(out / f"replay_{args.model}_{stamp}.json", "w"))
print(f"\n{used} sessions, model {args.model}")
for v in ("logged", "snapped"):
    R = rows[v]
    moved = [x for x in R if x["moved"]]
    m = lambda key, rr: np.nanmedian([x[key] for x in rr]) if rr else float("nan")
    print(f"{v:8} gestures {len(R)} (moved {len(moved)}): median P(onset) {m('p_onset', R):.3f}, entropy {m('entropy', R):.2f}, "
          f"P(played chord) {m('p_played', R):.4f}, rank {m('rank', R):.0f} | on beats: P(onset) on {np.mean(sharp[v][0]):.3f} "
          f"vs off {np.mean(sharp[v][1]):.3f}")
for v in ("logged", "snapped"):
    R = [x for x in rows[v] if x["moved"] or v == "logged"]
print("wrote", out / f"replay_{args.model}_{stamp}.json")
