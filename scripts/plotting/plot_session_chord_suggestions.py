"""Which chords does the model suggest in a ReaLJam complete-mode session, and which does the performer play?

At every chord gesture of one logged session, the model's next-chord distribution over chord onsets (renormalised)
is computed from the history as the server builds it (as in scripts/eval/session_timing_replay.py). Top panel:
heatmap over gestures (x) of the chords that are ever among the model's top 3 or played (y), colour = probability,
black box = the chord the performer played. Bottom panel: over the whole session, the model's top-1 suggestion
counts vs. the played chord counts.

    python scripts/plotting/plot_session_chord_suggestions.py [--session <file name substring>] [--model GAPT]
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from realchords.realjam import agent_interface  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session", default="1795017927936")
ap.add_argument("--model", default="GAPT")
ap.add_argument("--top", type=int, default=3)
args = ap.parse_args()

f = [x for x in glob.glob(str(REPO / "logs/realjam/GenJam Session*")) if args.session in x][0]
d = json.load(open(f))
agent = agent_interface.Agent(compile=False, voicings_path=None)
model, tk = agent.models[args.model], agent.tokenizer_for(args.model)
on_lo, on_hi = tk.chord_on_token_range
SIL = tk.name_to_id["SILENCE"]
chords = [SIL if c < 0 else int(c) for c in d["chordTokens"]]
notes = d["noteHistory"]
gest = [(c["frame"], c["symbol"]) for c in d.get("completionHistory") or []
        if c.get("symbol") and c["frame"] < len(chords) and on_lo <= chords[c["frame"]] <= on_hi]
name = lambda tid: tk.id_to_name[int(tid)].replace("CHORD_ON_", "")

dists, played = [], []
for k, _ in gest:
    past = [e for e in notes if e["frame"] < k]
    nh = agent.melody_to_frame_tokens(past, k).tolist()
    prompt, _, _ = agent._build_interleaved_prompt(nh, list(chords), k, agent.max_frames - 1, tokenizer=tk)
    with torch.no_grad():
        logits, _ = model.net(prompt, return_intermediates=True)
    p = torch.softmax(logits[0, -1, on_lo:on_hi + 1].float(), dim=-1).numpy()
    dists.append(p)
    played.append(chords[k] - on_lo)
D = np.array(dists)

rows = set(played)
for p in D:
    rows |= set(np.argsort(-p)[: args.top].tolist())
rows = sorted(rows, key=lambda i: -D[:, i].sum())
fig, (ax, bx) = plt.subplots(2, 1, figsize=(max(12, len(gest) * 0.22), 4 + 0.22 * len(rows)),
                             gridspec_kw={"height_ratios": [0.22 * len(rows) + 1, 3]})
im = ax.imshow(D[:, rows].T, aspect="auto", cmap="magma_r", vmin=0, vmax=min(1, D[:, rows].max()))
for x, pl in enumerate(played):
    y = rows.index(pl)
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, ec="#00c0ff", lw=1.8))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([name(i + on_lo) for i in rows], fontsize=7)
ax.set_xticks(range(len(gest)))
ax.set_xticklabels([f"{k // 16 + 1}.{k % 16 // 4 + 1}" + ("" if k % 4 == 0 else f"+{k % 4}") for k, _ in gest],
                   rotation=90, fontsize=6)
ax.set_xlabel("chord gesture (bar.beat, +16ths off the beat)")
ax.set_title(f"Model ({args.model}) chord suggestions at each of the performer's chord gestures; "
             f"blue box = the chord played. Session {Path(f).name[16:29]}")
fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="P(chord | change now)")

top1 = collections.Counter(name(int(np.argmax(p)) + on_lo) for p in D)
pl = collections.Counter(name(i + on_lo) for i in played)
labels = [c for c, _ in (top1 + pl).most_common(20)]
x = np.arange(len(labels))
bx.bar(x - 0.2, [top1[c] for c in labels], 0.4, label="model's top-1 suggestion", color="#7b2d8b")
bx.bar(x + 0.2, [pl[c] for c in labels], 0.4, label="chord played", color="#00a0e0")
bx.set_xticks(x)
bx.set_xticklabels(labels, rotation=45, fontsize=8)
bx.set_ylabel("gestures")
bx.legend(fontsize=8)
top1_hit = np.mean([int(np.argmax(p)) == q for p, q in zip(D, played)])
bx.set_title(f"Over the session: the model's top suggestion was the played chord at {100 * top1_hit:.0f}% "
             f"of {len(gest)} gestures; median P(played) {np.median([p[q] for p, q in zip(D, played)]):.3f}")
fig.tight_layout()
out = REPO / "logs/eval/session_timing_replay"
out.mkdir(parents=True, exist_ok=True)
png = out / f"chord_suggestions_{args.session}_{args.model}.png"
fig.savefig(png, dpi=120, bbox_inches="tight")
print("wrote", png)
