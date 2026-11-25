"""Complete mode: a /chord_candidates request sent during frame i (asking about frame i + 1) -- in which frame
does the answer arrive?

Times POST /chord_candidates on a running server with the frontend's body (format "probs": every chord's
probability, as requestCandidates sends it), saturated context (the worst case). The request leaves at a
uniformly random moment inside frame i (candidateLoop and gestures are not aligned to the frame grid), so with
round trip r and frame length F it arrives in frame i + floor(u + r / F), u ~ U[0, 1). Reports, per tempo, the
share of answers arriving in frame i, i + 1, i + 2, ... -- arriving in frame i means in time for frame i + 1.

    python scripts/eval/candidates_latency.py --url http://127.0.0.1:8899 [--model GAPT] [--bpms 80 120]
"""
import argparse
import json
import statistics
import time
import urllib.request

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--url", default="http://127.0.0.1:8899")
ap.add_argument("--model", default="GAPT")
ap.add_argument("--bpms", type=int, nargs="+", default=[80, 120])
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--repeats", type=int, default=60)
args = ap.parse_args()

frame = 300
notes = []
for i in range(frame // 4):
    pitch = [60, 62, 64, 65, 67, 65, 64, 62][i % 8]
    notes += [{"pitch": pitch, "frame": i * 4, "on": True}, {"pitch": pitch, "frame": i * 4 + 3, "on": False}]
vocab = json.loads(urllib.request.urlopen(f"{args.url}/chord_vocab?model={urllib.request.quote(args.model)}").read())
entries = vocab["chords"] if isinstance(vocab, dict) and "chords" in vocab else vocab
c_on = next(e["onsetToken"] for e in entries if e["symbol"] == "C")
c_hold = next(e["holdToken"] for e in entries if e["symbol"] == "C")
chord_tokens = [c_on if f % 16 == 0 else c_hold for f in range(frame)]
body = json.dumps({"model": args.model, "notes": notes, "chordTokens": chord_tokens, "frame": frame + 1,
                   "format": "probs"}).encode()


def call():
    req = urllib.request.Request(f"{args.url}/chord_candidates", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        json.loads(r.read())


for _ in range(args.warmup):
    call()
ms = []
for _ in range(args.repeats):
    t0 = time.perf_counter()
    call()
    ms.append((time.perf_counter() - t0) * 1000)
ms = np.array(ms)
print(f"/chord_candidates round trip ({args.model}, saturated context, {args.repeats} calls): "
      f"median {statistics.median(ms):.0f} ms, p10 {np.percentile(ms, 10):.0f}, p90 {np.percentile(ms, 90):.0f}, "
      f"max {ms.max():.0f} ms")
u = np.linspace(0, 1, 1001)[:-1]
for bpm in args.bpms:
    F = 60000 / bpm / 4
    k = np.floor(u[None, :] + ms[:, None] / F).astype(int).ravel()
    shares = {int(v): float(np.mean(k == v)) for v in np.unique(k)}
    print(f"{bpm} bpm (frame {F:.0f} ms): answer arrives in " +
          ", ".join(f"frame i+{v}: {s:.0%}" if v else f"frame i: {s:.0%}" for v, s in shares.items()))
