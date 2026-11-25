"""How many frames of imagined melody does a played chord depend on in ReaLJam's auto mode?

The frontend sends a /play request about once per frame with the current frame f; the server builds the
prompt from the real melody up to f - 1 and generates the lookahead window frame by frame, both lanes, so
every chord after the window's first depends on the model's imagined melody. Two cases:

  commit-ahead 0   every request regenerates the window. The chord sounding at frame i comes from the last
                   response that arrived before i, i.e. from a request started about ceil(latency / frame)
                   frames earlier, so it saw that many frames of imagined melody.
  commit-ahead c   frames f .. f+c-1 are fixed by earlier requests; in steady state (c = lookahead) a request
                   adds one new chord at the end of the window, after imagining the melody of the c-1
                   committed frames before it: c - 1 frames of imagined melody, independent of latency as
                   long as the latency stays below the commit window.

Times generate_live in its steady state (the previous call was one frame earlier) with a saturated context
(prompt at its length cap, the worst case), in-process (no HTTP), and reports the median latency, the frame
length at each tempo and the resulting number of imagined frames.

    python scripts/eval/live_response_gap.py [--models GAPT] [--compile | --mlx] [--bpms 80 120]
    python scripts/eval/live_response_gap.py --url http://127.0.0.1:8899   # through a running server (HTTP)

With --url the same steady-state requests go to a running ReaLJam server as the frontend sends them (POST
/play, the frontend's JSON body), so the time includes HTTP and JSON, and the server's own backend settings
(torch.compile by default) apply.
"""
import argparse
import json
import urllib.request
import math
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from realchords.realjam import agent_interface  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--models", nargs="+", default=["GAPT"])
ap.add_argument("--lookahead_beats", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--bpms", type=int, nargs="+", default=[80, 120])
ap.add_argument("--compile", action="store_true", help="torch.compile, as the live server does by default")
ap.add_argument("--mlx", action="store_true", help="MLX backend (Apple silicon), as server.py --mlx")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--repeats", type=int, default=15)
ap.add_argument("--url", default=None, help="time POST <url>/play on a running server instead of in-process")
args = ap.parse_args()

agent = agent_interface.Agent(compile=args.compile, mlx=args.mlx, voicings_path=None)


def post_play(model, notes, chord_tokens, frame, la, commit):
    """One /play request with the frontend's body (play_from_midi_input.js, syncWithServer)."""
    body = {"model": model, "notes": notes, "chordTokens": chord_tokens, "frame": frame, "lookahead": la,
            "commitahead": commit, "silenceTill": 0, "temperature": 0.8, "introSet": True,
            "useCustomVoicings": False, "bassLine": False, "prevVoicing": None}
    req = urllib.request.Request(args.url.rstrip("/") + "/play", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        json.loads(r.read())

frame = 300
notes = []
for i in range(frame // 4):
    pitch = [60, 62, 64, 65, 67, 65, 64, 62][i % 8]
    notes += [{"pitch": pitch, "frame": i * 4, "on": True}, {"pitch": pitch, "frame": i * 4 + 3, "on": False}]

rows = []
for model in args.models:
    tk = agent.tokenizer_for(model)
    c_on, c_hold = tk.name_to_id["CHORD_ON_C"], tk.name_to_id["CHORD_C"]
    history = [c_on if f % 16 == 0 else c_hold for f in range(frame)]
    for beats in args.lookahead_beats:
        la = 4 * beats
        for commit in (0, la):
            steady = history + [c_hold] * (la - 1)  # previous call at frame - 1 generated la frames

            def call():
                if args.url:
                    post_play(model, notes, list(steady), frame, la, commit)
                else:
                    agent.generate_live(model, notes, list(steady), frame, la, commit, 0.8,
                                        silence_till=0, intro_set=True)
            for _ in range(args.warmup):
                call()
            ms = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                call()
                ms.append((time.perf_counter() - t0) * 1000)
            rows.append((model, beats, commit, statistics.median(ms), max(ms)))
            print(f"{model} lookahead {beats} beat(s) commit {commit} frames: median {rows[-1][3]:.0f} ms, "
                  f"max {rows[-1][4]:.0f} ms", flush=True)

print("\nbackend: " + (f"HTTP to {args.url} (the server's own backend)" if args.url else
      ("MLX" if args.mlx else f"PyTorch ({agent.device})") + (' + torch.compile' if args.compile else '')
      + "; in-process, no HTTP"))
print("| model | lookahead | commit | median latency | " + " | ".join(
    f"imagined frames @ {b} bpm (frame {60000 / b / 4:.0f} ms)" for b in args.bpms) + " |")
print("|---|---|---|---|" + "---|" * len(args.bpms))
for model, beats, commit, med, _ in rows:
    la = 4 * beats
    cells = []
    for b in args.bpms:
        frame_ms = 60000 / b / 4
        if commit == 0:
            cells.append(f"{math.ceil(med / frame_ms)} (latency)")
        else:
            ok = med < commit * frame_ms
            cells.append(f"{commit - 1} (commit){'' if ok else ' -- latency exceeds the commit window'}")
    print(f"| {model} | {beats} beat(s) | {commit} | {med:.0f} ms | " + " | ".join(cells) + " |")
