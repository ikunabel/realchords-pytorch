#!/usr/bin/env python
"""How much of ReaLJam's real-time budget does generation use?

Times three things and compares them with the time a frame and a beat last at
given tempos -- the numbers that decide which lookahead a machine can sustain,
and whether extra generation work (e.g. K parallel plans) would fit:

  * baseline      -- one steady-state auto-mode call (generate_live with a full
                     commit window), i.e. what ReaLJam already does every frame;
  * K plans       -- K independent full-lookahead continuations of the same
                     history, run as ONE batch of shape [K, T];
  * press scoring -- a single prefill forward pass, the cost of asking the model
                     to rank candidates at the moment of a key press.

Plans only need refreshing once per beat, so their budget is a beat, whereas
the baseline's is a frame. Uses the same model stack as the live app.

Usage::

    python realchords/realjam/benchmark_realtime_latency.py              # PyTorch
    python realchords/realjam/benchmark_realtime_latency.py --mlx        # Apple silicon
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from realchords.realjam import agent_interface


def timed(fn, warmup: int, repeats: int) -> float:
    """Median wall time in ms."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlx", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the models, as the live server does by default")
    p.add_argument("--model", default="ReaLchords")
    p.add_argument("--lookahead", type=int, default=16, help="frames")
    p.add_argument("--batch_sizes", default="1,2,4,8,16")
    p.add_argument("--bpms", default="80,120")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    agent = agent_interface.Agent(compile=args.compile, mlx=args.mlx, voicings_path=None)
    model = agent.models[args.model]
    tk = agent.tokenizer
    la = args.lookahead

    # A saturated context: a steady melody, long enough that the prompt hits
    # its cap (256 - lookahead frames), which is the worst case for prefill.
    frame = 300
    notes = []
    for i in range(frame // 4):
        pitch = [60, 62, 64, 65, 67, 65, 64, 62][i % 8]
        notes.append({"pitch": pitch, "frame": i * 4, "on": True})
        notes.append({"pitch": pitch, "frame": i * 4 + 3, "on": False})
    c_on, c_hold = tk.name_to_id["CHORD_ON_C"], tk.name_to_id["CHORD_C"]
    history = [c_on if f % 16 == 0 else c_hold for f in range(frame)]

    note_hist = agent.melody_to_frame_tokens(notes, frame).tolist()
    prompt, _, _ = agent._build_interleaved_prompt(
        note_hist, list(history), frame, agent.max_frames - la)

    # -- baseline: steady-state generate_live (commit_frames = la - 1) --------
    # len(chord_tokens) = prev_frame + la with prev_frame = frame - 1
    steady = history + [c_hold] * (la - 1)
    def auto_call():
        agent.generate_live(args.model, notes, list(steady), frame, la, la,
                            0.8, silence_till=0, intro_set=True)
    baseline = timed(auto_call, args.warmup, args.repeats)

    # -- K plans: one batched free generation of the whole window -------------
    plans = {}
    for k in [int(x) for x in args.batch_sizes.split(",")]:
        batch = prompt.repeat(k, 1)
        def plan_call(batch=batch):
            agent.gen_online_model(model, batch, seq_len=2 * la - 1,
                                   temperature=0.8)
        plans[k] = timed(plan_call, args.warmup, args.repeats)

    # -- press-time scoring: one prefill over the full context ---------------
    press_prompt, _, _ = agent._build_interleaved_prompt(
        note_hist, list(history), frame, agent.max_frames - 1)
    if args.mlx:
        def press_call():
            model._forward_mx(press_prompt, cache=None)
    else:
        def press_call():
            with torch.no_grad():
                model.net(press_prompt, return_intermediates=True)
    press = timed(press_call, args.warmup, args.repeats)

    backend = "MLX" if args.mlx else f"PyTorch ({agent.device})"
    if args.compile:
        backend += " + torch.compile"
    print(f"\n{backend} · model {args.model} · lookahead {la} frames · "
          f"prompt {prompt.shape[1]} tokens · median of {args.repeats}\n")
    print(f"  auto-mode call (baseline)   {baseline:8.1f} ms")
    print(f"  press-time prefill          {press:8.1f} ms")
    print(f"\n  {'K plans':>8} {'batched':>10} {'per plan':>10} {'vs K=1':>8}")
    for k, ms in plans.items():
        print(f"  {k:>8} {ms:>8.1f}ms {ms / k:>8.1f}ms {ms / plans[1]:>7.2f}x")

    print("\n  Budgets:")
    for bpm in [int(b) for b in args.bpms.split(",")]:
        frame_ms, beat_ms = 60000 / bpm / 4, 60000 / bpm
        fits = [k for k, ms in plans.items() if ms < beat_ms / 2]
        print(f"    {bpm} BPM: frame {frame_ms:.0f} ms (baseline "
              f"{'fits' if baseline < frame_ms else 'OVER'}), beat {beat_ms:.0f} ms "
              f"-> largest K using <= half a beat: {max(fits) if fits else 'none'}")


if __name__ == "__main__":
    main()
