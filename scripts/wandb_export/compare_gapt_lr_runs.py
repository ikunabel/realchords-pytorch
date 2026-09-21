"""Compare LR schedule + train/eval curves of the GAPT-family RL runs from their
wandb exports (gapt 5e-6, gapt_paper_authentic 5e-7, gapt_multiscale 5e-7).
See journal/RL_LEARNING_RATE.md, "What the 5e-6 vs 5e-7 GAPT curves show"."""
import csv
import math

BASE = "/hpcwork/thes2192/realchords/logs/my_logs/rl"
RUNS = {
    "gapt 5e-6": f"{BASE}/gapt/wandb_export/history.csv",
    "gapt 5e-7": f"{BASE}/gapt_paper_authentic/wandb_export/history.csv",
    "gapt_ms 5e-7": f"{BASE}/gapt_multiscale/wandb_export/history.csv",
}


def f(x):
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


data = {k: list(csv.DictReader(open(p))) for k, p in RUNS.items()}


def series(rows, col, stepcol):
    out = []
    for r in rows:
        v, s = f(r.get(col)), f(r.get(stepcol))
        if v is not None and s is not None:
            out.append((s, v))
    return out


print("== LR schedule (train/actor_lr, train/critic_lr)")
for k, rows in data.items():
    for col in ("train/actor_lr", "train/critic_lr"):
        s = series(rows, col, "train/global_step")
        if not s:
            continue
        vals = [v for _, v in s]
        peak_step = max(s, key=lambda t: t[1])[0]
        print(f"  {k:13s} {col:16s} first={vals[0]:.2e} peak={max(vals):.2e}@{peak_step:.0f} "
              f"last={vals[-1]:.2e} min_after_peak={min(v for st, v in s if st >= peak_step):.2e} n={len(s)}")

TRAIN = ["train/reward", "train/entropy", "train/kl", "train/values", "train/policy_loss",
         "train/critic_loss", "train/grad_norm", "train/reward_gail_reward", "train/reward_loss",
         "train/reward_acc",
         "train/reward_discriminative_reward_0", "train/reward_contrastive_reward_0",
         "train/reward_multiscale_contrastive_reward",
         "train/reward_repetition_penalty", "train/reward_silence_penalty",
         "train/reward_invalid_output_penalty", "train/reward_long_note_penalty",
         "train/num_silence_tokens", "train/num_repetitive_generations"]
WINDOWS = [(1, 50), (200, 250), (450, 500), (700, 750), (950, 1010)]

print("\n== train metrics, mean over step windows")
for col in TRAIN:
    print(f"  {col}")
    for k, rows in data.items():
        s = series(rows, col, "train/global_step")
        if not s:
            continue
        cells = []
        for a, b in WINDOWS:
            w = [v for st, v in s if a <= st <= b]
            cells.append(f"{sum(w)/len(w):8.3f}" if w else "     n/a")
        print(f"    {k:13s} " + " ".join(cells))

EVAL = ["eval/reward", "eval/kl", "eval/note_in_chord_ratio_pred", "eval/note_in_chord_ratio_gt",
        "eval/avg_chord_duration_pred", "eval/avg_chord_duration_gt",
        "eval/reward_gail_reward", "eval/reward_discriminative_reward_0",
        "eval/reward_contrastive_reward_0", "eval/reward_multiscale_contrastive_reward",
        "eval/reward_contrastive_reward_rhythm_0",
        "eval/reward_repetition_penalty", "eval/reward_silence_penalty",
        "eval/reward_invalid_output_penalty", "eval/reward_long_note_penalty",
        "eval/invalid_ratio", "eval/num_silence_tokens"]
print("\n== eval metrics at each eval step")
for col in EVAL:
    print(f"  {col}")
    for k, rows in data.items():
        s = series(rows, col, "eval/global_step")
        if not s:
            continue
        print(f"    {k:13s} " + " ".join(f"{st:.0f}:{v:.3f}" for st, v in s))
