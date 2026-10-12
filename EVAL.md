# Running evaluations

Everything needed to generate from checkpoints and score the output. The journal
(`journal/METRICS.md`) explains what the metrics *mean*; this file is only how to run them.

---

## The short version

```bash
module load Python/3.12.3 && source .venv/bin/activate && export PYTHONPATH="$PWD"

# 1. generate + score the standard metrics, one config per test corpus
python scripts/eval/custom_eval/run_custom_eval.py configs/custom_eval/<study>/<corpus>.yml

# 2. the progression metrics, which read the dumps step 1 wrote
python scripts/eval/custom_eval/score_progressions.py --eval_dir logs/custom_eval/<study>
python scripts/eval/custom_eval/progression_match.py \
    --eval_dirs MLE=logs/custom_eval/<a> GAPT=logs/custom_eval/<b> --quality full --spans 3 4 5 6
```

Step 1 is slow (generation). Step 2 is minutes.

---

## Two entry points, and they are not interchangeable

`configs/custom_eval/` holds ymls for both. Feeding one to the wrong script fails; both now
refuse by name and point at the other, but know the difference:

| script | config key | use when |
|---|---|---|
| `run_custom_eval.py` | `model:` — list of `"label=path"` strings | normal case, all models in one process |
| `run_multi_model_eval.py` | `models:` — list of `{label, checkpoint}` mappings | the process would exceed a time limit, or models have different chord vocabs |

`run_multi_model_eval.py` runs each model as its own short subprocess and merges by
`seq_idx` afterwards. Use it for Hooktheory with many models: 2,785 windows x 26 models in
one process is well past the 30-minute login-node GPU limit.

**Its staging dir comes from the config stem**, so give each study a unique stem
(`gapt_loo_hooktheory_multimodel`, not `hooktheory_multimodel`) or runs share a staging
directory and leave stale siblings behind.

`run_custom_eval.py` also takes bare preset names; `--list` shows them.

---

## The protocol flags that matter

Set these in the config. The defaults are **not** the protocol we use:

```yaml
cropped_max_len: 512   # tokens, NOT frames. 512 tokens = 256 frames/lane = 16 bars
all_windows: true      # every non-overlapping window of every song, not one random crop
seed: 42               # fixes which windows; identical across models => paired comparison
```

`max_len` counts interleaved tokens and the loader halves it. The old default of 256 tokens
was 8 bars per lane while training used 16 — a train/test mismatch.

With `all_windows: true` the window, not the song, is the unit of analysis.

`run_multi_model_eval.py` forwards these to its subprocesses; it used not to, and silently
ran the old protocol into a directory labelled otherwise.

---

## What each stage produces

| file | written by | contents |
|---|---|---|
| `models/<label>/preds.pt` | generation | generated tokens — **the expensive artifact** |
| `gt/gt.pt` | generation | ground-truth tokens for the same windows |
| `means.json` | generation | note-in-chord, chord complexity, sync EMD, chord duration entropy, … |
| `metadata.jsonl` | generation | per-window scores, with `song_id` for clustered bootstraps |
| `progression_scores.json` | `score_progressions.py` | Δroot and quality divergence, excess over floor |
| `progression_match_*.json` | `progression_match.py` | share of n-chord progressions occurring in training |

**Adding a metric does not require regenerating.** Anything computable from `preds.pt` or the
per-frame `.pt` files can be scored after the fact — that is why the progression metrics are
separate scripts and run in minutes. Only a change of window length, eval seed or checkpoint
forces a regeneration.

---

## Progression metrics: build the reference first

```bash
# once per corpus; cached in data/cache/progression_reference/ and progression_events/
python scripts/eval/custom_eval/build_progression_reference.py --datasets all
```

**Rebuild the references whenever the histogram set changes.** They store which histograms
exist, and a stale reference raises `KeyError` on a removed name.

`progression_match.py --quality full` keeps extensions distinct (C7 != C13); `--quality
function` collapses them. Both need the matching event cache, built on first use.

---

## Cluster practicalities

**The login-node GPU kills a process after 30 minutes.** Anything longer goes to SLURM
(`sbatch scripts/jobscripts/eval/run_custom_eval.sh <config>`) or is split with
`run_multi_model_eval.py`.

**Do not trust `nvidia-smi` for which GPU is usable.** It has reported `Default` and `0%`
utilisation for a device that then refused work, and `[N/A]` utilisation for a perfectly
usable one. Test instead:

```bash
for g in 0 1 2 3; do printf "GPU $g: "; CUDA_VISIBLE_DEVICES=$g python -c "
import torch; torch.zeros(8,device='cuda'); f,t=torch.cuda.mem_get_info()
print(f'usable, {f/2**30:.0f} GB free')" 2>/dev/null || echo unusable; done
```

**Detach anything long**, or it dies when you log out:

```bash
setsid nohup python ... > logs/custom_eval/<name>.log 2>&1 < /dev/null &
```

Check it detached with `ps -o pid,sid,tty -p <pid>` — you want `sid == pid` and `tty = ?`.
A job started from the editor's shell is a descendant of the SSH session and will not survive.

The `thes2192` account may only use `c23g` / `c23g_low`. There is no CPU partition, so even
CPU-only work either runs on the login node or takes a GPU node.

---

## Plots

```bash
python scripts/eval/custom_eval/plot_alpha_sweep.py --eval_dir logs/custom_eval/<study> \
    --metric nicr [--zoom] [--overlay GAPT=logs/custom_eval/<other>]
python scripts/eval/custom_eval/plot_loo_raw.py --metric chord_complexity_mean
python scripts/eval/custom_eval/plot_loo_raw.py --metric droot --source progression
```

Both write `.pdf` and `.png`. `plot_loo_raw.py` shows raw per-seed values against GT with no
normalisation — prefer it for anything going in the thesis.

---

## Generating the configs

```bash
python scripts/data/generate_loo_configs.py          # LOO training configs
python scripts/data/select_loo_checkpoints.py        # pick checkpoints -> eval configs
python scripts/data/generate_gapt_loo_configs.py     # RL configs from the MLE grid
```

`select_loo_checkpoints.py` picks the lowest **aggregate** val/loss checkpoint — in a
leave-one-out run the held-out corpus has no validation loss, so a per-corpus rule is
undefined there and would let the held-out corpus influence selection.

Runs are **not synced to wandb**; history is read from the local `.wandb` datastore, where
wandb >= 0.25 stores history under repeated `nested_key` fields rather than `key`. Also note
`trainer/global_step` is 0-indexed while checkpoints are named 1-indexed, so the validation
at step 999 belongs to `step=1000.ckpt`.
