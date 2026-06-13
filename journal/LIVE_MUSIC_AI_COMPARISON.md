# Live music AI: ReaLchords / RealJam vs jam_bot vs Aria-Duet

Comparison of three real-time human–AI music systems: task, data, representation,
time encoding, inference, and error behavior. Based on public papers, repos, and this
codebase (March 2026).

| Project | Primary papers / artifacts | Public code |
|---------|---------------------------|-------------|
| **ReaLchords / RealJam** | Scarlatos et al. (ReaLJam, ReaLchords, GAPT) | **Full** — this repo, `pip install realjam` |
| **jam_bot** | Blanchard et al., ISMIR 2025 | **Partial** — no jam_bot stack; related [`anticipation`](https://github.com/lancelotblanchard/anticipation) fork |
| **Aria-Duet** (“Ghost in the Keys”) | Bradshaw et al., ISMIR 2025 + NeurIPS Creative AI 2025 | **Partial** — [`EleutherAI/aria`](https://github.com/EleutherAI/aria) (model + demo; Disklavier optional) |

---

## Task summary

| | **RealJam** | **jam_bot** | **Aria-Duet** |
|--|-------------|-------------|---------------|
| **Job** | **Accompany** live melody with **chord symbols** | **Improvise** with a virtuoso (lead / comp / call-response) | **Continue** solo piano after **turn handover** |
| **Human plays** | Monophonic melody (keyboard / UI) | Full keyboard MIDI | Full piano (Disklavier) |
| **AI outputs** | Chord labels → rule-based MIDI voicing | Symbolic performance MIDI | Symbolic performance MIDI |

---

## Time: fixed grid vs event stream

This difference drives much of the confusion around “infilling.”

### RealJam — harmonic grid

- `FRAME_PER_BEAT = 4` → in 4/4, **1 frame = one 16th note = 1/16 bar**.
- Melody and chord each have **one token per frame**, interleaved in the AR sequence:

```text
…  m₀   c₀   m₁   c₁   m₂   c₂  …
    └─ frame 0 ─┘ └─ frame 1 ─┘
    same 16th      +1/16 bar
```

- `m₀ → c₀`: **same** time step (melody lane vs chord lane).
- `c₀ → m₁`: **+1/16 bar**.
- Wall-clock time is **implicit in the frame index**, not encoded inside each token.

### jam_bot / Aria — performance events

- Tokens are **MIDI events** (pitch, onset, duration, velocity, pedal, …).
- **Time is inside the encoding** (e.g. ~10 ms steps in Aria; arrival-time style in AMT).
- **Next token ≠ +1/16 bar**: several tokens can fall in one 16th; gaps can span many 16ths.
- No fixed “one harmony decision per 16th” column.

| | **RealJam** | **jam_bot / Aria** |
|--|-------------|-------------------|
| **Clock** | Fixed **16th-note grid** | **Event-relative** time in tokens |
| **AR steps vs bar** | 2 tokens per 16th (melody + chord) | Variable tokens per 16th |
| **Harmony** | Explicit **one symbol per frame** | Implicit in note patterns |

---

## Co-design: data × model × interaction

| Axis | **RealJam** | **jam_bot** | **Aria-Duet** |
|------|-------------|-------------|---------------|
| **Data** | Hooktheory-style **melody + chord symbols** (+ POP909, etc.) | **Lakh MIDI** pretrain + **artist SFT** (Rudess) | **~100k h** solo-piano transcriptions (Aria-MIDI) |
| **Target** | Harmonic **labels** (lead sheet) | **Performance** notes | **Performance** notes |
| **Training** | MLE → **RL** (+ anchor; GAPT adds GAIL) | MLE / **SFT** | MLE pretrain + gen **SFT**; no public RL |
| **Interaction** | **Continuous** accompaniment (lookahead / commit) | **Continuous** free improv | **Turn-taking** (pedal / reset) |
| **Live inference** | Small model; **1 chord token / frame**; KV cache; ONNX | AMT-style streaming + systems engineering | **1B** model; MLX demo |

---

## Representation

| | **RealJam** | **jam_bot (AMT)** | **Aria** |
|--|-------------|-------------------|----------|
| **Harmony** | **`CHORD_{symbol}`** (~2.8k vocab) | Implicit in notes | Implicit in notes |
| **Melody** | **`NOTE_*`** per frame (mono) | Full MIDI (control stream) | Full MIDI |
| **Voicing** | Fixed decode (`CHORD_OCTAVE`, note_seq) | In token stream | In token stream |
| **Interpretability** | High for **chord symbols** | High for **sound**; low for chord labels | Same as jam_bot |

Hooktheory → symbol → token → RealJam MIDI: see [`CHORD_VOICING_FINDINGS.md`](CHORD_VOICING_FINDINGS.md).

---

## Interpretability, metrics, and educational use

RealJam / ReaLchords is built around **lead-sheet harmony**, not concert-grade piano performance. That shapes what you can see, measure, and learn from.

### Why it is more interpretable

| | **RealJam** | **jam_bot / Aria** |
|--|-------------|-------------------|
| **Harmony in the UI / logs** | Explicit **`CHORD_{symbol}`** per frame (e.g. `Am7`, `G`) | Harmony only as **note patterns** — must infer or transcribe |
| **What the AI decided** | One **named chord** per 16th column | Many MIDI events; no separate “chord decision” |
| **Wrong output** | Wrong **symbol** — readable and discussable | Wrong **notes** — heard, harder to label as a harmonic mistake |
| **Human input role** | **Monophonic melody** — input lane, not a performance target | **Full piano** — virtuoso duet / continuation quality |

The fixed grid makes the session a **harmonic spreadsheet**: each row is a 16th, columns are melody vs chord. That is closer to TheoryTab / lead-sheet thinking than to a dense performance transcript.

### Why metrics are easier

- **Aligned time:** one melody token + one chord token per frame → frame *t* in eval = frame *t* in live, no event-to-grid alignment step.
- **Discrete targets:** symbol accuracy, progression stats, note-in-chord ratio, repetition penalties — all defined on **token IDs** or decoded symbols (see `eval_utils.py`, contrastive reward models).
- **Reference labels:** Hooktheory-style annotations give GT chord symbols on the same grid; jam_bot / Aria improv and continuation have no parallel **symbol layer** to score against.

Performance LMs need extra extraction (chord recognition, onset detection) before you can talk about harmony quantitatively.

### Educational framing (not performance-grade piano)

RealJam treats **live keyboard input as a melody probe**, not as something that must sound like a polished performance:

- The user plays **one line** to explore “what harmony fits this?” — like humming over changes, not like a Disklavier duet.
- The AI’s job is **visible chord symbols** the user can read, compare to theory, and log — harmony as the **object of study**.
- Voicing is a **decode rule** (`CHORD_OCTAVE`, note_seq), not the learned target — appropriate when the goal is **understanding progressions**, not reproducing Rubinstein voicings.

jam_bot and Aria optimize **expressive joint performance** (virtuoso improv, solo continuation). RealJam optimizes **interpretable accompaniment behind a simple line** — a different product and a different learning loop.

---

## Inference: conditional generation vs AMT control/event

Both **paste** some AR steps and **sample** others. That is conditional / online generation.

| | **RealJam** | **AMT / jam_bot** |
|--|-------------|-------------------|
| **Pasted** | **Melody** tokens (human, per frame) | **Human MIDI** (control events) |
| **Sampled** | **Chord symbol** tokens (whole harmony lane) | **Generated MIDI** events (same vocab as control) |
| **Training for this layout** | Joint MLE on GT melody+chord; paste melody only at **inference / RL** | Trained on Lakh **control + completion** layouts |

RealJam is **not** BERT-style masked LM training. AMT is **not** a fixed melody/chord grid.

---

## Error accumulation and exposure bias

### Training vs inference (read this first)

These are different regimes. Mixing them up is what makes “human stays GT” confusing.

| | **Training (MLE)** | **Inference / live** |
|--|-------------------|----------------------|
| **Human?** | **No** — fixed dataset sequences | **Yes** — live player |
| **Ground truth?** | **Yes** — annotated melody + chords (RealJam) or MIDI files (Aria/jam_bot) | **No** for model outputs; chords/AI notes are **generated** |
| **Past tokens in context** | **All from dataset** (teacher forcing) | **Mix:** external input + **model’s own past samples** |
| **Exposure bias active?** | **No** — model never conditions on its own wrong chord/note history | **Yes** — core live problem |

**Exposure bias** = train on `P(yₜ | y_{<t})` with **dataset** `y_{<t}`, but deploy with **model-generated** `y_{<t}`.

**OOD (input)** = live human playing ≠ training distribution (rubato, wrong notes, style). Separate from exposure bias; all three can suffer it.

**RL rollouts (RealJam only):** melody usually **fixed from the batch** (dataset annotation); **chords sampled from the policy** — deliberately mimics the live asymmetry during training.

---

### Why RealJam compounding is worse at **inference** (not because training has a “human GT lane”)

At live time the question is: **which parts of the context come from the player vs from the model?**

| | **RealJam (live)** | **jam_bot (live)** | **Aria-Duet (live)** |
|--|-------------------|--------------------|-----------------------|
| **Externally supplied** (player, not model) | **Melody tokens** only — pasted each frame from live input | **Control MIDI** — full keyboard stream | **Human piano** until handover |
| **Model-generated and fed back** | **Entire chord lane** — all past `CHORD_*` tokens | **Generated notes** only | **Generated notes** during AI turn |
| **Separate harmony from player?** | **Yes** — user never sends chord labels | **No** — same event vocabulary | **No** |

“Externally supplied” is **not** ground truth — there is no correct live label. It means **the model did not write that part of the context**, so player mistakes cause **input OOD**, not **self-conditioning on wrong symbols**.

RealJam interleaved context when predicting the next chord:

```text
…  m₀  ĉ₀  m₁  ĉ₁  …  mₜ  → predict cₜ
     ↑   ↑       ↑
  player model   both in KV cache; ĉ’s are self-conditioning if wrong
```

- **m** slots: refreshed from **live melody** (external).
- **c** slots: **only** from past model samples — errors **accumulate in the harmony lane**.

jam_bot: **control MIDI** stays external; wrong tokens concentrate in the **generated** substream.

**Takeaway:** RealJam is a **self-conditioned harmony loop** at inference. jam_bot/Aria keep **player performance** as the stable external stream; they still have exposure bias on **generated** notes, but not on a **self-fed symbol layer**.

### Mechanism 1 — What gets re-ingested (inference only)

See table above. At **MLE training**, both melody and chord lanes are **dataset annotations** — no live player, no self-fed chord mistakes.

### Mechanism 2 — Session length and reset

| | **RealJam** | **jam_bot** | **Aria-Duet** |
|--|-------------|-------------|---------------|
| **Mode** | Continuous accompany for the whole session | Continuous improv | **Turn-based** |
| **Clear bad chord context mid-piece?** | **No** (intro / new session only) | Partial (roles, scheduling) | **Yes** (handover / reset) |
| **Wrong tokens in context** | Can persist for **minutes** | Can persist in generated stream | **Truncated** each AI turn |

Aria cuts error chains by design. RealJam **cannot** reset harmony without breaking “one progression through the tune.”

### Mechanism 3 — Severity of one wrong token

| | **RealJam** | **jam_bot / Aria** |
|--|-------------|-------------------|
| **One wrong token** | Wrong **chord symbol** (e.g. `A7` vs `Am7`) | Wrong **note** (pitch / time / velocity) |
| **Musical scope** | Wrong **harmonic function** for all frames until next change | Often **local** — passing tone, slight timing |
| **Stays identical in context?** | **`CHORD_X` hold tokens repeat** the same wrong label frame after frame | Note errors differ; dense stream mixes many small states |
| **Offline eval has reference labels?** | **Yes** — symbol accuracy vs annotations | Improv / continuation — softer “correctness” |

RealJam: **one symbol error = a discrete wrong harmonic state** repeated on the grid until the next change.

jam_bot / Aria: **more tokens per second** (more chances to err), but each error is usually **smaller** and there is **no separate self-fed symbol layer**.

### Why RealJam uses RL (and the others often do not)

| Issue | RealJam response | jam_bot / Aria response |
|-------|------------------|-------------------------|
| Self-fed wrong **chords** | **RL** on rollouts + reward models + anchor | N/A — no chord lane |
| Long continuous loop | RL + lookahead/commit | Control stream + (Aria) **reset** |
| Train never sees own mistakes | On-policy RL | Large pretrain + SFT; Aria **resets** |

RL targets **plausible chord sequences under the model’s own past outputs**. That problem is **central** to RealJam and **peripheral or absent** in the other two stacks.

### What does *not* explain the gap

- **Token density alone:** dense MIDI → more AR steps, not fewer exposure-bias problems in principle.
- **Infilling at inference:** all three paste conditions; the gap is **what** is pasted (mono melody vs full MIDI) and **what** is self-generated (symbols vs notes).
- **“jam_bot avoids exposure bias”:** it **reduces harm** via control grounding + softer errors + task; it does **not** remove AR bias on generated tokens.

---

## Open reproducibility

| | **RealJam** | **jam_bot** | **Aria** |
|--|-------------|-------------|----------|
| **Train → eval → live in repo** | Yes | No (concert stack closed) | Model + demo; Disklavier optional |
| **Weights public** | Yes | No | Hugging Face |
| **Training data public** | Hooktheory + converters | Lakh public; Rudess FT private | Aria-MIDI |
| **Modest CPU live demo** | Yes (small decoder, symbol/frame) | Concert-grade engineering | Real-time demo MLX / Apple Silicon; 1B |

---

## Advantages and disadvantages

### ReaLchords / RealJam

**Advantages:** Visible **chord symbols** on a fixed grid (interpretable live output); straightforward **symbol- and frame-level metrics**; suited to **harmony learning** (melody as probe, not performance-grade piano); open MLE/RL/GAPT stack; efficient chord-token live path; voicing tunable at decode without retraining symbols.

**Disadvantages:** No voicing in labels; chord-lane exposure bias; heavy RL pipeline; pop/rock data bias; melody-only live input vs training; not a piano-performance duet.

### jam_bot

**Advantages:** Expressive MIDI output; human control stream stays grounded; artist SFT for target style; AMT infilling architecture for accompaniment-style control.

**Disadvantages:** Stack not public; harmony not explicit; systems-heavy; no published chord-level RL story.

### Aria-Duet

**Advantages:** Public model, data, weights; large piano pretrain; turn-taking + reset; strong continuation with good prompts.

**Disadvantages:** Different task (continuation, not melody→chords); 1B note LM; MLX-focused real-time demo; prompt-sensitive; memorization risk on common repertoire.

---

## When each stack fits

| Goal | Fit |
|------|-----|
| Live **chord accompaniment** behind melody, **readable symbols**, grid metrics, **harmony education** | **RealJam** |
| **Free improv duet**, virtuoso style, full MIDI | **jam_bot** (reference; closed) |
| **Turn-taking piano continuation**, public foundation model | **Aria-Duet** |
| Voicing / voice-leading **in the training target** | Performance LMs (Aria, jam_bot, PiJAMA) — not Hooktheory symbols |

---

## References

- ReaLJam / ReaLchords / GAPT — Scarlatos et al. ([`README.md`](../README.md))
- jam_bot — Blanchard, Naseck, et al., ISMIR 2025; MIT “Symbiotic Virtuosity” (2024)
- Anticipatory Music Transformer — Thickstun et al. (Lakh MIDI)
- Aria — Bradshaw et al., [arxiv:2506.23869](https://arxiv.org/abs/2506.23869); [EleutherAI/aria](https://github.com/EleutherAI/aria)
- Ghost in the Keys — [arxiv:2511.01663](https://arxiv.org/html/2511.01663)
