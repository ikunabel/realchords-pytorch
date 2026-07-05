# Music tokenization — overview

Comparison of tokenization schemes relevant to this project: the repo's own grid-based
format, the miditok family, the Anticipatory Music Transformer control/event split, and
the Aria tokenizer.

---

## 1. ReaLchords grid tokenization (this repo)

**Source:** `realchords/dataset/hooktheory_tokenizer.py`

### Representation

Two parallel lanes, one token per 16th-note frame, interleaved:

```
[BOS, c₀, m₀, c₁, m₁, …, cₙ, mₙ, EOS, PAD…]
```

- **Grid resolution:** `FRAME_PER_BEAT = 4` → 16th note at any tempo (tempo not encoded)
- **Chord lane:** `CHORD_ON_{symbol}` (onset) / `CHORD_{symbol}` (hold)
- **Melody lane:** `NOTE_ON_{pitch}` (onset) / `NOTE_{pitch}` (hold) / `SILENCE`
- **Polyphony:** monophonic melody, one chord symbol per frame (no voicing)
- **Vocab size:** 2,962 tokens total

| Range | Tokens | Count |
|-------|--------|-------|
| PAD / BOS / EOS | special | 3 |
| SILENCE | melody silence | 1 |
| NOTE hold | 0–127 MIDI pitch | 128 |
| NOTE_ON | 0–127 MIDI pitch | 128 |
| CHORD hold | 1,351 symbols | 1,351 |
| CHORD_ON | 1,351 symbols | 1,351 |

### Benefits
- Extremely compact: 2 tokens per 16th note regardless of note density
- Explicit, readable chord symbols at every frame → interpretable UI and clean metrics
- Aligned lanes make RL reward definition trivial (one decision per frame)
- One token per lane per step → small model, fast inference, real-time on CPU
- Directly comparable to lead-sheet annotations (Hooktheory GT labels)

### Drawbacks
- Melody forced monophonic; polyphonic piano cannot be directly represented
- Chord symbols lose voicing, inversion (inversion dropped at training time), and register
- Tempo not encoded → no rhythm or swing feel; quantization artefacts at non-16th note timings
- Chord vocabulary closed: symbols outside `chord_names.json` are OOV and dropped
- Not suitable for raw MIDI without a melody/chord separation step

---

## 2. miditok formats

**Library:** [miditok](https://github.com/Natooz/MidiTok)  
**Paper refs:** REMI (Huang & Yang 2020), Compound Word (Hsiao et al. 2021), others

miditok provides a common API over several tokenization strategies. All operate on
multi-track MIDI and produce flat token sequences.

### 2a. REMI (Revamped MIDI Representation)

Bar, Position, Chord (optional), Tempo, Pitch, Velocity, Duration tokens per event.
Time is encoded as a position within a bar (subdivided by a fixed number of beats).

```
BAR  POSITION_4  TEMPO_120  PITCH_60  VELOCITY_80  DURATION_8
```

- **Resolution:** configurable position bins per bar (typically 16–32)
- **Polyphony:** full (multiple Pitch/Velocity/Duration tokens per position)
- **Vocab:** ~300–500 tokens depending on config

**Benefits:** Explicitly encodes bar structure; easy to reason about rhythm.  
**Drawbacks:** Bar token resets make long-range dependencies harder; tempo and chord are
optional extras not in the original formulation; position is relative to bar, not absolute.

### 2b. MIDI-Like

Note-On, Note-Off, Time-Shift, Velocity tokens — closest to the raw MIDI event stream.

```
NOTE_ON_60  VELOCITY_80  TIME_SHIFT_100ms  NOTE_OFF_60
```

- **Resolution:** time-shift token granularity (e.g. 10 ms steps → 100 tokens for 1 s)
- **Polyphony:** full
- **Vocab:** ~200–400 tokens

**Benefits:** Faithful to MIDI; minimal information loss.  
**Drawbacks:** Very long sequences for dense piano; Note-On/Off pairing creates implicit
structure that the model must learn; no explicit harmonic abstraction.

### 2c. Compound Word (CP)

Each timestep emits one "compound" token that bundles multiple attributes (family, pitch,
duration, velocity, position) as a tuple. The model predicts each attribute in the tuple
sequentially or jointly.

- **Polyphony:** full
- **Vocab:** per-attribute sub-vocabularies, combined at prediction time

**Benefits:** More information per AR step → shorter sequences than MIDI-Like; attributes
are explicitly factored.  
**Drawbacks:** Requires a modified model head (multi-task prediction per step); harder to
plug into standard transformer decoders.

### 2d. TSD / Structured / other variants

Variations on MIDI-Like with different time encoding (Time-Shift + Duration) or
structured note groups. Less commonly cited; trade-offs are similar to MIDI-Like.

### General miditok notes

- All formats handle multi-track MIDI natively (track/program tokens)
- Chord annotation is optional and requires an external chord extraction step for MIDI
  without explicit chord tracks
- All are data-agnostic: work on classical, jazz, pop, etc. without a closed vocabulary
- Sequences are significantly longer than grid-based formats for polyphonic content

---

## 3. Anticipatory Music Transformer (AMT) — control/event split

**Paper:** Thickstun et al. 2023 — *Anticipatory Music Transformer*  
**Repo:** [jthickstun/anticipation](https://github.com/jthickstun/anticipation)  
**Training data:** Lakh MIDI Dataset

### Representation

Tokens are split into two interleaved streams:

- **Event tokens:** actual MIDI notes (pitch, instrument, onset, duration)
- **Control tokens:** metadata that governs what comes next (timing, style, sometimes harmony)

The key idea is that at inference time, future control tokens can be *prepended* to the
current context window before generating the next event token — the model "anticipates"
what is about to happen. This is how jam_bot achieves real-time accompaniment: the human
MIDI stream is treated as control tokens that are inserted into the context ahead of the
AI generation steps.

```
[CONTROL: onset=1.0, pitch=60]  [EVENT: onset=1.0, pitch=72, dur=0.5]  …
```

- **Resolution:** continuous onset/duration (quantized to ~10 ms or similar)
- **Polyphony:** full (multi-instrument)
- **Vocab:** ~3,000–6,000 tokens depending on config

### Benefits
- The control/event factoring directly enables real-time anticipatory generation without
  separate training regimes
- Same vocabulary for human input and AI output → no modality gap
- Full performance-level expressiveness (dynamics, timing, multi-track)

### Drawbacks
- Harmony is implicit in note events, not explicit — no chord-label layer
- Longer sequences than grid formats; real-time requires the dedicated anticipation
  engineering (systems-heavy)
- Control/event interleaving requires careful training data construction
- The "melody" vs "accompaniment" split is a learned convention, not enforced by the format

---

## 4. Aria tokenizer (EleutherAI/aria)

**Paper:** Bradshaw et al. 2025 — *Ghost in the Keys / Aria-Duet* (ISMIR 2025)  
**Repo:** [EleutherAI/aria](https://github.com/EleutherAI/aria)  
**Training data:** Aria-MIDI (~100k+ hours solo piano)

### Representation

Custom tokenizer designed for solo piano transcriptions. Encodes MIDI events as:

- **Note-on** with pitch and velocity bucket
- **Duration** (quantized, typically 25 ms steps)
- **Time-delta** tokens (~10 ms resolution) between events
- Instrument/program tokens for the piano track
- Special segment and padding tokens

```
<NOTE_ON pitch=60 vel=4>  <DURATION_200ms>  <TIME_DELTA_50ms>  <NOTE_ON pitch=64 vel=3>  …
```

- **Resolution:** ~10 ms time-delta; ~25 ms duration steps
- **Polyphony:** full (overlapping notes handled naturally via time-delta)
- **Vocab:** ~3,000–5,000 tokens (piano-focused; no multi-instrument in base version)
- **Sequence length:** 1B parameter model; typical context ~8k–16k tokens covering
  several minutes of piano music

### Benefits
- Designed for and trained on the exact data you want to use (Aria-MIDI)
- High temporal resolution captures timing nuance and expressive performance
- Handles the polyphony of classical and jazz piano naturally
- Public weights on Hugging Face

### Drawbacks
- Piano-only in the base version; multi-instrument requires extension
- 1B parameters → not CPU-viable for real-time; requires Apple Silicon (MLX) or GPU
- No explicit chord lane → harmony must be inferred from note patterns
- Embedding space biased toward the solo piano distribution of Aria-MIDI

---

## Comparison summary

| Property | ReaLchords grid | REMI | MIDI-Like | AMT control/event | Aria |
|----------|----------------|------|-----------|-------------------|------|
| **Resolution** | 16th note (fixed) | Configurable bars/beats | ~10 ms | ~10 ms | ~10 ms |
| **Polyphony** | Mono melody + 1 chord | Full | Full | Full | Full (piano) |
| **Harmony** | Explicit symbol | Optional extracted | Implicit | Implicit | Implicit |
| **Vocab size** | ~3k | ~300–500 | ~200–400 | ~3–6k | ~3–5k |
| **Tokens / bar (4/4)** | 8 (2 lanes × 4) | ~30–80 | ~50–200+ | ~50–200+ | ~100–300+ |
| **Real-time feasibility** | Native (small model) | Needs engineering | Needs engineering | Native (anticipation) | MLX / GPU only |
| **Interpretability** | High (chord symbols) | Medium (bar/pos) | Low | Low | Low |
| **Multi-instrument** | No (2 lanes) | Yes | Yes | Yes | Piano only |
| **Tempo encoded** | No | Yes | Yes | Yes | Yes |
| **Public tooling** | This repo | miditok | miditok | anticipation repo | EleutherAI/aria |

---

## Relevance to ARIA-MIDI conversion

To convert ARIA-MIDI jazz/pop/funk to the ReaLchords grid format you need two steps that
none of the above formats provide out of the box:

1. **Melody extraction** from polyphonic piano (skyline or learned separator)
2. **Chord recognition** per 16th-note frame (music21 `chordify` or learned model)

The miditok formats are useful as an intermediate representation for analysis (t-SNE
coverage plots, genre comparison) because they provide a common, model-free vocabulary
across datasets. They are not a drop-in replacement for the grid format in the
ReaLchords training pipeline.
