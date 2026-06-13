# Chord representation and voicing — dataset findings

This document summarizes what we learned by exploring Hooktheory data in
`dataset_experiments/scripts/`. It answers: **what chord information exists at each
stage**, and **whether you can reconstruct how chords sound when played**.

For how to regenerate the JSON outputs, see [`output/README.md`](output/README.md).

---

## What “voicing reconstruction” means

**Voicing** = which pitch classes play, in **which octaves**, and how they are **spaced**
(open vs closed, spread across the keyboard).

**Voicing reconstruction** = starting from dataset files alone, recover the **exact MIDI
note list** (or spacing) you would hear on TheoryTab or in RealJam.

We use this term in `explore_raw_chords.py` (`voicing_reconstruction` in the JSON output)
to mean: *inventory what is and is not available for that task.*

**Finding:** you can reconstruct **harmonic identity** (which chord, which inversion) from
raw or processed Hooktheory. You **cannot** reconstruct **TheoryTab playback voicing**
from either dump — and RealJam applies yet another fixed playback rule at inference.

---

## The compression ladder

Each step throws away information the previous step might have implied:

```
Hookpad raw (Hooktheory_Raw.json)
  root (scale degree) + type + inversion + extensions
        ↓  external conversion (Sheet Sage / Hooktheory release; not in this repo)
Hooktheory.json.gz  annotations.harmony
  root_pitch_class + root_position_intervals + inversion   ← tertian stack encoding
        ↓  to_chord_name (inversion omitted in chord_to_frames)
Training tokens  CHORD_ON_{symbol} / CHORD_{symbol}
        ↓  decode_chord_token + note_seq + fixed octaves
RealJam MIDI  chord tones @ octave 4, bass @ octave 3
```

| Stage | What you keep | What you lose |
|-------|----------------|---------------|
| Raw → processed | Pitch classes, tertian interval stack, inversion | Scale-degree view; Hookpad `type` enum |
| Processed → tokens | Chord symbol (~1.3k–2.8k names) | Inversion; interval stack |
| Tokens → MIDI | Pitch-class set from note_seq | Register/spacing; Hooktheory voicing |

---

## Layer 1: Hookpad raw (`Hooktheory_Raw.json`)

**Script:** `explore_raw_chords.py` → `output/raw_chords/raw_chords_overview.json`

Each chord in `json.chords[]` is **symbolic**:

| Field | Meaning |
|-------|---------|
| `root` | Scale degree 1–7 in the current key (needs `json.keys` to get pitch class) |
| `type` | Hookpad quality enum (5 values in our corpus: 5, 7, 9, 11, 13 — triad through 13th shell) |
| `inversion` | 0 = root position, 1–3 = figured-bass style inversion |
| `applied`, `adds`, `omits`, … | Extensions (borrowed, suspensions, etc.) |
| `beat`, `duration` | Timing only |

**Not in chord objects:**

- Per-chord MIDI pitches or octave placement
- Open vs closed spacing
- Hookpad Band “octave centering” or voicing preset as a note list

**Elsewhere in raw JSON:**

- `json.notes` — **melody only** (`sd` + `octave`), not chord tones
- `json.bands[].harmony` / `bass` — instrument **style names** (e.g. `"Piano 1/4s"`), not pitches

**Counts (cache-eligible songs, MELODY + HARMONY, no TEMPO_CHANGES):**

- 23,462 songs; 383,985 chord events (matches processed 1:1)
- 3,606 unique raw chord shapes
- 56,108 inverted events; inversions 0–3

---

## Layer 2: Processed Hooktheory (`Hooktheory.json.gz`)

**Script:** `explore_chords.py` → `output/chords/chords_overview.json`

`annotations.harmony[]`:

```json
{
  "onset": 0,
  "offset": 4,
  "root_pitch_class": 7,
  "root_position_intervals": [4, 3],
  "inversion": 0
}
```

- **Pitch count** = `1 + len(root_position_intervals)` (1–7 in our corpus)
- Intervals are **stacked semitones** from the root — often chains of 3s and 4s (tertian
  spelling), but not exclusively
- **Inversion is stored** but **dropped** when building training frames (`chord_to_frames`
  calls `to_chord_name` without inversion)

**Counts:**

- 2,272 unique structural chords `(root, intervals, inversion)`
- Same 383,985 harmony events as raw

The raw → processed conversion script is **not in this repository** (published with
Sheet Sage / Hooktheory dataset). `scripts/convert_hooktheory_to_cache.py` only filters
and writes cache JSONL — it does not change harmony representation.

---

## Layer 3: Chord symbols and tokens

**Scripts:** `explore_chord_midi_coverage.py`, training code in `hooktheory_tokenizer.py`

- `to_chord_name(root, intervals)` → note_seq `pitches_to_chord_symbol` → e.g. `"G"`,
  `"Gb11"`, `"Fped"`
- Token IDs from `CHORD_ON_{name}` / `CHORD_{name}` in `chord_names.json`
- **Eval metrics operate here** — predicted symbol vs ground-truth symbol, not MIDI voicing

---

## Layer 4: RealJam / playtime MIDI

**Script:** `explore_chord_midi_coverage.py` → `output/chord_midi_coverage/chord_midi_coverage.json`

From `decode_chord_token` (`realjam/agent_interface.py`):

1. `chord_symbol_pitches(symbol)` → pitch classes 0–11
2. `chord_symbol_bass(symbol)` → bass pitch class (root for plain chords; slash bass when
   symbol has `/`)
3. Fixed octaves: `CHORD_OCTAVE = 4` (MIDI 48–59), `BASS_OCTAVE = 3` (MIDI 36–47)

**Playback rules:**

- Every chord tone goes to the **same** chord octave block
- A **separate bass note** is always added one octave lower
- Root-position triads **duplicate the root** (e.g. G at 55 and 43)

**Piano coverage:** MIDI 36–59 only (24 keys, all below middle C) — 27% of an 88-key
piano. This is a **project convention**, not Hooktheory annotation.

**Pitch-class collapse:** Hooktheory can stack tones across multiple octaves in absolute
pitch (e.g. F# with six stacked intervals). The symbol round-trip (`pitches_to_chord_symbol`
→ `chord_symbol_pitches`) reduces to pitch classes mod 12; RealJam then maps each class to
one fixed octave. Wide Hooktheory stacks become **dense clusters** at playtime.

---

## Author confirmation (Sheet Sage / Hooktheory conversion)

Email from the author of the raw → Hooktheory conversion (paraphrased):

- There is **no explicit voicing** for chords in the dataset aside from **inversion**
- TheoryTab rendered voicing is a **sophisticated default** in Hookpad’s player, not stored
  annotations
- The **tertian interval stack** in `Hooktheory.json.gz` was chosen for **simplicity**
- Sanity check: the same chord information should produce the **same voicing** on TheoryTab
  across tracks — consistent with voicing being a deterministic render function

This matches our exploration of raw JSON, processed harmony, and RealJam decode.

---

## Practical implications for ReaLchords

1. **Training and eval** — Models learn and are scored on **chord symbols (token IDs)**.
   Inversion and register are largely absent from the training target.

2. **TheoryTab sound ≠ training target ≠ RealJam sound** — Three different “playback”
   layers; only harmonic symbol is shared.

3. **Do not expect generated MIDI to match Hooktheory voicing** — Even with a perfect
   symbol prediction, RealJam uses fixed octaves + note_seq pitch classes, not Hookpad
   Band settings.

4. **Inversion in raw/processed is informative but unused in `chord_to_frames`** — Recovering
   slash-bass or inverted bass from Hooktheory would require passing inversion into
   `to_chord_name` or a separate bass model.

---

## Scripts and outputs (quick reference)

| Question | Script | Output |
|----------|--------|--------|
| Hookpad-native chords + voicing inventory | `explore_raw_chords.py` | `output/raw_chords/` |
| Processed interval chords + pitch counts | `explore_chords.py` | `output/chords/` |
| Symbol → token → MIDI pipeline | `explore_chord_midi_coverage.py` | `output/chord_midi_coverage/` |
| One raw song side-by-side | `explore_raw_hookpad.py` | `output/raw_json/` |
| One cache song + tokens | `explore_hooktheory.py` | `output/hooktheory/` |

Example voicing discrepancy (wide Hooktheory stack vs dense MIDI): see
`output/chord_midi_coverage/example_voicing_discrepancy.json` if present, or any
`Gb11` entry in `chord_midi_coverage.json` with `hooktheory_pitch_count: 6`.
