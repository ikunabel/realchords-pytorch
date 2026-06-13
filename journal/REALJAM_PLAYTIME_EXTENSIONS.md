# RealJam play-time extensions — structure & user harmony

Concise notes on **inference-time** ideas for adding lead-sheet-like structure and
user-played chords to RealJam. No retrain required for a first version; finetune/RL
optional later.

Related: [`LIVE_MUSIC_AI_COMPARISON.md`](LIVE_MUSIC_AI_COMPARISON.md),
[`CHORD_VOICING_FINDINGS.md`](CHORD_VOICING_FINDINGS.md).

---

## Current RealJam (baseline)

| Mechanism | Role |
|-----------|------|
| Live **melody** | Pasted into interleaved context each frame |
| Live **chords** | **Model-only** — sampled on chord frames |
| **Intro chords** | One-shot offline model seed before online gen |
| **Lookahead / commit** | Buffer stability — not musical form |
| **Structure** | No user key, chart, or sections in the loop |

Training data (Hooktheory `annotations.keys`, etc.) has structure; **inference does not expose it**.

---

## 1. Agreed structure at play-time (lead-sheet paradigm)

Real musicians agree on **key**, **changes**, and often **form** before improvising.
RealJam can stay symbol-based and add that as **session setup + constraints**, not a new token representation.

### User-facing controls (examples)

- **Key / mode** — e.g. G major  
- **Chart** — progression per section (`Verse: I–V–vi–IV`, or absolute symbols)  
- **Section marker** — button / pedal / bar count → “now chorus”  

### Inference-time use (no retrain)

| Lever | Effect |
|-------|--------|
| **Logit bias / mask** | Restrict or nudge `CHORD_*` toward chart in current key |
| **Snap on commit** | In commit window, align to nearest chart symbol |
| **Section chord reset** | On section change, clear or replace **chord** KV / history with chart tokens (melody context kept) — partial reset, not full Aria handover |

### Heavier options (later)

- **Load Hooktheory/cache progression** for a chosen song; scroll by BPM/bar  
- **Finetune** with `KEY_*` / `SECTION_*` prefix tokens if the model should *prefer* form without hard masks  

### Limits

- Constrains **harmony**, not melody — user can still play anything monophonically  
- Hard masks can fight melody-strong implications → prefer soft bias first  
- Modulations need explicit key-change events (Hooktheory has ~496 songs with key changes in cache)

---

## 2. User-played chords (paste like melody)

**Idea:** On chord frames, if the user played a chord → **paste** token into context; else **sample** model — same scheduling as melody `conditions` in `generate_online`.

```text
melody step  → paste if user note active
chord step   → paste if user chord active, else model sample
```

### Why it fits

- Same **interleaved prefix** mechanism as live melody  
- **Lead-sheet jam:** user holds `Am7`, AI comps around it or fills gaps  
- **Exposure bias:** user-set symbols **replace** self-fed wrong `CHORD_*` on those frames  

### Implementation (inference)

1. Per frame: active **polyphonic** MIDI → pitch-class set  
2. `note_seq` `pitches_to_chord_symbol` → string  
3. Map to `CHORD_ON_*` / `CHORD_*` via `chord_names.json` (+ simplify / nearest if OOV)  
4. Hook `generate_live` / `generate_online`: paste on chord steps when user chord present  
5. **Commit / lookahead:** do not overwrite user-locked frames  
6. Optional mode: auto / user-led / mixed  

### Training note

MLE today: chord lane is always **dataset GT** (train) or **model** (live). Mixed human chords at inference is **distribution shift** unless finetuned or covered by RL with overrides — often still works as a **strong hint**; continuation after user chords is the quality risk.

---

## 3. MIDI → symbol → vocab

Reverse of `decode_chord_token` (symbol → pitches):

```text
MIDI pitches → PCs mod 12 → pitches_to_chord_symbol → vocab lookup → token ID
```

### Works well when

- Clean **triad or 7th**, root position, simple grip (≤7 notes)  

### Not “always in vocab”

| Issue | Note |
|-------|------|
| OOV spelling | Rare symbols / aliases not in `chord_names.json` |
| Slash / inversion | e.g. `C/E` — may be missing from vocab |
| Ambiguous PCs | One pitch set → several symbols; note_seq picks one |
| Extensions / omissions | add9, sus, no5 → exotic or parse failure |
| `ChordSymbolError` | Need fallback: simplify, nearest in-vocab, or `SILENCE` |

Reuse patterns from Wikifonia `simplify_complex_chord` and online **`filter_invalid_tokens`** for generated *and* user-parsed symbols.

**Voicing:** user plays voiced MIDI; **token is symbol only** — playback still uses fixed RealJam decode rules.

---

## 4. Evaluation via session export

User steering does not block metrics if sessions are **logged and scored offline**.

### Log per frame (suggested)

- `frame`, `melody_token`, `chord_token`  
- **`chord_source`:** `model` | `user` | `silence`  
- Optional: raw user chord MIDI pitches, parsed symbol string  

### Offline metrics

- Symbol accuracy vs chart / Hooktheory GT where available  
- **Model-only subset** — frames where `chord_source == model`  
- Progression / repetition / rhythm rewards — same tools as batch eval, scoped to AI-filled regions  

User chords define **when the model is accountable**, not whether the session is analyzable.

---

## 5. Phased roadmap

| Phase | Work | Retrain? |
|-------|------|----------|
| **A** | UI: key, optional chart, section button; logit mask / snap; section chord reset | No |
| **B** | User chord lane: MIDI→symbol→paste; commit respects user locks; session export | No |
| **C** | Load cache progression for named song; bar/BPM sync | No |
| **D** | Finetune / RL with mixed user+model chord lanes, structure tokens | Yes |

---

## 6. Code touchpoints (this repo)

| Area | File(s) |
|------|---------|
| Live generation | `realchords/realjam/agent_interface.py` — `generate_live`, intro, interleave |
| Online AR paste/sample | `realchords/model/gen_model.py` — `generate_online` |
| Invalid chord filter | `realchords/model/sampling.py` — `filter_invalid_tokens_generate_online` |
| Symbol ↔ pitches | `note_seq.chord_symbols_lib`; `realchords/utils/data_utils.py` — `to_chord_name` |
| Vocab | `data/cache/chord_names.json`, `chord_names_augmented.json` |
| Frontend session | `realchords/realjam/frontend/play_from_midi_input.js` |

---

## Summary

- **Structure at play-time** = agreed key/chart/form via **inference constraints** and optional **section reset** on the chord lane — aligned with lead sheets, not endless free harmony.  
- **User-played chords** = **paste chord tokens** like melody — inference engineering; parse with **note_seq**, handle **OOV**, export sessions for **conditional metrics**.  
- Deeper integration (model learns form / mixed control) is **optional finetune/RL**, not required for a first prototype.
