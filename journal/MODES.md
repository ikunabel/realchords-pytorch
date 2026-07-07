# Modal chord–scale mapping

Reference for `realchords/utils/modes.py`: chord quality → modal scale(s), for a
planned **note-in-mode** metric alongside note-in-chord (`eval_utils.py`).

**Framing:** harmony modally, chord by chord. Each quality maps to one or more of
21 parent-scale modes. No song key, no Roman numerals, no corpus statistics.

---

## Purpose

**Note-in-chord** (existing) checks melody pitch class against spelled chord tones —
strict, chord tones only.

**Note-in-mode** (planned) checks against a modal scale from chord–scale theory.
`Cmaj7` → Ionian or Lydian; `F7` → Mixolydian or Lydian dominant. Melody notes
outside the chord but inside the mode count as idiomatic.

The curated map (`chord_quality_mode_map.jsonl`) is the working definition.

---

## Artifacts

```bash
python -m realchords.utils.modes
```

| File | Role |
|---|---|
| `modes.py` | Implementation |
| `pitch_class_chord_map.jsonl` | 3–7 pitch classes → `note_seq` chord symbols (3,199 resolved, 24 unresolved) |
| `chord_quality_mode_map.jsonl` | **Curated** quality → modes (use this) |
| `chord_quality_mode_map_all.jsonl` | Exhaustive subset matches (reference) |

166 qualities from `data/cache/chord_names.json`, root stripped via
`extract_chord_quality()` (`Cmaj7` → `maj7`, `C` → `""`).

---

## Twenty-one modes

`list_scale_modes()` — seven rotations each of major, harmonic minor, melodic minor:

- **Major:** Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian
- **Harmonic minor:** Harmonic minor, Locrian ♮6, Ionian augmented, Romanian minor, Phrygian dominant, Lydian ♯2, Ultralocrian
- **Melodic minor:** Melodic minor, Dorian ♭2, Lydian augmented, Lydian dominant, Mixolydian ♭6, Locrian ♯2, Altered

Each mode: `intervals`, `pitch_classes` (C root), `half_steps`. Lookups are
root-independent (mod 12).

---

## Mapping pipeline

Per quality: spell on C → pitch classes → find modes → curate → fallback if empty.

### 1. Exhaustive match

`chord_pitches ⊆ mode_pitches` — all mathematically possible modes. Written to
`chord_quality_mode_map_all.jsonl`.

### 2. Curated table

`CURATED_CHORD_QUALITY_MODES` in `modes.py` — hand-coded pedagogy (Nettles & Graf,
Levine, Berklee). First mode = `role: "primary"`, rest = `alternative`. Max 3 modes.

| Family | Modes (primary first) |
|---|---|
| `""` / `6` / `maj7` | Ionian, Lydian |
| `7` / `9` / `11` / `13` | Mixolydian, Lydian dominant |
| `m` / `m7` | Dorian, Aeolian |
| `m7b5` | Locrian, Locrian ♯2 |
| `o` / `o7` | Locrian, Locrian ♮6 |
| `+` / `sus` | Lydian augmented, Mixolydian, … |

Unknown qualities: strip extensions (`7(#9)` → `7`), else `FAMILY_MODE_PRIORITY`.

### 3. Intersect + narrow + filter

- **Intersect** curated candidates with exhaustive matches
- **Extension narrowing** on `#11`, `b13`, `#9`, `addb2` (e.g. `#9` → Altered,
  Phrygian dominant, Lydian ♯2)
- **Obscure filter** drops Ultralocrian, Lydian ♯2, Altered unless context warrants

### 4. Fallback chain (empty after curation)

1. `combinatorial` — exhaustive matches, no obscure filter (`(#9)` → Lydian ♯2)
2. `extension_hint` — alteration rules on all 21 modes (`7(#9)` → Phryg. dom., Lydian ♯2, Altered)
3. `best_overlap` — max shared pitch classes
4. `family_default` — family priority table

JSONL records include `"fallback": "<type>"` when used. All 166 qualities resolve
to ≥1 mode (66 curated, 80 combinatorial fallback, 20 extension_hint).

### Underdetermined sonorities

`ped`, `5`, ≤2 pitch classes: `"underdetermined": true`, skip curation, combinatorial
fallback (typically all 21 modes).

---

## JSONL fields

```json
{"chord_quality": "maj7", "pitch_classes": [0, 4, 7, 11],
 "modes": [{"name": "Ionian", "role": "primary"}, {"name": "Lydian", "role": "alternative"}]}
```

Optional: `underdetermined`, `fallback`.

---

## Examples

| Quality | Result |
|---|---|
| `maj7` | Ionian, Lydian (curated) |
| `7(#11)` | Lydian dominant (extension narrowing) |
| `(#9)` | Lydian ♯2 (`fallback: combinatorial`) |
| `7(#9)` | Phryg. dom., Lydian ♯2, Altered (`fallback: extension_hint`) |
| `ped` | All 21 modes (`underdetermined`) |

---

## Note-in-mode eval (planned)

Per frame: chord symbol → quality → lookup map → check melody ∈ mode pitch classes.

- Strict: primary mode only
- Lenient: union of listed modes
- `ped`/`5`: policy TBD (skip or loose)

Not yet in `eval_utils.py`.

---

## Out of scope

- Song key (`keys_overview.json` exists but unused)
- Roman numerals / functional harmony
- Corpus-driven mode frequencies
- Voicing / register (pitch-class sets only)

---

## Sources

Nettles & Graf 2015; Levine 1995; Mulholland & Hojnacki 2013; Haerle 1982;
Russell 1953/2001; [Open Music Theory — chord–scale theory](https://viva.pressbooks.pub/openmusictheory/chapter/chord-scale-theory/).

Curated table: `CURATED_CHORD_QUALITY_MODES` in `modes.py`.
