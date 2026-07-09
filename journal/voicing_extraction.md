# Voicing extraction from MIDI datasets

**Goal:** Build a lookup table mapping every chord name in `chord_names_augmented.json`
to a set of real-world piano voicings (concrete MIDI pitch sets) observed in actual
recordings / transcriptions.  
**Motivation:** RealJam's decoder maps all chord tones to a single fixed octave block
(MIDI 48–59), which sounds mechanical. Real voicings can spread notes across the
keyboard, add doublings, and vary by register.

---

## The idea in one sentence

Scan MIDI files, collect every group of ≥ 3 notes played simultaneously (within 50 ms),
reduce each group to its set of **pitch classes (mod 12)**, and find which chord in the
vocabulary has the **exact same pitch-class set**.

---

## Datasets used

| Dataset | Path | Files | Notes |
|---------|------|-------|-------|
| PIJAMA | `data/pijama/` | ~several hundred | Piano jazz transcriptions, full MIDI |
| Aria-MIDI (jazz) | `data/aria-midi-v1-deduped-ext/data/` | 18,839 of 371,053 | Genre filtered to `jazz` via `metadata.json` |

---

## Scripts

All scripts live in `scripts/extract_voicings/`.

### `extract_voicings.py` — general extractor

```
python scripts/extract_voicings/extract_voicings.py \
    --input_dir <dir>           # root dir, scanned recursively
    --output_dir <dir>          # where to write all_voicings.json
    [--metadata_json <path>]    # optional aria-midi metadata.json for genre filter
    [--genres jazz pop ...]     # only process files whose metadata genre matches
    [--onset_tolerance 0.05]    # seconds; max gap to group notes as a chord
    [--min_notes 3]             # minimum simultaneous notes to call it a chord
    [--workers N]               # parallel workers (default: CPU count − 1)
```

**Output:** `<output_dir>/all_voicings.json`

```json
[
  {"pitches": [48, 52, 55], "count": 12500},
  {"pitches": [52, 55, 60], "count": 8300},
  ...
]
```

Entries are sorted by descending count. Pitches are the raw MIDI numbers (no octave
normalisation at this stage — that happens during matching).

**How extraction works:**

1. Load all non-drum MIDI notes from every instrument track.
2. Sort by onset time.
3. Slide a window: notes within `onset_tolerance` seconds of the first note in the
   group are considered simultaneous. When the gap exceeds the tolerance, close the
   group and start a new one.
4. Keep only groups with ≥ `min_notes` distinct pitches.
5. Represent each group as a sorted tuple of unique MIDI pitches (intra-group
   duplicates removed; octave doublings kept because they carry register information).
6. Count occurrences of each unique tuple across all files.

**Aria-MIDI quirk:** `metadata.json` keys are plain integers (`"31357"`) but file
stems are zero-padded (`031357_0.mid`). The script normalises both to `int` for
comparison, so the genre filter works correctly.

**Performance note:** Multiprocessing (`mp.Pool`) works fine when run in a real
terminal session. When run inside Cursor's sandboxed shell via `required_permissions:
["all"]` with `&` backgrounding, worker processes were silently killed. Use a terminal
or `block_until_ms` large enough to run in foreground instead.

---

### `match_voicings_to_chords.py` — build the lookup table

```
python scripts/extract_voicings/match_voicings_to_chords.py \
    [--voicings data/voicings/merged/all_voicings.json]
    [--chord_names data/cache/chord_names_augmented.json]
    [--output data/voicings/merged/chord_voicings.json]
    [--min_count 3]         # drop voicings seen < 3 times
    [--min_notes 3]         # drop voicings with < 3 notes
    [--max_notes 8]         # drop dense clusters (> 8 notes)
    [--max_voicings_per_chord 500]
```

**Output:** `<output>` — JSON object mapping chord name → list of voicing dicts.

```json
{
  "C": [
    {"pitches": [48, 52, 55], "count": 12500},
    {"pitches": [52, 55, 60], "count": 8300}
  ],
  "Cm7": [...]
}
```

**Matching strategy (exact pitch-class set):**

1. Parse each chord name with `note_seq` → get root pitch class + interval list → derive
   the chord's **pitch-class set** (a 12-bit bitmask).
2. For each observed voicing, compute its **pitch-class bitmask** (`p % 12` for each
   pitch in the group).
3. Match iff `chord_bitmask == voicing_bitmask` — **exact equality**.
   - Octave doublings are OK (G3 and G4 both contribute pitch class G, counted once).
   - Extra notes beyond the chord definition → **not** a match.
   - Missing notes from the chord definition → **not** a match.

**Why exact, not subset?** We want voicings that express *only* the notes of that
chord, not voicings that happen to contain the chord plus extra extensions. If we used
subset matching, a C major voicing could match "C" even if it also contained a Bb
(making it really a C7).

**Multiple names sharing a pitch-class set:** slash chords like `C/E` have the same
pitch classes as `C`. The first alphabetical match is taken. This means slash chords
effectively never win the match, which is why they are mostly absent from the output
(see coverage section below).

---

### `merge_voicings.py` — combine multiple `all_voicings.json` files

```
python scripts/extract_voicings/merge_voicings.py \
    data/voicings/pijama/all_voicings.json \
    data/voicings/aria-midi-jazz/all_voicings.json \
    --output data/voicings/merged/all_voicings.json
```

Counts are summed for pitch tuples that appear in multiple sources. Output is sorted by
descending total count.

---

### `coverage_report.py` — per-chord statistics

```
python scripts/extract_voicings/coverage_report.py \
    [--chord_names data/cache/chord_names_augmented.json] \
    [--chord_voicings data/voicings/merged/chord_voicings.json]
```

Prints overall coverage, histogram of voicings per chord, breakdown of missing chords
by type (slash vs non-slash, top missing chord types).

---

## Results (Jul 9 2026)

### Per-dataset extraction

| Dataset | Files processed | Files skipped | Total events | Unique voicings |
|---------|----------------|---------------|--------------|-----------------|
| PIJAMA | ~several hundred | — | 1,695,035 | 330,367 |
| Aria-MIDI jazz | 18,808 / 18,839 | 31 (parse errors) | 3,832,975 | 485,848 |
| **Merged** | — | — | **5,528,010** | **657,643** |

Aria-MIDI jazz extraction ran in ~2 min with 4 workers (≈160 files/sec).

### Coverage after matching (merged dataset)

```
Chord vocab size          :   2,844
Chords with voicings      :   1,172  (41.2%)
Chords without voicings   :   1,672  (58.8%)
Total voicing entries     : 149,416
Total chord occurrences   : 4,517,269
```

**Voicings per chord (covered chords only):**

| Range | Count |
|-------|-------|
| 1 voicing | 52 chords |
| 2–5 voicings | 113 chords |
| 6–20 voicings | 199 chords |
| 21–100 voicings | 375 chords |
| 101–500 voicings | 433 chords |

**Missing chords (1,672 total):**

- **Slash chords:** 1,162 missing. Root cause: the exact-match strategy maps a voicing
  to the non-slash chord name that shares its pitch classes (e.g. `[C,E,G]` matches `C`
  not `C/E`).
- **Non-slash missing (510):** Mostly exotic types: `sus7`, `+7`, `5(addb7)`, `o(add7)`,
  `m13`, `maj13`, `6`, `m6`, `o7`, etc. These either lack enough real-world occurrences
  with exactly those pitch classes, or musicians voice them with extra extensions.

---

## Output files

| File | Description |
|------|-------------|
| `data/voicings/pijama/all_voicings.json` | Raw voicings from PIJAMA |
| `data/voicings/aria-midi-jazz/all_voicings.json` | Raw voicings from Aria-MIDI jazz |
| `data/voicings/merged/all_voicings.json` | Merged voicings (both datasets) |
| `data/voicings/merged/chord_voicings.json` | Lookup table: chord → voicings |
| `data/voicings/pijama/chord_voicings.json` | Same, PIJAMA-only (older, kept for reference) |

---

## Known limitations and future work

1. **Slash chords have no voicings.** Fix: for a slash chord `X/Y`, fall back to the
   root chord `X`'s voicings and re-voice them with pitch `Y` (or the nearest `Y`) in
   the bass.

2. **Exotic chord types are sparse.** `sus7`, `+7`, diminished types etc. rarely appear
   with exact pitch-class sets in real-world piano (musicians often add or omit tones).
   Fix: for unmatched chords, use a rule-based fallback that generates a close-position
   voicing from the theoretical pitch-class set.

3. **No filtering by instrument.** The extractor takes notes from all non-drum tracks,
   so orchestral MIDI can contribute string/brass clusters that look like "chords".
   Fix: filter to tracks with program numbers in the piano range (0–7) when processing
   non-jazz multi-track MIDI.

4. **Register distribution not captured.** The lookup stores actual MIDI pitches
   (preserving register), but no statistics on which registers are most common. For
   generation, sampling from the stored voicings weighted by count already handles this.

5. **`extract_pijama_voicings.py` is superseded** by the general `extract_voicings.py`.
   The PIJAMA-specific script can be deleted once the merged pipeline is confirmed.
