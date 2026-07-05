# Dataset summary

Melody–chord datasets for ReaLchords. **Supported** sets are converted to `data/cache/<name>/`
(JSONL, Hooktheory schema) and loaded via `HooktheoryDataset` (`scripts/convert_*_to_cache.py`).

**Cache splits:** 80 / 10 / 10 train / valid / test (seed 42). **Harmony after conversion:**
`root_pitch_class` + `root_position_intervals` + `inversion` → chord symbol tokens (inversion
dropped in training). Voicing not stored — see
[`CHORD_VOICING_FINDINGS.md`](../dataset_experiments/CHORD_VOICING_FINDINGS.md).

---

## At a glance

| Dataset | Status | Format | Genre | Train / val / test | Chords |
|---------|--------|--------|-------|-------------------|--------|
| Hooktheory | Supported | JSON (+ optional Hookpad raw JSON) | Pop/rock | 19 052 / 1 943 / 2 467 | Interval stack; ~2.8k symbols |
| POP909 | Supported | MIDI + TXT (chords, beats) | Pop | 727 / 90 / 92 | Symbols → note_seq; ~175 |
| Nottingham | Supported | ABC | Folk/trad | 815 / 101 / 103 | Simple symbols; ~32 |
| Wikifonia | Supported | MusicXML | Lead sheets | 401 / 50 / 51 | Symbols → note_seq; many simplified |
| Jazzvar | Candidate | MIDI | Jazz standards | ~505 files | Block chords; 4-bar refrains, 4/4 |
| JAZZMUS | Candidate | MusicXML | Jazz standards | ~163 files | Block chords |
| WJD (wjazzd) | Candidate | SQLite (+ CSV / MIDI) | Jazz solo + comp | ~456 solos | Block chord symbols on beats |
| Pijama | Candidate | MIDI (×2 transcriptions) | Jazz improv | **2 777** performances | AMT only; hawthorne + kong |

Melody + harmony: all supported sets have both; Pijama is piano-only AMT (no chord labels; use one of hawthorne/kong).

---

## Supported (detail)

**Hooktheory** — Primary corpus. Sheet Sage
[`Hooktheory.json.gz`](https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz);
23 462 songs pass cache filter (MELODY + HARMONY, no TEMPO_CHANGES). Keys and modulations in
`annotations.keys` (~496 songs with key changes). Raw Hookpad uses scale degree + chord type.

**POP909** — [POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset). Best alignment:
dedicated `MELODY` MIDI track, tab-separated chord + beat files, beat/downbeat sequences in cache.
Used with Hooktheory in GAPT training.

**Nottingham** — [ABC corpus](https://github.com/jukedeck/nottingham-dataset). Chord symbols in
quotes; music21 timing. Parser handles basic triads/7ths only; keys default to C major in cache.

**Wikifonia** — [Archive](https://github.com/andreamust/WikifoniaDataset) (~6k+ XML); ~502 parse
successfully. Wikifonia-specific symbol cleanup + note_seq; failures simplified or dropped.

**Shared:** `--augmentation` (±6 semitones), `WeightedJointDataset`, global `chord_names.json`.

---

## Jazz candidates (not integrated)

Need new `convert_<name>_to_cache.py` each.

**JAZZMUS** — MusicXML lead sheets, block chords. Closest template: Wikifonia converter + jazz
symbol rules.

**WJD** — `data/wjazzd/wjazzd.db`: `melody` + `beats` (chord symbols, times). ~456 solos /
~302 compositions. Explore: `dataset_experiments/scripts/explore_wjazzd.py`. Main work: beat
grid alignment, dense note quantization.

**Pijama** — [PiJAMA](https://almostimplemented.github.io/PiJAMA/) (Zenodo). **2 777 unique performances**
(783 live + 1 994 studio), 120 pianists; ~200+ hours. Under `data/pijama/`: `midi_hawthorne/midi/{live,studio}/…`
and `midi_kong/{live,studio}/…` — **same 2 777 titles, not byte duplicates**. Hawthorne = Onsets & Frames;
Kong = high-res transcription (different tick resolution, ~1.3× larger files on median). Pick **one** tree for
training unless you want dual transcriptions. No chord track; poor fit for melody–chord cache without separation.

---

## Quick pick

| Goal | Dataset |
|------|---------|
| Scale, pop/rock | Hooktheory |
| Clean alignment | POP909 |
| Folk / simple harmony | Nottingham |
| Lead sheets | Wikifonia |
| Jazz block changes | Jazzvar, JAZZMUS, WJD |
