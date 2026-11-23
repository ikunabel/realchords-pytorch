# Old selector behaviour (every voicing at every octave shift, no frequency
# term) vs new (as played first, shift as fallback, frequency weight 0.3) on
# real progressions, with ReaLJam's UI weights (vl 0.5, reg 0.2).
# Run from the repo root: python scripts/extract_voicings/compare_selector_shift_freq.py
# Results: journal/reference/VOICING_EXTRACTION.md, update 2026-11-16.
import sys, random, statistics as st, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from realchords.utils.voicing_selector import VoicingSelector, _vl_cost
from realchords.realjam.song_catalogue import _load_split, extract_reference_chords, extract_melody_notes
LK = "data/voicings/merged/chord_voicings.json"
old = VoicingSelector(LK, shift_as_fallback=False, freq_weight=0.0)
new = VoicingSelector(LK)
stored = {c: {tuple(e["pitches"]): e["count"] for e in v} for c, v in old._lookup.items()}
def run(sel, songs, bass_line):
    as_played, counts, moves, centroids = [], [], [], []
    for rec in songs:
        melody = extract_melody_notes(rec); prev = None
        for ch in extract_reference_chords(rec):
            sym = ch["symbol"]
            if sym not in stored: continue
            if bass_line:
                kw = dict(melody_pitch=55, melody_role="bass")
            else:
                sounding = [n["pitch"] for n in melody if n["onset"] < ch["offset"] and n["offset"] > ch["onset"]]
                kw = dict(melody_pitch=min(sounding) if sounding else None, melody_role="top")
            v = sel.select(sym, prev_voicing=prev, update_state=False, vl_weight=0.5, reg_weight=0.2, **kw)
            if not v: continue
            c = stored[sym].get(tuple(v)); as_played.append(c is not None); counts.append(c or 0)
            if prev: moves.append(_vl_cost(prev, v))
            centroids.append(sum(v) / len(v)); prev = v
    c = np.array(counts)
    return (f"{np.mean(as_played):6.1%} as played | count of chosen voicing: median {np.median(c[c>0]) if (c>0).any() else 0:6.0f}, "
            f"share seen <10x {np.mean((c>0)&(c<10)):5.1%} | voice movement {st.mean(moves):.2f} semitones | centroid {st.mean(centroids):.1f}")
random.seed(0)
hook = random.sample(_load_split(Path("data/cache"), "hooktheory", "test"), 300)
wjd = random.sample(_load_split(Path("data/cache"), "wjd", "test") + _load_split(Path("data/cache"), "wjd", "valid"), 60)
for name, songs, bl in [("Hooktheory, under melody", hook, False), ("WJD, above bass line", wjd, True)]:
    print(name); print("  old:", run(old, songs, bl)); print("  new:", run(new, songs, bl))
names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
nm = lambda ps: ' '.join(f"{names[p%12]}{p//12-1}" for p in ps)
for sel, lab in [(old, "old"), (new, "new")]:
    prev = None; out = []
    for sym in ["Dm7", "G7", "Cmaj7", "A7", "Dm7", "G7", "Cmaj7"]:
        v = sel.select(sym, prev_voicing=prev, update_state=False, vl_weight=0.5, reg_weight=0.2, melody_pitch=72, melody_role="top")
        out.append(f"{sym}: {nm(v)} ({stored[sym].get(tuple(v), 'shifted')})"); prev = v
    print(lab, "ii-V-I under C5:", " | ".join(out))
