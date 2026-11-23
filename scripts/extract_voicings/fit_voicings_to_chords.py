#!/usr/bin/env python3
"""Prototype: map voicings to chord names by a fit rule (name -> voicing), not by pitch-class set equality.

The current lookup (match_voicings_to_chords.py) reduces each extracted voicing to its pitch-class set and files it
under every vocabulary name with exactly that set. That loses the bass (F A C D is filed under F6 and Dm7 alike), is
ambiguous, and misses voicings that leave out the root or the fifth. Here we start from the name instead: its root,
degrees and bass come from ``chord_symbol_degrees`` (note_seq's parsing with added 7ths read as written), and a
voicing fits the name if

  1. bass:      its lowest note is the name's bass (the root, or the slash note);
  2. required:  it contains the chord's defining degrees: the third (or the sus 2nd/4th), the 7th (else the 6th),
                an altered 5th, the highest extension (9, 11, 13), and every degree the name adds or alters;
  3. optional:  the root, an unaltered 5th and lower extensions may be missing;
  4. extras:    "strict": nothing outside the chord's degrees; "tensions": also the tensions usual for the chord's
                quality (dominant 9 b9 #9 #11 13 b13; major 9 #11 13; minor 9 11 13; (half-)diminished 9 11 b13;
                sus 9 13; augmented 9).

Variants written: strict and tensions, each "as played" (only voicings whose lowest note already is the name's bass)
and "pooled" (every voicing moved by -6..+5 semitones so that its lowest note is the bass: all 12 keys pooled,
register kept within half an octave). The script also measures how much of the current lookup passes the rule.

    python scripts/extract_voicings/fit_voicings_to_chords.py --out_dir <dir>

Outputs <out_dir>/fit_<variant>.json (same format as chord_voicings.json) and prints a comparison. Journal:
reference/VOICING_EXTRACTION.md, "Fit rule".
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from note_seq import chord_symbols_lib as csl

from realchords.utils.data_utils import chord_symbol_degrees

REPO = Path(__file__).resolve().parents[2]
DEG_OFFSET = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 9: 2, 11: 5, 13: 9}
MIN_DISTINCT_PCS = 3  # a voicing of 3+ notes that are only 2 pitch classes (C F C) is not a chord
TENSIONS = {  # semitones above the root
    "dominant": {2, 1, 3, 6, 9, 8},
    "major": {2, 6, 9},
    "minor": {2, 5, 9},
    "diminished": {2, 5, 8},
    "sus": {2, 9},
    "augmented": {2},
    "other": set(),
}


def chord_rule(name: str) -> Optional[dict]:
    """root, bass, chord pitch classes, required pitch classes and allowed tensions of a name (relative to C)."""
    try:
        root, degrees, bass = chord_symbol_degrees(name)
        _, kind_str, mods, _ = csl._split_chord_symbol(name)
        kind = csl._parse_kind(kind_str)
    except Exception:  # noqa: BLE001
        return None
    pc = {d: (DEG_OFFSET[d] + a) % 12 for d, a in degrees.items()}
    required: Set[int] = set()
    if 3 in degrees:
        required.add(3)
    else:
        required |= {d for d in (2, 4) if d in degrees}
    if 7 in degrees:
        required.add(7)
    elif 6 in degrees:
        required.add(6)
    if degrees.get(5, 0) != 0:
        required.add(5)
    ext = [d for d in (9, 11, 13) if d in degrees]
    if ext:
        required.add(max(ext))
    required |= {d for d, a in degrees.items() if d not in kind or kind[d] != a}  # added / altered by the name
    if not required - {1}:  # power chord / pedal: everything it has
        required = set(degrees)
    if bass != root:  # slash chord: without its root the upper chord disappears (C F A is F/C, not Gsus7/C)
        required.add(1)
    third, fifth, seventh = degrees.get(3), degrees.get(5, 0), degrees.get(7)
    if third is None:
        quality = "sus" if (2 in degrees or 4 in degrees) else "other"
    elif third == 0 and fifth == 1:
        quality = "augmented"
    elif third == -1 and fifth == -1:
        quality = "diminished"
    elif third == -1:
        quality = "minor"
    elif seventh == -1:
        quality = "dominant"
    else:
        quality = "major"
    chord_pcs = {(root + p) % 12 for p in pc.values()}
    tension_set = TENSIONS[quality]
    if 6 not in degrees and 7 not in degrees:  # plain triad: a 6th or 7th would make it another token (C6, C7)
        tension_set = tension_set & {2}
    return {
        "root": root, "bass": bass,
        "chord": chord_pcs | {bass},
        "required": {(root + pc[d]) % 12 for d in required},
        "tensions": {(root + t) % 12 for t in tension_set} - chord_pcs,
        "quality": quality,
    }


def fits(voicing_pcs: Set[int], lowest_pc: int, rule: dict, tensions: bool) -> bool:
    if len(voicing_pcs) < MIN_DISTINCT_PCS and len(rule["chord"]) >= MIN_DISTINCT_PCS:
        return False
    if lowest_pc != rule["bass"] or not rule["required"] <= voicing_pcs:
        return False
    allowed = rule["chord"] | (rule["tensions"] if tensions else set())
    return voicing_pcs <= allowed


def load_voicings(path: Path, min_count: int, min_notes: int, max_notes: int):
    raw = json.load(open(path))
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    songs: Dict[Tuple[int, ...], int] = defaultdict(int)
    for v in raw:
        k = tuple(v["pitches"])
        counts[k] += v["count"]
        songs[k] += v.get("num_songs", 0)
    keep = [k for k, c in counts.items() if c >= min_count and min_notes <= len(k) <= max_notes]
    keep.sort(key=lambda k: -counts[k])
    return keep, np.array([counts[k] for k in keep]), np.array([songs[k] for k in keep])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voicings", default=str(REPO / "data/voicings/merged/all_voicings.json"))
    ap.add_argument("--chord_names", default=str(REPO / "data/cache/chord_names_augmented.json"))
    ap.add_argument("--current", default=str(REPO / "data/voicings/merged/chord_voicings.json"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--min_notes", type=int, default=3)
    ap.add_argument("--max_notes", type=int, default=8)
    ap.add_argument("--max_voicings_per_chord", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    voicings, counts, songs = load_voicings(Path(args.voicings), args.min_count, args.min_notes, args.max_notes)
    lowest = np.array([min(v) for v in voicings])
    low_pc = lowest % 12
    masks = np.array([sum(1 << x for x in {int(p - lo) % 12 for p in v}) for v, lo in zip(voicings, lowest)])
    by_mask: Dict[int, np.ndarray] = {}
    order = np.argsort(masks, kind="stable")
    uniq, starts = np.unique(masks[order], return_index=True)
    for m, s, e in zip(uniq, starts, list(starts[1:]) + [len(order)]):
        by_mask[int(m)] = np.sort(order[s:e])  # sorted indices = descending count
    print(f"{len(voicings):,} voicings, {len(by_mask):,} shapes, loaded in {time.time() - t0:.0f} s", flush=True)

    names = json.load(open(args.chord_names))
    names = names if isinstance(names, list) else list(names)
    rules = {n: chord_rule(n) for n in names}
    print(f"{sum(r is not None for r in rules.values()):,} of {len(names):,} names parsed", flush=True)

    results = {}
    for tensions in (False, True):
        for pooled in (False, True):
            tag = f"{'tensions' if tensions else 'strict'}_{'pooled' if pooled else 'as_played'}"
            lookup = {}
            for n, r in rules.items():
                if r is None:
                    continue
                ok_masks = [m for m in by_mask
                            if fits({(r["bass"] + i) % 12 for i in range(12) if m >> i & 1}, r["bass"], r, tensions)]
                if not ok_masks:
                    continue
                idx = np.concatenate([by_mask[m] for m in ok_masks])
                if not pooled:
                    idx = idx[low_pc[idx] == r["bass"]]
                if len(idx) == 0:
                    continue
                idx = idx[np.argsort(-counts[idx], kind="stable")][: args.max_voicings_per_chord * 10]
                agg: Dict[Tuple[int, ...], List[int]] = {}
                for i in idx:
                    shift = ((r["bass"] - low_pc[i] + 6) % 12) - 6
                    p = tuple(sorted(int(x + shift) for x in voicings[i]))
                    if p[0] < 28 or p[-1] > 108:
                        continue
                    a = agg.setdefault(p, [0, 0])
                    a[0] += int(counts[i]); a[1] += int(songs[i])
                best = sorted(agg.items(), key=lambda kv: -kv[1][0])[: args.max_voicings_per_chord]
                if best:
                    lookup[n] = [{"pitches": list(p), "count": c, "num_songs": s} for p, (c, s) in best]
            results[tag] = lookup
            json.dump(lookup, open(out / f"fit_{tag}.json", "w"))
            print(f"  {tag}: {len(lookup):,} names, {time.time() - t0:.0f} s", flush=True)

    current = json.load(open(args.current))
    results = {"current (pc-set match)": current, **results}
    print("\n| lookup | names with >= 1 | >= 10 | >= 100 voicings | median voicings/name | occurrences |")
    print("|---|---|---|---|---|---|")
    for tag, lk in results.items():
        sizes = [len(v) for v in lk.values()]
        occ = sum(e["count"] for v in lk.values() for e in v)
        print(f"| {tag} | {len(lk):,} | {sum(s >= 10 for s in sizes):,} | {sum(s >= 100 for s in sizes):,} | "
              f"{int(np.median(sizes)) if sizes else 0} | {occ:,} |")

    print("\nHow much of the current lookup passes the rule for the name it is filed under (occurrence-weighted):")
    for tensions in (False, True):
        tot = ok = 0
        for n, vs in current.items():
            r = rules.get(n)
            if r is None:
                continue
            for e in vs:
                tot += e["count"]
                ok += e["count"] * fits({p % 12 for p in e["pitches"]}, min(e["pitches"]) % 12, r, tensions)
        print(f"  {'tensions' if tensions else 'strict'}: {100 * ok / tot:.1f}%")
    why = defaultdict(int); tot = 0
    for n, vs in current.items():
        r = rules.get(n)
        if r is None:
            continue
        for e in vs:
            pcs, lo = {p % 12 for p in e["pitches"]}, min(e["pitches"]) % 12
            tot += e["count"]
            if fits(pcs, lo, r, False):
                continue
            if lo != r["bass"]:
                why["wrong bass (an inversion of the name, or another chord)"] += e["count"]
            elif len(pcs) < MIN_DISTINCT_PCS:
                why["fewer than 3 pitch classes"] += e["count"]
            else:
                why["other"] += e["count"]
    for k, v in why.items():
        print(f"    fails: {k}: {100 * v / tot:.1f}%")

    print("\nExamples (top 3 voicings):")
    for n in ("F6", "Dm7", "C9", "Asus(add7)", "C7", "Gsus7/C"):
        for tag, lk in results.items():
            top = [e["pitches"] for e in lk.get(n, [])[:3]]
            print(f"  {n:12} {tag:24} {len(lk.get(n, [])):4}  {top}")


if __name__ == "__main__":
    main()
