#!/usr/bin/env python3
"""Build the chord-conversion database: every chord annotation of every dataset
followed through each conversion step, to see exactly where information is
lost or changed (journal/reference/CHORD_CONVERSION_DB.md).

Chains:

    Hooktheory   Hookpad raw fields --(our renderer, checked vs Sheet Sage)-->
                 Sheet Sage processed (root + stacked intervals + inversion = the cache)
                 --to_chord_name (note_seq)--> chord name --> token (old / new vocabulary)
    others       source chord label --(each converter's own parser, traced)--> cache triple
                 --to_chord_name (note_seq)--> chord name --> token

For each event the database keeps an independent reading of the *source* label
(pitch classes, root, bass) next to what the cache and the final token contain,
and flags every difference.

Tables:
    events       one row per original chord event (all datasets; Hooktheory from
                 the cache's train/valid/test, others from the source songs that are
                 in the cache, label chain as the converter parses it)
    structures   every distinct (intervals, inversion) named in all 12 keys -- what
                 transposition augmentation produces
    vocab        the two chord vocabularies (old: 2,821 chords, new: 2,969)
    meta         build information
    v_loss_summary  per dataset: how many events each kind of loss affects

Usage (repo root, the .venv python; about 20-40 min, mostly music21 parsing)::

    python scripts/data/build_chord_conversion_db.py [--datasets hooktheory pop909 ...]
        [--out data/chord_conversion/chord_conversion.sqlite]
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "convert_data_to_cache"))
sys.path.insert(0, str(REPO / "scripts" / "data"))

from note_seq import chord_symbols_lib as csl  # noqa: E402

from realchords.utils.data_utils import to_chord_name  # noqa: E402
import hookpad_chords as hc  # noqa: E402

PC = hc.PC_NAMES
ALL_DATASETS = ["hooktheory", "pop909", "nottingham", "wikifonia", "chord_melody_dataset",
                "jazzmus", "emopia_plus", "wjd", "filobass"]
OLD_VOCAB = REPO / "checkpoints" / "chord_names_augmented.json"
NEW_VOCAB = REPO / "data" / "cache" / "chord_names_augmented.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pcs_str(pcs: Optional[Iterable[int]]) -> Optional[str]:
    if pcs is None:
        return None
    return ",".join(str(p) for p in sorted({int(p) % 12 for p in pcs}))


def stack_of(root: int, intervals: List[int]) -> List[int]:
    """Root-position stack in semitones above the root (as the cache encodes it)."""
    s = [0]
    for iv in intervals:
        s.append(s[-1] + iv)
    return s


def cache_info(root: int, intervals: List[int], inversion: int) -> Dict:
    """What the cache triple says: pitch classes and the bass its inversion implies."""
    st = stack_of(root, intervals)
    inv = int(inversion or 0)
    bass = (root + st[inv]) % 12 if inv < len(st) else root % 12
    return {"pcs": pcs_str(root + s for s in st), "bass": bass}


@functools.lru_cache(maxsize=None)
def token_info(root: int, intervals: Tuple[int, ...]) -> Dict:
    """The tokenization step: to_chord_name (note_seq) on the root-position stack,
    and what that name means again (pitch classes, root, bass)."""
    try:
        name = to_chord_name(root, list(intervals))
    except Exception as exc:  # noqa: BLE001
        return {"name": None, "error": str(exc)[:80]}
    try:
        pcs = {p % 12 for p in csl.chord_symbol_pitches(name)} | {csl.chord_symbol_bass(name) % 12}
        return {"name": name, "pcs": pcs_str(pcs), "root": csl.chord_symbol_root(name) % 12,
                "bass": csl.chord_symbol_bass(name) % 12}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "error": str(exc)[:80]}


def m21_reading(text: str) -> Optional[Dict]:
    """music21's reading of a chord symbol string: pitch classes, root, bass."""
    from music21 import harmony
    try:
        cs = harmony.ChordSymbol(text)
        ps = list(cs.pitches)
        if not ps:
            return None
        return {"pcs": pcs_str(p.pitchClass for p in ps), "root": cs.root().pitchClass,
                "bass": cs.bass().pitchClass if cs.bass() is not None else cs.root().pitchClass}
    except Exception:  # noqa: BLE001
        return None


def trace_noteseq_parse(symbol: str, transform) -> Dict:
    """Which path parse_chord_symbol_with_noteseq takes for `symbol` (it is called
    for its result separately; this only records the path): the transformed
    string, the string that produced the pitches, and direct / simplified /
    fallback_c_major / empty."""
    from convert_wikifonia_to_cache import simplify_complex_chord
    t = transform(symbol)
    if not t:
        return {"transformed": t, "used": None, "path": "empty"}
    try:
        if csl.chord_symbol_pitches(t):
            return {"transformed": t, "used": t, "path": "direct"}
        s2 = simplify_complex_chord(t)
    except Exception:  # noqa: BLE001
        s2 = simplify_complex_chord(t)
    try:
        if csl.chord_symbol_pitches(s2):
            return {"transformed": t, "used": s2, "path": "simplified"}
    except Exception:  # noqa: BLE001
        pass
    return {"transformed": t, "used": "C", "path": "fallback_c_major"}


# ---------------------------------------------------------------------------
# independent readings of source labels
# ---------------------------------------------------------------------------

HARTE = {"maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10], "min7": [0, 3, 7, 10],
         "maj7": [0, 4, 7, 11], "dim": [0, 3, 6], "dim7": [0, 3, 6, 9], "aug": [0, 4, 8],
         "sus4": [0, 5, 7], "sus2": [0, 2, 7], "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
         "hdim7": [0, 3, 6, 10], "minmaj7": [0, 3, 7, 11], "sus4(b7)": [0, 5, 7, 10]}
DEGREE = {"1": 0, "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7, "#5": 8,
          "b6": 8, "6": 9, "b7": 10, "7": 11}


def harte_reading(label: str) -> Optional[Dict]:
    """POP909's Harte-style label ('C:maj7/3') -> pitch classes, root, bass."""
    if ":" not in label:
        return None
    root_s, qual = label.split(":", 1)
    bass_deg = None
    if "/" in qual:
        qual, bass_deg = qual.split("/", 1)
    if qual not in HARTE:
        return None
    root = hc.tonic_pc(root_s)
    pcs = {(root + i) % 12 for i in HARTE[qual]}
    bass = root
    if bass_deg is not None and bass_deg in DEGREE:
        bass = (root + DEGREE[bass_deg]) % 12
        pcs.add(bass)
    return {"pcs": pcs_str(pcs), "root": root, "bass": bass}


def wjd_reading(label: str) -> Optional[Dict]:
    """A WJD chord label ('C-7', 'Abj7', 'A79b', 'Bb7911#', 'C7alt', 'G-7/F') ->
    pitch classes, root, bass. Accidentals follow their number (79b = 7 + b9)."""
    m = re.match(r"^([A-G][b#]?)(.*)$", label.strip())
    if not m or label.strip() in ("NC", ""):
        return None
    root = hc.tonic_pc(m.group(1))
    rest, bass = m.group(2), root
    if "/" in rest:
        rest, b = rest.split("/", 1)
        try:
            bass = hc.tonic_pc(b)
        except Exception:  # noqa: BLE001
            return None
    tones = set()
    if rest.startswith("m7b5"):
        tones |= {0, 3, 6, 10}; rest = rest[4:]
    elif rest.startswith("-"):
        tones |= {0, 3, 7}; rest = rest[1:]
    elif rest.startswith("+"):
        tones |= {0, 4, 8}; rest = rest[1:]
    elif rest.startswith("o"):
        tones |= {0, 3, 6}; rest = rest[1:]
        if rest.startswith("7"):
            tones.add(9); rest = rest[1:]
    elif rest.startswith("sus"):
        tones |= {0, 5, 7}; rest = rest[3:]
    else:
        tones |= {0, 4, 7}
    if rest.startswith("j7"):
        tones.add(11); rest = rest[2:]
    if rest.startswith("alt"):
        tones |= {10, 13, 15, 18, 20}; rest = rest[3:]
    for num, acc in re.findall(r"(13|11|9|7|6)([#b]?)", rest):
        base = {"6": 9, "7": 10, "9": 14, "11": 17, "13": 21}[num]
        tones.add(base + (1 if acc == "#" else -1 if acc == "b" else 0))
    if rest.endswith("alt"):
        tones |= {10, 13, 15, 18, 20}
    pcs = {(root + t) % 12 for t in tones} | {bass}
    return {"pcs": pcs_str(pcs), "root": root, "bass": bass}


# ---------------------------------------------------------------------------
# cache lookups
# ---------------------------------------------------------------------------

def cache_songs(dataset: str) -> Dict[str, Tuple[str, dict]]:
    """song id -> (split, cache record) from the non-augmented cache files."""
    out = {}
    for split in ("train", "valid", "test"):
        path = REPO / "data" / "cache" / dataset / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            d = json.loads(line)
            meta = d.get("hooktheory") or d.get(dataset) or d.get("wjazzd") or {}
            sid = str(meta.get("id", meta.get("melid")))
            out.setdefault(sid, (split, d))
    return out


# ---------------------------------------------------------------------------
# per-dataset event extraction: each yields dicts with the source/converter fields
# ---------------------------------------------------------------------------

def events_hooktheory(raw_pickle: Optional[Path]) -> Iterable[Dict]:
    if raw_pickle and raw_pickle.exists():
        raw = pickle.load(open(raw_pickle, "rb"))
    else:
        full = json.load(open(REPO / "data" / "hooktheory" / "Hooktheory_Raw.json"))
        raw = {k: {"keys": (r.get("json") or {}).get("keys"), "chords": (r.get("json") or {}).get("chords")}
               for k, r in full.items()}
        del full
    for sid, (split, d) in cache_songs("hooktheory").items():
        r = raw.get(sid) or {}
        by_beat = {}
        for c in r.get("chords") or []:
            if not c.get("isRest"):
                by_beat.setdefault(round(c["beat"] - 1, 4), c)
        for i, h in enumerate(d["annotations"]["harmony"]):
            ev = {"dataset": "hooktheory", "split": split, "song_id": sid, "event_index": i,
                  "onset": h["onset"], "offset": h["offset"],
                  "cache_root": h["root_pitch_class"], "cache_intervals": h["root_position_intervals"],
                  "cache_inversion": int(h.get("inversion") or 0), "converter_path": "sheetsage"}
            c = by_beat.get(round(h["onset"], 4))
            if c is None:
                ev["converter_path"] = "sheetsage_unaligned"
                yield ev
                continue
            key = hc.key_at(r.get("keys") or [], c["beat"])
            fields = {k: c.get(k) for k in ("root", "type", "inversion", "applied", "borrowed", "adds",
                                            "omits", "alterations", "suspensions")}
            fields["key"] = {"tonic": key.get("tonic"), "scale": key.get("scale")} if key else None
            ev["source_detail"] = json.dumps(fields)
            try:
                out = hc.render(c, key)
                ev.update(source_label=out["symbol"], source_roman=out["roman"],
                          source_pcs=pcs_str(out["pcs"]), source_root=out["root_pc"],
                          source_bass=out["bass_pc"])
            except Exception as exc:  # noqa: BLE001
                ev["source_label"] = f"(not rendered: {exc})"
            yield ev


def events_pop909() -> Iterable[Dict]:
    import convert_pop909_to_cache as cp
    songs = cache_songs("pop909")
    for sid, (split, _) in songs.items():
        f = REPO / "data" / "pop909" / "POP909" / "POP909" / sid / "chord_midi.txt"
        if not f.exists():
            continue
        i = 0
        for line in open(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            if label in ("N", "X") or start >= end:
                continue
            root, ivs, inv = cp.parse_chord_symbol(label)
            if not ivs:
                continue
            std = cp.pop909_to_standard_chord(label)
            path = "direct"
            if std is None:
                path = "fallback_c_major"
            else:
                try:
                    csl.chord_symbol_pitches(std)
                except Exception:  # noqa: BLE001
                    path = "fallback_c_major"
            src = harte_reading(label) or {}
            yield {"dataset": "pop909", "split": split, "song_id": sid, "event_index": i,
                   "onset": start, "offset": end, "source_label": label,
                   "source_pcs": src.get("pcs"), "source_root": src.get("root"), "source_bass": src.get("bass"),
                   "converter_input": std, "converter_used": std, "converter_path": path,
                   "cache_root": root, "cache_intervals": ivs, "cache_inversion": inv}
            i += 1


class _Capture:
    """Wraps a converter's parse function to record (song, label, result) per call."""

    def __init__(self):
        self.song = None
        self.calls: List[Tuple] = []


def events_musicxml(dataset: str) -> Iterable[Dict]:
    import convert_wikifonia_to_cache as cw
    cap = _Capture()
    original = cw.parse_chord_symbol_with_noteseq

    def wrapper(chord_symbol, *, chord_symbol_transform=cw.transform_wikifonia_chord_symbol):
        result = original(chord_symbol, chord_symbol_transform=chord_symbol_transform)
        cap.calls.append((cap.song, chord_symbol, chord_symbol_transform, result))
        return result

    cw.parse_chord_symbol_with_noteseq = wrapper
    try:
        songs = cache_songs(dataset)
        if dataset == "wikifonia":
            files = {Path(p).stem: Path(p) for p in glob.glob(str(REPO / "data/wikifonia/wikifonia/**/*.mxl"), recursive=True)}
            process = cw.process_wikifonia_file
        elif dataset == "chord_melody_dataset":
            # one MusicXML file per song and key (dir/<key>.xml); the cache id is
            # "<dir>__<key>" and train holds every key, valid/test one each
            import convert_chord_melody_dataset_to_cache as cc
            files = {}
            for sid_, (_, rec_) in songs.items():
                meta_ = rec_.get("chord_melody_dataset") or {}
                f_ = REPO / "data" / "chord-melody-dataset" / sid_.rsplit("__", 1)[0] / meta_.get("file", "")
                if f_.is_file():
                    files[sid_] = f_
            process = cc.process_chord_melody_xml_file
        elif dataset == "jazzmus":
            import convert_jazzmus_to_cache as cj
            files = {Path(p).stem: Path(p) for p in glob.glob(str(REPO / "data/jazzmus/*.musicxml"))}
            process = cj.process_jazzmus_file
        elif dataset == "filobass":
            import convert_filobass_to_cache as cf
            root = REPO / "data" / "FiloBass_v1.0.0" / "FiloBass ISMIR Publication"
            files = {Path(p).stem: Path(p) for p in glob.glob(str(root / "musicxml" / "*.xml"))}
            lookup = cf._load_soloist_lookup(root)
            process = lambda f: cf.process_filobass_file(f, lookup)  # noqa: E731
        else:
            raise ValueError(dataset)
        done = 0
        for sid, (split, rec) in songs.items():
            f = files.get(sid)
            if f is None:
                meta = rec.get(dataset) or {}
                f = files.get(Path(meta.get("file", "")).stem)
            if f is None:
                continue
            cap.song = (sid, split)
            try:
                process(f)
            except Exception:  # noqa: BLE001
                pass
            done += 1
        print(f"  {dataset}: {done} songs parsed, {len(cap.calls)} chord labels", flush=True)
    finally:
        cw.parse_chord_symbol_with_noteseq = original
    counters = Counter()
    for (sid, split), label, transform, (root, ivs, inv) in cap.calls:
        if not ivs:
            continue
        i = counters[sid]; counters[sid] += 1
        tr = trace_noteseq_parse(label, transform)
        src = m21_reading(label) or {}
        yield {"dataset": dataset, "split": split, "song_id": sid, "event_index": i,
               "source_label": label, "source_pcs": src.get("pcs"), "source_root": src.get("root"),
               "source_bass": src.get("bass"), "converter_input": tr["transformed"],
               "converter_used": tr["used"], "converter_path": tr["path"],
               "cache_root": root, "cache_intervals": list(ivs), "cache_inversion": inv}


def events_nottingham() -> Iterable[Dict]:
    import convert_nottingham_to_cache as cn
    cap = _Capture()
    original = cn.chord_symbol_to_intervals

    def wrapper(chord_symbol):
        result = original(chord_symbol)
        cap.calls.append((cap.song, chord_symbol, result))
        return result

    # The converter splits tunes 80/10/10 after parsing, so a file holds tunes of every split: identify each tune
    # (parse_abc_header runs once per tune, before its chords) and take its split from the cache. Tunes the
    # cache does not contain (dropped by the converter) are skipped, as for the other datasets.
    original_header = cn.parse_abc_header

    def header_wrapper(section):
        meta = original_header(section)
        sid = f"nottingham_{cap.file_stem}_{meta.get('index')}"
        cap.song = (sid, songs[sid][0]) if sid in songs else None
        return meta

    cn.chord_symbol_to_intervals = wrapper
    cn.parse_abc_header = header_wrapper
    try:
        songs = cache_songs("nottingham")
        files = sorted({(rec.get("nottingham") or {}).get("file") for _, rec in songs.values()} - {None})
        for fname in files:
            path = REPO / "data" / "nottingham" / "nottingham_abc" / fname
            if not path.exists():
                continue
            cap.file_stem = Path(fname).stem
            cap.song = None
            try:
                cn.parse_abc_file(str(path))
            except Exception:  # noqa: BLE001
                pass
        print(f"  nottingham: {len(files)} files, {len(cap.calls)} chord labels", flush=True)
    finally:
        cn.chord_symbol_to_intervals = original
        cn.parse_abc_header = original_header
    cap.calls = [c for c in cap.calls if c[0] is not None]
    counters = Counter()
    for (sid, split), label, (root, ivs, inv) in cap.calls:
        i = counters[sid]; counters[sid] += 1
        src = m21_reading(label) or {}
        yield {"dataset": "nottingham", "split": split, "song_id": sid, "event_index": i,
               "source_label": label, "source_pcs": src.get("pcs"), "source_root": src.get("root"),
               "source_bass": src.get("bass"), "converter_input": label, "converter_used": label,
               "converter_path": "own_parser", "cache_root": root, "cache_intervals": list(ivs),
               "cache_inversion": inv}


def events_emopia() -> Iterable[Dict]:
    import pretty_midi
    import convert_emopia_plus_to_cache as ce
    cap = _Capture()
    orig_parse, orig_p2s = ce.parse_chord_symbol_with_noteseq, csl.pitches_to_chord_symbol
    last = {}

    def p2s(pitches):
        sym = orig_p2s(pitches)
        last["pitches"], last["symbol"] = list(pitches), sym
        return sym

    def parse(symbol, *, chord_symbol_transform=None):
        result = orig_parse(symbol, chord_symbol_transform=chord_symbol_transform)
        cap.calls.append((cap.song, symbol, list(last.get("pitches", [])), result))
        return result

    ce.parse_chord_symbol_with_noteseq, csl.pitches_to_chord_symbol = parse, p2s
    try:
        for sid, (split, _) in cache_songs("emopia_plus").items():
            f = REPO / "data" / "EMOPIA+" / "EMOPIA+" / "midis" / f"{sid}.mid"
            if not f.exists():
                continue
            cap.song = (sid, split)
            try:
                ce.extract_harmony(pretty_midi.PrettyMIDI(str(f)))
            except Exception:  # noqa: BLE001
                pass
    finally:
        ce.parse_chord_symbol_with_noteseq, csl.pitches_to_chord_symbol = orig_parse, orig_p2s
    print(f"  emopia_plus: {len(cap.calls)} chords", flush=True)
    counters = Counter()
    for (sid, split), label, pitches, (root, ivs, inv) in cap.calls:
        if not ivs:
            continue
        i = counters[sid]; counters[sid] += 1
        try:
            sroot = csl.chord_symbol_root(label) % 12
        except Exception:  # noqa: BLE001
            sroot = None
        yield {"dataset": "emopia_plus", "split": split, "song_id": sid, "event_index": i,
               "source_label": label, "source_detail": json.dumps({"pitches": pitches}),
               "source_pcs": pcs_str(pitches), "source_root": sroot,
               "source_bass": min(pitches) % 12 if pitches else None,
               "converter_input": label, "converter_used": label, "converter_path": "direct",
               "cache_root": root, "cache_intervals": list(ivs), "cache_inversion": inv}


def events_wjd() -> Iterable[Dict]:
    import convert_wjd_to_cache as cw2
    conn = sqlite3.connect(str(REPO / "data" / "wjazzd" / "wjazzd.db"))
    for sid, (split, _) in cache_songs("wjd").items():
        beats = cw2._forward_fill_chords(cw2._get_beats(conn, int(sid)))
        prev, i = None, 0
        for bar, beat, chord_name, _ in beats:
            if chord_name == prev:
                continue
            prev = chord_name
            if not chord_name:
                continue
            parsed = cw2.parse_wjd_chord(chord_name)
            if parsed is None:
                continue
            root, ivs, inv = parsed
            tr = trace_noteseq_parse(cw2.transform_wjd_chord_symbol(chord_name), lambda s: s)
            src = wjd_reading(chord_name) or {}
            yield {"dataset": "wjd", "split": split, "song_id": sid, "event_index": i,
                   "onset": cw2._beat_number(bar, beat), "source_label": chord_name,
                   "source_pcs": src.get("pcs"), "source_root": src.get("root"), "source_bass": src.get("bass"),
                   "converter_input": cw2.transform_wjd_chord_symbol(chord_name),
                   "converter_used": tr["used"], "converter_path": tr["path"],
                   "cache_root": root, "cache_intervals": list(ivs), "cache_inversion": inv}
            i += 1


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

EVENT_COLUMNS = [
    "dataset", "split", "song_id", "event_index", "onset", "offset",
    "source_label", "source_roman", "source_detail", "source_pcs", "source_root", "source_bass",
    "converter_input", "converter_used", "converter_path",
    "cache_root", "cache_intervals", "cache_inversion", "cache_pcs", "cache_bass",
    "token_name", "token_root", "token_bass", "token_pcs", "token_error",
    "in_vocab_old", "in_vocab_new",
    "sheetsage_agrees",
    "f_fallback_c_major", "f_simplified", "f_converter_pcs", "f_converter_root", "f_converter_bass",
    "f_inversion_dropped", "f_token_rerooted", "f_token_pcs", "f_token_bass",
    "f_end_pcs", "f_end_root", "f_end_bass",
]


def finish(ev: Dict, old: set, new: set) -> Dict:
    """Cache and token steps, and the flags."""
    ivs = list(ev["cache_intervals"])
    ci = cache_info(ev["cache_root"], ivs, ev["cache_inversion"])
    tk = token_info(int(ev["cache_root"]), tuple(ivs))
    ev.update(cache_pcs=ci["pcs"], cache_bass=ci["bass"], cache_intervals=json.dumps(ivs),
              token_name=tk.get("name"), token_root=tk.get("root"), token_bass=tk.get("bass"),
              token_pcs=tk.get("pcs"), token_error=tk.get("error"))
    name = tk.get("name")
    ev["in_vocab_old"] = int(name in old) if name else 0
    ev["in_vocab_new"] = int(name in new) if name else 0

    def diff(a, b):
        return None if a is None or b is None else int(a != b)

    if ev["dataset"] == "hooktheory" and ev.get("source_pcs") is not None:
        ev["sheetsage_agrees"] = int(ev["source_pcs"] == ci["pcs"] and ev.get("source_root") == ev["cache_root"])
    path = ev.get("converter_path") or ""
    ev["f_fallback_c_major"] = int(path == "fallback_c_major")
    ev["f_simplified"] = int(path == "simplified")
    ev["f_converter_pcs"] = diff(ev.get("source_pcs"), ci["pcs"])
    ev["f_converter_root"] = diff(ev.get("source_root"), ev["cache_root"])
    ev["f_converter_bass"] = diff(ev.get("source_bass"), ci["bass"])
    ev["f_inversion_dropped"] = int(int(ev["cache_inversion"] or 0) > 0)
    ev["f_token_rerooted"] = diff(ev["cache_root"], tk.get("root"))
    ev["f_token_pcs"] = diff(ci["pcs"], tk.get("pcs"))
    ev["f_token_bass"] = diff(ci["bass"], tk.get("bass"))
    ev["f_end_pcs"] = diff(ev.get("source_pcs"), tk.get("pcs"))
    ev["f_end_root"] = diff(ev.get("source_root"), tk.get("root"))
    ev["f_end_bass"] = diff(ev.get("source_bass"), tk.get("bass"))
    return ev


SCHEMA = f"""
CREATE TABLE events (id INTEGER PRIMARY KEY, {", ".join(c + (" INTEGER" if c.startswith(("f_", "in_vocab", "sheetsage", "event_index", "cache_root", "cache_inversion", "cache_bass", "token_root", "token_bass", "source_root", "source_bass")) else " REAL" if c in ("onset", "offset") else " TEXT") for c in EVENT_COLUMNS)});
CREATE TABLE structures (intervals TEXT, inversion INTEGER, root INTEGER, token_name TEXT,
    token_root INTEGER, token_bass INTEGER, token_pcs TEXT, rerooted INTEGER,
    in_vocab_old INTEGER, in_vocab_new INTEGER, n_events INTEGER, datasets TEXT);
CREATE TABLE vocab (vocab TEXT, name TEXT, onset_id INTEGER, hold_id INTEGER, pcs TEXT, is_slash INTEGER);
CREATE TABLE meta (key TEXT, value TEXT);
CREATE INDEX ev_ds ON events(dataset);
CREATE INDEX ev_token ON events(token_name);
CREATE INDEX ev_src ON events(source_label);
CREATE VIEW v_loss_summary AS
SELECT dataset, COUNT(*) AS events,
  SUM(f_fallback_c_major) AS fallback_c_major,
  SUM(f_simplified) AS simplified,
  SUM(f_converter_pcs) AS converter_changed_pcs,
  SUM(f_converter_root) AS converter_changed_root,
  SUM(f_converter_bass) AS converter_changed_bass,
  SUM(f_inversion_dropped) AS inversion_dropped,
  SUM(f_token_rerooted) AS token_rerooted,
  SUM(f_token_pcs) AS token_changed_pcs,
  SUM(f_end_pcs) AS end_to_end_pcs_differ,
  SUM(f_end_bass) AS end_to_end_bass_differ,
  SUM(CASE WHEN in_vocab_old = 0 THEN 1 ELSE 0 END) AS not_in_old_vocab,
  SUM(CASE WHEN in_vocab_new = 0 THEN 1 ELSE 0 END) AS not_in_new_vocab,
  SUM(CASE WHEN source_pcs IS NULL THEN 1 ELSE 0 END) AS no_source_reading
FROM events GROUP BY dataset;
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--out", default=str(REPO / "data" / "chord_conversion" / "chord_conversion.sqlite"))
    ap.add_argument("--raw_pickle", default=None,
                    help="optional pickle of the raw Hookpad chords/keys per song (speeds up reruns)")
    args = ap.parse_args()

    old, new = set(json.load(open(OLD_VOCAB))), set(json.load(open(NEW_VOCAB)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    db = sqlite3.connect(str(out))
    db.executescript(SCHEMA)

    sources = {
        "hooktheory": lambda: events_hooktheory(Path(args.raw_pickle) if args.raw_pickle else None),
        "pop909": events_pop909, "nottingham": events_nottingham,
        "wikifonia": lambda: events_musicxml("wikifonia"),
        "chord_melody_dataset": lambda: events_musicxml("chord_melody_dataset"),
        "jazzmus": lambda: events_musicxml("jazzmus"), "filobass": lambda: events_musicxml("filobass"),
        "emopia_plus": events_emopia, "wjd": events_wjd,
    }
    structure_count: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    insert = f"INSERT INTO events ({', '.join(EVENT_COLUMNS)}) VALUES ({', '.join('?' * len(EVENT_COLUMNS))})"
    for ds in args.datasets:
        t0 = time.time()
        print(f"{ds} ...", flush=True)
        rows = []
        for ev in sources[ds]():
            if ev.get("cache_root") is None:
                rows.append(tuple(ev.get(c) for c in EVENT_COLUMNS))
                continue
            ev = finish(ev, old, new)
            structure_count[(ev["cache_intervals"], ev["cache_inversion"])][ds] += 1
            rows.append(tuple(ev.get(c) for c in EVENT_COLUMNS))
            if len(rows) >= 20000:
                db.executemany(insert, rows); rows = []
        if rows:
            db.executemany(insert, rows)
        db.commit()
        n = db.execute("SELECT COUNT(*) FROM events WHERE dataset=?", (ds,)).fetchone()[0]
        print(f"  {ds}: {n:,} events in {time.time() - t0:.0f} s", flush=True)

    # structures: every distinct (intervals, inversion) in all 12 keys
    srows = []
    for (ivs_json, inv), cnt in structure_count.items():
        ivs = tuple(json.loads(ivs_json))
        for r in range(12):
            tk = token_info(r, ivs)
            name = tk.get("name")
            srows.append((ivs_json, inv, r, name, tk.get("root"), tk.get("bass"), tk.get("pcs"),
                          None if tk.get("root") is None else int(tk.get("root") != r),
                          int(name in old) if name else 0, int(name in new) if name else 0,
                          sum(cnt.values()), json.dumps(dict(cnt))))
    db.executemany("INSERT INTO structures VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", srows)

    # vocabularies
    from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
    for tag, path in (("old", OLD_VOCAB), ("new", NEW_VOCAB)):
        names = json.load(open(path))
        tk = HooktheoryTokenizer(chord_names=names)
        vrows = []
        for n in names:
            try:
                pcs = pcs_str({p % 12 for p in csl.chord_symbol_pitches(n)} | {csl.chord_symbol_bass(n) % 12})
            except Exception:  # noqa: BLE001
                pcs = None
            vrows.append((tag, n, tk.name_to_id[f"CHORD_ON_{n}"], tk.name_to_id[f"CHORD_{n}"], pcs,
                          int("/" in n.replace("6/9", ""))))
        db.executemany("INSERT INTO vocab VALUES (?,?,?,?,?,?)", vrows)

    db.executemany("INSERT INTO meta VALUES (?,?)", [
        ("built", datetime.now().isoformat(timespec="seconds")),
        ("script", "scripts/data/build_chord_conversion_db.py"),
        ("datasets", json.dumps(args.datasets)),
        ("doc", "journal/reference/CHORD_CONVERSION_DB.md"),
    ])
    db.commit()
    print(f"written {out}")
    for row in db.execute("SELECT * FROM v_loss_summary"):
        print(row)


if __name__ == "__main__":
    main()
