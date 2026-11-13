"""Check every dataset cache for the defect classes we have actually been bitten by.

Each check exists because it missed something real (see journal/DATASET_SUMMARY.md):

  - **leakage / duplicates** -- JAZZMUS shipped exact-content duplicates across splits (23% of test
    duplicated a training song); the dedup code existed but the cache predated it.
  - **augmentation inflation** -- the Chord Melody Dataset cache used the corpus's own 12-key
    duplication as augmentation, so TRAIN held ~12 rows per song while VALID/TEST held 1, and its
    "4,252 songs" were 473 unique pieces.
  - **melody register** -- 43% of that corpus's melodies sat an octave too high because their source
    files omit `<transpose><octave-change>`. Note tokens are absolute pitch, so this is wrong input,
    not a cosmetic issue.
  - **metre** -- Nottingham's converter hardcoded 4/4 while 36% of the corpus is in compound metre.
  - **code/cache drift** -- every one of the above was a cache built by superseded code, so the last
    commit touching each converter is compared against the cache's build time.

Runs on the cache JSONL alone (stdlib only, no torch), so it is cheap enough to run before any
training batch.

Usage:
    python scripts/data/audit_caches.py
    python scripts/data/audit_caches.py --cache_dir data/cache --datasets jazzmus nottingham
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ZERO_OCTAVE = 60
SPLITS = ("train", "valid", "test")
# the metadata block is named after the dataset except where the converter predates that convention
META_KEY = {"wjd": "wjazzd"}
ID_FIELD = {"wjd": "melid"}
# only the Chord Melody Dataset names its transposed copies "<song>__<key>"; stripping a "__" suffix
# anywhere else merges distinct songs and manufactures false split collisions
STRIP_KEY_SUFFIX = {"chord_melody_dataset"}


def song_key(entry: dict, dataset: str) -> str:
    """Stable per-song identity, ignoring transposed-copy suffixes where the corpus uses them."""
    meta = entry.get(META_KEY.get(dataset, dataset)) or {}
    for field in (ID_FIELD.get(dataset), "id", "song_id", "melid", "file"):
        if field and meta.get(field) is not None:
            value = str(meta[field])
            return value.rsplit("__", 1)[0] if dataset in STRIP_KEY_SUFFIX else value
    raise KeyError(f"{dataset}: no identity field in metadata {sorted(meta)}")


def content_hash(entry: dict) -> str:
    """Hash of the actual musical content, so duplicates are found regardless of metadata."""
    ann = entry["annotations"]
    payload = [
        [(round(n["onset"], 4), round(n["offset"], 4), n["pitch_class"], n["octave"])
         for n in ann.get("melody", [])],
        [(round(h["onset"], 4), round(h["offset"], 4), h["root_pitch_class"],
          tuple(h.get("root_position_intervals", [])), h.get("inversion", 0))
         for h in ann.get("harmony", [])],
    ]
    return hashlib.md5(json.dumps(payload).encode()).hexdigest()


def load(path: Path, dataset: str) -> list:
    rows = []
    for line in path.open(encoding="utf-8"):
        entry = json.loads(line)
        ann = entry["annotations"]
        mel = ann.get("melody", [])
        rows.append({
            "song": song_key(entry, dataset),
            "hash": content_hash(entry),
            "n_melody": len(mel),
            "n_harmony": len(ann.get("harmony", [])),
            "median_midi": (statistics.median(ZERO_OCTAVE + n["octave"] * 12 + n["pitch_class"]
                                             for n in mel) if mel else None),
            "meters": ann.get("meters") or [],
            "bad_timing": sum(1 for n in mel if n["offset"] <= n["onset"] or n["onset"] < 0),
            "off_grid": sum(1 for n in mel if abs((n["onset"] * 4) - round(n["onset"] * 4)) > 1e-6),
        })
    return rows


def octave_split(medians: list) -> str:
    """Detect the CMD failure mode: two register modes an octave apart, not merely a high corpus.

    A threshold on register cannot distinguish a genuinely high corpus (EMOPIA+, median 73, one
    smooth mode) from one where part of the corpus is transposed (CMD before the fix, two modes 12
    apart). Bins at 3 semitones, then looks for a secondary mode 9-15 semitones from the main one
    holding at least 30% of its mass.
    """
    if len(medians) < 40:
        return ""
    hist = collections.Counter(int(m) // 3 * 3 for m in medians)
    peak = max(hist, key=lambda b: hist[b])
    for b, n in hist.items():
        if 9 <= abs(b - peak) <= 15 and n >= 0.30 * hist[peak]:
            nearer = [hist.get(x, 0) for x in range(min(b, peak) + 3, max(b, peak), 3)]
            if nearer and min(nearer) < 0.6 * min(n, hist[peak]):   # a real trough between them
                return f"bimodal register: modes at MIDI ~{peak} and ~{b} (octave defect?)"
    return ""


def converter_drift(dataset: str, cache_mtime: float) -> str:
    conv = Path(f"scripts/convert_data_to_cache/convert_{dataset}_to_cache.py")
    if not conv.exists():
        return "no converter"
    out = subprocess.run(["git", "log", "-1", "--format=%ct %h %s", "--", str(conv)],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        return "untracked"
    ts, sha, subject = out.split(" ", 2)
    return "ok" if int(ts) <= cache_mtime else f"DRIFT: {sha} {subject[:40]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    root = Path(args.cache_dir)
    names = args.datasets or sorted(d.name for d in root.iterdir()
                                    if d.is_dir() and (d / "train.jsonl").exists()
                                    and not d.name.endswith(("__pre_octave_fix", "__rebuild",
                                                             "__rebuild2")))
    problems = []
    for dataset in names:
        d = root / dataset
        per_split = {s: load(d / f"{s}.jsonl", dataset) for s in SPLITS if (d / f"{s}.jsonl").exists()}
        rows = [r for rs in per_split.values() for r in rs]
        if not rows:
            continue
        aug = {}
        for s in SPLITS:
            p = d / f"{s}_augmented.jsonl"
            if p.exists():
                aug[s] = sum(1 for _ in p.open())

        songs = {s: {r["song"] for r in rs} for s, rs in per_split.items()}
        hashes = {s: collections.Counter(r["hash"] for r in rs) for s, rs in per_split.items()}
        flags = []

        # 1. leakage by song identity and by exact content
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            if a in songs and b in songs:
                if songs[a] & songs[b]:
                    flags.append(f"{a}/{b} share {len(songs[a] & songs[b])} song ids")
                shared = set(hashes[a]) & set(hashes[b])
                if shared:
                    flags.append(f"{a}/{b} share {len(shared)} identical contents "
                                 f"({len(shared)/max(1,len(hashes[b])):.0%} of {b})")
        # 2. duplicates inside a split
        for s, h in hashes.items():
            dup = sum(n - 1 for n in h.values() if n > 1)
            if dup:
                flags.append(f"{s} has {dup} duplicate rows ({dup/len(per_split[s]):.0%})")
        # 3. augmentation inflation: rows per unique song should be 1 in the unaugmented files
        for s, rs in per_split.items():
            ratio = len(rs) / max(1, len(songs[s]))
            if ratio > 1.5:
                flags.append(f"{s} has {ratio:.1f} rows per song (built-in duplication?)")
        # 4. augmented/unaugmented relationship
        for s, n in aug.items():
            base = len(per_split.get(s, []))
            if s == "train" and base and not (1.2 <= n / base <= 15):
                flags.append(f"train_augmented/train = {n/base:.1f} (expected ~13, or 1 if none)")
            if s != "train" and base and n != base:
                flags.append(f"{s}_augmented has {n} rows vs {base} unaugmented")
        # 5. melody register
        meds = [r["median_midi"] for r in rows if r["median_midi"] is not None]
        corpus_med = statistics.median(meds) if meds else float("nan")
        high = sum(1 for m in meds if m >= 74) / max(1, len(meds))
        split_flag = octave_split(meds)
        if split_flag:
            flags.append(split_flag)
        # 6. degenerate rows and timing
        for label, key in (("no melody", "n_melody"), ("no harmony", "n_harmony")):
            n = sum(1 for r in rows if r[key] == 0)
            if n:
                flags.append(f"{n} rows with {label}")
        for label, key in (("bad onset/offset", "bad_timing"), ("off 16th grid", "off_grid")):
            n = sum(r[key] for r in rows)
            if n:
                flags.append(f"{n} melody notes {label}")
        # 7. metre
        units = collections.Counter(
            (m[0].get("beats_per_bar"), m[0].get("beat_unit")) if r["meters"] else None
            for r in rows for m in [r["meters"]])
        metre = "single 4/4 (hardcoded?)" if list(units) == [(4, 4)] else \
                f"{len(units)} distinct, top {units.most_common(2)}"
        drift = converter_drift(dataset, (d / "train.jsonl").stat().st_mtime)
        # NOT a flag: this test cannot distinguish a semantic change from a file move or a
        # metadata-only edit -- every drift has to be read before it means anything.

        print(f"\n=== {dataset}")
        print(f"  rows {len(rows):6d}  unique songs {len(set(r['song'] for r in rows)):6d}  "
              f"splits {'/'.join(str(len(per_split.get(s, []))) for s in SPLITS)}  "
              f"aug train {aug.get('train', '-')}")
        print(f"  median melody MIDI {corpus_med:.0f}  >=74: {high:.1%}   metre: {metre}")
        print(f"  converter: {drift}")
        if flags:
            for f in flags:
                print(f"  !! {f}")
            problems.append((dataset, flags))
        else:
            print("  clean")

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} dataset(s) with findings: " + ", ".join(d for d, _ in problems))
    else:
        print("all caches clean")


if __name__ == "__main__":
    main()
