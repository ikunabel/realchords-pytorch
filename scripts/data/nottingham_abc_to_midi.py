"""Render the split Nottingham ABC tunes to MIDI, for listening to the raw data.

Companion to `split_nottingham_abc.py`: that writes one `.abc` per tune, this renders each to
`.mid` via music21, keeping the melody and the realised chord symbols, so a raw tune can be
compared by ear with its cached/converted counterpart (whose chord and melody lanes are what the
model actually sees). File names keep the cache id, e.g. `nottingham_jigs_166__Lamb_Skinnet.mid`.

The tempo is whatever music21 defaults to; only the relative timing matters here. Tunes that music21
cannot parse are reported and skipped.

Usage:
    python scripts/data/nottingham_abc_to_midi.py
    python scripts/data/nottingham_abc_to_midi.py --src data/nottingham/nottingham_abc/all_songs \
        --out data/nottingham/nottingham_abc/all_songs_midi --limit 20
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/nottingham/nottingham_abc/all_songs")
    ap.add_argument("--out", default="data/nottingham/nottingham_abc/all_songs_midi")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N tunes")
    ap.add_argument("--overwrite", action="store_true", help="re-render tunes that already exist")
    args = ap.parse_args()

    from music21 import converter

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.abc"))[: args.limit]
    done = failed = skipped = 0
    for i, abc in enumerate(files, 1):
        out = out_dir / (abc.stem + ".mid")
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            converter.parse(str(abc)).write("midi", fp=str(out))
            done += 1
        except Exception as exc:  # noqa: BLE001 -- one bad tune shouldn't stop the batch
            failed += 1
            print(f"  failed: {abc.name} ({type(exc).__name__}: {exc})")
        if i % 100 == 0:
            print(f"  {i}/{len(files)} processed")
    print(f"\n{done} written, {skipped} already present, {failed} failed -> {out_dir}")


if __name__ == "__main__":
    main()
