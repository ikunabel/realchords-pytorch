"""Patch the real ABC metre into the existing Nottingham cache, in place.

`convert_nottingham_to_cache.py` used to hardcode `{"beats_per_bar": 4, "beat_unit": 4}` for every
tune, discarding the `M:` header: 36% of the corpus is in compound metre (6/8, 9/8), which makes any
bar- or pulse-relative metric misread it (see journal/DATASET_SUMMARY.md). The converter now parses
the header, but re-running it is not an option here: it would read a different source directory than
the cache was built from (1,034 tunes vs the cache's 1,019) and therefore reshuffle the splits.

This script instead rewrites only the `annotations.meters` field of each cached entry, looked up by
the tune's own provenance (`nottingham.file` + `nottingham.index`) in the raw ABC collections. Note
content, ordering and splits are untouched, so existing checkpoints and evaluations stay valid --
the model never sees this field.

Usage:
    python scripts/data/fix_nottingham_meters.py --dry_run
    python scripts/data/fix_nottingham_meters.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "convert_data_to_cache"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from convert_nottingham_to_cache import parse_meter  # noqa: E402


def abc_meters(abc_dir: Path) -> dict:
    """{(collection file, X index): M: header} for every tune in the raw collections."""
    out = {}
    for abc in sorted(abc_dir.glob("*.abc")):
        index = None
        for line in abc.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("X:"):
                index = line[2:].strip()
            elif line.startswith("M:") and index is not None:
                out.setdefault((abc.name, index), line[2:].strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", default="data/cache/nottingham")
    ap.add_argument("--abc_dir", default="data/nottingham/nottingham_abc")
    ap.add_argument("--dry_run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    meters = abc_meters(Path(args.abc_dir))
    print(f"{len(meters)} tunes found in {args.abc_dir}")

    stats = collections.Counter()
    for path in sorted(Path(args.cache_dir).glob("*.jsonl")):
        lines, changed, missing = [], 0, 0
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            key = (entry["nottingham"]["file"], str(entry["nottingham"]["index"]))
            header = meters.get(key)
            if header is None:
                missing += 1
            else:
                new = [parse_meter(header)]
                if entry["annotations"].get("meters") != new:
                    entry["annotations"]["meters"] = new
                    changed += 1
                stats[header] += 1
            lines.append(json.dumps(entry))
        if not args.dry_run:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {path.name:26s} {len(lines):5d} entries, {changed:5d} metres corrected, "
              f"{missing:3d} not found in the ABC sources")

    print("\nmetre distribution now stored in the cache:")
    total = sum(stats.values())
    for meter, n in stats.most_common():
        print(f"  {meter:6s} {n:5d}  {n / max(1, total):5.1%}")
    if args.dry_run:
        print("\n(dry run: nothing written)")


if __name__ == "__main__":
    main()
