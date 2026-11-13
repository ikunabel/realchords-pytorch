"""Split the Nottingham ABC collections into one file per tune.

The raw corpus ships as 14 multi-tune files (`jigs.abc`, `waltzes.abc`, ...), each holding hundreds
of tunes separated by their `X:` index line. This writes one `.abc` per tune so individual pieces
can be opened, played or compared against their cached counterpart.

File names match the ids used in the cache (`nottingham_<collection>_<X>`), with the title appended
for readability:  `nottingham_jigs_166__Lamb_Skinnet.abc`

Also prints the meter (`M:`) distribution, which the cache does not preserve --
`convert_nottingham_to_cache.py` hardcodes 4/4 for every tune (see journal/DATASET_SUMMARY.md).

Usage:
    python scripts/data/split_nottingham_abc.py
    python scripts/data/split_nottingham_abc.py --src data/nottingham/nottingham_abc \
        --out data/nottingham/nottingham_abc/all_songs
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


def sanitize(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"\s+", "_", text)[:limit] or "untitled"


def field(lines: list[str], tag: str) -> str:
    for line in lines:
        if line.startswith(f"{tag}:"):
            return line[len(tag) + 1 :].strip()
    return ""


def split_file(path: Path, out_dir: Path) -> list[dict]:
    """One dict per tune written; tunes start at a line beginning with 'X:'."""
    tunes, current = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("X:"):
            if current:
                tunes.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        tunes.append(current)

    written = []
    for lines in tunes:
        index = field(lines, "X")
        title = field(lines, "T")
        song_id = f"nottingham_{path.stem}_{index}"
        out = out_dir / f"{song_id}__{sanitize(title)}.abc"
        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        written.append({"id": song_id, "title": title, "meter": field(lines, "M"),
                        "key": field(lines, "K"), "path": str(out)})
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/nottingham/nottingham_abc")
    ap.add_argument("--out", default="data/nottingham/nottingham_abc/all_songs")
    args = ap.parse_args()

    src, out_dir = Path(args.src), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tunes = []
    for abc in sorted(src.glob("*.abc")):
        tunes = split_file(abc, out_dir)
        all_tunes += tunes
        print(f"{abc.name:16s} {len(tunes):4d} tunes")

    meters = collections.Counter(t["meter"] for t in all_tunes)
    print(f"\n{len(all_tunes)} tunes written to {out_dir}\n")
    print("meter (M:) distribution -- not preserved in the cache, which hardcodes 4/4:")
    for meter, n in meters.most_common():
        print(f"  {meter or '(none)':8s} {n:4d}  {n / len(all_tunes):5.1%}")
    compound = sum(n for m, n in meters.items() if m.endswith("/8") and m.split("/")[0] in {"6", "9", "12"})
    print(f"\ncompound metres (pulse = dotted quarter = 6 frames): {compound} ({compound/len(all_tunes):.1%})")


if __name__ == "__main__":
    main()
