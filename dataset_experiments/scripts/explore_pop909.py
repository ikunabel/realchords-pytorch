#!/usr/bin/env python3
"""Explore POP909 chord symbol parsing."""

from __future__ import annotations

import argparse

from _common import setup_imports, write_json, write_text

setup_imports()

from scripts.convert_pop909_to_cache import parse_chord_symbol


DEFAULT_SYMBOLS = [
    "B:maj",
    "C",
    "Am",
    "G7",
    "F#m7",
    "Dbmaj7",
    "E7(b9)",
    "A/C#",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbols",
        nargs="*",
        default=DEFAULT_SYMBOLS,
        help="Chord symbols to parse (default: built-in examples)",
    )
    args = parser.parse_args()

    results = {}
    lines = ["POP909 parse_chord_symbol results", ""]
    for symbol in args.symbols:
        root_pc, intervals, inversion = parse_chord_symbol(symbol)
        results[symbol] = {
            "root_pitch_class": root_pc,
            "root_position_intervals": intervals,
            "inversion": inversion,
        }
        lines.append(
            f"{symbol!r} -> root={root_pc}, intervals={intervals}, inversion={inversion}"
        )

    write_json("pop909", "chord_parsing.json", results)
    write_text("pop909", "chord_parsing.txt", "\n".join(lines) + "\n")

    summary = write_text(
        "pop909",
        "run_summary.txt",
        f"Parsed {len(args.symbols)} symbols -> output/pop909/chord_parsing.txt\n",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
