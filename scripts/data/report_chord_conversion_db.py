#!/usr/bin/env python3
"""Print the loss overview of the chord-conversion database
(scripts/data/build_chord_conversion_db.py; journal/reference/CHORD_CONVERSION_DB.md).

    python scripts/data/report_chord_conversion_db.py [--db ...] [--examples 6] [--markdown]

Sections: per dataset, the share of events each kind of loss affects; for each
kind, the most frequent examples; the vocabulary tokens whose name, read back
by note_seq, gives other notes than the chord structure that produced it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
N = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LOSSES = [
    ("f_fallback_c_major", "converter fell back to C major"),
    ("f_simplified", "converter simplified the label"),
    ("f_converter_pcs", "converter changed the notes"),
    ("f_converter_root", "converter changed the root"),
    ("f_converter_bass", "converter changed / lost the bass"),
    ("f_inversion_dropped", "inversion dropped at tokenization"),
    ("f_token_rerooted", "note_seq name has another root (re-rooted)"),
    ("f_token_pcs", "note_seq name reads back as other notes"),
    ("f_end_pcs", "token's notes differ from the source"),
    ("f_end_bass", "token's bass differs from the source"),
]


def notes(s):
    return " ".join(N[int(x)] for x in s.split(",")) if s else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "data" / "chord_conversion" / "chord_conversion.sqlite"))
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    db = sqlite3.connect(args.db)

    datasets = [r[0] for r in db.execute("SELECT dataset FROM events GROUP BY dataset ORDER BY COUNT(*) DESC")]
    print("# Share of events affected, per dataset\n")
    head = "| loss | " + " | ".join(datasets) + " |"
    print(head)
    print("|" + "---|" * (len(datasets) + 1))
    totals = {d: db.execute("SELECT COUNT(*) FROM events WHERE dataset=?", (d,)).fetchone()[0] for d in datasets}
    print("| events | " + " | ".join(f"{totals[d]:,}" for d in datasets) + " |")
    for col, label in LOSSES:
        cells = []
        for d in datasets:
            n, known = db.execute(f"SELECT SUM({col}), COUNT({col}) FROM events WHERE dataset=?", (d,)).fetchone()
            cells.append("–" if not known else f"{100 * (n or 0) / known:.2f}%")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for vocab in ("old", "new"):
        cells = []
        for d in datasets:
            n = db.execute(f"SELECT SUM(in_vocab_{vocab} = 0) FROM events WHERE dataset=?", (d,)).fetchone()[0]
            cells.append(f"{100 * (n or 0) / totals[d]:.2f}%")
        print(f"| token not in the {vocab} vocabulary | " + " | ".join(cells) + " |")
    hk = db.execute("SELECT SUM(sheetsage_agrees), COUNT(sheetsage_agrees) FROM events WHERE dataset='hooktheory'").fetchone()
    if hk[1]:
        print(f"\nHooktheory: our Hookpad renderer agrees with Sheet Sage on {100 * hk[0] / hk[1]:.2f}% of {hk[1]:,} chords "
              f"(disagreements count as 'converter changed the notes').")

    print("\n# Most frequent examples per loss\n")
    for col, label in LOSSES:
        print(f"## {label}\n")
        for d in datasets:
            rows = db.execute(
                f"""SELECT source_label, converter_used, converter_path, source_pcs, cache_pcs,
                           cache_inversion, token_name, token_pcs, COUNT(*) n
                    FROM events WHERE dataset=? AND {col}=1
                    GROUP BY source_label, cache_root, cache_intervals, cache_inversion
                    ORDER BY n DESC LIMIT ?""", (d, args.examples)).fetchall()
            if not rows:
                continue
            print(f"{d}:")
            for src, used, path, spcs, cpcs, inv, tok, tpcs, n in rows:
                print(f"  {n:6d}×  {src!s:20s} [{notes(spcs)}] → conv {used!s:14s} ({path}) → cache [{notes(cpcs)}] inv {inv}"
                      f" → token {tok!s:20s} [{notes(tpcs)}]")
        print()

    print("# Vocabulary tokens that don't read back to their chord\n")
    rows = db.execute("SELECT intervals, root, token_name, token_pcs, n_events, in_vocab_old, in_vocab_new FROM structures").fetchall()
    bad = {}
    for ivs, root, name, tpcs, n, iold, inew in rows:
        st = [0]
        for iv in json.loads(ivs):
            st.append(st[-1] + iv)
        spcs = ",".join(str(p) for p in sorted({(root + s) % 12 for s in st}))
        if name and tpcs and spcs != tpcs:
            b = bad.setdefault(name, [spcs, tpcs, 0, iold, inew])
            b[2] += n
    for vocab in ("old", "new"):
        names = {r[0] for r in db.execute("SELECT name FROM vocab WHERE vocab=?", (vocab,))}
        hit = [k for k in bad if k in names]
        print(f"{vocab} vocabulary: {len(hit)} of {len(names)} chord names read back as other notes than the chord they were made from")
    for name, (spcs, tpcs, n, _, _) in sorted(bad.items(), key=lambda x: -x[1][2])[:15]:
        print(f"  {name:22s} made from [{notes(spcs)}]  reads back as [{notes(tpcs)}]")


if __name__ == "__main__":
    main()
