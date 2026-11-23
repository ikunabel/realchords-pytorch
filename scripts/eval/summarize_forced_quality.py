"""Summarise --forced_quality results of analyze_chord_onset_timing.py by position class.

Usage: python scripts/eval/summarize_forced_quality.py logs/eval/chord_onset_timing/forced_quality
"""
import json, glob, sys
CLASSES = [("downbeat", [0]), ("beat 3", [8]), ("beats 2, 4", [4, 12]),
           ("and", [2, 6, 10, 14]), ("16ths", [1, 3, 5, 7, 9, 11, 13, 15])]
f = sorted(glob.glob(sys.argv[1] + "/*.json"))[-1]
d = json.load(open(f))
print(f)
for r in d["results"]:
    fq = r["forced_quality"]
    bp = {x["bucket"]: x for x in fq["bar_position"]}
    po = {x["bucket"]: x for x in r["bar_position"]}
    rc = fq["real_change"]
    print(f"\n{r['model']} ({r['songs']} songs), one forced draw at T {fq['temperature']}")
    print(f"  {'':16s} {'frames':>8s} {'P(chg)':>7s} {'-logP':>6s} {'re-strike':>9s} {'next real':>9s} {'other':>6s} {'note-in-chord':>13s} {'in key':>7s}")
    def row(label, rows, prows):
        n = sum(x["n_frames"] for x in rows)
        w = lambda key: sum(x[key] * x["n_frames"] for x in rows) / n
        nk = sum(x["n_key_frames"] for x in rows)
        ik = sum((x["in_key"] or 0) * x["n_key_frames"] for x in rows) / nk if nk else float("nan")
        p = sum(x["mean_p_onset"] * x["n_frames"] for x in prows) / sum(x["n_frames"] for x in prows) if prows else float("nan")
        print(f"  {label:16s} {n:8,d} {p:7.3f} {w('surprisal'):6.2f} {w('m_current'):9.3f} {w('m_next'):9.3f} {w('m_other'):6.3f} {w('note_in_chord'):13.3f} {ik:7.3f}")
    for label, bs in CLASSES:
        row(label, [bp[b] for b in bs], [po[b] for b in bs])
    row("real changes", rc, [])
    for label, bs in CLASSES[:2]:
        sub = [x for x in rc if x["bucket"] in bs]
        if sub: row(f"  real, {label}", sub, [])
