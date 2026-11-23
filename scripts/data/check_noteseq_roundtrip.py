"""Hooktheory: name every chord event with note_seq, read it back, and attribute each disagreement to
note_seq bug A (reader flattens every add7) or bug B (writer drops degrees named like the absolute bass pitch
class). See journal/reference/CHORD_CONVERSION_DB.md, bug 1.

    python scripts/data/check_noteseq_roundtrip.py
"""
import sqlite3, collections, json, re, copy, sys
from note_seq import chord_symbols_lib as c

DB = "data/chord_conversion/chord_conversion.sqlite"

def stack(root, iv):
    p=[root]; cur=root
    for i in iv: cur+=i; p.append(cur)
    return p

# --- patched writer: bass degree relative to the chosen root (bug B fix) ---
import inspect, textwrap
src = inspect.getsource(c.pitches_to_chord_symbol)
assert src.count("_SCALE_DEGREES[bass]") == 2
src = src.replace("_SCALE_DEGREES[bass]", "_SCALE_DEGREES[(bass - best_root) % 12]").replace(
    "def pitches_to_chord_symbol", "def writer_fixed")
ns = dict(vars(c)); exec(src, ns); writer_fixed = ns["writer_fixed"]

# --- patched reader: no implicit flat on add7 (bug A fix) ---
def _add_fixed(degrees, degree, alter):
    if degree in degrees: raise c.ChordSymbolError("dup")
    degrees[degree] = alter
def reader(name, fixed):
    orig = dict(c._DEGREE_MODIFICATIONS)
    if fixed:
        for k in ("add","add#","addb"): c._DEGREE_MODIFICATIONS[k] = (_add_fixed, orig[k][1])
    try: return set(c.chord_symbol_pitches(name))
    except Exception as e: return ("ERR", str(e))
    finally: c._DEGREE_MODIFICATIONS.clear(); c._DEGREE_MODIFICATIONS.update(orig)

con = sqlite3.connect(DB)
rows = con.execute("select cache_root, cache_intervals from events where dataset='hooktheory'").fetchall()
cnt = collections.Counter((r, iv) for r, iv in rows)
stats = collections.Counter(); ex = collections.defaultdict(list); struct = collections.Counter()
for (root, iv), n in cnt.items():
    p = stack(root, json.loads(iv)); pcs = set(x % 12 for x in p)
    name = c.pitches_to_chord_symbol(p)
    ok = reader(name, False) == pcs
    if ok: stats["ok"] += n; continue
    nameB = writer_fixed(p)
    okB = reader(nameB, False) == pcs           # writer fix alone
    okA = reader(name, True) == pcs              # reader fix alone
    okAB = reader(nameB, True) == pcs            # both
    cat = "B (bass filter)" if okB else "A (add7)" if okA else "A+B" if okAB else "other"
    stats[cat] += n; struct[cat] += 1
    if len(ex[cat]) < 6: ex[cat].append((n, name, nameB, sorted(pcs), reader(name, False)))
tot = sum(cnt.values())
print("events", tot, "distinct (root,intervals)", len(cnt))
for k, v in stats.most_common(): print(f"{k:16} {v:7} events ({100*v/tot:.2f}%)  {struct.get(k,'')} structures")
for k, L in ex.items():
    print("\n", k)
    for e in L: print("  ", e)
