"""
Analyse wjazzd chord symbol coverage against the project's chord_names.json vocab.
Writes results to dataset_experiments/output/aria/wjazzd_chord_coverage.json.

Notation note: wjazzd uses '-' for minor (e.g. C-7) vs 'm' in the project vocab.
The script checks coverage both raw and after applying a simple '-' -> 'm' normalisation.
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]
DB = ROOT / "data" / "wjazzd" / "wjazzd.db"
VOCAB_FILE = ROOT / "data" / "cache" / "chord_names.json"
OUTPUT_DIR = Path(__file__).parents[1] / "output" / "wjazzd"

with open(VOCAB_FILE) as f:
    vocab = set(json.load(f))

conn = sqlite3.connect(DB)
rows = conn.execute("SELECT chord, signature FROM beats WHERE chord != ''").fetchall()
conn.close()

# Count raw symbols (all time signatures)
raw_counts = Counter(r[0] for r in rows)

# Count only 4/4 symbols
counts_4_4 = Counter(r[0] for r in rows if r[1] == "4/4")

def normalise(symbol: str) -> str:
    """Convert wjazzd minor notation C-7 -> Cm7."""
    return symbol.replace("-", "m")

def coverage_report(counts: Counter) -> dict:
    total_events = sum(counts.values())
    unique_symbols = set(counts.keys())

    in_vocab_raw  = {s for s in unique_symbols if s in vocab}
    oov_raw       = unique_symbols - in_vocab_raw

    in_vocab_norm = {s for s in unique_symbols if normalise(s) in vocab}
    oov_norm      = unique_symbols - in_vocab_norm

    events_covered_raw  = sum(counts[s] for s in in_vocab_raw)
    events_covered_norm = sum(counts[s] for s in in_vocab_norm)

    top_oov_raw  = sorted(oov_raw,  key=lambda s: -counts[s])[:30]
    top_oov_norm = sorted(oov_norm, key=lambda s: -counts[s])[:30]

    return {
        "total_events": total_events,
        "unique_symbols": len(unique_symbols),
        "raw": {
            "in_vocab": len(in_vocab_raw),
            "oov": len(oov_raw),
            "event_coverage_pct": round(100 * events_covered_raw / total_events, 2),
            "top_oov": {s: counts[s] for s in top_oov_raw},
        },
        "normalised": {
            "in_vocab": len(in_vocab_norm),
            "oov": len(oov_norm),
            "event_coverage_pct": round(100 * events_covered_norm / total_events, 2),
            "top_oov": {normalise(s): counts[s] for s in top_oov_norm},
        },
    }

result = {
    "vocab_size": len(vocab),
    "all_signatures": coverage_report(raw_counts),
    "four_four_only": coverage_report(counts_4_4),
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUTPUT_DIR / "wjazzd_chord_coverage.json"
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"Wrote {out}")
