"""Distance from the tonic on the Tonnetz, frame by frame: does the model wander, end at home, stay home?

For free-running runs (plot_onset_hold_free_running.py JSON logs: melody from a Hooktheory song, chords sampled) and
the song's real chords, every frame's sounding chord is reduced to a major or minor triad (7ths and added notes
dropped: C7 -> C, Am7 -> Am; sus, dim, aug, power chords and silence are "unplaced") and its neo-Riemannian
distance from the song's tonic triad is computed: the fewest P/L/R flips between triangles on the Tonnetz
(C -> Am, Em, Cm 1; C -> F, G 2; C -> Dm 3). The tonic triad comes from the annotated key (tonic, and major/minor
from the scale's third).

Per run: share of placed frames at the tonic, mean distance, distance of the last chord, number and mean length of
tonic visits (frames), share unplaced. Plot: per song, the distance curve of the real chords (black) and of each
model's seeds.

    python scripts/eval/tonnetz_tonic_distance.py <free_running_*.json> [...] [--out_dir DIR]
"""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]
from realchords.utils.data_utils import chord_symbol_degrees  # noqa: E402


def triad(name):
    """(root pc, 'M'|'m') or None."""
    try:
        root, deg, _ = chord_symbol_degrees(name)
    except Exception:  # noqa: BLE001
        return None
    if 3 not in deg or deg.get(5, 0) != 0:
        return None
    return (root, "M" if deg[3] == 0 else "m" if deg[3] == -1 else None) if deg[3] in (0, -1) else None


def plr(t):
    r, q = t
    if q == "M":
        return [(r, "m"), ((r + 9) % 12, "m"), ((r + 4) % 12, "m")]  # P, R (relative minor), L (mediant minor)
    return [(r, "M"), ((r + 3) % 12, "M"), ((r + 8) % 12, "M")]


def all_distances(src):
    dist, frontier = {src: 0}, [src]
    while frontier:
        nxt = []
        for t in frontier:
            for u in plr(t):
                if u not in dist:
                    dist[u] = dist[t] + 1
                    nxt.append(u)
        frontier = nxt
    return dist


def sounding(onsets, n):
    out, cur, d = [None] * n, None, dict(onsets)
    for k in range(n):
        cur = d.get(k, cur)
        out[k] = cur
    return out


def stats(curve):
    placed = [x for x in curve if x is not None]
    visits, k = [], 0
    while k < len(curve):
        if curve[k] == 0:
            j = k
            while j < len(curve) and curve[j] == 0:
                j += 1
            visits.append(j - k)
            k = j
        else:
            k += 1
    last = next((x for x in reversed(curve) if x is not None), None)
    return {"at_tonic": np.mean([x == 0 for x in placed]) if placed else np.nan,
            "mean_dist": np.mean(placed) if placed else np.nan, "last": last, "visits": len(visits),
            "visit_len": np.mean(visits) if visits else np.nan, "unplaced": 1 - len(placed) / len(curve)}


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+")
ap.add_argument("--out_dir", default=str(REPO / "logs/eval/onset_hold_session"))
args = ap.parse_args()

keys = {}
for split in ("test",):
    for line in open(REPO / f"data/cache/hooktheory/{split}.jsonl"):
        d = json.loads(line)
        k = d["annotations"]["keys"]
        if len(k) == 1:
            third = sum(k[0]["scale_degree_intervals"][:2])
            keys[d["hooktheory"]["urls"]["song"]] = (k[0]["tonic_pitch_class"], "M" if third == 4 else "m")

curves = collections.defaultdict(dict)  # song -> label -> list of curves
rows = []
for path in args.runs:
    meta = json.load(open(path))
    prime = meta.get("prime_bars", 0)
    for r in meta["runs"]:
        key = keys.get(r["song_url"])
        if key is None:
            continue
        D = all_distances(key)
        n = len(r["p_onset"])
        to_curve = lambda onsets: [None if c is None or triad(c) is None else D.get(triad(c)) for c in sounding(onsets, n)]
        title = r["song_url"].rsplit("/", 1)[-1]
        gt = to_curve(r["real_onsets"])
        curves[title]["real chords"] = [gt]
        label = f"{r['model']}{' primed' if prime else ''}"
        curves[title].setdefault(label, []).append(to_curve(r["model_onsets"]))
        rows.append({"song": title, "label": label, "seed": meta.get("seed"), **stats(curves[title][label][-1])})
        rows.append({"song": title, "label": "real chords", "seed": None, **stats(gt)})

# table: per label, mean over songs and seeds (real chords once per song)
seen, table = set(), collections.defaultdict(list)
for x in rows:
    if x["label"] == "real chords":
        if x["song"] in seen:
            continue
        seen.add(x["song"])
    table[x["label"]].append(x)
print("| chords | runs | at tonic (placed frames) | mean distance | last chord at tonic | tonic visits | mean visit length (frames) | unplaced |")
print("|---|---|---|---|---|---|---|---|")
for lab, xs in table.items():
    m = lambda k: np.nanmean([x[k] for x in xs])
    print(f"| {lab} | {len(xs)} | {100 * m('at_tonic'):.0f}% | {m('mean_dist'):.2f} | "
          f"{100 * np.mean([x['last'] == 0 for x in xs]):.0f}% | {m('visits'):.1f} | {m('visit_len'):.1f} | {100 * m('unplaced'):.0f}% |")
for song in curves:
    print(song, {lab: [stats(c)["last"] for c in cs] for lab, cs in curves[song].items()})

out = Path(args.out_dir)
fig, axes = plt.subplots(len(curves), 1, figsize=(15, 2.6 * len(curves)), sharex=False)
axes = np.atleast_1d(axes)
colors = {"GAPT 7 datasets": "#d62728", "Online MLE": "#1f77b4", "GAPT 7 datasets primed": "#ff9896",
          "Online MLE primed": "#aec7e8"}
for ax, (song, labs) in zip(axes, curves.items()):
    for lab, cs in labs.items():
        if lab == "real chords":
            continue
        for i, c in enumerate(cs):
            y = [np.nan if v is None else v + 0.08 * (i - 1) for v in c]
            ax.step(range(len(y)), y, where="post", color=colors.get(lab, "gray"), lw=1, alpha=0.8,
                    label=lab if i == 0 else None)
    y = [np.nan if v is None else v for v in labs["real chords"][0]]
    ax.step(range(len(y)), y, where="post", color="k", lw=2.2, label="real chords")
    for k in range(0, len(y) + 1, 16):
        ax.axvline(k, color="k", lw=0.4, alpha=0.3)
    ax.set_yticks(range(0, 7))
    ax.set_ylim(-0.3, 6.3)
    ax.invert_yaxis()
    ax.set_ylabel("Tonnetz flips\nfrom tonic")
    ax.set_title(song, fontsize=9)
axes[0].legend(fontsize=7, ncol=5, loc="upper right")
axes[-1].set_xlabel("frame (bar lines every 16); 0 = tonic triad, gaps = chord not placeable (sus/dim/aug/silence)")
fig.tight_layout()
png = out / "tonnetz_tonic_distance.png"
fig.savefig(png, dpi=110, bbox_inches="tight")
print("wrote", png)
