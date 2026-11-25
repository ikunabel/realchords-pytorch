"""Tonnetz helpers: chord name -> major/minor triad, and neo-Riemannian (P/L/R) distance between triads."""
import sys
from functools import lru_cache
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from realchords.utils.data_utils import chord_symbol_degrees  # noqa: E402

TRIADS = [(r, q) for r in range(12) for q in ("M", "m")]


@lru_cache(maxsize=None)
def triad(name):
    """(root pc, 'M'|'m') for chords with a major or minor third and a perfect (or no) fifth; else None
    (sus, dim, aug, power chords). 7ths and added notes are dropped: C7 -> C, Am7 -> Am."""
    try:
        root, deg, _ = chord_symbol_degrees(name)
    except Exception:  # noqa: BLE001
        return None
    if deg.get(3) not in (0, -1) or deg.get(5, 0) != 0:
        return None
    return (root, "M" if deg[3] == 0 else "m")


def _plr(t):
    r, q = t
    if q == "M":
        return [(r, "m"), ((r + 9) % 12, "m"), ((r + 4) % 12, "m")]  # P, R, L
    return [(r, "M"), ((r + 3) % 12, "M"), ((r + 8) % 12, "M")]


@lru_cache(maxsize=None)
def _from(src):
    dist, frontier = {src: 0}, [src]
    while frontier:
        nxt = []
        for t in frontier:
            for u in _plr(t):
                if u not in dist:
                    dist[u] = dist[t] + 1
                    nxt.append(u)
        frontier = nxt
    return dist


def plr_distance(a, b):
    """Fewest P/L/R flips from triad a to triad b (C -> Am 1, C -> G 2, C -> Dm 3, C -> F# 4)."""
    return _from(a)[b]
