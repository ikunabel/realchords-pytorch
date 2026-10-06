"""Chord-progression statistics: is the harmony *grammatical*, not just consonant?

Motivation. `note_in_chord_ratio` is trivially maximised by a degenerate strategy --
pick, at every frame, any chord that happens to contain the melody note. That scores
~1.0 while being musically absurd, and ground truth itself only scores 0.65-0.79, so
the metric's optimum is not its maximum. It also moves with chord-change *rate*: a
model that changes chords rarely gives the melody fewer chances to fall outside the
chord, which is how a mixture missing its largest corpus scored *better* on it than
the full mixture (journal/DATASET_MIX_LOO.md).

What this module adds is a measure of chord *order and function* that the degenerate
strategy fails: the distribution of chord-to-chord moves, compared against the same
distribution estimated from real music.

Key-free by construction. A chord move is represented by the *interval* between
consecutive roots (mod 12) plus the function class of each chord, which is invariant
to transposition -- so no key annotation is needed. That matters because only
Hooktheory has real keys; every other converter writes a placeholder (see
journal/METRICS.md). METRICS.md considered a tonic-relative transition metric and
rejected it, partly for needing keys and partly on optimal-transport cost; the
relative-motion formulation needs neither.

Representation
--------------
A chord is (root pitch class, quality class). Quality is the *function* triple
(third, fifth, seventh) -- extensions (9/11/13/add) are deliberately dropped: a C9
and a C7 have the same harmonic function, and keeping colour tones triples the state
space for no grammatical gain. Of the 21 function classes that occur, the top 9 cover
99.3% of Hooktheory, but the corpora differ enormously (Nottingham uses *three*
qualities in total; JAZZMUS is 39% dominant sevenths against Hooktheory's 6%), so the
class list is derived per corpus from its own training reference rather than
hard-coded.

Histograms, in increasing resolution (K = 12 roots, Q = len(classes)):
  droot        K       root motion only            -- enough data on every corpus
  quality      Q       chord vocabulary
  droot_qto    K*Q     *the headline*: where it moved and what it landed on
  qual_trans   Q*Q     quality grammar
  joint        K*Q*Q   adds which quality it came *from* (preparation)

Comparison
----------
Jensen-Shannon divergence, base 2, so values are in [0, 1]. Not optimal transport:
EMD needs a ground distance between bins, and for root motion there is no defensible
one (a tritone is not "nearly" a fifth just because the intervals are adjacent).

Two things are not optional when reporting these numbers:

  * **Matched n.** An estimated divergence is biased upward by roughly K/n even when
    the two distributions are identical. Models emit different numbers of chord
    changes (on Hooktheory, GAPT produced 49k against ground truth's 32k), so
    unmatched counts make models incomparable. `compare` subsamples every input to a
    common event count and averages over repeats.
  * **The floor.** The same divergence between held-out ground truth and the training
    reference, at the same K and the same n, carries the same bias. Report
    `distance - floor`; the leading term cancels, and the floor is also the honest
    answer to "how close could a perfect model get from this many samples?".

A chord *event* is a change in (root, quality). Re-articulating the same chord is not
a progression event, and rests are skipped rather than breaking the progression --
both are rhythm, measured elsewhere.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from realchords.utils.eval_utils import chord_symbols_lib, _parse_chord_symbol

ChordEvent = Tuple[int, str]          # (root pitch class, quality class)
OTHER = "other"
NUM_ROOTS = 12

# Function classes, most frequent first across the corpora. Used only as a stable
# ordering; the active list for a corpus comes from its reference file.
KNOWN_CLASSES = [
    "MP", "mP", "mP7", "MP7", "MPM7", "sP", "md7", "sP7", "md", "mdd7", "mPM7", "MA",
]


def quality_class(intervals: frozenset) -> str:
    """Function triple (third, fifth, seventh) from root-relative pitch classes.

    third:  M major (4) / m minor (3) / s neither, i.e. sus or power chord
    fifth:  P perfect (7) / d diminished (6) / A augmented (8) / - absent
    seventh: 7 minor (10) / M7 major (11) / d7 diminished (9, with a flat fifth) / none
    """
    third = "M" if 4 in intervals else "m" if 3 in intervals else "s"
    fifth = "P" if 7 in intervals else "d" if 6 in intervals else "A" if 8 in intervals else "-"
    if 10 in intervals:
        seventh = "7"
    elif 11 in intervals:
        seventh = "M7"
    elif 9 in intervals and 6 in intervals:
        seventh = "d7"
    else:
        seventh = ""
    return f"{third}{fifth}{seventh}"


def chord_symbol_to_event(symbol: str) -> ChordEvent | None:
    """(root, raw function class) for a chord symbol, or None if unparseable."""
    try:
        root = chord_symbols_lib.chord_symbol_root(symbol)
        pitches = chord_symbols_lib.chord_symbol_pitches(symbol)
    except Exception:
        return None
    intervals = frozenset((p - root) % 12 for p in pitches)
    return root % 12, quality_class(intervals)


def chord_events(
    chord_tokens: Sequence[int],
    id_to_name: Dict[int, str],
    classes: Sequence[str] | None = None,
) -> List[ChordEvent]:
    """Chord-change events from one sequence's chord lane.

    Non-chord tokens (silence, padding, specials) are skipped, and a repeated chord
    is not an event -- only a change in (root, quality) is.
    """
    allowed = set(classes) if classes is not None else None
    events: List[ChordEvent] = []
    previous: ChordEvent | None = None
    for token in chord_tokens:
        symbol = _parse_chord_symbol(id_to_name.get(int(token), ""))
        if symbol is None:
            continue
        event = chord_symbol_to_event(symbol)
        if event is None:
            continue
        if allowed is not None and event[1] not in allowed:
            event = (event[0], OTHER)
        if event != previous:
            events.append(event)
        previous = event
    return events


def chord_lane(interleaved: Sequence[int], chord_first: bool = True) -> List[int]:
    """The chord half of an interleaved sequence, dropping the leading BOS.

    Verified against both the dataloader and custom_evaluation's dumps: the token
    after BOS is a chord (BOS, CHORD_ON_F, NOTE_ON_69, CHORD_F, ...), so the chord
    lane is the *even* offsets of the body. Padding and specials stay in and are
    dropped later by symbol parsing -- filtering them here would shift the parity
    and silently yield the melody lane.
    """
    body = list(interleaved)[1:]
    return body[0::2] if chord_first else body[1::2]


# --------------------------------------------------------------------------- #
# histograms
# --------------------------------------------------------------------------- #

def histograms(
    sequences: Iterable[Sequence[ChordEvent]],
    classes: Sequence[str],
) -> Dict[str, Counter]:
    """The five histograms over all transitions in a collection of sequences."""
    index = {q: i for i, q in enumerate(classes)}
    out = {k: Counter() for k in ("droot", "quality", "droot_qto", "qual_trans", "joint")}
    for events in sequences:
        for i, (root, qual) in enumerate(events):
            out["quality"][index.get(qual, index[OTHER])] += 1
            if i == 0:
                continue
            prev_root, prev_qual = events[i - 1]
            d = (root - prev_root) % 12
            qf = index.get(prev_qual, index[OTHER])
            qt = index.get(qual, index[OTHER])
            out["droot"][d] += 1
            out["droot_qto"][(d, qt)] += 1
            out["qual_trans"][(qf, qt)] += 1
            out["joint"][(d, qf, qt)] += 1
    return out


def num_bins(name: str, num_classes: int) -> int:
    return {
        "droot": NUM_ROOTS,
        "quality": num_classes,
        "droot_qto": NUM_ROOTS * num_classes,
        "qual_trans": num_classes * num_classes,
        "joint": NUM_ROOTS * num_classes * num_classes,
    }[name]


def transitions(sequences: Iterable[Sequence[ChordEvent]]) -> List[Tuple]:
    """Flat list of (prev_root, prev_qual, root, qual), the unit that gets subsampled."""
    out = []
    for events in sequences:
        for i in range(1, len(events)):
            out.append((events[i - 1][0], events[i - 1][1], events[i][0], events[i][1]))
    return out


# --------------------------------------------------------------------------- #
# divergence
# --------------------------------------------------------------------------- #

def jensen_shannon(p: Counter, q: Counter) -> float:
    """JS divergence in bits, in [0, 1]. Empty input on either side gives nan."""
    n_p, n_q = sum(p.values()), sum(q.values())
    if n_p == 0 or n_q == 0:
        return float("nan")
    total = 0.0
    for key in set(p) | set(q):
        pi, qi = p.get(key, 0) / n_p, q.get(key, 0) / n_q
        mi = 0.5 * (pi + qi)
        if pi > 0:
            total += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            total += 0.5 * qi * math.log2(qi / mi)
    return max(0.0, min(1.0, total))


def _histograms_from_transitions(
    trans: Sequence[Tuple], classes: Sequence[str]
) -> Dict[str, Counter]:
    index = {q: i for i, q in enumerate(classes)}
    out = {k: Counter() for k in ("droot", "quality", "droot_qto", "qual_trans", "joint")}
    for prev_root, prev_qual, root, qual in trans:
        d = (root - prev_root) % 12
        qf = index.get(prev_qual, index[OTHER])
        qt = index.get(qual, index[OTHER])
        out["droot"][d] += 1
        out["quality"][qt] += 1
        out["droot_qto"][(d, qt)] += 1
        out["qual_trans"][(qf, qt)] += 1
        out["joint"][(d, qf, qt)] += 1
    return out


def compare(
    model_trans: Sequence[Tuple],
    reference: Dict[str, Counter],
    classes: Sequence[str],
    match_n: int | None = None,
    repeats: int = 20,
    seed: int = 0,
    min_per_bin: float = 5.0,
) -> Dict[str, float | None]:
    """JS divergence per resolution, model vs reference.

    `match_n` subsamples the model's transitions to a common count so that two models
    emitting different numbers of chord changes are compared at equal sample size;
    the estimate is averaged over `repeats` draws. A resolution is reported as None
    when it has fewer than `min_per_bin` transitions per bin, rather than returning a
    number that is mostly counting noise -- on JAZZMUS the full joint would have 0.5
    transitions per bin.
    """
    if not model_trans:
        return {k: None for k in reference}
    rng = random.Random(seed)
    n = min(match_n or len(model_trans), len(model_trans))
    out: Dict[str, float | None] = {}
    draws = [rng.sample(list(model_trans), n) for _ in range(repeats if n < len(model_trans) else 1)]
    per_draw = [_histograms_from_transitions(d, classes) for d in draws]
    for name in reference:
        if n < min_per_bin * num_bins(name, len(classes)):
            out[name] = None
            continue
        values = [jensen_shannon(h[name], reference[name]) for h in per_draw]
        out[name] = sum(values) / len(values)
    return out


# --------------------------------------------------------------------------- #
# named conditionals
# --------------------------------------------------------------------------- #
# Single interpretable numbers pulled out of the joint. Each conditions on a slice
# with thousands of samples behind it, which the full 1200-bin divergence does not
# have, and each says *what* went wrong rather than only how far off it is.

def dominant_preparation_rate(trans: Sequence[Tuple]) -> float | None:
    """P(came from a dominant 7th | root rose a fourth, landed on a major chord).

    The textbook V7 -> I cadence. On Hooktheory ground truth this is 11.6%; the
    pre-trained decoder over-prepares at 18.8% and the RL policy under-prepares at
    8.1%, a difference invisible to any resolution that ignores the source quality.
    """
    hits = total = 0
    for prev_root, prev_qual, root, qual in trans:
        if (root - prev_root) % 12 == 5 and qual in ("MP", "MPM7"):
            total += 1
            hits += prev_qual == "MP7"
    return hits / total if total else None


def two_five_rate(trans: Sequence[Tuple]) -> float | None:
    """Share of all transitions that are a ii-V: m7 up a fourth to a dominant 7th."""
    if not trans:
        return None
    hits = sum(
        1 for pr, pq, r, q in trans
        if (r - pr) % 12 == 5 and pq == "mP7" and q == "MP7"
    )
    return hits / len(trans)


def fifth_motion_rate(trans: Sequence[Tuple]) -> float | None:
    """Share of root moves by a fifth in either direction -- the backbone of tonal
    harmony, and the single number the degenerate 'any chord containing the melody
    note' strategy cannot fake (it lands near uniform, ~1/6 across the two bins)."""
    if not trans:
        return None
    hits = sum(1 for pr, _, r, _ in trans if (r - pr) % 12 in (5, 7))
    return hits / len(trans)


def tritone_rate(trans: Sequence[Tuple]) -> float | None:
    """Share of root moves by a tritone. Rare in real music (Hooktheory GT 1.3%);
    an elevated value indicates scattered, non-functional progressions."""
    if not trans:
        return None
    return sum(1 for pr, _, r, _ in trans if (r - pr) % 12 == 6) / len(trans)


CONDITIONALS = {
    "dominant_preparation_rate": dominant_preparation_rate,
    "two_five_rate": two_five_rate,
    "fifth_motion_rate": fifth_motion_rate,
    "tritone_rate": tritone_rate,
}


def conditionals(trans: Sequence[Tuple]) -> Dict[str, float | None]:
    return {name: fn(trans) for name, fn in CONDITIONALS.items()}


def derive_classes(quality_counts: Counter, coverage: float = 0.99) -> List[str]:
    """Quality classes covering `coverage` of a corpus, most frequent first, + OTHER.

    Derived per corpus: Nottingham uses three qualities in total, Hooktheory 21, and
    JAZZMUS is dominated by dominant sevenths where Hooktheory is dominated by plain
    triads. One hard-coded list would misrepresent most of them.
    """
    total = sum(quality_counts.values())
    if not total:
        return [OTHER]
    kept, running = [], 0
    for name, count in quality_counts.most_common():
        kept.append(name)
        running += count
        if running / total >= coverage:
            break
    order = {q: i for i, q in enumerate(KNOWN_CLASSES)}
    kept.sort(key=lambda q: (order.get(q, len(KNOWN_CLASSES)), q))
    return kept + [OTHER]


# --------------------------------------------------------------------------- #
# event cache
# --------------------------------------------------------------------------- #
# Extracting chord events means walking the dataloader, which is ~4 minutes for
# hooktheory's training split and dominates every run. The events themselves are
# small (a few MB per corpus), so they are cached and everything downstream --
# new n-gram orders, different quality groupings, corpus-to-corpus distances --
# becomes instant. It also decouples the metrics from the dataloader, which is
# where both of this module's early bugs lived (lane parity, mask filtering).

EVENTS_CACHE_DIR = "data/cache/progression_events"


def events_cache_path(dataset_name: str, split: str, root: str = EVENTS_CACHE_DIR) -> "Path":
    from pathlib import Path
    return Path(root) / f"{dataset_name}.{split}.json"


def save_events(path, sequences, meta: Dict) -> None:
    """Write event sequences plus the provenance needed to detect a stale cache."""
    import json
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "sequences": [[[r, q] for r, q in seq] for seq in sequences],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_events(path, meta: Dict | None = None):
    """Cached event sequences, or None if absent or built under different settings.

    `meta` is compared field by field; a mismatch returns None rather than silently
    serving events extracted at a different window length or chord vocabulary.
    """
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta is not None:
        cached = payload.get("meta", {})
        for key, value in meta.items():
            if cached.get(key) != value:
                return None
    return [[(int(r), str(q)) for r, q in seq] for seq in payload["sequences"]]
