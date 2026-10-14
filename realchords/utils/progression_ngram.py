"""An n-gram model of chord progressions, used as an evaluation yardstick.

Why this exists alongside progression_metrics. Those histograms are bigrams: they see
each chord move in isolation, so a model that produces every correct two-chord step
but strings them into incoherent phrases scores well. Shuffling ground truth's chord
order confirms the limit -- it moves the root-motion histogram by only 0.038, and the
chord-quality histogram not at all. Longer context is needed to say anything about
phrase structure, and that is what an n-gram adds.

Why an n-gram rather than a neural model. A trained decoder as evaluator is circular:
it is the same model family as the anchor the RL policy is already KL-regularised
towards, so the policy has effectively been optimised against it. A count-based model
is deterministic, inspectable (you can look up which contexts a model gets wrong) and
cannot be trained into agreement with the thing it judges.

No truncated vocabulary. Symbols are (root interval, full function class) with every
class the corpus uses -- 24 in Hooktheory, 3 in Nottingham. Rare classes are not
lumped into "other"; instead the estimate backs off to shorter context and then to
root motion alone, so resolution degrades gracefully where counts are thin rather
than being thrown away up front.

Key-free, like the histograms: a root *interval* is the same in every key, which
matters because only Hooktheory carries real key annotations.

Scoring. Do NOT minimise perplexity. It has a degenerate optimum -- a model looping
I-V-vi-IV forever is extremely predictable and would score best, which is the same
trap as maximising note-in-chord ratio. Real progressions have a characteristic
*spread* of surprisal, so the comparison is two-sample: the distribution of
per-sequence log-likelihood from the model against the same distribution from
held-out real music. Too predictable shifts high and narrows; too scattered shifts
low. Both are failures and a two-sided comparison sees both.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

ChordEvent = Tuple[int, str]
Symbol = Tuple[int, str]          # (root interval from previous chord, quality)
BOS = (-1, "<s>")


def to_symbols(events: Sequence[ChordEvent]) -> List[Symbol]:
    """A chord sequence as a sequence of *moves*, which is what carries the grammar.

    The first chord has no predecessor, so the sequence starts at the first move.
    Quality is kept verbatim; no class is dropped.
    """
    out: List[Symbol] = []
    for i in range(1, len(events)):
        prev_root, _ = events[i - 1]
        root, qual = events[i]
        out.append(((root - prev_root) % 12, qual))
    return out


def coarsen(symbol: Symbol) -> int:
    """The backoff view of a symbol: root motion only, quality discarded."""
    return symbol[0]


class ProgressionNGram:
    """Interpolated n-gram over chord moves, with a root-motion-only backoff.

    Smoothing is Bayesian (Dirichlet) interpolation, applied recursively:

        P_k(x | ctx) = (count(ctx, x) + alpha * P_{k-1}(x | ctx[1:])) / (count(ctx) + alpha)

    so an unseen context contributes nothing and the estimate falls through to shorter
    context automatically. The chain bottoms out at the unigram over symbols, which
    itself backs off to the root-motion marginal and finally to uniform -- so a
    quality this corpus has never used still receives non-zero probability rather than
    making the log-likelihood infinite.
    """

    def __init__(self, order: int = 3, alpha: float = 1.0):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.alpha = alpha
        self.ngrams: List[Dict[Tuple, Counter]] = [defaultdict(Counter) for _ in range(order)]
        self.context_totals: List[Dict[Tuple, int]] = [defaultdict(int) for _ in range(order)]
        self.unigram: Counter = Counter()
        self.root_marginal: Counter = Counter()
        self.qualities: set = set()
        self.num_symbols_seen = 0

    # ---------------------------------------------------------------- fitting
    def fit(self, sequences: Iterable[Sequence[ChordEvent]]) -> "ProgressionNGram":
        for events in sequences:
            symbols = to_symbols(events)
            if not symbols:
                continue
            padded = [BOS] * (self.order - 1) + list(symbols)
            for symbol in symbols:
                self.unigram[symbol] += 1
                self.root_marginal[coarsen(symbol)] += 1
                self.qualities.add(symbol[1])
                self.num_symbols_seen += 1
            for k in range(self.order):
                for i in range(self.order - 1, len(padded)):
                    context = tuple(padded[i - k:i]) if k else ()
                    self.ngrams[k][context][padded[i]] += 1
                    self.context_totals[k][context] += 1
        self._vocab_size = max(1, 12 * max(1, len(self.qualities)))
        return self

    # --------------------------------------------------------------- scoring
    def _p_base(self, symbol: Symbol) -> float:
        """Unigram backing off to root motion, then uniform. Never zero."""
        total = max(1, self.num_symbols_seen)
        n_roots = 12
        p_root = (self.root_marginal.get(coarsen(symbol), 0) + 1.0 / n_roots) / (total + 1.0)
        p_quality_given_root = 1.0 / max(1, len(self.qualities))
        floor = p_root * p_quality_given_root
        return (self.unigram.get(symbol, 0) + self.alpha * floor) / (total + self.alpha)

    def _p_recursive(self, k: int, context: Tuple, symbol: Symbol) -> float:
        if k == 0:
            return self._p_base(symbol)
        lower = self._p_recursive(k - 1, context[1:], symbol)
        counts = self.ngrams[k].get(context)
        total = self.context_totals[k].get(context, 0)
        hits = counts.get(symbol, 0) if counts else 0
        return (hits + self.alpha * lower) / (total + self.alpha)

    def logprob(self, events: Sequence[ChordEvent]) -> Tuple[float, int]:
        """(total log2 probability, number of moves scored) for one progression."""
        symbols = to_symbols(events)
        if not symbols:
            return 0.0, 0
        padded = [BOS] * (self.order - 1) + list(symbols)
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - (self.order - 1):i])
            p = self._p_recursive(self.order - 1, context, padded[i])
            total += math.log2(max(p, 1e-12))
        return total, len(symbols)

    def per_sequence_logprob(
        self, sequences: Iterable[Sequence[ChordEvent]], min_moves: int = 3
    ) -> List[float]:
        """Mean log2 probability per move, one value per progression.

        Normalised by length so long and short songs are comparable, and sequences
        with fewer than `min_moves` moves are dropped -- their average is dominated by
        the start-of-sequence context rather than by any progression.
        """
        out = []
        for events in sequences:
            total, n = self.logprob(events)
            if n >= min_moves:
                out.append(total / n)
        return out


def distribution_summary(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "p10": None, "p50": None, "p90": None}
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in ordered) / (n - 1)) if n > 1 else 0.0

    def q(p: float) -> float:
        return ordered[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {"n": n, "mean": mean, "sd": sd, "p10": q(0.10), "p50": q(0.50), "p90": q(0.90)}


def two_sample(model: Sequence[float], reference: Sequence[float]) -> Dict[str, float | None]:
    """Compare two log-likelihood distributions.

    `mean_shift` is signed on purpose: positive means the model's progressions are
    *more* predictable than real ones (the cliche failure), negative means less (the
    scattered failure). Reporting only its magnitude would merge two opposite problems.
    `ks` is the largest gap between the two cumulative distributions, which also
    responds to a change in spread at equal means.
    """
    if not model or not reference:
        return {"mean_shift": None, "sd_ratio": None, "ks": None}
    ms, rs = distribution_summary(model), distribution_summary(reference)
    merged = sorted(set(model) | set(reference))
    n_m, n_r = len(model), len(reference)
    sorted_m, sorted_r = sorted(model), sorted(reference)
    ks = 0.0
    i = j = 0
    for value in merged:
        while i < n_m and sorted_m[i] <= value:
            i += 1
        while j < n_r and sorted_r[j] <= value:
            j += 1
        ks = max(ks, abs(i / n_m - j / n_r))
    return {
        "mean_shift": ms["mean"] - rs["mean"],
        "sd_ratio": (ms["sd"] / rs["sd"]) if rs["sd"] else None,
        "ks": ks,
    }
