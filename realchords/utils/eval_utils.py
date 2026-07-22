"""Evaluation utilities for ReaLchords."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

import bisect

import numpy as np
import torch
import note_seq.chord_symbols_lib as chord_symbols_lib
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as _scipy_entropy
from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance_nd

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.modes import (
    DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
    extract_chord_quality,
    list_scale_modes,
)

SPECIAL_TOKENS = frozenset({"PAD", "BOS", "EOS", "SILENCE"})
ModeRef = Tuple[str, int]
ModePitchClasses = Dict[ModeRef, FrozenSet[int]]
QualityModeEntry = Dict[str, object]


@lru_cache(maxsize=1)
def _load_quality_mode_lookup(
    map_path: str = str(DEFAULT_CHORD_QUALITY_MODE_MAP_PATH),
) -> Tuple[Dict[str, QualityModeEntry], ModePitchClasses]:
    """Load chord-quality → mode map and C-root pitch-class lookup."""
    mode_pitch_classes: ModePitchClasses = {}
    for parent_scale, mode_list in list_scale_modes().items():
        for mode in mode_list:
            mode_pitch_classes[(parent_scale, mode["mode_index"])] = frozenset(
                mode["pitch_classes"]
            )

    quality_map: Dict[str, QualityModeEntry] = {}
    for line in Path(map_path).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        quality_map[entry["chord_quality"]] = entry
    return quality_map, mode_pitch_classes


def _split_melody_chord_lanes(
    seq: Sequence[int],
    sequence_order: str,
) -> Tuple[List[int], List[int]]:
    if sequence_order == "chord_first":
        return seq[1::2], seq[::2]
    return seq[::2], seq[1::2]


def _parse_chord_symbol(token_name: str) -> str | None:
    if token_name.startswith("CHORD_ON_"):
        return token_name[len("CHORD_ON_") :]
    if token_name.startswith("CHORD_"):
        return token_name[len("CHORD_") :]
    return None


def _parse_note_pitch(token_name: str) -> int | None:
    if token_name.startswith("NOTE_ON_"):
        prefix = "NOTE_ON_"
    elif token_name.startswith("NOTE_"):
        prefix = "NOTE_"
    else:
        return None
    try:
        return int(token_name[len(prefix) :])
    except ValueError:
        return None


def _transpose_pitch_classes(
    pitch_classes: FrozenSet[int],
    root_pc: int,
) -> FrozenSet[int]:
    return frozenset((pc + root_pc) % 12 for pc in pitch_classes)


def _melody_pitch_histogram(
    melody_tokens: Sequence[int],
    start: int,
    end: int,
    tokenizer: HooktheoryTokenizer,
) -> Tuple[List[float], float]:
    histogram = [0.0] * 12
    total_weight = 0.0
    for frame in range(start, end):
        token_name = tokenizer.id_to_name.get(melody_tokens[frame], "")
        if token_name in SPECIAL_TOKENS:
            continue
        pitch = _parse_note_pitch(token_name)
        if pitch is None:
            continue
        pitch_class = pitch % 12
        histogram[pitch_class] += 1.0
        total_weight += 1.0
    return histogram, total_weight


def _mode_strict_score(
    histogram: Sequence[float],
    total_weight: float,
    mode_pcs: FrozenSet[int],
) -> float:
    if total_weight <= 0:
        return 0.0
    for pc, weight in enumerate(histogram):
        if weight > 0 and pc not in mode_pcs:
            return 0.0
    return 1.0


def _candidate_mode_pitch_classes(
    chord_symbol: str,
    quality_entry: QualityModeEntry,
    mode_pitch_classes: ModePitchClasses,
) -> List[FrozenSet[int]]:
    root_pc = chord_symbols_lib.chord_symbol_root(chord_symbol)
    candidate_sets: List[FrozenSet[int]] = []
    for mode in quality_entry["modes"]:  # type: ignore[index]
        mode_ref = (mode["parent_scale"], mode["mode_index"])
        pitch_classes = mode_pitch_classes.get(mode_ref)
        if pitch_classes is None:
            continue
        candidate_sets.append(_transpose_pitch_classes(pitch_classes, root_pc))
    return candidate_sets


def _best_mode_fit_score(
    histogram: Sequence[float],
    total_weight: float,
    candidate_mode_pcs: Sequence[FrozenSet[int]],
) -> float:
    if total_weight <= 0 or not candidate_mode_pcs:
        return 0.0

    best_score = 0.0
    for mode_pcs in candidate_mode_pcs:
        score = _mode_strict_score(histogram, total_weight, mode_pcs)
        best_score = max(best_score, score)
    return best_score


def _chord_onset_indices(
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    onset_indices = []
    for frame, token in enumerate(chord_tokens):
        token_name = tokenizer.id_to_name.get(token, "")
        if token_name in SPECIAL_TOKENS:
            continue
        if tokenizer.is_chord_on(token):
            onset_indices.append(frame)
    return onset_indices


def _note_onset_indices(
    melody_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    onset_indices = []
    for frame, token in enumerate(melody_tokens):
        token_name = tokenizer.id_to_name.get(token, "")
        if token_name in SPECIAL_TOKENS:
            continue
        if tokenizer.is_note_on(token):
            onset_indices.append(frame)
    return onset_indices


def _sequence_mode_fit_score(
    melody_tokens: Sequence[int],
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
    quality_map: Dict[str, QualityModeEntry],
    mode_pitch_classes: ModePitchClasses,
    skip_underdetermined: bool,
    min_melody_weight: float,
) -> Tuple[float, float, int]:
    segment_pass_sum = 0.0
    segment_count = 0

    onset_indices = _chord_onset_indices(chord_tokens, tokenizer)
    if not onset_indices:
        return np.nan, 0.0, 0

    for index, start in enumerate(onset_indices):
        end = (
            onset_indices[index + 1]
            if index + 1 < len(onset_indices)
            else len(chord_tokens)
        )
        chord_name = tokenizer.id_to_name.get(chord_tokens[start], "")
        chord_symbol = _parse_chord_symbol(chord_name)
        if chord_symbol is None:
            continue

        quality = extract_chord_quality(chord_symbol)
        quality_entry = quality_map.get(quality)
        if quality_entry is None:
            continue
        if skip_underdetermined and quality_entry.get("underdetermined"):
            continue

        histogram, total_weight = _melody_pitch_histogram(
            melody_tokens,
            start,
            end,
            tokenizer,
        )
        if total_weight < min_melody_weight:
            continue

        candidate_mode_pcs = _candidate_mode_pitch_classes(
            chord_symbol,
            quality_entry,
            mode_pitch_classes,
        )
        segment_score = _best_mode_fit_score(
            histogram,
            total_weight,
            candidate_mode_pcs,
        )
        segment_count += 1
        segment_pass_sum += segment_score

    if segment_count <= 0:
        return np.nan, 0.0, segment_count
    return segment_pass_sum / segment_count, float(segment_count), segment_count


def _frame_mode_fit_score(
    pitch_class: int,
    chord_symbol: str,
    quality_map: Dict[str, QualityModeEntry],
    mode_pitch_classes: ModePitchClasses,
    skip_underdetermined: bool,
) -> Tuple[bool, float]:
    """Score one melody frame against candidate modes for the active chord."""
    quality = extract_chord_quality(chord_symbol)
    quality_entry = quality_map.get(quality)
    if quality_entry is None:
        return False, float("nan")
    if skip_underdetermined and quality_entry.get("underdetermined"):
        return False, float("nan")

    candidate_mode_pcs = _candidate_mode_pitch_classes(
        chord_symbol,
        quality_entry,
        mode_pitch_classes,
    )
    if not candidate_mode_pcs:
        return False, float("nan")

    histogram = [0.0] * 12
    histogram[pitch_class] = 1.0
    score = _best_mode_fit_score(
        histogram,
        1.0,
        candidate_mode_pcs,
    )
    return True, score


def evaluate_melody_mode_fit_per_frame(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
    skip_underdetermined: bool = True,
    mode_map_path: Path | str = DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
) -> Dict[str, torch.Tensor]:
    """Per-frame note-in-mode for interleaved chord-first sequences.

    Each active melody frame is scored against candidate modes for the chord
    sounding at that frame (same chord-quality lookup as the segment metric).
    A frame scores 1.0 only if its pitch class is in at least one candidate
    mode for the active chord, else 0.0 (strict fit).

    Returns a dict with:
        matches: float tensor [batch, num_frames] — score in [0, 1], NaN if unscorable
        valid: bool tensor [batch, num_frames]
        mean: float tensor [batch] — mean over valid frames (NaN if none)
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    quality_map, mode_pitch_classes = _load_quality_mode_lookup(str(mode_map_path))

    batch_size = sequences.size(0)
    num_frames = sequences.size(1) // 2
    matches = torch.full((batch_size, num_frames), float("nan"))
    valid = torch.zeros((batch_size, num_frames), dtype=torch.bool)
    means: List[float] = []

    for row_idx, seq in enumerate(sequences):
        seq_list = seq.cpu().tolist()
        melody_tokens, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)

        score_sum = 0.0
        valid_count = 0
        for frame_idx, (note_token, chord_token) in enumerate(
            zip(melody_tokens, chord_tokens)
        ):
            note_name = tokenizer.id_to_name.get(note_token, "")
            chord_name = tokenizer.id_to_name.get(chord_token, "")
            if note_name in SPECIAL_TOKENS or chord_name in SPECIAL_TOKENS:
                continue

            pitch = _parse_note_pitch(note_name)
            chord_symbol = _parse_chord_symbol(chord_name)
            if pitch is None or chord_symbol is None:
                continue

            is_valid, score = _frame_mode_fit_score(
                pitch % 12,
                chord_symbol,
                quality_map,
                mode_pitch_classes,
                skip_underdetermined=skip_underdetermined,
            )
            if not is_valid:
                continue
            valid[row_idx, frame_idx] = True
            matches[row_idx, frame_idx] = float(score)
            score_sum += score
            valid_count += 1
        means.append(score_sum / valid_count if valid_count > 0 else float("nan"))

    return {
        "matches": matches,
        "valid": valid,
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def evaluate_melody_mode_fit_ratio(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    return_count: bool = False,
    sequence_order: str = "chord_first",
    skip_underdetermined: bool = True,
    min_melody_weight: float = 1.0,
    mode_map_path: Path | str = DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score how well melody notes over each chord region fit a mode.

    Unlike note-in-chord, this metric is segment-based: for each span between
    consecutive ``CHORD_ON`` events, melody pitch classes are compared against
    candidate modes from ``chord_quality_mode_map.jsonl``. A segment scores 1.0
    only if every melody pitch class in that span is in at least one candidate
    mode for the active chord (strict fit), else 0.0; the sequence score is the
    unweighted mean over segments.

    Args:
        sequences: Tensor of shape ``[batch, seq_len]`` with alternating melody and
            chord tokens.
        tokenizer: Hooktheory tokenizer with ``id_to_name`` mapping.
        model_part: ``"melody"`` or ``"chord"`` (validated for API parity).
        return_count: If True, also return segment counts (twice, for API parity).
        sequence_order: ``"chord_first"`` or ``"melody_first"``.
        skip_underdetermined: Skip segments for sparse qualities like ``ped``/``5``.
        min_melody_weight: Minimum weighted melody mass required to score a segment.
        mode_map_path: Path to curated chord-quality mode JSONL.

    Returns:
        Tensor of mode-fit ratios per sequence, or a tuple with segment counts
        (twice) when ``return_count=True``.
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )
    if model_part not in {"melody", "chord"}:
        raise ValueError(f"Invalid model_part: {model_part}")

    quality_map, mode_pitch_classes = _load_quality_mode_lookup(str(mode_map_path))

    ratios = []
    melody_weights = []
    segment_counts = []
    for seq in sequences:
        seq_list = seq.cpu().tolist()
        melody_tokens, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        ratio, melody_weight, segment_count = _sequence_mode_fit_score(
            melody_tokens=melody_tokens,
            chord_tokens=chord_tokens,
            tokenizer=tokenizer,
            quality_map=quality_map,
            mode_pitch_classes=mode_pitch_classes,
            skip_underdetermined=skip_underdetermined,
            min_melody_weight=min_melody_weight,
        )
        ratios.append(ratio)
        melody_weights.append(melody_weight)
        segment_counts.append(segment_count)

    ratio_tensor = torch.tensor(ratios)
    if return_count:
        return (
            ratio_tensor,
            torch.tensor(melody_weights),
            torch.tensor(segment_counts),
        )
    return ratio_tensor


def _note_in_chord_pair(
    note_token: int,
    chord_token: int,
    tokenizer: HooktheoryTokenizer,
) -> Tuple[bool, bool]:
    """Return (valid, in_chord) for one melody/chord frame pair."""
    note_name = tokenizer.id_to_name.get(note_token, "")
    chord_name_full = tokenizer.id_to_name.get(chord_token, "")

    if note_name in SPECIAL_TOKENS or chord_name_full in SPECIAL_TOKENS:
        return False, False

    if not (note_name.startswith("NOTE_") or note_name.startswith("NOTE_ON_")):
        return False, False

    if chord_name_full.startswith("CHORD_ON_"):
        chord_str = chord_name_full[len("CHORD_ON_") :]
    elif chord_name_full.startswith("CHORD_"):
        chord_str = chord_name_full[len("CHORD_") :]
    else:
        return False, False

    if note_name.startswith("NOTE_ON_"):
        try:
            note_pitch = int(note_name[len("NOTE_ON_") :])
        except ValueError:
            return False, False
    elif note_name.startswith("NOTE_"):
        try:
            note_pitch = int(note_name[len("NOTE_") :])
        except ValueError:
            return False, False
    else:
        return False, False

    chord_pitches = chord_symbols_lib.chord_symbol_pitches(chord_str)
    in_chord = note_pitch % 12 in [p % 12 for p in chord_pitches]
    return True, in_chord


def evaluate_note_in_chord_per_frame(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, torch.Tensor]:
    """Per-frame note-in-chord for interleaved chord-first sequences.

    Returns a dict with:
        matches: float tensor [batch, num_frames] — 1.0 in-chord, 0.0 not, NaN invalid
        valid: bool tensor [batch, num_frames]
        mean: float tensor [batch] — sequence-level NiCR (NaN if no valid frames)
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    batch_size = sequences.size(0)
    num_frames = sequences.size(1) // 2
    matches = torch.full((batch_size, num_frames), float("nan"))
    valid = torch.zeros((batch_size, num_frames), dtype=torch.bool)
    means: List[float] = []

    for row_idx, seq in enumerate(sequences):
        seq_list = seq.cpu().tolist()
        if sequence_order == "chord_first":
            chord_tokens = seq_list[::2]
            melody_tokens = seq_list[1::2]
        else:
            melody_tokens = seq_list[::2]
            chord_tokens = seq_list[1::2]

        correct = 0
        valid_count = 0
        for frame_idx, (note_token, chord_token) in enumerate(
            zip(melody_tokens, chord_tokens)
        ):
            is_valid, in_chord = _note_in_chord_pair(
                note_token, chord_token, tokenizer
            )
            if not is_valid:
                continue
            valid[row_idx, frame_idx] = True
            matches[row_idx, frame_idx] = float(in_chord)
            valid_count += 1
            if in_chord:
                correct += 1
        means.append(correct / valid_count if valid_count > 0 else float("nan"))

    return {
        "matches": matches,
        "valid": valid,
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def evaluate_chord_symbols_per_frame(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, object]:
    """Per-frame chord symbols for interleaved sequences.

    Returns a dict with:
        symbols: list[list[str]] — chord symbol per frame, ``""`` if none
        is_onset: bool tensor [batch, num_frames] — ``CHORD_ON`` at this frame
        valid: bool tensor [batch, num_frames] — frame carries a chord token
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    batch_size = sequences.size(0)
    num_frames = sequences.size(1) // 2
    symbols: List[List[str]] = []
    is_onset = torch.zeros((batch_size, num_frames), dtype=torch.bool)
    valid = torch.zeros((batch_size, num_frames), dtype=torch.bool)

    for row_idx, seq in enumerate(sequences):
        seq_list = seq.cpu().tolist()
        _, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        row_symbols: List[str] = []
        for frame_idx, chord_token in enumerate(chord_tokens):
            token_name = tokenizer.id_to_name.get(chord_token, "")
            if token_name in SPECIAL_TOKENS:
                row_symbols.append("")
                continue
            chord_symbol = _parse_chord_symbol(token_name)
            if chord_symbol is None:
                row_symbols.append("")
                continue
            row_symbols.append(chord_symbol)
            valid[row_idx, frame_idx] = True
            if tokenizer.is_chord_on(chord_token):
                is_onset[row_idx, frame_idx] = True
        symbols.append(row_symbols)

    return {
        "symbols": symbols,
        "is_onset": is_onset,
        "valid": valid,
    }


def chord_type_distribution(
    chords: Dict[str, object],
    weighting: str = "onset",
) -> Dict[str, int]:
    """Chord-symbol usage histogram from ``evaluate_chord_symbols_per_frame``'s output.

    Args:
        chords: Output of ``evaluate_chord_symbols_per_frame`` (needs
            ``symbols``, ``is_onset``, ``valid``).
        weighting: ``"onset"`` counts each chord *change* once (progression
            vocabulary -- what chords get chosen, regardless of how long they
            last). ``"frame"`` counts every frame the chord is held (dwell-time
            weighted -- long chords dominate the count).

    Returns:
        Dict mapping chord symbol -> count.
    """
    if weighting not in {"onset", "frame"}:
        raise ValueError(f"Unsupported weighting '{weighting}'. Expected 'onset' or 'frame'.")

    counts: Counter = Counter()
    symbols = chords["symbols"]
    is_onset = chords["is_onset"]
    valid = chords["valid"]
    for row_idx, row in enumerate(symbols):
        for frame_idx, sym in enumerate(row):
            if not sym or not bool(valid[row_idx, frame_idx].item()):
                continue
            if weighting == "frame":
                counts[sym] += 1
            elif bool(is_onset[row_idx, frame_idx].item()):
                counts[sym] += 1
    return dict(counts)


def chord_type_js_distance(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    base: float = 2.0,
) -> float:
    """Jensen-Shannon distance between two chord-type usage distributions.

    Unlike Vendi score (which measures how diverse ONE source's own chord
    usage is against itself), this directly compares whether TWO sources --
    e.g. a model's output vs. the GT test set, or two different datasets --
    choose chords in similar proportions. A model can have high Vendi
    (genuinely varied output) while still favoring a completely different
    chord palette than real data; this metric is what catches that, not
    Vendi. Missing symbols in one distribution are treated as zero count, not
    excluded.

    Args:
        counts_a: Chord symbol -> count, e.g. ``chord_type_distribution(...)``.
        counts_b: Chord symbol -> count, same format, other source.
        base: Logarithm base for the underlying entropy calculation. With the
            default of 2.0, the result is bounded in [0, 1]: 0 = identical
            usage proportions, 1 = completely disjoint chord vocabularies.

    Returns:
        Jensen-Shannon distance (the square root of the JS divergence, a true
        metric satisfying the triangle inequality, unlike divergence itself).
    """
    vocab = sorted(set(counts_a) | set(counts_b))
    if not vocab:
        raise ValueError("Cannot compute JS distance: both distributions are empty.")

    a = np.array([counts_a.get(sym, 0) for sym in vocab], dtype=np.float64)
    b = np.array([counts_b.get(sym, 0) for sym in vocab], dtype=np.float64)
    if a.sum() == 0 or b.sum() == 0:
        raise ValueError(
            "Cannot compute JS distance: a distribution has zero total count."
        )
    return float(jensenshannon(a, b, base=base))


def chord_root_pitch_class_distribution(counts: Dict[str, int]) -> Dict[int, int]:
    """Collapse a chord-symbol usage histogram down to root pitch class (0-11).

    Same input/purpose as ``chord_type_distribution``'s output, but discards
    chord quality entirely -- useful when quality itself isn't the thing being
    compared (see ``chord_root_distribution_emd``, which needs *some* ordered
    axis to move mass along, and root pitch class is the natural one: chord
    symbols themselves have no inherent distance between them).
    """
    root_counts: Counter = Counter()
    for symbol, count in counts.items():
        try:
            root_pc = chord_symbols_lib.chord_symbol_root(symbol) % 12
        except Exception:
            continue
        root_counts[root_pc] += count
    return dict(root_counts)


# Unit-circle embedding of the 12 pitch classes -- consecutive pitch classes
# (and pc 11 <-> pc 0) are adjacent on the circle, so Euclidean distance
# between two embedded points respects chromatic-circle proximity instead of
# treating pitch classes as an arbitrary, unordered 0-11 integer labeling.
_PITCH_CLASS_CIRCLE = np.stack(
    [np.cos(2 * np.pi * np.arange(12) / 12), np.sin(2 * np.pi * np.arange(12) / 12)],
    axis=1,
)


def chord_root_distribution_emd(
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
) -> float:
    """Circular Earth Mover's Distance between two chord-root usage distributions.

    Unlike ``chord_type_js_distance`` (categorical -- every distinct chord
    symbol is equally "different" from every other), this compares only the
    *root* pitch class each chord uses, on the chromatic circle: moving mass
    from C to C# costs less than moving it from C to F#. Complements
    JS-distance rather than replacing it -- this is blind to chord quality
    (a Cmaj7 and a Cm7 are indistinguishable here), JS-distance is blind to
    harmonic proximity between different roots.

    Args:
        counts_a: Chord symbol -> count, e.g. ``chord_type_distribution(...)``.
        counts_b: Chord symbol -> count, same format, other source.

    Returns:
        Wasserstein-1 distance on the unit circle (0 = identical root usage
        proportions; max possible is 2, two point masses on opposite sides).
    """
    root_a = chord_root_pitch_class_distribution(counts_a)
    root_b = chord_root_pitch_class_distribution(counts_b)
    weights_a = np.array([root_a.get(pc, 0) for pc in range(12)], dtype=np.float64)
    weights_b = np.array([root_b.get(pc, 0) for pc in range(12)], dtype=np.float64)
    if weights_a.sum() == 0 or weights_b.sum() == 0:
        raise ValueError(
            "Cannot compute root-distribution EMD: a distribution has zero total count."
        )
    return float(
        wasserstein_distance_nd(
            _PITCH_CLASS_CIRCLE,
            _PITCH_CLASS_CIRCLE,
            u_weights=weights_a,
            v_weights=weights_b,
        )
    )


def evaluate_note_in_chord_ratio(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    return_count: bool = False,
    sequence_order: str = "chord_first",
) -> torch.Tensor:
    """
    Compute the note in chord ratio for each sequence in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain note tokens and odd indices contain chord tokens (or reverse).
    For each note-chord pair, frames where either token is a special token (PAD, BOS, EOS, SILENCE)
    are ignored. For valid pairs, the function checks if the note (modulo 12) is one of the chord's pitches.
    The ratio for each sequence is the average of valid time steps with a note-in-chord match.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.
        model_part (str): Specifies which part of the model to evaluate ("melody" or "chord").
        return_count (bool): If True, returns a tuple with the ratio, valid counts, and correct counts.

        torch.Tensor: A list of note in chord ratios, one for each sequence in the batch.
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    ratios = []
    valid_counts = []
    correct_counts = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        if sequence_order == "chord_first":
            chord_tokens = seq[::2]
            melody_tokens = seq[1::2]
        else:
            melody_tokens = seq[::2]
            chord_tokens = seq[1::2]

        if model_part not in {"melody", "chord"}:
            raise ValueError(f"Invalid model_part: {model_part}")

        valid_count = 0
        correct_count = 0

        for note_token, chord_token in zip(melody_tokens, chord_tokens):
            is_valid, in_chord = _note_in_chord_pair(
                note_token, chord_token, tokenizer
            )
            if not is_valid:
                continue
            if in_chord:
                correct_count += 1
            valid_count += 1

        ratio = correct_count / valid_count if valid_count > 0 else np.nan
        ratios.append(ratio)

        valid_counts.append(valid_count)
        correct_counts.append(correct_count)
    if return_count:
        return torch.tensor(ratios), torch.tensor(valid_counts), torch.tensor(correct_counts)
    else:
        return torch.tensor(ratios)


def _chord_to_note_onset_intervals(
    melody_tokens: Sequence[int],
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    """Chord-onset -> nearest preceding melody-note-onset interval, in frames.

    For each chord onset, find the closest melody note onset at or before it
    and record the gap between them. Chord onsets with no preceding melody
    note onset (nothing has been played yet) are skipped -- there is no
    "preceding note" to measure from.
    """
    note_onsets = _note_onset_indices(melody_tokens, tokenizer)
    chord_onsets = _chord_onset_indices(chord_tokens, tokenizer)

    intervals: List[int] = []
    for chord_frame in chord_onsets:
        # Rightmost note onset at or before chord_frame.
        pos = bisect.bisect_right(note_onsets, chord_frame) - 1
        if pos < 0:
            continue
        intervals.append(chord_frame - note_onsets[pos])
    return intervals


def evaluate_chord_to_note_onset_intervals(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, object]:
    """Chord-to-note onset intervals (synchronization), per RealChords §"Synchronization".

    For each chord onset, measures the gap (in frames) to the nearest
    preceding melody note onset. A model that places chords tightly relative
    to melody rhythm should produce a distribution of these gaps similar to
    real data; compare distributions across sequences with
    ``synchronization_emd`` rather than just comparing means.

    Returns a dict with:
        intervals: list[list[int]] — one list of gaps (frames) per sequence,
            ragged since chord-onset counts vary; empty if no chord onset has
            a preceding melody note onset.
        intervals_flat: int64 tensor [total_onsets] — all intervals across the
            batch, flattened, for feeding directly into ``synchronization_emd``.
        mean: float tensor [batch] — per-sequence mean interval (NaN if none).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    intervals_per_seq: List[List[int]] = []
    means: List[float] = []
    flat: List[int] = []

    for seq in sequences:
        seq_list = seq.cpu().tolist()
        melody_tokens, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        intervals = _chord_to_note_onset_intervals(melody_tokens, chord_tokens, tokenizer)
        intervals_per_seq.append(intervals)
        flat.extend(intervals)
        means.append(float(np.mean(intervals)) if intervals else float("nan"))

    return {
        "intervals": intervals_per_seq,
        "intervals_flat": torch.tensor(flat, dtype=torch.int64),
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def _as_numpy(values: torch.Tensor | Sequence[int]) -> np.ndarray:
    return np.asarray(values.cpu().numpy() if isinstance(values, torch.Tensor) else values)


def synchronization_emd(
    model_intervals: torch.Tensor | Sequence[int],
    reference_intervals: torch.Tensor | Sequence[int],
    max_interval: int = 18,
) -> float:
    """Earth Mover's Distance between two chord-to-note onset interval distributions.

    Compares the full distribution of chord-to-note onset intervals (see
    ``evaluate_chord_to_note_onset_intervals``) between a model's output and a
    reference (e.g. test-set GT) -- lower is better, 0 means identical
    distributions.

    Matches the paper's methodology: intervals are binned into
    ``[0, 1, ..., max_interval - 1, inf]`` (an exact bin per frame gap up to
    ``max_interval - 1``, then one overflow bin for everything at or beyond
    ``max_interval`` -- same ``max_interval`` convention as
    ``chord_to_note_onset_interval_histogram``: paper default 18, i.e. exact
    bins ``0..17`` then an overflow bin for ``>=18``), and EMD is computed on
    the resulting histograms. In practice this is implemented as
    clip-then-Wasserstein rather than literally building histograms first --
    for 1D data the two give an identical result (both only depend on the
    empirical CDF of the clipped values), but clipping first is simpler and
    avoids a redundant intermediate representation. This *does* change the
    result vs. computing EMD on raw unclipped samples: an outlier beyond the
    cutoff no longer contributes its true magnitude, only "beyond the cutoff"
    -- deliberately capping how much any single extreme event can skew the
    score. Note the paper reports this value multiplied by 1000 (``x10^3``)
    purely for table readability; this function returns the unscaled value in
    frames.

    Args:
        model_intervals: Flattened chord-to-note onset intervals from the model.
        reference_intervals: Flattened chord-to-note onset intervals from the
            reference distribution (e.g. GT test set).
        max_interval: Number of exact-gap bins (``0..max_interval-1``); gaps
            ``>= max_interval`` are folded into one overflow bin.

    Returns:
        The Earth Mover's Distance (Wasserstein-1) between the two clipped
        distributions, in frames (unscaled -- multiply by 1000 to match the
        paper's table convention).
    """
    model_arr = _as_numpy(model_intervals)
    reference_arr = _as_numpy(reference_intervals)
    if model_arr.size == 0 or reference_arr.size == 0:
        raise ValueError(
            "Cannot compute EMD: both model_intervals and reference_intervals "
            "must be non-empty."
        )
    model_clipped = np.minimum(model_arr, max_interval)
    reference_clipped = np.minimum(reference_arr, max_interval)
    return float(wasserstein_distance(model_clipped, reference_clipped))


def chord_to_note_onset_interval_histogram(
    intervals: torch.Tensor | Sequence[int],
    max_interval: int = 18,
    density: bool = True,
) -> np.ndarray:
    """Histogram of chord-to-note onset intervals, one bin per frame gap.

    For visualization/inspection alongside ``synchronization_emd`` -- same
    ``max_interval`` binning convention as that function (default 18 matches
    its default), so this histogram now actually reflects what the EMD sees.

    Args:
        intervals: Flattened chord-to-note onset intervals (frames), e.g.
            ``evaluate_chord_to_note_onset_intervals(...)["intervals_flat"]``.
        max_interval: Number of exact-gap bins, for gaps ``0, 1, ..., max_interval - 1``.
            Gaps ``>= max_interval`` are folded into one overflow bin.
        density: If True (default), normalize to a probability distribution
            (sums to 1) so histograms from batches of different sizes are
            directly comparable, e.g. model output vs. a GT test set.

    Returns:
        1D array of length ``max_interval + 1``: counts (or densities) for
        gaps ``0, 1, ..., max_interval - 1``, then the ``>= max_interval``
        overflow bin last.
    """
    if max_interval < 1:
        raise ValueError(f"max_interval must be >= 1, got {max_interval}")

    arr = np.asarray(
        intervals.cpu().numpy() if isinstance(intervals, torch.Tensor) else intervals
    )
    clipped = np.minimum(arr, max_interval)
    counts = np.bincount(clipped, minlength=max_interval + 1)[: max_interval + 1]

    if density:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


def _chord_durations(
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    """Duration (in frames) of each chord segment, onset to next onset.

    The final segment runs to the end of ``chord_tokens`` (i.e. to the end of
    whatever window was passed in -- a crop boundary, not necessarily the
    chord's true end, same convention as the mode-fit segment scorer).
    """
    onsets = _chord_onset_indices(chord_tokens, tokenizer)
    durations: List[int] = []
    for index, start in enumerate(onsets):
        end = onsets[index + 1] if index + 1 < len(onsets) else len(chord_tokens)
        durations.append(end - start)
    return durations


def evaluate_chord_durations(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, object]:
    """Chord segment durations (frames), for the "Rhythmic diversity" metric.

    Returns a dict with:
        durations: list[list[int]] — one list of segment durations (frames)
            per sequence, ragged since chord counts vary; empty if no chord
            onset was found.
        durations_flat: int64 tensor [total_segments] — all durations across
            the batch, flattened, ready for ``duration_entropy``.
        mean: float tensor [batch] — per-sequence mean duration (NaN if none).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    durations_per_seq: List[List[int]] = []
    means: List[float] = []
    flat: List[int] = []

    for seq in sequences:
        seq_list = seq.cpu().tolist()
        _, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        durations = _chord_durations(chord_tokens, tokenizer)
        durations_per_seq.append(durations)
        flat.extend(durations)
        means.append(float(np.mean(durations)) if durations else float("nan"))

    return {
        "durations": durations_per_seq,
        "durations_flat": torch.tensor(flat, dtype=torch.int64),
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def duration_entropy(
    durations: torch.Tensor | Sequence[int],
    base: float = np.e,
) -> float:
    """Shannon entropy of a distribution of segment durations (rhythmic diversity
    / rhythmic regularity).

    Generic over chord durations (``evaluate_chord_durations``) or melody note
    durations (``evaluate_note_durations``) -- each distinct duration value
    (in frames) is treated as a category. Low entropy means the source
    repeatedly defaults to one or two durations (e.g. a walking bass that's
    almost always exactly one quarter note, as in WJD); high entropy means a
    wide, varied mix of note/chord lengths (typical of freer melodic rhythm in
    the other datasets). This is the same quantity either way -- "rhythmic
    diversity" and "rhythmic regularity" are just low/high entropy read in
    opposite directions.

    Pool ``durations_flat`` across a whole batch/test-set for a stable
    estimate -- entropy from a single sequence's handful of segments is a
    noisy, biased estimate of the true diversity, same small-sample issue as
    the synchronization histograms.

    Args:
        durations: Segment durations in frames, e.g.
            ``evaluate_chord_durations(...)["durations_flat"]`` or
            ``evaluate_note_durations(...)["durations_flat"]``.
        base: Logarithm base. Defaults to natural log (``nats``), matching
            the paper's convention; pass ``2.0`` for bits instead.

    Returns:
        Shannon entropy of the empirical duration distribution.
    """
    arr = np.asarray(
        durations.cpu().numpy() if isinstance(durations, torch.Tensor) else durations
    )
    if arr.size == 0:
        raise ValueError("Cannot compute entropy of an empty duration distribution.")
    _, counts = np.unique(arr, return_counts=True)
    return float(_scipy_entropy(counts, base=base))


def duration_emd(
    model_durations: torch.Tensor | Sequence[int],
    reference_durations: torch.Tensor | Sequence[int],
    max_duration: int = 34,
) -> float:
    """Earth Mover's Distance between two segment-duration distributions.

    Same idea as ``synchronization_emd``, generic over chord or note
    durations (see ``duration_entropy``'s docstring for that generality).
    Matches the paper's chord-length methodology: durations are binned into
    ``[0, 1, ..., max_duration - 1, inf]`` (paper default 34, i.e. exact bins
    ``0..33`` then an overflow bin for ``>=34``), implemented as
    clip-then-Wasserstein (see ``synchronization_emd`` for why this is
    equivalent to histogram-then-EMD for 1D data). As with
    ``synchronization_emd``, the paper reports this multiplied by 1000 for
    table readability; this function returns the unscaled value in frames.

    Args:
        model_durations: Flattened segment durations from the model, e.g.
            ``evaluate_chord_durations(...)["durations_flat"]``.
        reference_durations: Flattened segment durations from the reference
            distribution (e.g. GT test set), same kind (chord or note) as
            ``model_durations``.
        max_duration: Number of exact-length bins (``0..max_duration-1``);
            durations ``>= max_duration`` are folded into one overflow bin.

    Returns:
        The Earth Mover's Distance (Wasserstein-1) between the two clipped
        distributions, in frames (unscaled -- multiply by 1000 to match the
        paper's table convention).
    """
    model_arr = _as_numpy(model_durations)
    reference_arr = _as_numpy(reference_durations)
    if model_arr.size == 0 or reference_arr.size == 0:
        raise ValueError(
            "Cannot compute EMD: both model_durations and reference_durations "
            "must be non-empty."
        )
    model_clipped = np.minimum(model_arr, max_duration)
    reference_clipped = np.minimum(reference_arr, max_duration)
    return float(wasserstein_distance(model_clipped, reference_clipped))


def _note_durations(
    melody_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    """Duration (in frames) of each melody note, onset to next onset.

    Same convention as ``_chord_durations``: the final note runs to the end
    of ``melody_tokens`` (i.e. to the end of whatever window was passed in).
    A gap that includes trailing SILENCE before the next onset is still
    counted as part of the note's inter-onset interval, same as real rhythm
    analysis (IOI) -- this isn't distinguishing legato/staccato articulation,
    just "how long until the next note starts."
    """
    onsets = _note_onset_indices(melody_tokens, tokenizer)
    durations: List[int] = []
    for index, start in enumerate(onsets):
        end = onsets[index + 1] if index + 1 < len(onsets) else len(melody_tokens)
        durations.append(end - start)
    return durations


def evaluate_note_durations(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, object]:
    """Melody note durations (frames) -- the melody-side rhythmic diversity/
    regularity metric (see ``duration_entropy``).

    Returns a dict with:
        durations: list[list[int]] — one list of note durations (frames) per
            sequence, ragged since note counts vary; empty if no note onset
            was found.
        durations_flat: int64 tensor [total_notes] — all durations across the
            batch, flattened, ready for ``duration_entropy``.
        mean: float tensor [batch] — per-sequence mean duration (NaN if none).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    durations_per_seq: List[List[int]] = []
    means: List[float] = []
    flat: List[int] = []

    for seq in sequences:
        seq_list = seq.cpu().tolist()
        melody_tokens, _ = _split_melody_chord_lanes(seq_list, sequence_order)
        durations = _note_durations(melody_tokens, tokenizer)
        durations_per_seq.append(durations)
        flat.extend(durations)
        means.append(float(np.mean(durations)) if durations else float("nan"))

    return {
        "durations": durations_per_seq,
        "durations_flat": torch.tensor(flat, dtype=torch.int64),
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def _chord_pitch_counts(
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> List[int]:
    """Number of distinct pitch classes in each chord segment, onset to next onset.

    E.g. a triad (maj/min/dim/aug) has 3, a 7th chord has 4, a 9th chord has 5.
    Same segment convention as ``_chord_durations``: one value per chord
    change, not per frame.
    """
    onsets = _chord_onset_indices(chord_tokens, tokenizer)
    counts: List[int] = []
    for start in onsets:
        chord_name = tokenizer.id_to_name.get(chord_tokens[start], "")
        chord_symbol = _parse_chord_symbol(chord_name)
        if chord_symbol is None:
            continue
        try:
            pitches = chord_symbols_lib.chord_symbol_pitches(chord_symbol)
        except Exception:
            continue
        counts.append(len(set(p % 12 for p in pitches)))
    return counts


def evaluate_chord_complexity(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> Dict[str, object]:
    """Chord complexity: how many distinct pitch classes each chord uses.

    One value per chord segment (a change, not a frame -- a long 9th chord
    counts once, the same as a short one), pooled and averaged the same way
    as ``evaluate_chord_durations``. Left unnormalized on purpose -- the raw
    count is directly readable (a mean of 3 means the song is mostly triads,
    3.7 means a mix with some richer chords, etc.), which a rescaled [0, 1]
    version would obscure. Verified empirically against the full chord
    vocabulary that raw counts fall in ``[1, 7]`` (1 = single-note/pedal, 7 =
    fully-stacked 13th chord), so that range is a real fact about the data,
    not a guessed bound.

    Returns a dict with:
        pitch_counts: list[list[int]] — one list of pitch counts per
            sequence, ragged since chord counts vary; empty if no chord
            onset was found or none parsed.
        pitch_counts_flat: int64 tensor [total_segments] — all pitch counts
            across the batch, flattened.
        mean: float tensor [batch] — per-sequence mean pitch count (NaN if none).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    counts_per_seq: List[List[int]] = []
    means: List[float] = []
    flat: List[int] = []

    for seq in sequences:
        seq_list = seq.cpu().tolist()
        _, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        counts = _chord_pitch_counts(chord_tokens, tokenizer)
        counts_per_seq.append(counts)
        flat.extend(counts)
        means.append(float(np.mean(counts)) if counts else float("nan"))

    return {
        "pitch_counts": counts_per_seq,
        "pitch_counts_flat": torch.tensor(flat, dtype=torch.int64),
        "mean": torch.tensor(means, dtype=torch.float32),
    }


def evaluate_sequence_num_frames(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> torch.Tensor:
    """Number of valid (non-PAD) frames per sequence -- i.e. the song's length.

    Padding is applied identically to both lanes (``pad_and_get_mask``), so
    counting non-PAD frames in either lane gives the same answer; the chord
    lane is used here. SILENCE is real content (nothing sounding, but still
    part of the song) and is counted, unlike PAD.

    Args:
        sequences: Tensor of shape ``[batch, seq_len]`` with alternating melody
            and chord tokens.
        tokenizer: Hooktheory tokenizer with ``id_to_name`` mapping.
        sequence_order: ``"chord_first"`` or ``"melody_first"``.

    Returns:
        int64 tensor [batch] of frame counts, one per sequence.
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    counts: List[int] = []
    for seq in sequences:
        seq_list = seq.cpu().tolist()
        _, chord_tokens = _split_melody_chord_lanes(seq_list, sequence_order)
        num_frames = sum(
            1 for token in chord_tokens if tokenizer.id_to_name.get(token, "") != "PAD"
        )
        counts.append(num_frames)
    return torch.tensor(counts, dtype=torch.int64)


_NON_SONG_TOKENS = frozenset({"PAD", "BOS", "EOS"})


def _silence_ratio(
    lane_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
) -> float:
    """Fraction of valid (non-PAD/BOS/EOS) frames in one lane that are SILENCE."""
    total = 0
    silence = 0
    for token in lane_tokens:
        name = tokenizer.id_to_name.get(token, "")
        if name in _NON_SONG_TOKENS:
            continue
        total += 1
        if name == "SILENCE":
            silence += 1
    return silence / total if total > 0 else float("nan")


def evaluate_chord_silence_ratio(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> torch.Tensor:
    """Fraction of chord-lane frames that are SILENCE, per sequence (e.g. 0.01 = 1%).

    Denominator is valid song frames (excludes PAD/BOS/EOS, but SILENCE itself
    counts toward the total since it's real content, not padding).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )
    ratios = [
        _silence_ratio(_split_melody_chord_lanes(seq.cpu().tolist(), sequence_order)[1], tokenizer)
        for seq in sequences
    ]
    return torch.tensor(ratios, dtype=torch.float32)


def evaluate_melody_silence_ratio(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
) -> torch.Tensor:
    """Fraction of melody-lane frames that are SILENCE, per sequence (e.g. 0.01 = 1%).

    Denominator is valid song frames (excludes PAD/BOS/EOS, but SILENCE itself
    counts toward the total since it's real content, not padding).
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )
    ratios = [
        _silence_ratio(_split_melody_chord_lanes(seq.cpu().tolist(), sequence_order)[0], tokenizer)
        for seq in sequences
    ]
    return torch.tensor(ratios, dtype=torch.float32)


def evaluate_initial_silence(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """
    Compute the number initial silence tokens in each sequence in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain model tokens and odd indices contain context tokens.
    For each sequence, the function counts the number of initial silence tokens.
    The count is the number of consecutive silence tokens at the beginning of the sequence.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.

    Returns:
        torch.Tensor: A list of initial silence, one for each sequence in the batch.
    """
    silence_count_all = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        model_part_tokens = seq[::2]

        silence_count = 0

        # Get token names using tokenizer's id_to_name mapping.
        for token in model_part_tokens:
            token_name = tokenizer.id_to_name.get(token, "")
            if token_name == "SILENCE":
                silence_count += 1
            else:
                break

        silence_count_all.append(silence_count)

    return torch.tensor(silence_count_all).float()


def evaluate_average_duration(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """
    Compute the average duration of the model part in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain model tokens and odd indices contain context tokens.
    For each sequence, the function computes the average duration of the model part.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.

    Returns:
        torch.Tensor: A list of average durations, one for each sequence in the batch.
    """
    special_tokens = {"PAD", "BOS", "EOS", "SILENCE"}

    durations = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        model_part_tokens = seq[::2]

        num_onset = 0
        num_tokens = 0
        num_silence = 0

        # Get token names using tokenizer's id_to_name mapping.
        for token in model_part_tokens:
            token_name = tokenizer.id_to_name.get(token, "")
            if token_name in special_tokens:
                continue
            else:
                num_tokens += 1
                if token_name.startswith("NOTE_ON_") or token_name.startswith(
                    "CHORD_ON_"
                ):
                    num_onset += 1
                elif token_name == "SILENCE":
                    num_silence += 1

        # Compute the average duration of the model part.
        if num_onset > 0:
            duration = (num_tokens - num_silence) / num_onset
        else:
            duration = 0.0

        durations.append(duration)

    return torch.tensor(durations)
