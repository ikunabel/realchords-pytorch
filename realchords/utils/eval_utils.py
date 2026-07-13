"""Evaluation utilities for ReaLchords."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np
import torch
import note_seq.chord_symbols_lib as chord_symbols_lib

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


def _circular_semitone_distance(pc_a: int, pc_b: int) -> int:
    distance = abs(pc_a - pc_b) % 12
    return min(distance, 12 - distance)


def _min_distance_to_mode(pc: int, mode_pcs: FrozenSet[int]) -> int:
    return min(_circular_semitone_distance(pc, mode_pc) for mode_pc in mode_pcs)


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


def _mode_coverage_score(
    histogram: Sequence[float],
    total_weight: float,
    mode_pcs: FrozenSet[int],
) -> float:
    if total_weight <= 0:
        return 0.0
    in_mode_weight = sum(histogram[pc] for pc in mode_pcs)
    return in_mode_weight / total_weight


def _mode_distance_score(
    histogram: Sequence[float],
    total_weight: float,
    mode_pcs: FrozenSet[int],
    sigma: float,
) -> float:
    if total_weight <= 0:
        return 0.0
    variance = 2.0 * sigma * sigma
    score = 0.0
    for pc, weight in enumerate(histogram):
        if weight <= 0:
            continue
        distance = _min_distance_to_mode(pc, mode_pcs)
        score += weight * math.exp(-(distance * distance) / variance)
    return score / total_weight


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
    scoring: str,
    sigma: float,
) -> float:
    if total_weight <= 0 or not candidate_mode_pcs:
        return 0.0

    best_score = 0.0
    for mode_pcs in candidate_mode_pcs:
        if scoring == "coverage":
            score = _mode_coverage_score(histogram, total_weight, mode_pcs)
        elif scoring == "distance":
            score = _mode_distance_score(histogram, total_weight, mode_pcs, sigma)
        elif scoring == "strict":
            score = _mode_strict_score(histogram, total_weight, mode_pcs)
        else:
            raise ValueError(
                f"Unsupported scoring '{scoring}'. Expected 'coverage', 'distance', or 'strict'."
            )
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


def _sequence_mode_fit_score(
    melody_tokens: Sequence[int],
    chord_tokens: Sequence[int],
    tokenizer: HooktheoryTokenizer,
    quality_map: Dict[str, QualityModeEntry],
    mode_pitch_classes: ModePitchClasses,
    scoring: str,
    sigma: float,
    skip_underdetermined: bool,
    min_melody_weight: float,
) -> Tuple[float, float, int]:
    weighted_score_sum = 0.0
    melody_weight_sum = 0.0
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
            scoring=scoring,
            sigma=sigma,
        )
        segment_count += 1
        if scoring == "strict":
            segment_pass_sum += segment_score
        else:
            weighted_score_sum += segment_score * total_weight
            melody_weight_sum += total_weight

    if segment_count <= 0:
        return np.nan, 0.0, segment_count
    if scoring == "strict":
        return segment_pass_sum / segment_count, float(segment_count), segment_count
    if melody_weight_sum <= 0:
        return np.nan, 0.0, segment_count
    return weighted_score_sum / melody_weight_sum, melody_weight_sum, segment_count


def _frame_mode_fit_score(
    pitch_class: int,
    chord_symbol: str,
    quality_map: Dict[str, QualityModeEntry],
    mode_pitch_classes: ModePitchClasses,
    scoring: str,
    sigma: float,
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
        scoring=scoring,
        sigma=sigma,
    )
    return True, score


def evaluate_melody_mode_fit_per_frame(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    sequence_order: str = "chord_first",
    scoring: str = "strict",
    sigma: float = 1.5,
    skip_underdetermined: bool = True,
    mode_map_path: Path | str = DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
) -> Dict[str, torch.Tensor]:
    """Per-frame note-in-mode for interleaved chord-first sequences.

    Each active melody frame is scored against candidate modes for the chord
    sounding at that frame (same chord-quality lookup as the segment metric).

    Returns a dict with:
        matches: float tensor [batch, num_frames] — score in [0, 1], NaN if unscorable
        valid: bool tensor [batch, num_frames]
        mean: float tensor [batch] — mean over valid frames (NaN if none)
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )
    if scoring not in {"coverage", "distance", "strict"}:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. Expected 'coverage', 'distance', or 'strict'."
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
                scoring=scoring,
                sigma=sigma,
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
    scoring: str = "strict",
    sigma: float = 1.5,
    skip_underdetermined: bool = True,
    min_melody_weight: float = 1.0,
    mode_map_path: Path | str = DEFAULT_CHORD_QUALITY_MODE_MAP_PATH,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score how well melody notes over each chord region fit a mode.

    Unlike note-in-chord, this metric is segment-based: for each span between
    consecutive ``CHORD_ON`` events, melody pitch classes (duration-weighted by
    frame holds) are compared against candidate modes from
    ``chord_quality_mode_map.jsonl``. The best-fitting candidate mode yields a
    per-segment score; sequence score aggregates across segments (unweighted mean
    for ``strict``, melody-weighted mean otherwise).

    Args:
        sequences: Tensor of shape ``[batch, seq_len]`` with alternating melody and
            chord tokens.
        tokenizer: Hooktheory tokenizer with ``id_to_name`` mapping.
        model_part: ``"melody"`` or ``"chord"`` (validated for API parity).
        return_count: If True, also return melody-weight and segment counts.
        sequence_order: ``"chord_first"`` or ``"melody_first"``.
        scoring: ``"strict"`` (all notes must be in mode or segment scores 0),
            ``"coverage"`` (fraction of weighted melody in mode), or
            ``"distance"`` (Gaussian kernel on semitone distance to nearest mode
            tone).
        sigma: Kernel width for ``scoring="distance"``.
        skip_underdetermined: Skip segments for sparse qualities like ``ped``/``5``.
        min_melody_weight: Minimum weighted melody mass required to score a segment.
        mode_map_path: Path to curated chord-quality mode JSONL.

    Returns:
        Tensor of mode-fit ratios per sequence, or a tuple with weights and
        segment counts when ``return_count=True``.
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )
    if model_part not in {"melody", "chord"}:
        raise ValueError(f"Invalid model_part: {model_part}")
    if scoring not in {"coverage", "distance", "strict"}:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. Expected 'coverage', 'distance', or 'strict'."
        )

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
            scoring=scoring,
            sigma=sigma,
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
