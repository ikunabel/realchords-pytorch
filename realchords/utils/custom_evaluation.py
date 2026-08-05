#!/usr/bin/env python3
"""Paired chord evaluation: compare GT chords vs model-predicted chords for the same melodies.

Every song in the validation split is processed once. The GT melody is used as
conditioning for every specified model, producing predictions that share the same
melodic context and can be compared row-by-row.  Full provenance (song URL, dataset
index) is saved alongside the tensors.

Usage (argbind, config-driven -- see configs/custom_eval/ and
scripts/eval/custom_eval/run_custom_eval.py for the presets that replace the old
custom_eval.sh shell functions):
    python realchords/utils/custom_evaluation.py \
        --args.load configs/custom_eval/paired_hooktheory.yml

    python realchords/utils/custom_evaluation.py \
        --args.load configs/custom_eval/gt_hooktheory.yml

Or equivalently, without a config file:
    python realchords/utils/custom_evaluation.py \
        --base_model logs/mle_chord/step=10000.ckpt \
        --model "MLE=base RealJam=logs/realchords/actor.pth GAPT=logs/gapt/actor.pth" \
        --dataset_name hooktheory \
        --dataset_split test \
        --save_dir logs/custom_eval/hooktheory \
        --num_batches 32 \
        --batch_size 64 \
        --seed 42

    python realchords/utils/custom_evaluation.py \
        --gt_only \
        --dataset_name hooktheory \
        --dataset_split test \
        --save_dir logs/custom_eval/gt/hooktheory

MIDI export is a separate step -- see scripts/eval/custom_eval/export_paired_midis.py,
which reads this script's saved tensors/metadata and writes MIDI without
re-running generation or metrics.

Every run produces two variants under --save_dir, each with its own full set
of outputs (gt.pt, {slug}.pt, etc.):
    <save_dir>/cropped_songs/   legacy 256-frame (8-bar melody + 8-bar chord)
                                random crop, aligned to a chord onset
    <save_dir>/full_songs/      whole songs, uncropped (sized to the longest
                                song in the split) -- for listening to full
                                MIDIs or computing frame-level GT metrics over
                                entire songs rather than a random window
Model generation works for both (the target length is derived from the
batch, not hardcoded), but a model's own positional-embedding limit -- if
any -- is a separate constraint that "full_songs" doesn't route around.

Model spec format  (--model):
    Label=base          use the base MLE checkpoint directly
    Label=path.ckpt     load a Lightning checkpoint (needs args.yml alongside)
    Label=path.pth      load RL actor weights on top of the base model

Outputs (all in --save_dir):
    metadata.jsonl                    one JSON line per row: {seq_idx, song_url, nicr}
    gt.pt                             GT sequences  [N, seq_len]
    {slug}.pt                         model predictions [N, seq_len] for each --model
    gt_nicr_per_frame.pt              per-frame NiCR for GT (matches/valid/mean)
    {slug}_nicr_per_frame.pt          per-frame NiCR for each model
    gt_mode_per_frame.pt              per-frame note-in-mode for GT
    {slug}_mode_per_frame.pt          per-frame note-in-mode for each model
    gt_chords_per_frame.pt            per-frame chord symbols for GT
    {slug}_chords_per_frame.pt        per-frame chord symbols for each model
    gt_chord_distribution.json        GT chord counts (onset + frame) for cross-dataset analysis
    gt_chord_durations.pt             chord segment durations (frames) for GT
    gt_note_durations.pt              melody note durations (frames) for GT
    gt_chord_silence_ratio.pt         per-sequence chord-lane SILENCE fraction for GT
    gt_melody_silence_ratio.pt        per-sequence melody-lane SILENCE fraction for GT
    gt_num_frames.pt                  per-sequence valid (non-PAD) frame count for GT
    gt_sync_intervals.pt              chord-to-note onset intervals (synchronization) for GT
    gt_chord_complexity.pt            distinct-pitch-class count per chord segment for GT
    {slug}_chord_durations.pt         same, per model
    {slug}_note_durations.pt          same, per model
    {slug}_chord_silence_ratio.pt     same, per model
    {slug}_melody_silence_ratio.pt    same, per model
    {slug}_num_frames.pt              same, per model
    {slug}_sync_intervals.pt          same, per model
    {slug}_chord_complexity.pt        same, per model
    means.json                        summary: rhythm/silence/note-in-chord/note-in-mode/
                                       chord-complexity means per source, plus model-vs-GT
                                       comparisons (sync EMD, chord-type JS distance) -- see
                                       realchords/utils/eval_utils.py
    chord_names_augmented.json        vocab snapshot for downstream decoding
    model_labels.json                 {label: slug} map + dataset_name, for
                                       scripts/eval/custom_eval/export_paired_midis.py
"""

import copy
import json
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import types

import argbind
import numpy as np
import torch
from lightning import seed_everything
from tqdm import tqdm

from realchords.constants import FRAME_PER_BEAT
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.decoder_only import LitDecoder
from realchords.lit_module.enc_dec import LitEncoderDecoder
from realchords.utils.chord_diversity_analysis import (
    compute_vendi_score,
    load_contrastive_reward_direct,
)
from realchords.utils.eval_utils import (
    chord_type_distribution,
    chord_type_js_distance,
    duration_entropy,
    evaluate_chord_complexity,
    evaluate_chord_durations,
    evaluate_chord_silence_ratio,
    evaluate_chord_symbols_per_frame,
    evaluate_chord_to_note_onset_intervals,
    evaluate_melody_mode_fit_per_frame,
    evaluate_melody_mode_fit_ratio,
    evaluate_melody_silence_ratio,
    evaluate_note_durations,
    evaluate_note_in_chord_per_frame,
    evaluate_note_in_chord_ratio,
    evaluate_sequence_num_frames,
    synchronization_emd,
)
from realchords.utils.experiment_utils import (
    DATASET_CACHE_DIRS,
    _random_crop_on_chord_onset,
    create_dataset_dataloaders,
    replace_eos_with_pad,
    resolve_chord_names_path_for_dataset,
    save_vocab_snapshot,
)
from realchords.utils.experiment_utils_model_data import (
    generate_from_data,
    generate_from_data_enc_dec,
)
from realchords.utils.inference_utils import load_lit_model, load_rl_model
from realchords.utils.midi_export import (
    resolve_include_chord_bass,
    select_midi_indices,
    write_all_source_midis,
    write_source_midis,
)
from realchords.utils.sequence_penalty_analysis import strip_bos


def _save_nicr_per_frame(
    tensor: torch.Tensor,
    tokenizer,
    path: Path,
) -> Tuple[torch.Tensor, float]:
    """Compute per-frame NiCR, save .pt, return (per-sequence means [N], pooled ratio).

    The pooled ratio matches ``evaluate_generated_sequences.py`` (the original
    repo's script): total correct frames / total valid frames across the
    whole batch -- every *frame* weighted equally, not every *sequence*. This
    differs from a mean of per-sequence ratios whenever sequence lengths
    vary; kept consistent with that script rather than the paper's own
    per-song-then-averaged recipe (Appendix K), per user preference.
    """
    stripped = strip_bos(tensor, tokenizer)
    nicr = evaluate_note_in_chord_per_frame(stripped, tokenizer)
    torch.save(nicr, path)
    _, valid_counts, correct_counts = evaluate_note_in_chord_ratio(
        stripped, tokenizer, model_part="chord", return_count=True
    )
    total_valid = int(valid_counts.sum().item())
    pooled_ratio = (
        float(correct_counts.sum().item()) / total_valid
        if total_valid > 0
        else float("nan")
    )
    return nicr["mean"], pooled_ratio


def _save_mode_per_frame(
    tensor: torch.Tensor,
    tokenizer,
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Compute per-frame note-in-mode, save .pt, return
    (per-sequence per-frame means [N], per-sequence per-segment ratios [N], pooled ratio).

    The pooled ratio matches ``evaluate_generated_sequences.py``: a
    segment-weighted average of ``evaluate_melody_mode_fit_ratio``'s
    per-sequence ratios across the whole batch -- every chord *segment*
    weighted equally, not every *sequence*. This uses the segment-based
    mode-fit function, not the per-frame one saved to ``path`` -- the two are
    different measurements (per-chord-segment histogram fit vs. per-note
    fit). The per-segment per-sequence ratios are also returned (not just the
    per-frame ones) so a "mean of per-song ratios" can be reported on the
    *same* segment-based definition as the pooled number, rather than mixing
    the two different measurements.
    """
    stripped = strip_bos(tensor, tokenizer)
    mode_fit = evaluate_melody_mode_fit_per_frame(
        stripped,
        tokenizer,
    )
    torch.save(mode_fit, path)

    seq_ratio, _, segment_counts = evaluate_melody_mode_fit_ratio(
        stripped, tokenizer, model_part="chord", return_count=True
    )
    scorable = torch.isfinite(seq_ratio) & (segment_counts > 0)
    total_segments = int(segment_counts[scorable].sum().item()) if scorable.any() else 0
    pooled_ratio = (
        float((seq_ratio[scorable] * segment_counts[scorable].float()).sum().item())
        / total_segments
        if total_segments > 0
        else float("nan")
    )
    return mode_fit["mean"], seq_ratio, pooled_ratio


def _save_chords_per_frame(
    tensor: torch.Tensor,
    tokenizer,
    path: Path,
) -> Dict[str, object]:
    """Compute per-frame chord symbols and save .pt."""
    stripped = strip_bos(tensor, tokenizer)
    chords = evaluate_chord_symbols_per_frame(stripped, tokenizer)
    torch.save(chords, path)
    return chords


def _chord_distribution(chords: Dict[str, object]) -> Dict[str, object]:
    """Aggregate per-frame chord symbols into onset and frame counts."""
    onset_counts = chord_type_distribution(chords, weighting="onset")
    frame_counts = chord_type_distribution(chords, weighting="frame")
    return {
        "onset_counts": onset_counts,
        "frame_counts": frame_counts,
        "num_onsets": int(sum(onset_counts.values())),
        "num_chord_frames": int(sum(frame_counts.values())),
        "num_unique_chords_onset": len(onset_counts),
        "num_unique_chords_frame": len(frame_counts),
    }


def _save_chord_distribution(
    chords: Dict[str, object],
    path: Path,
    *,
    dataset_name: str,
    dataset_split: str,
    num_sequences: int,
) -> None:
    dist = _chord_distribution(chords)
    dist["dataset_name"] = dataset_name
    dist["dataset_split"] = dataset_split
    dist["num_sequences"] = num_sequences
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dist, fh, indent=2, sort_keys=True)


def _nanmean(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return float("nan")
    return float(tensor.float().nanmean().item())


def _nansum(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().nansum().item())


def _compute_vendi_score(
    tensor: torch.Tensor,
    reward_wrapper,
    reward_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> Optional[float]:
    """Vendi diversity score for one population: one source's (GT, or one
    model's) full accumulated tensor within one crop-variant run.

    Vendi is a population-level metric (an eigenvalue-entropy-of-similarity-
    kernel quantity over many embedded sequences, not defined for a single
    sequence on its own) -- ``tensor`` here already *is* that population,
    the same way it already is for ``chord_duration_entropy`` /
    ``note_duration_entropy`` below. Embeds each sequence's chord lane via
    the contrastive reward model's ``get_chord_embed``, same computation as
    ``chord_diversity_analysis.accumulate_diversity_metrics``, just reading
    an in-memory tensor instead of a list of .pt files.

    Ignores the ``device`` argument in favor of ``reward_model``'s own
    device: ``load_contrastive_reward_direct`` can silently fall back to CPU on a
    CUDA OOM, so trusting the model's actual parameter device (rather than
    whatever device the rest of this run is using) avoids a device
    mismatch in that case.
    """
    del device
    model_device = next(reward_model.parameters()).device
    pad = reward_wrapper.pad_token_id
    bos = reward_wrapper.bos_token_id
    eos = reward_wrapper.eos_token_id
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.size(0), batch_size):
            batch = tensor[start : start + batch_size]
            if batch.size(1) == 0:
                continue
            model_tokens, _, model_mask, _ = reward_wrapper.get_inputs_from_sequence(batch)
            chord_mask = (
                (model_tokens != pad) & (model_tokens != bos) & (model_tokens != eos)
            )
            non_empty = chord_mask.sum(dim=1) > 0
            if not non_empty.any():
                continue
            embed_batch = reward_model.get_chord_embed(
                chord=model_tokens[non_empty].to(model_device),
                chord_mask=model_mask[non_empty].to(model_device),
            )
            embeddings.append(embed_batch.cpu().numpy())
    return compute_vendi_score(embeddings)


def _save_mean_metrics(
    tensor: torch.Tensor,
    tokenizer,
    save_dir: Path,
    prefix: str,
    chords: Dict[str, object],
    *,
    reward_wrapper: Optional[object],
    reward_model: Optional[torch.nn.Module],
    device: torch.device,
) -> Dict[str, object]:
    """Rhythm/silence/synchronization metrics for one source (GT or a model).

    Saves per-sequence .pt files (durations, silence ratios, frame counts,
    synchronization intervals) and returns a summary dict -- both scalar
    means for printing, and the raw pooled distributions needed to compare
    this source against another one (synchronization_emd, chord_type_js_distance).

    prefix is prepended to every filename as-is (include a trailing "_" for
    e.g. "gt_chord_durations.pt"; pass "" when save_dir's own name already
    disambiguates the source, e.g. a per-model models/<slug>/ folder).

    reward_wrapper/reward_model: None (no --contrastive_checkpoint given)
    skips Vendi entirely (vendi_score = None in the result) rather than
    computing it against a checkpoint that might not actually match
    tensor's chord vocab -- see main()'s contrastive_checkpoint docstring.
    """
    stripped = strip_bos(tensor, tokenizer)

    chord_durations = evaluate_chord_durations(stripped, tokenizer)
    torch.save(chord_durations, save_dir / f"{prefix}chord_durations.pt")
    chord_duration_entropy = (
        duration_entropy(chord_durations["durations_flat"])
        if chord_durations["durations_flat"].numel() > 0
        else float("nan")
    )

    note_durations = evaluate_note_durations(stripped, tokenizer)
    torch.save(note_durations, save_dir / f"{prefix}note_durations.pt")
    note_duration_entropy = (
        duration_entropy(note_durations["durations_flat"])
        if note_durations["durations_flat"].numel() > 0
        else float("nan")
    )

    chord_silence = evaluate_chord_silence_ratio(stripped, tokenizer)
    torch.save(chord_silence, save_dir / f"{prefix}chord_silence_ratio.pt")

    melody_silence = evaluate_melody_silence_ratio(stripped, tokenizer)
    torch.save(melody_silence, save_dir / f"{prefix}melody_silence_ratio.pt")

    num_frames = evaluate_sequence_num_frames(stripped, tokenizer)
    torch.save(num_frames, save_dir / f"{prefix}num_frames.pt")

    sync = evaluate_chord_to_note_onset_intervals(stripped, tokenizer)
    torch.save(sync, save_dir / f"{prefix}sync_intervals.pt")

    complexity = evaluate_chord_complexity(stripped, tokenizer)
    torch.save(complexity, save_dir / f"{prefix}chord_complexity.pt")

    # Population-level (not per-sequence): the whole `tensor` (this source,
    # this crop variant) is the population Vendi is computed over -- see
    # _compute_vendi_score's docstring. Skipped entirely (None) when no
    # contrastive checkpoint was given.
    vendi_score = (
        _compute_vendi_score(tensor, reward_wrapper, reward_model, device)
        if reward_model is not None
        else None
    )

    return {
        "chord_duration_entropy": chord_duration_entropy,
        "note_duration_entropy": note_duration_entropy,
        "chord_silence_ratio_mean": _nanmean(chord_silence),
        "melody_silence_ratio_mean": _nanmean(melody_silence),
        "num_frames_mean": _nanmean(num_frames),
        "num_frames_total": _nansum(num_frames),
        "chord_complexity_mean": _nanmean(complexity["mean"]),
        "vendi_score": vendi_score,
        "sync_intervals_flat": sync["intervals_flat"],
        "chord_dist_onset": chord_type_distribution(chords, weighting="onset"),
        "chord_dist_frame": chord_type_distribution(chords, weighting="frame"),
    }


def _compare_to_gt(model_means: Dict[str, object], gt_means: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Cross-source comparison metrics: model vs. GT (this run's reference)."""
    model_sync = model_means["sync_intervals_flat"]
    gt_sync = gt_means["sync_intervals_flat"]
    sync_emd = (
        synchronization_emd(model_sync, gt_sync)
        if model_sync.numel() > 0 and gt_sync.numel() > 0
        else None
    )

    def _safe_js(a: Dict[str, int], b: Dict[str, int]) -> Optional[float]:
        if not a or not b:
            return None
        return chord_type_js_distance(a, b)

    return {
        "sync_emd_vs_gt": sync_emd,
        "chord_type_js_distance_onset_vs_gt": _safe_js(
            model_means["chord_dist_onset"], gt_means["chord_dist_onset"]
        ),
        "chord_type_js_distance_frame_vs_gt": _safe_js(
            model_means["chord_dist_frame"], gt_means["chord_dist_frame"]
        ),
    }


def _fmt_nicr(mean: float) -> Optional[float]:
    if mean != mean:  # NaN
        return None
    return round(float(mean), 4)


def _fmt_metric(value: Optional[float]) -> Optional[float]:
    """NaN/None-safe rounding for JSON output (NaN isn't valid JSON)."""
    if value is None or value != value:  # None or NaN
        return None
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GROUP = __file__
bind = partial(argbind.bind, group=GROUP)

_VALID_DATASET_SPLITS = {"train", "valid", "test", "all"}


def _slugify(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "model"


def _parse_model_arg(raw: str) -> Tuple[str, str]:
    """Parse 'Label=path_or_keyword' → (label, path_or_keyword)."""
    if "=" not in raw:
        raise ValueError(f"Invalid --model value '{raw}'. Expected Label=path or Label=base")
    label, spec = raw.split("=", 1)
    return label.strip(), spec.strip()


def _validate_args(
    *,
    base_model: str,
    model: List[str],
    gt_only: bool,
    dataset_name: str,
    dataset_split: str,
    save_dir: str,
) -> None:
    if dataset_name not in DATASET_CACHE_DIRS:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Expected one of: {sorted(DATASET_CACHE_DIRS)}"
        )
    if dataset_split not in _VALID_DATASET_SPLITS:
        raise ValueError(
            f"Invalid dataset_split '{dataset_split}'. "
            f"Expected one of: {sorted(_VALID_DATASET_SPLITS)}"
        )
    if not save_dir:
        raise ValueError("save_dir must be provided.")
    if gt_only:
        if base_model or model:
            print("Note: base_model / model are ignored in gt_only mode.")
        return
    if not base_model:
        raise ValueError("base_model is required unless gt_only is set.")
    if not model:
        raise ValueError("At least one --model entry is required unless gt_only is set.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _max_song_frames(
    dataset_name: str,
    dataset_split: str,
    outlier_factor: float = 2.0,
) -> int:
    """Longest song (in frames, per melody/chord lane) in the given split(s),
    robust to a single mis-annotated outlier blowing up padding for every song.

    Used to size an uncropped ("full song") dataloader: `HooktheoryDataset
    .random_crop` (and its chord-onset-aligned variant) only crop when a
    song's frame count exceeds `max_len_per_part`, so passing a `max_len`
    derived from this never truncates anything (for the songs it's sized to),
    while still using the same fixed-size padding/collation the dataloader
    already relies on.

    Every song gets padded to whatever this returns, so one bad `num_beats`
    value (e.g. hooktheory's largest is 1716 beats vs. a 544-beat runner-up
    and a 36-beat median -- almost certainly a data error, not a real song)
    would otherwise force ~100x wasted padding on every other song in the
    split. Rather than a blanket percentile cutoff (which would also catch
    genuinely long songs in the long tail, not just the error), this compares
    each candidate max only against its immediate runner-up: a real outlier
    shows up as one isolated, dramatic jump (1716 vs. 544 -- more than 3x),
    whereas the legitimate long tail grows gradually (544 vs. 532 -- barely
    above 1x). Only isolated jumps past `outlier_factor` get dropped, and
    dropping stops the moment a step is no longer dramatic, so the real long
    tail is left untouched. Dropped songs just get legitimately cropped like
    any over-long song, same as the legacy crop path already does for songs
    longer than the crop length.
    """
    cache_dir = Path(DATASET_CACHE_DIRS[dataset_name.lower()])
    splits = ["train", "valid", "test"] if dataset_split == "all" else [dataset_split]
    all_beats: List[int] = []
    for split in splits:
        split_path = cache_dir / f"{split}.jsonl"
        if not split_path.exists():
            continue
        with open(split_path, encoding="utf-8") as handle:
            for line in handle:
                all_beats.append(json.loads(line)["annotations"].get("num_beats", 0))

    if not all_beats:
        return 0

    all_beats.sort()
    excluded = 0
    while (
        len(all_beats) >= 2
        and all_beats[-2] > 0
        and all_beats[-1] > outlier_factor * all_beats[-2]
    ):
        excluded += 1
        all_beats.pop()

    if excluded:
        print(
            f"_max_song_frames: excluded {excluded} outlier song(s) whose "
            f"num_beats was more than {outlier_factor}x their runner-up "
            f"from full_songs sizing -- using {all_beats[-1]} instead. "
            "These will be cropped like any over-long song instead of "
            "forcing huge padding on every other song in the split."
        )

    return all_beats[-1] * FRAME_PER_BEAT


_LIT_MODULE_CLS = {
    "decoder_only": LitDecoder,
    "encoder_decoder": LitEncoderDecoder,
}


def _detect_model_type(checkpoint_path: str) -> str:
    """Read 'model_type' ('decoder_only' or 'encoder_decoder') from a
    checkpoint's own args.yml, so the right lit_module_cls (and later, the
    right generation function) gets used automatically instead of assuming
    every checkpoint is decoder-only."""
    experiment_dir = Path(checkpoint_path).parent
    args = argbind.load_args(experiment_dir / "args.yml")
    model_type = args.get("model_type", "decoder_only")
    if model_type not in _LIT_MODULE_CLS:
        raise ValueError(
            f"{checkpoint_path}: unsupported model_type '{model_type}' in args.yml "
            f"(expected one of {sorted(_LIT_MODULE_CLS)})"
        )
    return model_type


def _load_models(
    base_ckpt: str,
    model_specs: List[Tuple[str, str]],   # [(label, path_or_keyword), ...]
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, torch.nn.Module], object, Dict[str, str]]:
    """Load all requested models. Returns (label → model, tokenizer, label → model_type).

    Each checkpoint's architecture (decoder-only vs encoder-decoder) is
    detected independently from its own args.yml -- base_model and every
    Label=path.ckpt entry can be either, mixed freely in one run, as long as
    they share a compatible chord vocab (see dataset_chord_names_path in
    main()). RL actors (Label=path.pth) are always decoder-only -- this
    codebase has no encoder-decoder RL fine-tuning path.
    """
    base_model_type = _detect_model_type(base_ckpt)
    print(f"Loading base model from {base_ckpt} (model_type={base_model_type}) …")
    base_model, tokenizer, _ = load_lit_model(
        model_path=base_ckpt,
        lit_module_cls=_LIT_MODULE_CLS[base_model_type],
        batch_size=batch_size,
        compile=False,
    )
    base_model.eval().to(device)

    models: Dict[str, torch.nn.Module] = {}
    model_types: Dict[str, str] = {}
    for label, spec in model_specs:
        slug = _slugify(label)
        if spec == "base":
            print(f"  {label}: using base MLE model")
            models[label] = base_model
            model_types[label] = base_model_type
        elif spec.endswith(".ckpt"):
            spec_model_type = _detect_model_type(spec)
            print(f"  {label}: loading Lightning checkpoint {spec} (model_type={spec_model_type})")
            m, _, _ = load_lit_model(
                model_path=spec,
                lit_module_cls=_LIT_MODULE_CLS[spec_model_type],
                batch_size=batch_size,
                compile=False,
            )
            m.eval().to(device)
            models[label] = m
            model_types[label] = spec_model_type
        elif spec.endswith(".pth"):
            print(f"  {label}: loading RL actor {spec}")
            m = load_rl_model(
                model_path=spec,
                model=copy.deepcopy(base_model),
                compile=False,
            )
            m.eval().to(device)
            models[label] = m
            model_types[label] = "decoder_only"
        else:
            raise ValueError(
                f"Unrecognised model spec '{spec}' for '{label}'. "
                "Expected 'base', a '.ckpt' path, or a '.pth' path."
            )
        print(f"    → loaded '{label}' as '{slug}'")

    return models, tokenizer, model_types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_eval(
    args: types.SimpleNamespace,
    *,
    device: torch.device,
    models: Dict[str, torch.nn.Module],
    model_tokenizer,
    model_specs: List[Tuple[str, str]],
    model_types: Dict[str, str],
    max_len: int,
    save_dir: Path,
    run_label: str,
    reward_wrapper: Optional[object],
    reward_model: Optional[torch.nn.Module],
    dataset_chord_names_path: Optional[str] = None,
    midi_samples: int = 10,
    seed: int = 42,
) -> None:
    """Run one full eval pass (dataloader → generation → save) into save_dir.

    Called once per crop variant ("cropped_songs", "full_songs") from main().
    """
    # Chord dataloader — shuffle=False (already the default in create_dataset_dataloaders).
    # dataset_chord_names_path pins this to the loaded models' own vocab
    # (set by main() when models are loaded) instead of letting it
    # auto-resolve to whatever the current global vocab is -- see the
    # comment where main() builds it.
    _, val_loader = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="chord",
        batch_size=args.batch_size,
        max_len=max_len,
        chord_names_path=dataset_chord_names_path,
    )
    dataset_tokenizer = val_loader.dataset.tokenizer
    tokenizer = dataset_tokenizer if args.gt_only else model_tokenizer
    # Ensure every crop starts on a CHORD_ON event so the strict decoder
    # never sees a hold token at frame 0 (same fix as handle_data_only_mode).
    val_loader.dataset.random_crop = types.MethodType(
        _random_crop_on_chord_onset, val_loader.dataset
    )

    n_batches = len(val_loader) if args.num_batches == -1 else args.num_batches
    print(f"\n=== {run_label} ===")
    print(f"Dataset: {args.dataset_name} / {args.dataset_split}")
    print(f"Batches to process: {n_batches}  (total available: {len(val_loader)})")
    if args.gt_only:
        print("Mode: GT only")
    else:
        print(f"Models: {[l for l, _ in model_specs]}")
    print(f"Saving to: {save_dir}\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Output layout: gt_only keeps the original flat "gt_<name>" files
    # directly in save_dir (many downstream consumers depend on this exact
    # shape -- run_clamp2_probe.sh, write_dataset_statistics.py, the
    # midi_gt_<dataset> presets, ...). Model-comparison mode uses gt/ +
    # models/<slug>/ subfolders instead, with no "gt_"/"<slug>_" filename
    # prefix needed since the folder itself disambiguates -- see main()'s
    # save_dir docstring.
    gt_out = save_dir if args.gt_only else save_dir / "gt"
    gt_out.mkdir(parents=True, exist_ok=True)
    gt_prefix = "gt_" if args.gt_only else ""
    models_out: Dict[str, Path] = {}
    if not args.gt_only:
        for label, _ in model_specs:
            d = save_dir / "models" / _slugify(label)
            d.mkdir(parents=True, exist_ok=True)
            models_out[label] = d

    # Resume support: if generation already ran and saved gt.pt/preds.pt +
    # metadata_raw.json (e.g. a prior run crashed later, during metrics
    # computation), skip the slow generation loop and load those straight
    # from disk instead of regenerating.
    metadata_raw_path = save_dir / "metadata_raw.json"
    gt_tensor_path = gt_out / "gt.pt"
    model_paths = {label: models_out[label] / "preds.pt" for label, _ in model_specs}
    can_resume = (
        gt_tensor_path.exists()
        and metadata_raw_path.exists()
        and (args.gt_only or all(p.exists() for p in model_paths.values()))
    )

    if can_resume:
        print(f"Found existing gt.pt/preds.pt + metadata_raw.json in {save_dir} -- "
              "skipping generation, reusing saved tensors.")
        gt_tensor = torch.load(gt_tensor_path)
        model_rows: Dict[str, List[torch.Tensor]] = {
            label: [torch.load(model_paths[label])] for label, _ in model_specs
        }
        with metadata_raw_path.open(encoding="utf-8") as fh:
            metadata: List[Dict] = json.load(fh)
        seq_idx = len(metadata)
    else:
        # Accumulators
        gt_rows: List[torch.Tensor] = []
        model_rows = {label: [] for label, _ in model_specs}
        metadata = []
        seq_idx = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, total=n_batches, desc="Batches")
            ):
                if batch_idx >= n_batches:
                    break

                # GT sequences: [batch, seq_len] with BOS+EOS stripped to [BOS, tokens…]
                gt_seq = batch["targets"].to(device)          # [B, seq_len] with BOS+EOS
                gt_seq_stripped = gt_seq[:, 1:-1]             # drop leading BOS + trailing EOS
                gt_seq_stripped = replace_eos_with_pad(gt_seq_stripped, tokenizer)

                song_urls: List[str] = batch.get("song_url", ["unknown"] * gt_seq.size(0))

                # Record metadata for each sequence in the batch
                for b in range(gt_seq.size(0)):
                    metadata.append({
                        "seq_idx": seq_idx + b,
                        "song_url": song_urls[b] if isinstance(song_urls, list) else song_urls,
                        "batch_idx": batch_idx,
                        "batch_pos": b,
                        "dataset_name": args.dataset_name,
                    })

                # Save GT (with BOS, without EOS)
                gt_with_bos = torch.cat([
                    torch.full((gt_seq.size(0), 1), tokenizer.bos_token,
                               dtype=torch.long, device=device),
                    gt_seq_stripped,
                ], dim=1)
                gt_rows.append(gt_with_bos.cpu())

                if not args.gt_only:
                    target_seq_len = gt_with_bos.size(1)
                    for label, model in models.items():
                        if model_types[label] == "encoder_decoder":
                            preds = generate_from_data_enc_dec(
                                model=model,
                                sequences=gt_seq_stripped,
                                tokenizer=model_tokenizer,
                                target_seq_len=target_seq_len,
                            )
                        else:
                            preds = generate_from_data(
                                model=model,
                                sequences=gt_seq_stripped,
                                tokenizer=model_tokenizer,
                                prompt_steps=0,
                                target_seq_len=target_seq_len,
                            )
                        model_rows[label].append(preds.cpu())

                seq_idx += gt_seq.size(0)

        gt_tensor = torch.cat(gt_rows, dim=0)

    # ---- Save ---------------------------------------------------------------
    print(f"\nSaving {seq_idx} sequences to {save_dir} …")

    # GT
    torch.save(gt_tensor, gt_tensor_path)
    print(f"  {gt_tensor_path.relative_to(save_dir)}  {tuple(gt_tensor.shape)}")

    # Per-model predictions
    model_tensors: Dict[str, torch.Tensor] = {}
    if not args.gt_only:
        for label, rows in model_rows.items():
            tensor = torch.cat(rows, dim=0)
            model_tensors[label] = tensor
            torch.save(tensor, model_paths[label])
            print(f"  {model_paths[label].relative_to(save_dir)}  {tuple(tensor.shape)}")

    # Raw per-sequence metadata (song_url/batch provenance only, not yet
    # enriched with nicr/mode_fit) -- persisted now so a crash during metrics
    # computation below doesn't lose generation results; enables the resume
    # path above on the next run.
    with metadata_raw_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh)

    gt_nicr_means, gt_nicr_pooled = _save_nicr_per_frame(
        gt_tensor, dataset_tokenizer, gt_out / f"{gt_prefix}nicr_per_frame.pt"
    )
    print(f"  gt/{gt_prefix}nicr_per_frame.pt  pooled={gt_nicr_pooled:.4f}")

    gt_mode_means, gt_mode_seg_ratios, gt_mode_pooled = _save_mode_per_frame(
        gt_tensor,
        dataset_tokenizer,
        gt_out / f"{gt_prefix}mode_per_frame.pt",
    )
    print(f"  gt/{gt_prefix}mode_per_frame.pt  pooled={gt_mode_pooled:.4f}")

    gt_chords = _save_chords_per_frame(
        gt_tensor, dataset_tokenizer, gt_out / f"{gt_prefix}chords_per_frame.pt"
    )
    print(f"  gt/{gt_prefix}chords_per_frame.pt")

    dist_path = gt_out / f"{gt_prefix}chord_distribution.json"
    _save_chord_distribution(
        gt_chords,
        dist_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        num_sequences=gt_tensor.size(0),
    )
    print(f"  gt/{gt_prefix}chord_distribution.json  ({dist_path})")

    gt_means = _save_mean_metrics(
        gt_tensor, dataset_tokenizer, gt_out, gt_prefix, gt_chords,
        reward_wrapper=reward_wrapper, reward_model=reward_model, device=device,
    )
    print(
        f"  gt/{gt_prefix}chord_durations.pt / {gt_prefix}note_durations.pt  "
        f"chord_entropy={gt_means['chord_duration_entropy']:.4f} "
        f"note_entropy={gt_means['note_duration_entropy']:.4f}"
    )
    print(f"  gt vendi_score={gt_means['vendi_score']}")
    print(
        f"  gt/{gt_prefix}chord_silence_ratio.pt / {gt_prefix}melody_silence_ratio.pt  "
        f"chord={gt_means['chord_silence_ratio_mean']:.4f} "
        f"melody={gt_means['melody_silence_ratio_mean']:.4f}"
    )
    print(
        f"  gt/{gt_prefix}num_frames.pt  mean={gt_means['num_frames_mean']:.1f} "
        f"total={gt_means['num_frames_total']:.0f}"
    )
    print(f"  gt/{gt_prefix}sync_intervals.pt")
    print(f"  gt/{gt_prefix}chord_complexity.pt  mean={gt_means['chord_complexity_mean']:.4f}")

    model_nicr_means: Dict[str, torch.Tensor] = {}
    model_nicr_pooled: Dict[str, float] = {}
    model_mode_means: Dict[str, torch.Tensor] = {}
    model_mode_seg_ratios: Dict[str, torch.Tensor] = {}
    model_mode_pooled: Dict[str, float] = {}
    model_means_by_label: Dict[str, Dict[str, object]] = {}
    if not args.gt_only:
        for label, rows in model_rows.items():
            slug = _slugify(label)
            out_dir = models_out[label]
            tensor = torch.cat(rows, dim=0)
            means, pooled = _save_nicr_per_frame(
                tensor, model_tokenizer, out_dir / "nicr_per_frame.pt"
            )
            model_nicr_means[label] = means
            model_nicr_pooled[label] = pooled
            print(f"  models/{slug}/nicr_per_frame.pt  pooled={pooled:.4f}")
            mode_means, mode_seg_ratios, mode_pooled = _save_mode_per_frame(
                tensor,
                model_tokenizer,
                out_dir / "mode_per_frame.pt",
            )
            model_mode_means[label] = mode_means
            model_mode_seg_ratios[label] = mode_seg_ratios
            model_mode_pooled[label] = mode_pooled
            print(f"  models/{slug}/mode_per_frame.pt  pooled={mode_pooled:.4f}")
            model_chords = _save_chords_per_frame(
                tensor, model_tokenizer, out_dir / "chords_per_frame.pt"
            )
            print(f"  models/{slug}/chords_per_frame.pt")

            model_means = _save_mean_metrics(
                tensor, model_tokenizer, out_dir, "", model_chords,
                reward_wrapper=reward_wrapper, reward_model=reward_model, device=device,
            )
            model_means_by_label[label] = model_means
            print(
                f"  models/{slug}/chord_durations.pt / note_durations.pt  "
                f"chord_entropy={model_means['chord_duration_entropy']:.4f} "
                f"note_entropy={model_means['note_duration_entropy']:.4f}"
            )
            print(f"  {slug} vendi_score={model_means['vendi_score']}")
            print(
                f"  models/{slug}/chord_silence_ratio.pt / melody_silence_ratio.pt  "
                f"chord={model_means['chord_silence_ratio_mean']:.4f} "
                f"melody={model_means['melody_silence_ratio_mean']:.4f}"
            )
            print(
                f"  models/{slug}/num_frames.pt  mean={model_means['num_frames_mean']:.1f} "
                f"total={model_means['num_frames_total']:.0f}"
            )
            print(f"  models/{slug}/sync_intervals.pt")
            print(f"  models/{slug}/chord_complexity.pt  mean={model_means['chord_complexity_mean']:.4f}")

    # Cross-source comparison metrics (model vs. GT), one small summary file
    means_summary = {
        "gt": {
            "chord_duration_entropy": _fmt_metric(gt_means["chord_duration_entropy"]),
            "note_duration_entropy": _fmt_metric(gt_means["note_duration_entropy"]),
            "chord_silence_ratio_mean": _fmt_metric(gt_means["chord_silence_ratio_mean"]),
            "melody_silence_ratio_mean": _fmt_metric(gt_means["melody_silence_ratio_mean"]),
            "num_frames_mean": _fmt_metric(gt_means["num_frames_mean"]),
            "num_frames_total": _fmt_metric(gt_means["num_frames_total"]),
            "note_in_chord_ratio_mean": _fmt_metric(gt_nicr_pooled),
            "note_in_chord_ratio_per_song_mean": _fmt_metric(_nanmean(gt_nicr_means)),
            "note_in_mode_ratio_mean": _fmt_metric(gt_mode_pooled),
            "note_in_mode_ratio_per_song_mean": _fmt_metric(_nanmean(gt_mode_seg_ratios)),
            "chord_complexity_mean": _fmt_metric(gt_means["chord_complexity_mean"]),
            "vendi_score": _fmt_metric(gt_means["vendi_score"]),
            # Always 0.0 by construction (GT compared against itself) -- included
            # explicitly as the literal parallel to the ReaLchords paper's Table 1
            # "Test set" row, which is the same trivial self-comparison.
            "sync_emd_vs_gt": 0.0,
        },
        "models": {},
    }
    for label, model_means in model_means_by_label.items():
        slug = _slugify(label)
        comparison = _compare_to_gt(model_means, gt_means)
        means_summary["models"][slug] = {
            "chord_duration_entropy": _fmt_metric(model_means["chord_duration_entropy"]),
            "note_duration_entropy": _fmt_metric(model_means["note_duration_entropy"]),
            "chord_silence_ratio_mean": _fmt_metric(model_means["chord_silence_ratio_mean"]),
            "melody_silence_ratio_mean": _fmt_metric(model_means["melody_silence_ratio_mean"]),
            "num_frames_mean": _fmt_metric(model_means["num_frames_mean"]),
            "num_frames_total": _fmt_metric(model_means["num_frames_total"]),
            "note_in_chord_ratio_mean": _fmt_metric(model_nicr_pooled[label]),
            "note_in_chord_ratio_per_song_mean": _fmt_metric(_nanmean(model_nicr_means[label])),
            "note_in_mode_ratio_mean": _fmt_metric(model_mode_pooled[label]),
            "note_in_mode_ratio_per_song_mean": _fmt_metric(
                _nanmean(model_mode_seg_ratios[label])
            ),
            "chord_complexity_mean": _fmt_metric(model_means["chord_complexity_mean"]),
            "vendi_score": _fmt_metric(model_means["vendi_score"]),
            **{key: _fmt_metric(value) for key, value in comparison.items()},
        }
    means_path = save_dir / "means.json"
    with means_path.open("w", encoding="utf-8") as fh:
        json.dump(means_summary, fh, indent=2, sort_keys=True)
    print(f"  means.json  ({means_path})")

    # Metadata (with per-sequence NiCR / mode-fit means)
    meta_path = save_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as fh:
        for entry in metadata:
            idx = entry["seq_idx"]
            entry["nicr"] = {
                "gt": _fmt_nicr(gt_nicr_means[idx].item()),
            }
            entry["mode_fit"] = {
                "gt": _fmt_nicr(gt_mode_means[idx].item()),
            }
            for label, means in model_nicr_means.items():
                slug = _slugify(label)
                entry["nicr"][slug] = _fmt_nicr(means[idx].item())
            for label, means in model_mode_means.items():
                slug = _slugify(label)
                entry["mode_fit"][slug] = _fmt_nicr(means[idx].item())
            fh.write(json.dumps(entry) + "\n")
    print(f"  metadata.jsonl  ({len(metadata)} rows, with per-sequence NiCR/mode-fit)")

    # Vocab snapshot so the .pt files can always be decoded correctly
    snapshot = save_vocab_snapshot(str(save_dir))
    print(f"  vocab snapshot → {snapshot}")

    # Label→slug mapping (and dataset_name) so scripts/eval/custom_eval/export_paired_midis.py
    # can find each model's {slug}.pt without re-running generation.
    model_labels_path = save_dir / "model_labels.json"
    with model_labels_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_name": args.dataset_name,
                "labels": {label: _slugify(label) for label, _ in model_specs},
            },
            fh,
            indent=2,
        )
    print(f"  model_labels.json  ({model_labels_path})")

    # MIDI preview -- a small random sample every run, not the whole split
    # (use scripts/eval/custom_eval/export_paired_midis.py separately for a
    # full/custom-size export without re-running generation).
    midi_indices = select_midi_indices(
        gt_tensor.size(0), midi_samples, default_all=False, seed=seed
    )
    if midi_indices:
        midi_dir = save_dir / "midi"
        if args.gt_only:
            write_source_midis(
                gt_tensor, dataset_tokenizer, midi_dir, "gt", metadata, midi_indices,
                strict_chords=True,
                include_chord_bass=resolve_include_chord_bass(False, False, args.dataset_name),
            )
        else:
            write_all_source_midis(
                gt_tensor=gt_tensor,
                model_tensors=model_tensors,
                ordered_labels=[label for label, _ in model_specs],
                metadata=metadata,
                gt_tokenizer=dataset_tokenizer,
                model_tokenizer=model_tokenizer,
                midi_dir=midi_dir,
                indices=midi_indices,
                include_chord_bass=resolve_include_chord_bass(False, False, args.dataset_name),
            )
        print(f"  midi/  ({len(midi_indices)} song(s) x {1 + len(model_tensors)} source(s))")

    print(f"\nDone with {run_label}.")


@bind(without_prefix=True)
def main(
    args,
    base_model: str = "",
    model: List[str] = [],
    gt_only: bool = False,
    dataset_name: str = "hooktheory",
    dataset_split: str = "test",
    save_dir: str = "",
    num_batches: int = -1,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "auto",
    contrastive_checkpoint: str = "",
    run_full_songs: bool = True,
    midi_samples: int = 10,
) -> None:
    """
    Args:
        base_model: Lightning checkpoint (.ckpt) for the MLE chord baseline.
            Required unless gt_only.
        model: Models to evaluate, as 'Label=base' (base MLE model),
            'Label=path.ckpt' (Lightning checkpoint), or 'Label=path.pth'
            (RL actor). Not used with gt_only.
        gt_only: Collect GT sequences and chord distributions only (no
            model loading).
        dataset_name: One of the keys in DATASET_CACHE_DIRS.
        dataset_split: One of train / valid / test / all.
        save_dir: Output root. In gt_only mode, cropped_songs/ and
            full_songs/ are written underneath it (see run_full_songs). In
            model-comparison mode, output goes straight into save_dir:
            gt/ (GT tensors/metrics), models/<slug>/ (one subfolder per
            --model entry), means.json, metadata.jsonl, and midi/ at the
            top level -- no cropped_songs/full_songs split, always one flat
            pass at the legacy 256-frame crop length.
        run_full_songs: gt_only only. Also run the full_songs (uncropped)
            variant, not just cropped_songs -- cheap here since gt_only
            does no generation. Ignored in model-comparison mode, which
            never runs a full_songs pass (see save_dir above).
        num_batches: Number of batches to process. -1 = all.
        contrastive_checkpoint: Direct path to a contrastive reward
            Lightning checkpoint (.ckpt), for the Vendi diversity score.
            Optional -- if omitted, Vendi is not computed at all
            (vendi_score: null in means.json) rather than computed against
            a checkpoint that might not match this run's chord vocab. Must
            actually be trained on the same dataset(s) as whatever's being
            evaluated -- a checkpoint trained on different/fewer datasets
            either fails a fast vocab-size check at startup or, if the
            vocab happens to match, would still be scoring chords/patterns
            it never saw in its own training. No fallback/remapping is
            attempted here on purpose; see
            realchords/utils/chord_diversity_analysis.py's
            load_contrastive_reward_direct.
        midi_samples: Auto-render this many randomly chosen songs (seeded by
            seed) to MIDI every run. In gt_only mode: one file per song
            under <save_dir>/<mode>/midi/gt/. In model-comparison mode: one
            *folder* per song under <save_dir>/midi/, each containing
            gt.mid plus one <slug>.mid per model -- every source's take on
            the same song side by side, for direct A/B listening. 0
            disables. -1 renders every song (slow for large splits --
            prefer running scripts/eval/custom_eval/export_paired_midis.py
            separately for that instead of setting this to -1).
    """
    _validate_args(
        base_model=base_model,
        model=model,
        gt_only=gt_only,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        save_dir=save_dir,
    )
    seed_everything(seed, workers=True)
    device_obj = _resolve_device(device)
    save_dir_path = Path(save_dir)

    model_specs: List[Tuple[str, str]] = []
    models: Dict[str, torch.nn.Module] = {}
    model_tokenizer = None
    model_types: Dict[str, str] = {}
    dataset_chord_names_path: Optional[str] = None
    if not gt_only:
        model_specs = [_parse_model_arg(m) for m in model]
        models, model_tokenizer, model_types = _load_models(
            base_model, model_specs, device_obj, batch_size
        )
        # Pin the eval dataset to the *loaded models'* own chord vocab
        # (baked into their checkpoints at training time) instead of
        # create_dataset_dataloaders' default auto-resolution, which picks
        # whatever the *current* global vocab is -- silently wrong once
        # that's grown past what these checkpoints were actually trained
        # on. See journal/SEQUENCE_EVALUATION.md.
        save_dir_path.mkdir(parents=True, exist_ok=True)
        dataset_chord_names_path = str(save_dir_path / "_model_chord_vocab.json")
        with open(dataset_chord_names_path, "w", encoding="utf-8") as fh:
            json.dump(model_tokenizer.chord_names, fh)
        eval_vocab_num_tokens = model_tokenizer.num_tokens
    else:
        cache_dir = DATASET_CACHE_DIRS[dataset_name.lower()]
        dataset_chord_names_path = resolve_chord_names_path_for_dataset(cache_dir)
        with open(dataset_chord_names_path, encoding="utf-8") as fh:
            eval_vocab_num_tokens = HooktheoryTokenizer(chord_names=json.load(fh)).num_tokens

    # Vendi score is optional: it needs a contrastive reward checkpoint
    # whose chord embedding table (fixed at whatever vocab it was trained
    # on) matches eval_vocab_num_tokens exactly, since raw chord token ids
    # get fed straight into it. No remapping/fallback on purpose -- scoring
    # with a mismatched checkpoint would be both technically broken
    # (out-of-range embedding lookup, CUDA device-side assert) and
    # conceptually meaningless (the reward model never saw that
    # vocabulary's chords during its own training -- e.g. this repo
    # currently has no contrastive/discriminative reward model actually
    # trained on the 7-dataset mix some decoder-only/enc-dec checkpoints
    # use). If no matching checkpoint is available, omit
    # --contrastive_checkpoint and Vendi is simply not computed
    # (vendi_score: null in means.json) rather than computed wrong.
    reward_wrapper = reward_model = None
    if contrastive_checkpoint:
        print(f"Loading contrastive reward ({contrastive_checkpoint}) for Vendi score …")
        reward_wrapper, reward_model, vendi_device = load_contrastive_reward_direct(
            contrastive_checkpoint, device_obj, eval_vocab_num_tokens
        )
        print(f"  → on {vendi_device}")
    else:
        print("No --contrastive_checkpoint given -- Vendi score will not be computed.")

    run_args = types.SimpleNamespace(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        batch_size=batch_size,
        num_batches=num_batches,
        gt_only=gt_only,
    )

    if gt_only:
        # gt_only is the only mode where the cropped_songs/full_songs split
        # still matters (cross-dataset chord-distribution analysis benefits
        # from both) -- each gets its own subfolder, as before.
        _run_eval(
            run_args,
            device=device_obj,
            models=models,
            model_tokenizer=model_tokenizer,
            model_specs=model_specs,
            model_types=model_types,
            max_len=256,  # legacy crop length: 8-bar melody + 8-bar chord
            save_dir=save_dir_path / "cropped_songs",
            run_label="cropped_songs",
            reward_wrapper=reward_wrapper,
            reward_model=reward_model,
            dataset_chord_names_path=dataset_chord_names_path,
            midi_samples=midi_samples,
            seed=seed,
        )
        if run_full_songs:
            # +1 bar of margin: num_beats can under-count by a frame or two
            # at the tail depending on quantization, and max_len must
            # strictly exceed the longest song for random_crop's guard to
            # skip cropping.
            max_frames = _max_song_frames(dataset_name, dataset_split)
            full_songs_max_len = 2 * (max_frames + FRAME_PER_BEAT * 4)
            print(
                f"full_songs: sizing dataloader for uncropped songs "
                f"(longest song ~{max_frames} frames/lane, max_len={full_songs_max_len})"
            )
            _run_eval(
                run_args,
                device=device_obj,
                models=models,
                model_tokenizer=model_tokenizer,
                model_specs=model_specs,
                model_types=model_types,
                max_len=full_songs_max_len,
                save_dir=save_dir_path / "full_songs",
                run_label="full_songs",
                reward_wrapper=reward_wrapper,
                reward_model=reward_model,
                dataset_chord_names_path=dataset_chord_names_path,
                midi_samples=midi_samples,
                seed=seed,
            )
        else:
            print("run_full_songs=False -- skipping full_songs, cropped_songs only.")
    else:
        # Model-comparison mode: always one flat pass at the legacy crop
        # length, straight into save_dir (no cropped_songs/full_songs
        # subfolder -- that split only exists for gt_only). See
        # gt/ + models/<slug>/ layout inside _run_eval.
        _run_eval(
            run_args,
            device=device_obj,
            models=models,
            model_tokenizer=model_tokenizer,
            model_specs=model_specs,
            model_types=model_types,
            max_len=256,  # legacy crop length: 8-bar melody + 8-bar chord
            save_dir=save_dir_path,
            run_label="eval",
            reward_wrapper=reward_wrapper,
            reward_model=reward_model,
            dataset_chord_names_path=dataset_chord_names_path,
            midi_samples=midi_samples,
            seed=seed,
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    args = argbind.parse_args(group=GROUP)
    if args.get("save_dir"):
        argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        main(args)
