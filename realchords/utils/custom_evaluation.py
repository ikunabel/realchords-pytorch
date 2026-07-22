#!/usr/bin/env python3
"""Paired chord evaluation: compare GT chords vs model-predicted chords for the same melodies.

Every song in the validation split is processed once. The GT melody is used as
conditioning for every specified model, producing predictions that share the same
melodic context and can be compared row-by-row.  Full provenance (song URL, dataset
index) is saved alongside the tensors.

Usage:
    python realchords/utils/custom_evaluation.py \
        --base_model logs/mle_chord/step=10000.ckpt \
        --model "MLE=base" \
        --model "RealJam=logs/realchords/actor.pth" \
        --model "GAPT=logs/gapt/actor.pth" \
        --dataset_name hooktheory \
        --dataset_split test \
        --save_dir logs/paired_eval/hooktheory \
        --num_batches 32 \
        --batch_size 64 \
        --seed 42

    python realchords/utils/custom_evaluation.py \
        --gt_only \
        --dataset_name hooktheory \
        --dataset_split test \
        --save_dir logs/paired_eval/gt/hooktheory

Every run produces two variants under --save_dir, each with its own full set
of outputs (gt.pt, {slug}.pt, midi/, etc.):
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
    {slug}_chord_durations.pt         same, per model
    {slug}_note_durations.pt          same, per model
    {slug}_chord_silence_ratio.pt     same, per model
    {slug}_melody_silence_ratio.pt    same, per model
    {slug}_num_frames.pt              same, per model
    {slug}_sync_intervals.pt          same, per model
    extra_metrics.json                summary: rhythm/silence means per source, plus
                                       model-vs-GT comparisons (sync EMD, chord-type JS
                                       distance) -- see realchords/utils/eval_utils.py
    chord_names_augmented.json        vocab snapshot for downstream decoding
    midi/                             GT-only: 10 random sanity-check MIDIs (see --midi_samples)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pretty_midi
import note_seq.chord_symbols_lib as _chord_lib
import types

import torch
from lightning import seed_everything
from tqdm import tqdm

from realchords.constants import FRAME_PER_BEAT
from realchords.dataset.hooktheory_tokenizer import to_midi_pitch
from realchords.lit_module.decoder_only import LitDecoder
from realchords.utils.eval_utils import (
    chord_type_distribution,
    chord_type_js_distance,
    duration_entropy,
    evaluate_chord_durations,
    evaluate_chord_silence_ratio,
    evaluate_chord_symbols_per_frame,
    evaluate_chord_to_note_onset_intervals,
    evaluate_melody_mode_fit_per_frame,
    evaluate_melody_silence_ratio,
    evaluate_note_durations,
    evaluate_note_in_chord_per_frame,
    evaluate_sequence_num_frames,
    synchronization_emd,
)
from realchords.utils.experiment_utils import (
    DATASET_CACHE_DIRS,
    _random_crop_on_chord_onset,
    create_dataset_dataloaders,
    replace_eos_with_pad,
    save_vocab_snapshot,
)
from realchords.utils.experiment_utils_model_data import generate_from_data
from realchords.utils.inference_utils import load_lit_model, load_rl_model
from realchords.utils.sequence_penalty_analysis import strip_bos


def _save_nicr_per_frame(
    tensor: torch.Tensor,
    tokenizer,
    path: Path,
) -> torch.Tensor:
    """Compute per-frame NiCR, save .pt, return per-sequence means [N]."""
    stripped = strip_bos(tensor, tokenizer)
    nicr = evaluate_note_in_chord_per_frame(stripped, tokenizer)
    torch.save(nicr, path)
    return nicr["mean"]


def _save_mode_per_frame(
    tensor: torch.Tensor,
    tokenizer,
    path: Path,
) -> torch.Tensor:
    """Compute per-frame note-in-mode, save .pt, return per-sequence means [N]."""
    stripped = strip_bos(tensor, tokenizer)
    mode_fit = evaluate_melody_mode_fit_per_frame(
        stripped,
        tokenizer,
    )
    torch.save(mode_fit, path)
    return mode_fit["mean"]


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


def _save_extra_metrics(
    tensor: torch.Tensor,
    tokenizer,
    save_dir: Path,
    prefix: str,
    chords: Dict[str, object],
) -> Dict[str, object]:
    """Rhythm/silence/synchronization metrics for one source (GT or a model).

    Saves per-sequence .pt files (durations, silence ratios, frame counts,
    synchronization intervals) and returns a summary dict -- both scalar
    means for printing, and the raw pooled distributions needed to compare
    this source against another one (synchronization_emd, chord_type_js_distance).
    """
    stripped = strip_bos(tensor, tokenizer)

    chord_durations = evaluate_chord_durations(stripped, tokenizer)
    torch.save(chord_durations, save_dir / f"{prefix}_chord_durations.pt")
    chord_duration_entropy = (
        duration_entropy(chord_durations["durations_flat"])
        if chord_durations["durations_flat"].numel() > 0
        else float("nan")
    )

    note_durations = evaluate_note_durations(stripped, tokenizer)
    torch.save(note_durations, save_dir / f"{prefix}_note_durations.pt")
    note_duration_entropy = (
        duration_entropy(note_durations["durations_flat"])
        if note_durations["durations_flat"].numel() > 0
        else float("nan")
    )

    chord_silence = evaluate_chord_silence_ratio(stripped, tokenizer)
    torch.save(chord_silence, save_dir / f"{prefix}_chord_silence_ratio.pt")

    melody_silence = evaluate_melody_silence_ratio(stripped, tokenizer)
    torch.save(melody_silence, save_dir / f"{prefix}_melody_silence_ratio.pt")

    num_frames = evaluate_sequence_num_frames(stripped, tokenizer)
    torch.save(num_frames, save_dir / f"{prefix}_num_frames.pt")

    sync = evaluate_chord_to_note_onset_intervals(stripped, tokenizer)
    torch.save(sync, save_dir / f"{prefix}_sync_intervals.pt")

    return {
        "chord_duration_entropy": chord_duration_entropy,
        "note_duration_entropy": note_duration_entropy,
        "chord_silence_ratio_mean": _nanmean(chord_silence),
        "melody_silence_ratio_mean": _nanmean(melody_silence),
        "num_frames_mean": _nanmean(num_frames),
        "sync_intervals_flat": sync["intervals_flat"],
        "chord_dist_onset": chord_type_distribution(chords, weighting="onset"),
        "chord_dist_frame": chord_type_distribution(chords, weighting="frame"),
    }


def _compare_to_gt(model_extra: Dict[str, object], gt_extra: Dict[str, object]) -> Dict[str, Optional[float]]:
    """Cross-source comparison metrics: model vs. GT (this run's reference)."""
    model_sync = model_extra["sync_intervals_flat"]
    gt_sync = gt_extra["sync_intervals_flat"]
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
            model_extra["chord_dist_onset"], gt_extra["chord_dist_onset"]
        ),
        "chord_type_js_distance_frame_vs_gt": _safe_js(
            model_extra["chord_dist_frame"], gt_extra["chord_dist_frame"]
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
# MIDI rendering constants and helpers
# ---------------------------------------------------------------------------

_CHORD_OCTAVE   = 4   # default chord-tone octave (MIDI 48–59)
_BASS_OCTAVE    = 3   # bass note one octave below chord root voicing
_MELODY_VEL     = 90
_CHORD_VEL      = 64


def _dedup(pitches: List[int]) -> List[int]:
    seen: set = set()
    out = []
    for p in pitches:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _naive_pitches(
    chord_name: str,
    *,
    include_bass: bool = True,
    chord_octave: int = _CHORD_OCTAVE,
) -> List[int]:
    chord_pcs = _chord_lib.chord_symbol_pitches(chord_name)
    pitches = [p % 12 + chord_octave * 12 for p in chord_pcs]
    if include_bass:
        bass_pc = _chord_lib.chord_symbol_bass(chord_name) % 12
        pitches.append(bass_pc + _BASS_OCTAVE * 12)
    return _dedup(pitches)


def _lenient_decode_chord_frames(chord_frames, tokenizer):
    """Decode chord frames leniently: only CHORD_ON tokens start a chord.

    Unlike ``tokenizer.decode_chord_frames``, this never raises for
    hold-only transitions or crops that start mid-chord.  Hold tokens
    that don't belong to the current ongoing chord are simply skipped,
    so the output starts cleanly from the first genuine chord onset.
    """
    fpb = tokenizer.frame_per_beat
    chords = []
    ongoing = None
    for i, tok in enumerate(chord_frames):
        name = tokenizer.id_to_name.get(int(tok), "")
        if "CHORD_ON_" in name:
            if ongoing is not None:
                ongoing["offset"] = i / fpb
                chords.append(ongoing)
            ongoing = {
                "chord_name": name[len("CHORD_ON_"):],
                "onset": i / fpb,
            }
        elif name == "SILENCE" and ongoing is not None:
            ongoing["offset"] = i / fpb
            chords.append(ongoing)
            ongoing = None
        # CHORD_HOLD tokens for a different chord, PAD, BOS, EOS → silently skip
    if ongoing is not None:
        ongoing["offset"] = len(chord_frames) / fpb
        chords.append(ongoing)
    return chords


def _decode_chord_anns(chord_frames, tokenizer, *, strict: bool):
    """Decode chord frames, using strict tokenizer decode when possible."""
    if strict:
        try:
            return tokenizer.decode_chord_frames(chord_frames)
        except ValueError:
            pass
    return _lenient_decode_chord_frames(chord_frames, tokenizer)


def _append_section(
    seq: torch.Tensor,          # 1-D, BOS already stripped
    tokenizer,
    spb: float,
    t0: float,
    label: str,
    melody_instr: pretty_midi.Instrument,
    chord_instr: pretty_midi.Instrument,
    midi_obj: pretty_midi.PrettyMIDI,
    *,
    strict_chords: bool = False,
    include_chord_bass: bool = True,
    chord_octave: int = _CHORD_OCTAVE,
    melody_octave: int = 0,
) -> float:
    """Render one section (melody + chords) into instruments.  Returns section duration (s)."""
    chord_frames = seq[0::2].numpy()
    melody_frames = seq[1::2].numpy()

    chord_anns = _decode_chord_anns(chord_frames, tokenizer, strict=strict_chords)
    try:
        melody_anns = tokenizer.decode_melody_frames(melody_frames)
    except ValueError:
        melody_anns = []

    for note in melody_anns:
        melody_instr.notes.append(pretty_midi.Note(
            velocity=_MELODY_VEL,
            pitch=to_midi_pitch(note["octave"] + melody_octave, note["pitch_class"]),
            start=note["onset"] * spb + t0,
            end=note["offset"] * spb + t0,
        ))

    for chord in chord_anns:
        for p in _naive_pitches(
            chord["chord_name"],
            include_bass=include_chord_bass,
            chord_octave=chord_octave,
        ):
            chord_instr.notes.append(pretty_midi.Note(
                velocity=_CHORD_VEL,
                pitch=max(0, min(127, p)),
                start=chord["onset"] * spb + t0,
                end=chord["offset"] * spb + t0,
            ))
        midi_obj.lyrics.append(pretty_midi.Lyric(
            text=chord["chord_name"],
            time=chord["onset"] * spb + t0,
        ))

    all_offsets = [c["offset"] for c in chord_anns] + [m["offset"] for m in melody_anns]
    return max(all_offsets) * spb if all_offsets else 0.0


def _select_midi_indices(
    num_sequences: int,
    midi_samples: Optional[int],
    *,
    gt_only: bool,
    seed: int,
) -> List[int]:
    """Pick which sequence indices to export as MIDI."""
    if num_sequences <= 0:
        return []
    if midi_samples is None:
        target = 10 if gt_only else num_sequences
    elif midi_samples < 0:
        target = num_sequences
    else:
        target = midi_samples
    target = min(target, num_sequences)
    if target >= num_sequences:
        return list(range(num_sequences))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_sequences), target))


def write_paired_midis(
    gt_tensor: torch.Tensor,
    model_tensors: Dict[str, torch.Tensor],
    ordered_labels: List[str],
    metadata: List[Dict],
    gt_tokenizer,
    model_tokenizer,
    midi_dir: Path,
    bpm: int = 120,
    pause_bars: float = 0.5,
    indices: Optional[List[int]] = None,
    include_chord_bass: bool = True,
    chord_octave: int = _CHORD_OCTAVE,
    melody_octave: int = 0,
) -> None:
    """Write one MIDI file per song.

    Layout:
        [GT melody + GT chords]
        [pause]
        [GT melody + Model-A chords]
        [pause]
        [GT melody + Model-B chords]  …

    The melody track is identical in every section (GT melody used as
    conditioning for all models).  Naive fixed-octave voicings are used
    throughout so the only difference between sections is the chord symbols.
    """
    midi_dir.mkdir(parents=True, exist_ok=True)
    spb       = 60.0 / bpm
    pause_sec = pause_bars * 4 * spb  # 4 beats per bar

    export_indices = indices if indices is not None else list(range(gt_tensor.size(0)))
    for out_idx, i in enumerate(tqdm(export_indices, desc="Writing MIDIs")):
        midi_obj     = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
        melody_instr = pretty_midi.Instrument(program=0, name="Melody")
        chord_instr  = pretty_midi.Instrument(program=0, name="Chords")

        t_cursor = 0.0

        # GT section: decode with the dataset vocab that encoded gt.pt
        gt_seq = gt_tensor[i, 1:]
        dur = _append_section(
            gt_seq, gt_tokenizer, spb, t_cursor, "GT",
            melody_instr, chord_instr, midi_obj, strict_chords=True,
            include_chord_bass=include_chord_bass,
            chord_octave=chord_octave,
            melody_octave=melody_octave,
        )
        t_cursor += dur + pause_sec

        # Model sections: decode with the model checkpoint vocab
        for label in ordered_labels:
            model_seq = model_tensors[label][i, 1:]
            dur = _append_section(
                model_seq, model_tokenizer, spb, t_cursor, label,
                melody_instr, chord_instr, midi_obj, strict_chords=False,
                include_chord_bass=include_chord_bass,
                chord_octave=chord_octave,
                melody_octave=melody_octave,
            )
            t_cursor += dur + pause_sec

        midi_obj.instruments.extend([melody_instr, chord_instr])

        song_url = metadata[i].get("song_url", "unknown")
        safe     = re.sub(r"[^a-zA-Z0-9_-]", "_", song_url)[-60:]
        midi_obj.write(str(midi_dir / f"{out_idx:04d}_seq{i:04d}_{safe}.mid"))

    print(f"  Wrote {len(export_indices)} MIDI files to {midi_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _slugify(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_") or "model"


def _parse_model_arg(raw: str) -> Tuple[str, str]:
    """Parse 'Label=path_or_keyword' → (label, path_or_keyword)."""
    if "=" not in raw:
        raise ValueError(f"Invalid --model value '{raw}'. Expected Label=path or Label=base")
    label, spec = raw.split("=", 1)
    return label.strip(), spec.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model",
        metavar="PATH",
        help="Lightning checkpoint (.ckpt) for the MLE chord baseline. "
             "Required unless --gt_only.",
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="Label=PATH",
        help=(
            "Model to evaluate. Use 'Label=base' for the base MLE model, "
            "'Label=path.ckpt' for a Lightning checkpoint, "
            "or 'Label=path.pth' for an RL actor. Repeat for multiple models. "
            "Not used with --gt_only."
        ),
    )
    parser.add_argument(
        "--gt_only",
        action="store_true",
        help="Collect GT sequences and chord distributions only (no model loading).",
    )
    parser.add_argument(
        "--dataset_name",
        default="hooktheory",
        choices=list(DATASET_CACHE_DIRS.keys()),
    )
    parser.add_argument("--dataset_split", default="test",
                        choices=["train", "valid", "test", "all"])
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--num_batches", type=int, default=-1,
                        help="Number of batches to process. -1 = all.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--midi_dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory for MIDI output. Defaults to <save_dir>/midi.",
    )
    parser.add_argument(
        "--midi_samples",
        type=int,
        default=None,
        metavar="N",
        help="Export N randomly chosen sequences as MIDI (seeded by --seed). "
             "Default: 10 for --gt_only, all sequences otherwise. Use -1 for all.",
    )
    parser.add_argument(
        "--bpm", type=int, default=120,
        help="Tempo for MIDI rendering (default: 120).",
    )
    parser.add_argument(
        "--pause_bars", type=float, default=0.5,
        help="Silence between sections in bars (default: 0.5).",
    )
    parser.add_argument(
        "--melody_octave",
        type=int,
        default=0,
        help="Offset added to each melody note's stored octave for MIDI export.",
    )
    parser.add_argument(
        "--chord_octave",
        type=int,
        default=_CHORD_OCTAVE,
        help="Octave for naive chord-tone voicings (default: 4).",
    )
    parser.add_argument(
        "--no_chord_bass",
        action="store_true",
        help="Omit the separate bass note from chord voicings (default for wjd).",
    )
    parser.add_argument(
        "--include_chord_bass",
        action="store_true",
        help="Force the separate chord bass note even for wjd.",
    )
    return parser.parse_args()


def _resolve_include_chord_bass(args: argparse.Namespace) -> bool:
    if args.include_chord_bass:
        return True
    if args.no_chord_bass:
        return False
    return args.dataset_name != "wjd"


def _validate_args(args: argparse.Namespace) -> None:
    if args.gt_only:
        if args.base_model or args.model:
            print("Note: --base_model / --model are ignored in --gt_only mode.")
        return
    if not args.base_model:
        raise SystemExit("--base_model is required unless --gt_only is set.")
    if not args.model:
        raise SystemExit("At least one --model is required unless --gt_only is set.")


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


def _load_models(
    base_ckpt: str,
    model_specs: List[Tuple[str, str]],   # [(label, path_or_keyword), ...]
    device: torch.device,
    batch_size: int,
) -> Tuple[Dict[str, torch.nn.Module], object]:
    """Load all requested models. Returns (label → model, tokenizer)."""
    print(f"Loading base model from {base_ckpt} …")
    base_model, tokenizer, _ = load_lit_model(
        model_path=base_ckpt,
        lit_module_cls=LitDecoder,
        batch_size=batch_size,
        compile=False,
    )
    base_model.eval().to(device)

    models: Dict[str, torch.nn.Module] = {}
    for label, spec in model_specs:
        slug = _slugify(label)
        if spec == "base":
            print(f"  {label}: using base MLE model")
            models[label] = base_model
        elif spec.endswith(".ckpt"):
            print(f"  {label}: loading Lightning checkpoint {spec}")
            m, _, _ = load_lit_model(
                model_path=spec,
                lit_module_cls=LitDecoder,
                batch_size=batch_size,
                compile=False,
            )
            m.eval().to(device)
            models[label] = m
        elif spec.endswith(".pth"):
            print(f"  {label}: loading RL actor {spec}")
            m = load_rl_model(
                model_path=spec,
                model=copy.deepcopy(base_model),
                compile=False,
            )
            m.eval().to(device)
            models[label] = m
        else:
            raise ValueError(
                f"Unrecognised model spec '{spec}' for '{label}'. "
                "Expected 'base', a '.ckpt' path, or a '.pth' path."
            )
        print(f"    → loaded '{label}' as '{slug}'")

    return models, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_eval(
    args: argparse.Namespace,
    *,
    device: torch.device,
    models: Dict[str, torch.nn.Module],
    model_tokenizer,
    model_specs: List[Tuple[str, str]],
    max_len: int,
    save_dir: Path,
    run_label: str,
) -> None:
    """Run one full eval pass (dataloader → generation → save) into save_dir.

    Called once per crop variant ("cropped_songs", "full_songs") from main().
    """
    # Chord dataloader — shuffle=False (already the default in create_dataset_dataloaders)
    _, val_loader = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="chord",
        batch_size=args.batch_size,
        max_len=max_len,
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

    # Accumulators
    gt_rows: List[torch.Tensor] = []
    model_rows: Dict[str, List[torch.Tensor]] = {label: [] for label, _ in model_specs}
    metadata: List[Dict] = []
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
                    preds = generate_from_data(
                        model=model,
                        sequences=gt_seq_stripped,
                        tokenizer=model_tokenizer,
                        prompt_steps=0,
                        target_seq_len=target_seq_len,
                    )
                    model_rows[label].append(preds.cpu())

            seq_idx += gt_seq.size(0)

    # ---- Save ---------------------------------------------------------------
    print(f"\nSaving {seq_idx} sequences to {save_dir} …")

    # GT
    gt_tensor = torch.cat(gt_rows, dim=0)
    torch.save(gt_tensor, save_dir / "gt.pt")
    print(f"  gt.pt  {tuple(gt_tensor.shape)}")

    # Per-model predictions
    if not args.gt_only:
        for label, rows in model_rows.items():
            slug = _slugify(label)
            tensor = torch.cat(rows, dim=0)
            torch.save(tensor, save_dir / f"{slug}.pt")
            print(f"  {slug}.pt  {tuple(tensor.shape)}")

    gt_nicr_means = _save_nicr_per_frame(
        gt_tensor, dataset_tokenizer, save_dir / "gt_nicr_per_frame.pt"
    )
    print(f"  gt_nicr_per_frame.pt  mean={gt_nicr_means.nanmean().item():.4f}")

    gt_mode_means = _save_mode_per_frame(
        gt_tensor,
        dataset_tokenizer,
        save_dir / "gt_mode_per_frame.pt",
    )
    print(f"  gt_mode_per_frame.pt  mean={gt_mode_means.nanmean().item():.4f}")

    gt_chords = _save_chords_per_frame(
        gt_tensor, dataset_tokenizer, save_dir / "gt_chords_per_frame.pt"
    )
    print("  gt_chords_per_frame.pt")

    dist_path = save_dir / "gt_chord_distribution.json"
    _save_chord_distribution(
        gt_chords,
        dist_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        num_sequences=gt_tensor.size(0),
    )
    print(f"  gt_chord_distribution.json  ({dist_path})")

    gt_extra = _save_extra_metrics(gt_tensor, dataset_tokenizer, save_dir, "gt", gt_chords)
    print(
        f"  gt_chord_durations.pt / gt_note_durations.pt  "
        f"chord_entropy={gt_extra['chord_duration_entropy']:.4f} "
        f"note_entropy={gt_extra['note_duration_entropy']:.4f}"
    )
    print(
        f"  gt_chord_silence_ratio.pt / gt_melody_silence_ratio.pt  "
        f"chord={gt_extra['chord_silence_ratio_mean']:.4f} "
        f"melody={gt_extra['melody_silence_ratio_mean']:.4f}"
    )
    print(f"  gt_num_frames.pt  mean={gt_extra['num_frames_mean']:.1f}")
    print("  gt_sync_intervals.pt")

    model_nicr_means: Dict[str, torch.Tensor] = {}
    model_mode_means: Dict[str, torch.Tensor] = {}
    model_extras: Dict[str, Dict[str, object]] = {}
    if not args.gt_only:
        for label, rows in model_rows.items():
            slug = _slugify(label)
            tensor = torch.cat(rows, dim=0)
            means = _save_nicr_per_frame(
                tensor, model_tokenizer, save_dir / f"{slug}_nicr_per_frame.pt"
            )
            model_nicr_means[label] = means
            print(
                f"  {slug}_nicr_per_frame.pt  mean={means.nanmean().item():.4f}"
            )
            mode_means = _save_mode_per_frame(
                tensor,
                model_tokenizer,
                save_dir / f"{slug}_mode_per_frame.pt",
            )
            model_mode_means[label] = mode_means
            print(
                f"  {slug}_mode_per_frame.pt  mean={mode_means.nanmean().item():.4f}"
            )
            model_chords = _save_chords_per_frame(
                tensor, model_tokenizer, save_dir / f"{slug}_chords_per_frame.pt"
            )
            print(f"  {slug}_chords_per_frame.pt")

            model_extra = _save_extra_metrics(
                tensor, model_tokenizer, save_dir, slug, model_chords
            )
            model_extras[label] = model_extra
            print(
                f"  {slug}_chord_durations.pt / {slug}_note_durations.pt  "
                f"chord_entropy={model_extra['chord_duration_entropy']:.4f} "
                f"note_entropy={model_extra['note_duration_entropy']:.4f}"
            )
            print(
                f"  {slug}_chord_silence_ratio.pt / {slug}_melody_silence_ratio.pt  "
                f"chord={model_extra['chord_silence_ratio_mean']:.4f} "
                f"melody={model_extra['melody_silence_ratio_mean']:.4f}"
            )
            print(f"  {slug}_num_frames.pt  mean={model_extra['num_frames_mean']:.1f}")
            print(f"  {slug}_sync_intervals.pt")

    # Cross-source comparison metrics (model vs. GT), one small summary file
    extra_metrics_summary = {
        "gt": {
            "chord_duration_entropy": _fmt_metric(gt_extra["chord_duration_entropy"]),
            "note_duration_entropy": _fmt_metric(gt_extra["note_duration_entropy"]),
            "chord_silence_ratio_mean": _fmt_metric(gt_extra["chord_silence_ratio_mean"]),
            "melody_silence_ratio_mean": _fmt_metric(gt_extra["melody_silence_ratio_mean"]),
            "num_frames_mean": _fmt_metric(gt_extra["num_frames_mean"]),
        },
        "models": {},
    }
    for label, model_extra in model_extras.items():
        slug = _slugify(label)
        comparison = _compare_to_gt(model_extra, gt_extra)
        extra_metrics_summary["models"][slug] = {
            "chord_duration_entropy": _fmt_metric(model_extra["chord_duration_entropy"]),
            "note_duration_entropy": _fmt_metric(model_extra["note_duration_entropy"]),
            "chord_silence_ratio_mean": _fmt_metric(model_extra["chord_silence_ratio_mean"]),
            "melody_silence_ratio_mean": _fmt_metric(model_extra["melody_silence_ratio_mean"]),
            "num_frames_mean": _fmt_metric(model_extra["num_frames_mean"]),
            **{key: _fmt_metric(value) for key, value in comparison.items()},
        }
    extra_metrics_path = save_dir / "extra_metrics.json"
    with extra_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(extra_metrics_summary, fh, indent=2, sort_keys=True)
    print(f"  extra_metrics.json  ({extra_metrics_path})")

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

    # ---- MIDI output --------------------------------------------------------
    # If the caller passed an explicit --midi_dir, nest it per run_label so
    # the cropped_songs/full_songs variants don't overwrite each other.
    midi_dir = (args.midi_dir / run_label) if args.midi_dir is not None else save_dir / "midi"
    ordered_labels = [label for label, _ in model_specs]
    model_tensors = {
        label: torch.cat(model_rows[label], dim=0) for label in ordered_labels
    } if not args.gt_only else {}
    midi_indices = _select_midi_indices(
        gt_tensor.size(0),
        args.midi_samples,
        gt_only=args.gt_only,
        seed=args.seed,
    )

    print(f"\nWriting MIDI files to {midi_dir} …")
    include_chord_bass = _resolve_include_chord_bass(args)
    write_paired_midis(
        gt_tensor=gt_tensor,
        model_tensors=model_tensors,
        ordered_labels=ordered_labels,
        metadata=metadata,
        gt_tokenizer=dataset_tokenizer,
        model_tokenizer=model_tokenizer or dataset_tokenizer,
        midi_dir=midi_dir,
        bpm=args.bpm,
        pause_bars=args.pause_bars,
        indices=midi_indices,
        include_chord_bass=include_chord_bass,
        chord_octave=args.chord_octave,
        melody_octave=args.melody_octave,
    )

    print(f"\nDone with {run_label}.")
    if not args.gt_only:
        print("Model labels saved:")
        for label, _ in model_specs:
            print(f"  {label}  →  {_slugify(label)}.pt")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    seed_everything(args.seed, workers=True)
    device = _resolve_device(args.device)

    model_specs: List[Tuple[str, str]] = []
    models: Dict[str, torch.nn.Module] = {}
    model_tokenizer = None
    if not args.gt_only:
        model_specs = [_parse_model_arg(m) for m in args.model]
        models, model_tokenizer = _load_models(
            args.base_model, model_specs, device, args.batch_size
        )

    # +1 bar of margin: num_beats can under-count by a frame or two at the
    # tail depending on quantization, and max_len must strictly exceed the
    # longest song for random_crop's guard to skip cropping.
    max_frames = _max_song_frames(args.dataset_name, args.dataset_split)
    full_songs_max_len = 2 * (max_frames + FRAME_PER_BEAT * 4)
    print(
        f"full_songs: sizing dataloader for uncropped songs "
        f"(longest song ~{max_frames} frames/lane, max_len={full_songs_max_len})"
    )

    _run_eval(
        args,
        device=device,
        models=models,
        model_tokenizer=model_tokenizer,
        model_specs=model_specs,
        max_len=256,  # legacy crop length: 8-bar melody + 8-bar chord
        save_dir=args.save_dir / "cropped_songs",
        run_label="cropped_songs",
    )
    _run_eval(
        args,
        device=device,
        models=models,
        model_tokenizer=model_tokenizer,
        model_specs=model_specs,
        max_len=full_songs_max_len,
        save_dir=args.save_dir / "full_songs",
        run_label="full_songs",
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
