#!/usr/bin/env python3
"""Unified harmony, note-in-mode, and diversity evaluation for generated sequence folders.

This script is the recommended public entry point for reproducing the Figure 4
evaluation workflow from user-generated checkpoints.

Usage: normally driven by scripts/eval/evaluate_sequences/run_evaluate_sequences.py, which
resolves system names/categories against configs/generate_sequences/*/*.yml (the same
config presets scripts/generate_sequences.py uses -- system name == preset
name, directory == that preset's save_dir) and calls this script with
--args.load configs/eval_sequences/generated_sequences/default.yml plus a resolved --system list. To
call this script directly:
    python scripts/eval/evaluate_sequences/evaluate_generated_sequences.py \
        --args.load configs/eval_sequences/generated_sequences/default.yml \
        --system "Online MLE=logs/generated/online_mle ReaLchords=logs/generated/realchords"

Or equivalently, without a config file:
    python scripts/eval/evaluate_sequences/evaluate_generated_sequences.py \
        --system "Online MLE=logs/generated/online_mle ReaLchords=logs/generated/realchords GAPT=logs/generated/gapt" \
        --analysis_root logs/figure4_eval \
        --summary_path logs/figure4_eval/summary.json \
        --config configs/single_agent_rl/realchords.yml

Inputs:
  Each `--system` value must be `LABEL=DIR`, where DIR contains `.pt` files
  produced by `scripts/generate_sequences.py`.

Outputs:
  - Per-file intermediate artifacts under:
      <analysis_root>/<system-slug>/penalties/...
      <analysis_root>/<system-slug>/diversity/...
  - A system-level summary JSON at `--summary_path` with this shape:
      {
        "systems": {
          "Label": {
            "input_dir": "...",
            "num_sequence_files": 3,
            "num_sequences": 1024,
            "overall_note_in_chord_ratio": 0.48,
            "overall_mode_fit_ratio": 0.72,
            "overall_vendi_score": 21.7,
            "sources": [{"path": "...", "num_sequences": 256}, ...]
          }
        }
      }
"""

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import argbind
import numpy as np
import torch

from realchords.utils.chord_diversity_analysis import (
    DEFAULT_CHORD_NAMES,
    DEFAULT_CONFIG,
    accumulate_chord_counts,
    analyze_diversity_file,
    compute_entropy,
    compute_vendi_score,
    load_contrastive_reward,
    resolve_device,
)
from realchords.utils.eval_utils import (
    SPECIAL_TOKENS,
    evaluate_melody_mode_fit_ratio,
    evaluate_note_in_chord_ratio,
)
from realchords.utils.experiment_utils_reward_analysis import (
    build_hooktheory_tokenizer,
)
from realchords.utils.sequence_penalty_analysis import (
    AnalysisConfig,
    MODEL_PART_CHOICES,
    SEQUENCE_ORDER_CHOICES,
    analyze_penalty_file,
    collect_sequence_files,
    load_sequences,
    load_tokenizer,
    strip_bos,
)
from realchords.utils.train_utils import AttrDict


@dataclass(frozen=True)
class SystemSpec:
    label: str
    directory: Path
    slug: str


# ---------------------------------------------------------------------------
# Per-sequence auxiliary stats
# ---------------------------------------------------------------------------

_PER_SEQ_FIELDNAMES = [
    "system",
    "source_file",
    "seq_idx",
    "note_in_chord_ratio",
    "valid_melody_frames",
    "correct_melody_frames",
    "mode_fit_ratio",
    "mode_fit_segments",
    "melody_silence_ratio",
    "melody_active_frames",
    "num_chord_changes",
    "num_unique_chords",
]


def _aux_stats_batch(
    sequences: torch.Tensor,
    tokenizer,
    sequence_order: str,
) -> Dict[str, List]:
    """Lightweight per-sequence stats derived purely from token names.

    Returns equal-length lists (one entry per sequence):
      num_chord_changes     — CHORD_ON event count
      num_unique_chords     — distinct chord symbols used
      melody_silence_ratio  — SILENCE / (SILENCE + NOTE) frames
      melody_active_frames  — frames carrying a NOTE token
    """
    if sequence_order == "chord_first":
        chord_slice, melody_slice = slice(0, None, 2), slice(1, None, 2)
    else:
        chord_slice, melody_slice = slice(1, None, 2), slice(0, None, 2)

    num_chord_changes: List[int] = []
    num_unique_chords: List[int] = []
    melody_silence_ratio: List[float] = []
    melody_active_frames: List[int] = []

    for seq in sequences:
        chord_tokens = seq[chord_slice].tolist()
        melody_tokens = seq[melody_slice].tolist()

        chord_on_count = 0
        unique_names: set = set()
        for tok in chord_tokens:
            name = tokenizer.id_to_name.get(tok, "")
            if name in SPECIAL_TOKENS:
                continue
            if tokenizer.is_chord_on(tok):
                chord_on_count += 1
                unique_names.add(name[len("CHORD_ON_"):])
        num_chord_changes.append(chord_on_count)
        num_unique_chords.append(len(unique_names))

        silence_f = note_f = 0
        for tok in melody_tokens:
            name = tokenizer.id_to_name.get(tok, "")
            if name in {"PAD", "BOS", "EOS"}:
                continue
            if name == "SILENCE":
                silence_f += 1
            elif name.startswith("NOTE"):
                note_f += 1
        total = silence_f + note_f
        melody_silence_ratio.append(silence_f / total if total > 0 else math.nan)
        melody_active_frames.append(note_f)

    return {
        "num_chord_changes": num_chord_changes,
        "num_unique_chords": num_unique_chords,
        "melody_silence_ratio": melody_silence_ratio,
        "melody_active_frames": melody_active_frames,
    }


def _pct(vals: List[float], p: int) -> str:
    if not vals:
        return "n/a"
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return f"{s[idx]:.3f}"


def _fmt_stats(vals: List[float]) -> str:
    if not vals:
        return "n/a"
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (
        f"mean={mu:.3f}  std={sd:.3f}  "
        f"p10={_pct(vals, 10)}  p50={_pct(vals, 50)}  p90={_pct(vals, 90)}  "
        f"[{min(vals):.3f} … {max(vals):.3f}]"
    )


def _print_per_seq_summary(rows: List[Dict]) -> None:
    by_system: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    def _floats(lst: List[Dict], key: str) -> List[float]:
        out = []
        for r in lst:
            v = r.get(key, "")
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                pass
        return out

    print("\n" + "=" * 72)
    print("Per-sequence distribution summary")
    print("=" * 72)
    for label, sys_rows in by_system.items():
        print(f"\n{label}  (n={len(sys_rows)})")
        print(f"  NiCR             {_fmt_stats(_floats(sys_rows, 'note_in_chord_ratio'))}")
        print(f"  Mode fit         {_fmt_stats(_floats(sys_rows, 'mode_fit_ratio'))}")
        print(f"  Melody silence   {_fmt_stats(_floats(sys_rows, 'melody_silence_ratio'))}")
        print(f"  Chord changes    {_fmt_stats(_floats(sys_rows, 'num_chord_changes'))}")
        print(f"  Unique chords    {_fmt_stats(_floats(sys_rows, 'num_unique_chords'))}")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "system"


def parse_system_arg(raw: str) -> SystemSpec:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --system value '{raw}'. Expected the form LABEL=DIR"
        )
    label, directory = raw.split("=", 1)
    label = label.strip()
    path = Path(directory).expanduser().resolve()
    if not label:
        raise ValueError("System label cannot be empty")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"System directory not found: {path}")
    return SystemSpec(label=label, directory=path, slug=slugify(label))


GROUP = __file__
bind = partial(argbind.bind, group=GROUP)


def _validate_choice_args(model_part: str, sequence_order: str) -> None:
    if model_part not in MODEL_PART_CHOICES:
        raise ValueError(
            f"Invalid model_part '{model_part}'. Expected one of: {MODEL_PART_CHOICES}"
        )
    if sequence_order not in SEQUENCE_ORDER_CHOICES:
        raise ValueError(
            f"Invalid sequence_order '{sequence_order}'. Expected one of: {SEQUENCE_ORDER_CHOICES}"
        )


def build_penalty_config(args, input_dir: Path, output_dir: Path) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        chord_names_path=args.chord_names_path,
        model_part=args.model_part,
        long_note_threshold=args.long_note_threshold,
        repetition_threshold=args.repetition_threshold,
        max_frames=args.max_frames,
        frames_per_beat=args.frames_per_beat,
        sequence_order=args.sequence_order,
    )


def accumulate_harmony_metrics(
    sequence_files: List[Path],
    tokenizer,
    model_part: str,
    sequence_order: str,
    system_label: str = "",
    per_sequence_rows: Optional[List[Dict]] = None,
) -> Dict[str, object]:
    total_sequences = 0
    total_valid_frames = 0
    total_correct_frames = 0
    weighted_mode_fit_sum = 0.0
    total_mode_fit_segments = 0
    sources = []

    for path in sequence_files:
        sequences = load_sequences(path)
        total_sequences += int(sequences.size(0))
        sources.append(
            {"path": str(path), "num_sequences": int(sequences.size(0))}
        )
        sequences = strip_bos(sequences, tokenizer)
        _, valid_counts, correct_counts = evaluate_note_in_chord_ratio(
            sequences,
            tokenizer,
            model_part=model_part,
            return_count=True,
            sequence_order=sequence_order,
        )
        mode_fit, _, segment_counts = evaluate_melody_mode_fit_ratio(
            sequences,
            tokenizer,
            model_part=model_part,
            return_count=True,
            sequence_order=sequence_order,
        )
        total_valid_frames += int(valid_counts.sum().item())
        total_correct_frames += int(correct_counts.sum().item())
        scorable = torch.isfinite(mode_fit) & (segment_counts > 0)
        if scorable.any():
            total_mode_fit_segments += int(segment_counts[scorable].sum().item())
            weighted_mode_fit_sum += float(
                (mode_fit[scorable] * segment_counts[scorable].float()).sum().item()
            )

        # ---- per-sequence rows (optional) ----
        if per_sequence_rows is not None:
            aux = _aux_stats_batch(sequences, tokenizer, sequence_order)
            vc = valid_counts.tolist()
            cc = correct_counts.tolist()
            mf = mode_fit.tolist()
            sc = segment_counts.tolist()
            for i in range(sequences.size(0)):
                has_melody = vc[i] > 0
                has_mode = sc[i] > 0 and math.isfinite(mf[i])
                per_sequence_rows.append({
                    "system": system_label,
                    "source_file": path.name,
                    "seq_idx": i,
                    "note_in_chord_ratio": f"{cc[i] / max(vc[i], 1):.4f}" if has_melody else "",
                    "valid_melody_frames": int(vc[i]),
                    "correct_melody_frames": int(cc[i]),
                    "mode_fit_ratio": f"{mf[i]:.4f}" if has_mode else "",
                    "mode_fit_segments": int(sc[i]),
                    "melody_silence_ratio": (
                        f"{aux['melody_silence_ratio'][i]:.4f}"
                        if math.isfinite(aux["melody_silence_ratio"][i])
                        else ""
                    ),
                    "melody_active_frames": aux["melody_active_frames"][i],
                    "num_chord_changes": aux["num_chord_changes"][i],
                    "num_unique_chords": aux["num_unique_chords"][i],
                })

    overall_note_in_chord_ratio = (
        float(total_correct_frames / total_valid_frames)
        if total_valid_frames
        else None
    )
    overall_mode_fit_ratio = (
        float(weighted_mode_fit_sum / total_mode_fit_segments)
        if total_mode_fit_segments
        else None
    )
    return {
        "num_sequences": total_sequences,
        "total_valid_frames": total_valid_frames,
        "total_correct_frames": total_correct_frames,
        "overall_note_in_chord_ratio": overall_note_in_chord_ratio,
        "total_mode_fit_segments": total_mode_fit_segments,
        "overall_mode_fit_ratio": overall_mode_fit_ratio,
        "sources": sources,
    }


def accumulate_diversity_metrics(
    sequence_files: List[Path],
    tokenizer,
    reward_wrapper,
    reward_model,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    chord_counter: Counter = Counter()
    embeddings: List[np.ndarray] = []
    total_sequences = 0
    total_embedded_sequences = 0
    pad = tokenizer.pad_token
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    with torch.no_grad():
        for path in sequence_files:
            sequences = load_sequences(path)
            total_sequences += int(sequences.size(0))
            for start in range(0, sequences.size(0), batch_size):
                batch = sequences[start : start + batch_size]
                if batch.size(1) == 0:
                    continue
                if torch.all(batch[:, 0] != bos):
                    bos_column = torch.full(
                        (batch.size(0), 1),
                        bos,
                        dtype=batch.dtype,
                    )
                    batch = torch.cat([bos_column, batch], dim=1)
                elif not torch.all(batch[:, 0] == bos):
                    raise ValueError(
                        f"Inconsistent BOS handling in {path}; either all rows should contain BOS or none should."
                    )

                model_tokens, _, model_mask, _ = (
                    reward_wrapper.get_inputs_from_sequence(batch)
                )
                accumulate_chord_counts(model_tokens, tokenizer, chord_counter)

                chord_mask = (
                    (model_tokens != pad)
                    & (model_tokens != bos)
                    & (model_tokens != eos)
                )
                non_empty = chord_mask.sum(dim=1) > 0
                if not non_empty.any():
                    continue

                tokens_to_embed = model_tokens[non_empty].to(device)
                mask_to_embed = model_mask[non_empty].to(device)
                embed_batch = reward_model.get_chord_embed(
                    chord=tokens_to_embed,
                    chord_mask=mask_to_embed,
                )
                embeddings.append(embed_batch.cpu().numpy())
                total_embedded_sequences += int(non_empty.sum().item())

    entropy, normalized_entropy, observed_norm = compute_entropy(
        chord_counter, tokenizer
    )
    vendi_score = compute_vendi_score(embeddings)
    return {
        "num_sequences": total_sequences,
        "num_sequences_with_embeddings": total_embedded_sequences,
        "total_chord_frames": int(sum(chord_counter.values())),
        "num_unique_chords": len(chord_counter),
        "entropy_nats": float(entropy),
        "normalized_entropy_all_chords": float(normalized_entropy),
        "normalized_entropy_observed": float(observed_norm),
        "overall_vendi_score": vendi_score,
    }


@bind(without_prefix=True)
def main(
    args,
    system: List[str] = [],
    analysis_root: str = "logs/generated_sequence_eval",
    summary_path: str = "logs/generated_sequence_eval/summary.json",
    config: str = str(DEFAULT_CONFIG),
    contrastive_index: int = 0,
    chord_names_path: str = str(DEFAULT_CHORD_NAMES),
    device: str = "auto",
    batch_size: int = 128,
    model_part: str = "chord",
    sequence_order: str = "chord_first",
    long_note_threshold: int = 32,
    repetition_threshold: int = 4,
    max_frames: int = 256,
    frames_per_beat: int = 4,
    skip_intermediate_artifacts: bool = False,
    per_sequence_csv: str = "",
) -> None:
    """
    Args:
        system: System specs, each 'LABEL=DIR'. At least one required.
        analysis_root: Directory where per-file intermediate analysis
            artifacts are written.
        summary_path: Path of the combined system-level summary JSON.
        config: RL config file listing contrastive reward checkpoints for
            Vendi embeddings.
        contrastive_index: Which contrastive reward checkpoint to use for
            Vendi embeddings.
        chord_names_path: Tokenizer chord-name mapping JSON.
        batch_size: Embedding batch size for Vendi score computation.
        model_part: chord or melody -- model part used for harmony analysis.
        sequence_order: chord_first or melody_first -- token ordering
            within a frame for harmony analysis.
        long_note_threshold: Frame threshold (inclusive) for long-note penalties.
        repetition_threshold: Consecutive identical event threshold for
            repetition penalty.
        max_frames: Maximum number of frames per sequence for per-beat padding.
        skip_intermediate_artifacts: If set, compute only the system
            summary and skip writing per-file artifacts.
        per_sequence_csv: If set, write a CSV with one row per sequence
            (NiCR, mode-fit ratio, melody silence ratio, chord-change
            count, unique-chord count) and print a per-system distribution
            summary (mean/std/percentiles) to stdout.
    """
    if not system:
        raise ValueError("At least one --system LABEL=DIR entry is required.")
    _validate_choice_args(model_part, sequence_order)

    args = AttrDict(
        system=system,
        analysis_root=Path(analysis_root),
        summary_path=Path(summary_path),
        config=Path(config),
        contrastive_index=contrastive_index,
        chord_names_path=Path(chord_names_path),
        device=device,
        batch_size=batch_size,
        model_part=model_part,
        sequence_order=sequence_order,
        long_note_threshold=long_note_threshold,
        repetition_threshold=repetition_threshold,
        max_frames=max_frames,
        frames_per_beat=frames_per_beat,
        skip_intermediate_artifacts=skip_intermediate_artifacts,
        per_sequence_csv=Path(per_sequence_csv) if per_sequence_csv else None,
    )

    systems = [parse_system_arg(item) for item in args.system]
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = build_hooktheory_tokenizer(args.chord_names_path)
    harmony_tokenizer = load_tokenizer(args.chord_names_path)
    device = resolve_device(args.device)
    reward_wrapper, reward_model, checkpoint, device = load_contrastive_reward(
        args.config,
        args.contrastive_index,
        tokenizer,
        device,
    )

    per_seq_rows: Optional[List[Dict]] = [] if args.per_sequence_csv else None

    existing_systems: Dict[str, object] = {}
    if args.summary_path.exists():
        with args.summary_path.open("r", encoding="utf-8") as _fh:
            _existing = json.load(_fh)
            existing_systems = _existing.get("systems", {})

    summary: Dict[str, object] = {
        "analysis_root": str(args.analysis_root.resolve()),
        "config": str(Path(args.config).resolve()),
        "contrastive_index": int(args.contrastive_index),
        "contrastive_checkpoint": checkpoint,
        "model_part": args.model_part,
        "sequence_order": args.sequence_order,
        "scoring": "strict",
        "systems": existing_systems,
    }

    for system in systems:
        print(f"Evaluating {system.label} from {system.directory}")
        sequence_files = collect_sequence_files(system.directory)
        penalties_dir = args.analysis_root / system.slug / "penalties"
        diversity_dir = args.analysis_root / system.slug / "diversity"

        if not args.skip_intermediate_artifacts:
            penalty_config = build_penalty_config(
                args, system.directory, penalties_dir
            )
            penalties_dir.mkdir(parents=True, exist_ok=True)
            diversity_dir.mkdir(parents=True, exist_ok=True)
            for path in sequence_files:
                analyze_penalty_file(path, penalty_config, harmony_tokenizer)
                analyze_diversity_file(
                    path=path,
                    output_dir=diversity_dir,
                    input_root=system.directory,
                    checkpoint=checkpoint,
                    checkpoint_index=args.contrastive_index,
                    tokenizer=tokenizer,
                    reward_wrapper=reward_wrapper,
                    reward_model=reward_model,
                    device=device,
                    batch_size=args.batch_size,
                )

        harmony_metrics = accumulate_harmony_metrics(
            sequence_files,
            tokenizer=harmony_tokenizer,
            model_part=args.model_part,
            sequence_order=args.sequence_order,
            system_label=system.label,
            per_sequence_rows=per_seq_rows,
        )
        diversity_metrics = accumulate_diversity_metrics(
            sequence_files,
            tokenizer=tokenizer,
            reward_wrapper=reward_wrapper,
            reward_model=reward_model,
            device=device,
            batch_size=args.batch_size,
        )

        summary["systems"][system.label] = {
            "input_dir": str(system.directory),
            "num_sequence_files": len(sequence_files),
            "num_sequences": harmony_metrics["num_sequences"],
            "num_sequences_with_embeddings": diversity_metrics[
                "num_sequences_with_embeddings"
            ],
            "total_valid_frames": harmony_metrics["total_valid_frames"],
            "total_correct_frames": harmony_metrics["total_correct_frames"],
            "overall_note_in_chord_ratio": harmony_metrics[
                "overall_note_in_chord_ratio"
            ],
            "overall_mode_fit_ratio": harmony_metrics["overall_mode_fit_ratio"],
            "total_mode_fit_segments": harmony_metrics["total_mode_fit_segments"],
            "overall_vendi_score": diversity_metrics["overall_vendi_score"],
            "entropy_nats": diversity_metrics["entropy_nats"],
            "normalized_entropy_all_chords": diversity_metrics[
                "normalized_entropy_all_chords"
            ],
            "normalized_entropy_observed": diversity_metrics[
                "normalized_entropy_observed"
            ],
            "total_chord_frames": diversity_metrics["total_chord_frames"],
            "num_unique_chords": diversity_metrics["num_unique_chords"],
            "sources": harmony_metrics["sources"],
            "analysis_dirs": {
                "penalties": str(penalties_dir),
                "diversity": str(diversity_dir),
            },
        }

    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote summary to {args.summary_path.resolve()}")

    if per_seq_rows is not None and args.per_sequence_csv is not None:
        args.per_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sequence_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_PER_SEQ_FIELDNAMES)
            writer.writeheader()
            writer.writerows(per_seq_rows)
        print(f"Wrote per-sequence CSV to {args.per_sequence_csv.resolve()}")
        _print_per_seq_summary(per_seq_rows)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parsed_args = argbind.parse_args(group=GROUP)
    with argbind.scope(parsed_args):
        main(parsed_args)
