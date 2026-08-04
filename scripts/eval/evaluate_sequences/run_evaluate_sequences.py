#!/usr/bin/env python3
"""Dispatcher for evaluating generated-sequence systems.

Replaces scripts/eval/evaluate_sequences_functions.sh and
scripts/eval/generated_systems.sh's registry for the eval side. System
names and their on-disk directories are resolved from configs/generate_sequences/*/*.yml
(the same presets scripts/generate_sequences.py uses) -- the eval side no
longer keeps a separate registry; add a generation preset once and it's
usable from both run_generate_sequences.py and this script.

Usage:
    python scripts/eval/evaluate_sequences/run_evaluate_sequences.py --list
    python scripts/eval/evaluate_sequences/run_evaluate_sequences.py gt
    python scripts/eval/evaluate_sequences/run_evaluate_sequences.py gt_vs_mle gt_vs_gapt
    python scripts/eval/evaluate_sequences/run_evaluate_sequences.py 'model_vs_model/*gapt_multiscale*'
    python scripts/eval/evaluate_sequences/run_evaluate_sequences.py gt/hooktheory gt_vs_gapt/hooktheory

A bare category name (gt, gt_vs_mle, gt_vs_realchords, gt_vs_gapt,
model_vs_model) expands to every preset in that folder, same as
run_generate_sequences.py. A glob pattern (contains *, ?, or [) matches
against every known preset name. Only presets that actually have generated
.pt output on disk are passed through to evaluate_generated_sequences.py --
others are silently skipped (use --list to see on-disk status per preset).

Selected systems are split into two evaluate_generated_sequences.py calls
(both writing into the same summary_path, so results merge): jazzmus/wjd get
--chord_names_path pointed at the current global chord vocab, everything
else uses configs/eval_sequences/generated_sequences/default.yml's own (older, reward-model-checkpoint-
matched) default. See the NEW_CHORD_NAMES_PATH comment below for why.
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

# _generate_presets.py lives in the sibling generate_sequences/ folder (shared
# by both dispatchers, so it isn't duplicated into each pipeline's own folder).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generate_sequences"))
from _generate_presets import (  # noqa: E402
    REPO_ROOT,
    discover_presets,
    expand,
    preset_dataset_name,
    preset_save_dir,
)

EVALUATE_SCRIPT = Path(__file__).resolve().parent / "evaluate_generated_sequences.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval_sequences" / "generated_sequences" / "default.yml"

# jazzmus/wjd's GT dumps were generated against the (then-current) *growing*
# global chord vocab, not the historical hooktheory/pop909/nottingham-only
# snapshot the reward-model checkpoint in configs/eval_sequences/generated_sequences/default.yml
# expects (data/cache/old_chord_names_augmented.json) -- using that file for
# these two datasets either crashes (jazzmus has chords outside the
# hooktheory/pop909/nottingham union) or silently misdecodes (wjd's chords
# are all present in both files, but not necessarily at the same indices,
# since the global file has been re-sorted since). Route these two through
# the current global file instead; every other dataset keeps the config's
# own default.
_NEW_VOCAB_DATASETS = {"jazzmus", "wjd"}
NEW_CHORD_NAMES_PATH = "data/cache/chord_names_augmented.json"


def has_outputs(directory: Path) -> bool:
    return directory.is_dir() and any(directory.rglob("*.pt"))


def resolve_glob(pattern: str, presets: dict) -> list:
    return sorted(n for n in presets if fnmatch.fnmatch(n, pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="Preset names, categories, or glob patterns.")
    parser.add_argument("--list", action="store_true", help="List every preset with its on-disk status and exit.")
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG),
        help="Base config for evaluate_generated_sequences.py (default: configs/eval_sequences/generated_sequences/default.yml).",
    )
    parser.add_argument("--analysis_root", type=str, default=None, help="Override the config's analysis_root.")
    parser.add_argument("--summary_path", type=str, default=None, help="Override the config's summary_path.")
    args = parser.parse_args()

    presets = discover_presets()

    if args.list or not args.names:
        for name in sorted(presets):
            try:
                save_dir = preset_save_dir(presets[name])
                status = "ready" if has_outputs(save_dir) else ("empty" if save_dir.is_dir() else "missing")
            except Exception as exc:  # noqa: BLE001 -- surfaced directly to the user
                status = f"error ({exc})"
            print(f"[{status:7}] {name}")
        sys.exit(0 if args.list else 2)

    selected = []
    for item in args.names:
        if any(ch in item for ch in "*?["):
            matches = resolve_glob(item, presets)
            if not matches:
                print(f"WARNING: pattern matched no presets: {item}")
            selected.extend(matches)
        else:
            try:
                selected.extend(expand([item], presets))
            except ValueError as exc:
                print(f"{exc} Run with --list to see valid names.")
                sys.exit(2)
    seen = set()
    selected = [n for n in selected if not (n in seen or seen.add(n))]

    # Group by which chord vocab each system needs (see NEW_CHORD_NAMES_PATH
    # comment above) -- each group becomes its own evaluate_generated_sequences.py
    # call, all writing into the same summary_path so the results merge (the
    # script already merges into any pre-existing summary's "systems" dict).
    groups = {"default": [], "new_vocab": []}
    skipped = []
    for name in selected:
        save_dir = preset_save_dir(presets[name])
        if not has_outputs(save_dir):
            skipped.append(name)
            continue
        group = "new_vocab" if preset_dataset_name(presets[name]) in _NEW_VOCAB_DATASETS else "default"
        groups[group].append(f"{name}={save_dir}")

    total = len(groups["default"]) + len(groups["new_vocab"])
    if total == 0:
        print("ERROR: none of the selected systems have generated .pt output on disk.")
        if skipped:
            print("Skipped (no outputs):", ", ".join(skipped))
        sys.exit(1)

    print(f"Evaluating {total} system(s) ({len(skipped)} skipped, no outputs)")

    exit_code = 0
    for group, system_specs in groups.items():
        if not system_specs:
            continue
        cmd = [
            sys.executable, str(EVALUATE_SCRIPT),
            "--args.load", args.config,
            "--system", " ".join(system_specs),
        ]
        if group == "new_vocab":
            cmd += ["--chord_names_path", NEW_CHORD_NAMES_PATH]
        if args.analysis_root:
            cmd += ["--analysis_root", args.analysis_root]
        if args.summary_path:
            cmd += ["--summary_path", args.summary_path]

        print(f"--- {group}: {len(system_specs)} system(s) ---")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        exit_code = exit_code or result.returncode

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
