#!/usr/bin/env python3
"""Dispatcher for the custom-eval config presets in configs/custom_eval/.

Replaces the old scripts/eval/custom_eval.sh (one hardcoded shell function
per dataset) -- presets are discovered from configs/custom_eval/*.yml (paired
model-vs-model configs) and configs/custom_eval/gt/*.yml (GT-only configs),
not from a hardcoded list. Add a new comparison by dropping a .yml in one of
those two locations -- no code change needed.

Usage:
    python scripts/eval/custom_eval/run_custom_eval.py <preset> [<preset> ...]
    python scripts/eval/custom_eval/run_custom_eval.py <path/to/config.yml>
    python scripts/eval/custom_eval/run_custom_eval.py --list

Presets:
    <name> for every configs/custom_eval/<name>.yml
        Paired model-vs-model comparison (e.g. paired_hooktheory, or any
        other top-level config such as realchords_vs_realchords_multiscale).

    gt_<dataset> for every configs/custom_eval/gt/<dataset>.yml, plus gt_all
        GT-only chord-distribution collection (no models). gt_all runs every
        discovered gt_<dataset> preset in sequence.

    midi_gt_<dataset>, plus midi_gt_all
        MIDI export for an already-completed gt_<dataset> run (separate
        step -- see scripts/eval/custom_eval/export_paired_midis.py), reusing
        the same gt/<dataset>.yml config as gt_<dataset>. midi_gt_all runs
        every discovered midi_gt_<dataset> preset in sequence.

    A path to an existing .yml file (registered or not) can also be passed
    directly and will be run through the paired-comparison script.

Example:
    python scripts/eval/custom_eval/run_custom_eval.py paired_hooktheory
    python scripts/eval/custom_eval/run_custom_eval.py gt_all
    python scripts/eval/custom_eval/run_custom_eval.py gt_hooktheory midi_gt_hooktheory
    python scripts/eval/custom_eval/run_custom_eval.py configs/custom_eval/realchords_vs_realchords_multiscale.yml
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "custom_eval"
GT_CONFIG_DIR = CONFIG_DIR / "gt"

_EVAL_SCRIPT = REPO_ROOT / "realchords" / "utils" / "custom_evaluation.py"
_MIDI_SCRIPT = REPO_ROOT / "scripts" / "eval" / "custom_eval" / "export_paired_midis.py"


def _discover_presets():
    """preset name -> (script, config_path), built from what's actually on disk."""
    presets = {}
    if CONFIG_DIR.is_dir():
        for path in sorted(CONFIG_DIR.glob("*.yml")):
            presets[path.stem] = (_EVAL_SCRIPT, path)
    if GT_CONFIG_DIR.is_dir():
        for path in sorted(GT_CONFIG_DIR.glob("*.yml")):
            presets[f"gt_{path.stem}"] = (_EVAL_SCRIPT, path)
            presets[f"midi_gt_{path.stem}"] = (_MIDI_SCRIPT, path)
    return presets


def _aggregates(presets):
    return {
        "gt_all": sorted(n for n in presets if n.startswith("gt_") and not n.startswith("gt_all")),
        "midi_gt_all": sorted(n for n in presets if n.startswith("midi_gt_") and not n.startswith("midi_gt_all")),
    }


def _expand(names, aggregates):
    expanded = []
    for name in names:
        expanded.extend(aggregates.get(name, [name]))
    return expanded


def _run(script: Path, config_path: Path, label: str) -> int:
    cmd = [sys.executable, str(script), "--args.load", str(config_path)]
    print(f"\n=== {label}: {' '.join(cmd)} ===")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def _run_name(name: str, presets: dict) -> int:
    if name in presets:
        script, config_path = presets[name]
        return _run(script, config_path, name)
    # Fall back to treating the argument as a raw config path (registered or not).
    path = Path(name)
    if path.is_file():
        return _run(_EVAL_SCRIPT, path, str(path))
    print(f"Unknown preset or config path '{name}'. Run with --list to see valid presets.")
    return 2


def main() -> None:
    args = sys.argv[1:]
    presets = _discover_presets()
    aggregates = _aggregates(presets)

    if not args or args == ["--list"]:
        print("Available presets:")
        for name in sorted(list(presets) + list(aggregates)):
            print(f"  {name}")
        sys.exit(0 if args else 2)

    names = _expand(args, aggregates)
    failures = []
    for name in names:
        code = _run_name(name, presets)
        if code != 0:
            failures.append((name, code))

    if failures:
        print("\nFailed presets:")
        for name, code in failures:
            print(f"  {name} (exit {code})")
        sys.exit(1)


if __name__ == "__main__":
    main()
