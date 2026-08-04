#!/usr/bin/env python3
"""Dispatcher for the generate-sequences config presets in configs/generate_sequences/.

Replaces scripts/eval/generate_sequences_functions.sh (one hardcoded shell
function per generation) and scripts/eval/generated_systems.sh's
registry-of-function-names for the generate side. Presets are discovered
directly from configs/generate_sequences/**/*.yml -- add a new generation by adding a
config file, not a new shell function.

Usage:
    python scripts/eval/generate_sequences/run_generate_sequences.py --list
    python scripts/eval/generate_sequences/run_generate_sequences.py gt/hooktheory
    python scripts/eval/generate_sequences/run_generate_sequences.py gt_vs_mle model_vs_model/gapt_melody_vs_gapt_chord_free_generation
    python scripts/eval/generate_sequences/run_generate_sequences.py model_vs_model --max_parallel 4 --log_dir scripts/jobscripts/slurm_logs/generate_sequences

A bare category name (gt, gt_vs_mle, gt_vs_realchords, gt_vs_gapt,
model_vs_model -- i.e. any subfolder of configs/generate_sequences/) expands to every
preset in that folder. Mix category names and individual preset names freely.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _generate_presets import REPO_ROOT, discover_presets, expand  # noqa: E402

GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_sequences.py"


def run_one(name: str, config_path: Path, log_dir: str):
    cmd = [sys.executable, str(GENERATE_SCRIPT), "--args.load", str(config_path)]
    print(f"=== starting: {name} ===")
    if log_dir:
        log_path = Path(log_dir) / f"{name.replace('/', '__')}.out"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        proc._log_handle = handle  # closed once the process finishes, see main()
    else:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names", nargs="*", help="Preset names (category/name) or bare category names."
    )
    parser.add_argument("--list", action="store_true", help="List every discovered preset and exit.")
    parser.add_argument(
        "--max_parallel", type=int, default=1,
        help="Max concurrent generations on one GPU (default: 1, sequential).",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Write each preset's stdout/stderr to <log_dir>/<name>.out instead of the console "
             "(useful with --max_parallel > 1, so concurrent output doesn't interleave).",
    )
    args = parser.parse_args()

    presets = discover_presets()

    if args.list or not args.names:
        by_category: dict = {}
        for name in presets:
            by_category.setdefault(name.split("/")[0], []).append(name)
        for category in sorted(by_category):
            print(f"{category}/ ({len(by_category[category])})")
            for name in sorted(by_category[category]):
                print(f"  {name}")
        sys.exit(0 if args.list else 2)

    try:
        to_run = expand(args.names, presets)
    except ValueError as exc:
        print(f"{exc} Run with --list to see valid names.")
        sys.exit(2)
    print(f"=== {len(to_run)} generation(s), max {args.max_parallel} parallel ===")

    running = []
    failures = []
    queue = list(to_run)
    while queue or running:
        while queue and len(running) < args.max_parallel:
            name = queue.pop(0)
            proc = run_one(name, presets[name], args.log_dir)
            running.append((name, proc))
        time.sleep(2)
        still_running = []
        for name, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((name, proc))
                continue
            if getattr(proc, "_log_handle", None):
                proc._log_handle.close()
            if code != 0:
                failures.append((name, code))
                print(f"=== finished (FAILED, exit {code}): {name} ===")
            else:
                print(f"=== finished: {name} ===")
        running = still_running

    if failures:
        print("\nFailed presets:")
        for name, code in failures:
            print(f"  {name} (exit {code})")
        sys.exit(1)
    print("=== all generations finished ===")


if __name__ == "__main__":
    main()
