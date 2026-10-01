"""Write <experiment dir>/summary.json (metric -> model -> condition) for a custom-eval experiment.

Normally not needed by hand: custom_evaluation.py and run_multi_model_eval.py call this
automatically at the end of every run. See realchords/utils/experiment_summary.py for the format.

Usage:
    python scripts/eval/custom_eval/summarize_experiment.py logs/custom_eval/dataset_loo
    python scripts/eval/custom_eval/summarize_experiment.py logs/custom_eval/gt --include '*/full_songs'
"""

from __future__ import annotations

import argparse

from realchords.utils.experiment_summary import write_experiment_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment_dir")
    ap.add_argument("--include", nargs="*", default=None,
                    help="fnmatch patterns on the condition name (run path relative to "
                         "experiment_dir), e.g. '*/full_songs' to skip the cropped-song runs")
    args = ap.parse_args()
    out = write_experiment_summary(args.experiment_dir, include=args.include)
    if out is None:
        raise SystemExit("no means.json found")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
