"""Temperature-scaled dataset sampling weights, derived from per-dataset frame counts.

weight_i ∝ effective_frames_i ** alpha, normalized to sum to 1.
alpha=1.0 -> proportional to dataset size (frame count)
alpha=0.0 -> uniform, regardless of size
alpha=0.5 -> sqrt scaling; a common middle ground (mT5/XLM-R-style temperature sampling)
"""

import json
import os
from typing import Dict, List

from realchords.constants import CACHE_DIR

DEFAULT_FRAME_COUNTS_PATH = os.path.join(CACHE_DIR, "dataset_frame_counts.json")


def load_effective_frame_counts(
    frame_counts_path: str = DEFAULT_FRAME_COUNTS_PATH,
) -> Dict[str, float]:
    with open(frame_counts_path) as f:
        data = json.load(f)
    return {name: info["effective_frames"] for name, info in data["datasets"].items()}


def compute_alpha_weights(
    datasets: List[str],
    alpha: float,
    frame_counts_path: str = DEFAULT_FRAME_COUNTS_PATH,
) -> List[float]:
    """Normalized sampling weights for datasets at the given alpha

    Each dataset's effective frame count is raised to the power of `alpha`, then
    the results are normalized to sum to 1. Raising to a fractional power shrinks
    the gap between large and small counts (since x**alpha grows sub-linearly for
    0 < alpha < 1), which is why alpha interpolates between "uniform" (alpha=0,
    every count collapses to 1) and "proportional to size" (alpha=1, counts are
    used as-is).

    alpha=0.5 -> sqrt temperature-scaled sampling weights, computed at runtime from
    data/cache/dataset_frame_counts.json via
    realchords.dataset.dataset_weights.compute_alpha_weights. Baseline for comparison
    is the sibling *.uniform.weights.yml config (alpha=0.0). alpha=1.0 would be pure
    proportional-to-size (hooktheory would dominate at ~66%, jazzmus ~2%); alpha=0.5
    is a middle ground (standard "temperature sampling", as in mT5/XLM-R) that still
    favors bigger datasets but keeps small ones from being drowned out.
    """
    frame_counts = load_effective_frame_counts(frame_counts_path)
    missing = [d for d in datasets if d not in frame_counts]
    if missing:
        raise ValueError(
            f"No frame count entry for datasets {missing} in {frame_counts_path}"
        )

    scaled = {d: frame_counts[d] ** alpha for d in datasets}
    total = sum(scaled.values())
    return [scaled[d] / total for d in datasets]
