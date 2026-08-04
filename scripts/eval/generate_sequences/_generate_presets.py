"""Shared preset discovery over configs/generate_sequences/*/*.yml.

Used by both scripts/eval/generate_sequences/run_generate_sequences.py (to run generation
presets) and scripts/eval/evaluate_sequences/run_evaluate_sequences.py (to resolve system names
into on-disk directories for evaluation) -- one registry, not two, so a new
preset only needs to be added in one place (a config file) to be usable from
both commands.
"""

import argbind

from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_ROOT = REPO_ROOT / "configs" / "generate_sequences"


def discover_presets() -> Dict[str, Path]:
    """preset name ("category/name") -> config path, for every configs/generate_sequences/*/*.yml."""
    presets = {}
    for path in sorted(CONFIG_ROOT.glob("*/*.yml")):
        name = f"{path.parent.name}/{path.stem}"
        presets[name] = path
    return presets


def preset_save_dir(config_path: Path) -> Path:
    """The save_dir a given generate preset writes its output to."""
    data = argbind.load_args(config_path)
    save_dir = data.get("save_dir")
    if not save_dir:
        raise ValueError(f"{config_path} has no save_dir")
    return (REPO_ROOT / save_dir).resolve()


def preset_dataset_name(config_path: Path) -> str:
    """The dataset_name a given generate preset conditions on (empty for
    model-vs-model presets that don't set one)."""
    data = argbind.load_args(config_path)
    return data.get("dataset_name", "") or ""


def expand(requested: List[str], presets: Dict[str, Path]) -> List[str]:
    """Resolve a mix of exact preset names and bare category names (folder
    names under configs/generate_sequences/) into an ordered, de-duplicated list of
    exact preset names."""
    categories = sorted({name.split("/")[0] for name in presets})
    expanded = []
    for item in requested:
        if item in categories:
            expanded.extend(n for n in presets if n.startswith(f"{item}/"))
        elif item in presets:
            expanded.append(item)
        else:
            raise ValueError(
                f"Unknown preset or category '{item}'. Known categories: {categories}"
            )
    seen = set()
    result = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result
