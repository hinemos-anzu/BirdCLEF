"""Shared pipeline utilities for Codex experiment 001.

This is intentionally lightweight. Replace the dry-run summary with the actual
feature extraction, training, or inference code for a specific experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class Environment(str, Enum):
    KAGGLE = "kaggle"
    COLAB = "colab"


@dataclass(frozen=True)
class ExperimentPaths:
    input_root: Path
    output_dir: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict[str, Any], env: Environment) -> ExperimentPaths:
    paths = config["paths"]
    env_prefix = "kaggle" if env == Environment.KAGGLE else "colab"

    input_root = Path(
        os.environ.get(
            "BIRDCLEF_INPUT_ROOT",
            paths[f"{env_prefix}_input_root"],
        )
    )
    output_dir = Path(
        os.environ.get(
            "BIRDCLEF_OUTPUT_DIR",
            paths[f"{env_prefix}_output_dir"],
        )
    )
    return ExperimentPaths(input_root=input_root, output_dir=output_dir)


def summarize_dataset(input_root: Path, sample_files: int) -> pd.DataFrame:
    audio_dirs = [
        input_root / "train_soundscapes",
        input_root / "test_soundscapes",
    ]
    rows: list[dict[str, Any]] = []
    for audio_dir in audio_dirs:
        files = sorted(audio_dir.glob("*.ogg")) if audio_dir.exists() else []
        rows.append(
            {
                "directory": str(audio_dir),
                "exists": audio_dir.exists(),
                "ogg_count": len(files),
                "sample_files": ";".join(p.name for p in files[:sample_files]),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(env: Environment, config_path: Path) -> None:
    config = load_config(config_path)
    paths = resolve_paths(config, env)
    runtime = config.get("runtime", {})

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_dataset(
        paths.input_root,
        sample_files=int(runtime.get("sample_files", 5)),
    )

    summary_path = paths.output_dir / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"experiment_id={config['experiment_id']}")
    print(f"environment={env.value}")
    print(f"input_root={paths.input_root}")
    print(f"output_dir={paths.output_dir}")
    print(f"wrote={summary_path}")
    print(summary.to_string(index=False))
