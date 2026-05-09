"""Kaggle entrypoint for Codex experiment 001."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import Environment, run_experiment


if __name__ == "__main__":
    run_experiment(Environment.KAGGLE, ROOT / "configs" / "experiment.yaml")
