"""Google Colab entrypoint for Codex experiment 001.

In Colab, mount Drive before running this file if your data lives there:

    from google.colab import drive
    drive.mount('/content/drive')
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import Environment, run_experiment


if __name__ == "__main__":
    run_experiment(Environment.COLAB, ROOT / "configs" / "experiment.yaml")
