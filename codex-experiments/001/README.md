# Codex Experiment 001

Purpose: keep Codex-generated BirdCLEF experiment code in a small, reproducible folder that can be run from Kaggle or Google Colab.

## Layout

```text
codex-experiments/001/
  README.md
  requirements.txt
  configs/experiment.yaml
  kaggle_runner.py
  colab_runner.py
  src/pipeline.py
  outputs/.gitkeep
```

## Kaggle

1. Add this GitHub repository as a Kaggle input or copy this folder into a Kaggle notebook.
2. Set `BIRDCLEF_INPUT_ROOT` if your dataset path differs from `/kaggle/input/competitions/birdclef-2026`.
3. Run:

```bash
python codex-experiments/001/kaggle_runner.py
```

The runner writes outputs under `/kaggle/working/codex_experiment_001` by default.

## Google Colab

1. Clone the repository.
2. Mount Google Drive if your input data is stored there.
3. Set `BIRDCLEF_INPUT_ROOT` to the dataset directory.
4. Run:

```bash
python codex-experiments/001/colab_runner.py
```

## Notes

Large datasets, model weights, Kaggle inputs, and generated artifacts should not be committed to GitHub. Keep only code, configuration, and lightweight documentation here.
