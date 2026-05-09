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
  cv_blend_gate_runner.py
  blend_cv_gate_cell.py
  v18_cv_gate_kaggle.ipynb
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

## Kaggle Notebook

Use `v18_cv_gate_kaggle.ipynb` when you want the full guided workflow:

1. Clone `codex/experiment-001`.
2. Force V18 into `MODE = "train"` in the Kaggle working copy.
3. Run the V18 script with `%run` so notebook variables remain available.
4. Save `sed_preds_tr_aligned.npy`, `oof_proto_probs.npy`, and `perch_meta.parquet`.
5. Run the CV blend gate for `w_proto=0.65` vs `w_proto=0.60`.

Notebook path:

```text
codex-experiments/001/v18_cv_gate_kaggle.ipynb
```

## CV Blend Gate

Use this to decide whether `w_proto=0.65` is worth spending a limited LB submission.

First run the V18 notebook/script far enough to produce these artifacts:

```text
/kaggle/working/cache/oof_proto_probs.npy
/kaggle/working/cache/perch_meta.parquet
```

The SED train predictions may only exist as an in-memory notebook variable. If so, run this one line after Cell 11/12:

```python
np.save('/kaggle/working/cache/sed_preds_tr_aligned.npy', sed_preds_tr_aligned)
```

Then run:

```bash
python BirdCLEF/codex-experiments/001/cv_blend_gate_runner.py \
  --sed-oof /kaggle/working/cache/sed_preds_tr_aligned.npy \
  --candidate-w 0.65 \
  --baseline-w 0.60
```

Outputs:

```text
/kaggle/working/codex_experiment_001/blend_cv_gate_table.csv
/kaggle/working/codex_experiment_001/blend_cv_gate_decision.json
```

The conservative submit gate is:

```text
candidate_lb_proxy_gain >= +0.0010
bootstrap_p_win >= 0.65
```

If `should_submit` is `false`, keep `w_proto=0.60` or move to the next improvement candidate instead of spending LB.

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
