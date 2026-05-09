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
  a2_classwise_correction_patch.py
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

## A2 Classwise Correction

A2 replaces the scalar ResidualSSM `correction_weight` with classwise correction weights learned on the training fold. It keeps the original scalar value as a prior and shrinks per-class weights toward it to reduce overfit.

In Kaggle, run this after clone and after forcing `MODE = "train"`, but before `%run` executes the V18 script:

```bash
python /kaggle/working/BirdCLEF/codex-experiments/001/a2_classwise_correction_patch.py \
  --script /kaggle/working/BirdCLEF/birdclef_v18_full_pipeline_oof_codex_ready_fixed.py
```

Expected output:

```text
A2 patch applied: /kaggle/working/BirdCLEF/birdclef_v18_full_pipeline_oof_codex_ready_fixed.py
```

Then run V18 normally:

```python
%run /kaggle/working/BirdCLEF/birdclef_v18_full_pipeline_oof_codex_ready_fixed.py
```

The log should include:

```text
A2 classwise correction weights: tuned=... classes, moved=..., mean=..., range=[..., ...]
```

The patcher is idempotent. If it has already been applied, it prints `A2 patch already present` and leaves the script unchanged.

## Kaggle Notebook

Use `v18_cv_gate_kaggle.ipynb` when you want the full guided workflow:

1. Clone `codex/experiment-001`.
2. Force V18 into `MODE = "train"` in the Kaggle working copy.
3. Optionally apply A2 with `a2_classwise_correction_patch.py` before `%run`.
4. Run the V18 script with `%run` so notebook variables remain available.
5. Save `sed_preds_tr_aligned.npy`, `oof_proto_probs.npy`, and `perch_meta.parquet`.
6. Run the CV blend gate for `w_proto=0.65` vs `w_proto=0.60`.

Notebook path:

```text
codex-experiments/001/v18_cv_gate_kaggle.ipynb
```

## CV Blend Gate

Use this to decide whether a limited LB submission is worth spending.

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

If `should_submit` is `false`, keep the baseline or move to the next improvement candidate instead of spending LB.

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
