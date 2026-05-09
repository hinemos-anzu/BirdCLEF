"""Patch V18 script with A2 classwise ResidualSSM correction weights.

This patcher edits the Kaggle working copy of
birdclef_v18_full_pipeline_oof_codex_ready_fixed.py. It is intentionally
idempotent and keeps the original scalar correction path as a fallback.

Usage on Kaggle:
  python BirdCLEF/codex-experiments/001/a2_classwise_correction_patch.py \
    --script /kaggle/working/BirdCLEF/birdclef_v18_full_pipeline_oof_codex_ready_fixed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "A2_CLASSWISE_CORRECTION_WEIGHTS"

HELPER_CODE = r'''
# ── A2: Classwise ResidualSSM correction weights ─────────────────────────────
def learn_classwise_correction_weights(
    first_pass_flat,
    correction_flat,
    Y_full,
    base_weight=0.30,
    grid=None,
    min_pos=3,
    shrink=0.50,
    verbose=True,
):
    """
    Learn one ResidualSSM correction weight per class on the training fold.

    The scalar correction_weight is kept as the prior. For classes with enough
    positives and negatives, choose the best weight on a small AUC grid, then
    shrink toward base_weight to reduce fold overfit. Classes without enough
    support keep base_weight.
    """
    if grid is None:
        grid = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=np.float32)
    else:
        grid = np.asarray(grid, dtype=np.float32)

    weights = np.full(N_CLASSES, float(base_weight), dtype=np.float32)
    selected = 0
    improved = 0

    for c in range(N_CLASSES):
        y = Y_full[:, c]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos < min_pos or neg < min_pos:
            continue

        base_scores = first_pass_flat[:, c]
        corr_scores = correction_flat[:, c]
        base_auc = roc_auc_score(y, base_scores + float(base_weight) * corr_scores)
        best_w = float(base_weight)
        best_auc = float(base_auc)

        for w in grid:
            auc = roc_auc_score(y, base_scores + float(w) * corr_scores)
            if auc > best_auc + 1e-6:
                best_auc = float(auc)
                best_w = float(w)

        weights[c] = float(base_weight + shrink * (best_w - base_weight))
        selected += 1
        if abs(best_w - base_weight) > 1e-6:
            improved += 1

    weights = np.clip(weights, 0.0, 0.60).astype(np.float32)
    if verbose:
        print(
            f"A2 classwise correction weights: tuned={selected} classes, "
            f"moved={improved}, mean={weights.mean():.3f}, "
            f"range=[{weights.min():.2f}, {weights.max():.2f}]"
        )
    return weights


def apply_classwise_correction(first_pass_flat, correction_flat, correction_weights):
    """Apply classwise ResidualSSM correction weights with broadcasting."""
    cw = np.asarray(correction_weights, dtype=np.float32)
    if cw.ndim == 0:
        return first_pass_flat + float(cw) * correction_flat
    return first_pass_flat + cw[None, :] * correction_flat
# ── /A2_CLASSWISE_CORRECTION_WEIGHTS ─────────────────────────────────────────
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def patch_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"A2 patch already present: {path}")
        return

    insert_after = 'print("✅ ResidualSSM defined (~439K params, ~20s training)")'
    text = replace_once(text, insert_after, insert_after + "\n" + HELPER_CODE, "ResidualSSM marker")

    # Full-train/test path: train_residual_ssm currently returns scalar correction_weight.
    old_full = '''res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr,
    first_pass_flat=first_pass,
    Y_full=Y_FULL_aligned,
    site_ids=site_ids,
    hour_ids=hour_ids,
    n_epochs=CFG["residual_ssm"]["n_epochs"],
    patience=CFG["residual_ssm"]["patience"],
    lr=CFG["residual_ssm"]["lr"],
    correction_weight=CFG["residual_ssm"]["correction_weight"],
    verbose=CFG["verbose"],
)'''
    new_full = '''res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr,
    first_pass_flat=first_pass,
    Y_full=Y_FULL_aligned,
    site_ids=site_ids,
    hour_ids=hour_ids,
    n_epochs=CFG["residual_ssm"]["n_epochs"],
    patience=CFG["residual_ssm"]["patience"],
    lr=CFG["residual_ssm"]["lr"],
    correction_weight=CFG["residual_ssm"]["correction_weight"],
    verbose=CFG["verbose"],
)

res_model.eval()
with torch.no_grad():
    train_correction = res_model(
        torch.tensor(emb_tr.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32),
        torch.tensor(first_pass.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32),
        site_ids=torch.tensor(site_ids, dtype=torch.long),
        hours=torch.tensor(hour_ids, dtype=torch.long),
    ).numpy().reshape(-1, N_CLASSES)

correction_weight = learn_classwise_correction_weights(
    first_pass_flat=first_pass,
    correction_flat=train_correction,
    Y_full=Y_FULL_aligned,
    base_weight=correction_weight,
    min_pos=3,
    shrink=0.50,
    verbose=True,
)'''
    text = replace_once(text, old_full, new_full, "full-train ResidualSSM block")

    old_apply_full = '''final_scores = first_pass + correction_weight * correction'''
    new_apply_full = '''final_scores = apply_classwise_correction(first_pass, correction, correction_weight)'''
    text = replace_once(text, old_apply_full, new_apply_full, "full correction application")

    old_oof_train = '''res_model, correction_weight = train_residual_ssm(
            emb_full=emb_tr_f,
            first_pass_flat=first_pass_tr_f,
            Y_full=Y_tr_f,
            site_ids=tr_site_ids_f,
            hour_ids=tr_hour_ids_f,
            n_epochs=res_n_epochs, patience=res_patience,
            lr=1e-3, correction_weight=0.30, verbose=False,
        )'''
    new_oof_train = '''res_model, correction_weight = train_residual_ssm(
            emb_full=emb_tr_f,
            first_pass_flat=first_pass_tr_f,
            Y_full=Y_tr_f,
            site_ids=tr_site_ids_f,
            hour_ids=tr_hour_ids_f,
            n_epochs=res_n_epochs, patience=res_patience,
            lr=1e-3, correction_weight=0.30, verbose=False,
        )

        res_model.eval()
        with torch.no_grad():
            tr_correction_f = res_model(
                torch.tensor(emb_tr_f.reshape(n_tr, N_WINDOWS, -1), dtype=torch.float32),
                torch.tensor(first_pass_tr_f.reshape(n_tr, N_WINDOWS, -1), dtype=torch.float32),
                site_ids=torch.tensor(tr_site_ids_f, dtype=torch.long),
                hours=torch.tensor(tr_hour_ids_f, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        correction_weight = learn_classwise_correction_weights(
            first_pass_flat=first_pass_tr_f,
            correction_flat=tr_correction_f,
            Y_full=Y_tr_f,
            base_weight=correction_weight,
            min_pos=3,
            shrink=0.50,
            verbose=False,
        )'''
    text = replace_once(text, old_oof_train, new_oof_train, "OOF ResidualSSM block")

    old_oof_apply = '''final_va = first_pass_va + correction_weight * va_correction'''
    new_oof_apply = '''final_va = apply_classwise_correction(first_pass_va, va_correction, correction_weight)'''
    text = replace_once(text, old_oof_apply, new_oof_apply, "OOF correction application")

    path.write_text(text, encoding="utf-8")
    print(f"A2 patch applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    patch_script(args.script)


if __name__ == "__main__":
    main()
