"""Patch V18 script with A2 classwise ResidualSSM correction weights.

This edits only the Kaggle working copy of
birdclef_v18_full_pipeline_oof_codex_ready_fixed.py. The patch is idempotent
and uses regex anchors around ResidualSSM blocks so it tolerates formatting
changes in the V18 script.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = "A2_CLASSWISE_CORRECTION_WEIGHTS"

HELPER_CODE = r'''
# A2_CLASSWISE_CORRECTION_WEIGHTS
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
    """Learn one ResidualSSM correction weight per class."""
    if grid is None:
        grid = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=np.float32)
    else:
        grid = np.asarray(grid, dtype=np.float32)

    weights = np.full(N_CLASSES, float(base_weight), dtype=np.float32)
    tuned = 0
    moved = 0

    for c in range(N_CLASSES):
        y = Y_full[:, c]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos < min_pos or neg < min_pos:
            continue

        base_scores = first_pass_flat[:, c]
        corr_scores = correction_flat[:, c]
        best_w = float(base_weight)
        best_auc = roc_auc_score(y, base_scores + float(base_weight) * corr_scores)

        for w in grid:
            auc = roc_auc_score(y, base_scores + float(w) * corr_scores)
            if auc > best_auc + 1e-6:
                best_auc = float(auc)
                best_w = float(w)

        weights[c] = float(base_weight + shrink * (best_w - base_weight))
        tuned += 1
        if abs(best_w - base_weight) > 1e-6:
            moved += 1

    weights = np.clip(weights, 0.0, 0.60).astype(np.float32)
    if verbose:
        print(
            f"A2 classwise correction weights: tuned={tuned} classes, "
            f"moved={moved}, mean={weights.mean():.3f}, "
            f"range=[{weights.min():.2f}, {weights.max():.2f}]"
        )
    return weights


def apply_classwise_correction(first_pass_flat, correction_flat, correction_weights):
    cw = np.asarray(correction_weights, dtype=np.float32)
    if cw.ndim == 0:
        return first_pass_flat + float(cw) * correction_flat
    return first_pass_flat + cw[None, :] * correction_flat
# /A2_CLASSWISE_CORRECTION_WEIGHTS
'''

FULL_A2 = r'''

# A2: learn classwise correction weights on full training data.
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
)
'''

OOF_A2 = r'''

        # A2: learn classwise correction weights on this training fold.
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
        )
'''


def sub_once(pattern: str, repl: str, text: str, label: str, flags: int = 0) -> str:
    text2, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text2


def patch_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"A2 patch already present: {path}")
        return

    text = sub_once(
        r'(print\("✅ ResidualSSM defined \(~439K params, ~20s training\)"\))',
        lambda m: m.group(1) + "\n" + HELPER_CODE,
        text,
        "ResidualSSM helper insertion",
    )

    full_pattern = (
        r'(res_model, correction_weight = train_residual_ssm\(\n'
        r'(?:(?!\n\)).)*?emb_full=emb_tr,\n'
        r'(?:(?!\n\)).)*?first_pass_flat=first_pass,\n'
        r'(?:(?!\n\)).)*?Y_full=Y_FULL_aligned,\n'
        r'(?:(?!\n\)).)*?\n\))'
    )
    text = sub_once(full_pattern, lambda m: m.group(1) + FULL_A2, text, "full-train ResidualSSM block", re.S)

    oof_pattern = (
        r'(res_model, correction_weight = train_residual_ssm\(\n'
        r'(?:(?!\n        \)).)*?emb_full=emb_tr_f,\n'
        r'(?:(?!\n        \)).)*?first_pass_flat=first_pass_tr_f,\n'
        r'(?:(?!\n        \)).)*?Y_full=Y_tr_f,\n'
        r'(?:(?!\n        \)).)*?\n        \))'
    )
    text = sub_once(oof_pattern, lambda m: m.group(1) + OOF_A2, text, "OOF ResidualSSM block", re.S)

    text = sub_once(
        r'final_scores\s*=\s*first_pass\s*\+\s*correction_weight\s*\*\s*correction',
        'final_scores = apply_classwise_correction(first_pass, correction, correction_weight)',
        text,
        "full correction application",
    )
    text = sub_once(
        r'final_va\s*=\s*first_pass_va\s*\+\s*correction_weight\s*\*\s*va_correction',
        'final_va = apply_classwise_correction(first_pass_va, va_correction, correction_weight)',
        text,
        "OOF correction application",
    )

    path.write_text(text, encoding="utf-8")
    print(f"A2 patch applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    patch_script(args.script)


if __name__ == "__main__":
    main()
