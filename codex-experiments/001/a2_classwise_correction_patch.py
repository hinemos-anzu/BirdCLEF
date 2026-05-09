"""Patch V18 script with A2 classwise ResidualSSM correction weights.

Edits only the Kaggle working copy of birdclef_v18_full_pipeline_oof_codex_ready_fixed.py.
This patch intentionally modifies only the OOF/CV path. The full-train submission
path stays scalar until A2 is validated by CV.
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
            verbose=True,
        )
'''


def find_call_end(text: str, open_paren: int) -> int:
    depth = 0
    in_str: str | None = None
    escape = False
    for i in range(open_paren, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in {"'", '"'}:
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise RuntimeError("Unbalanced train_residual_ssm call")


def find_train_call(text: str, required_tokens: list[str]) -> tuple[int, int] | None:
    needle = "res_model, correction_weight = train_residual_ssm("
    pos = 0
    while True:
        start = text.find(needle, pos)
        if start < 0:
            return None
        open_paren = text.find("(", start)
        end = find_call_end(text, open_paren)
        compact = re.sub(r"\s+", "", text[start:end])
        if all(token in compact for token in required_tokens):
            return start, end
        pos = end


def sub_required(pattern: str, repl, text: str, label: str) -> str:
    text2, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text2


def patch_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"A2 patch already present: {path}")
        return

    text = sub_required(
        r'print\("✅ ResidualSSM defined \(~439K params, ~20s training\)"\)',
        lambda m: m.group(0) + "\n" + HELPER_CODE,
        text,
        "ResidualSSM helper insertion",
    )

    found = find_train_call(
        text,
        ["emb_full=emb_tr_f", "first_pass_flat=first_pass_tr_f", "Y_full=Y_tr_f"],
    )
    if found is None:
        raise RuntimeError("Could not find OOF ResidualSSM block")
    _, end = found
    text = text[:end] + OOF_A2 + text[end:]

    text = sub_required(
        r'final_va\s*=\s*first_pass_va\s*\+\s*correction_weight\s*\*\s*va_correction',
        'final_va = apply_classwise_correction(first_pass_va, va_correction, correction_weight)',
        text,
        "OOF correction application",
    )

    path.write_text(text, encoding="utf-8")
    print(f"A2 OOF-only patch applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    patch_script(args.script)


if __name__ == "__main__":
    main()
