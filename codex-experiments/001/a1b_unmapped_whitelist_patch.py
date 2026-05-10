"""Patch V18 script with A1b whitelist-based unmapped prototype boost.

A1b is a safer replacement for the rejected A1 prototype boost.
It edits only the Kaggle working copy and modifies only the OOF/CV path.

Design:
- Target only no-signal unmapped classes.
- Run an inner GroupKFold on the OOF training fold.
- Whitelist a class only when prototype scores beat the current fold-training
  first-pass baseline by a margin.
- Apply a weak boost only to the high-confidence validation tail.

Usage:
  python BirdCLEF/codex-experiments/001/a1b_unmapped_whitelist_patch.py \
    --script /kaggle/working/BirdCLEF/birdclef_v18_full_pipeline_oof_codex_ready_fixed.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "A1B_UNMAPPED_WHITELIST_PROTO"

HELPER_CODE = r'''
# A1B_UNMAPPED_WHITELIST_PROTO
def _a1b_l2_normalize(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _a1b_sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def get_no_signal_unmapped_positions():
    proxy_keys = set(proxy_map.keys()) if 'proxy_map' in globals() else set()
    return np.array([int(i) for i in UNMAPPED_POS if int(i) not in proxy_keys], dtype=np.int32)


def _a1b_proto_scores(x_train, y_train, x_valid, min_pos=2, min_neg=20, scale=7.0):
    y = y_train.astype(bool)
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos < min_pos or neg < min_neg:
        return None
    pos_cent = _a1b_l2_normalize(x_train[y].mean(axis=0, keepdims=True))[0]
    neg_cent = _a1b_l2_normalize(x_train[~y].mean(axis=0, keepdims=True))[0]
    raw = x_valid @ pos_cent - x_valid @ neg_cent
    return _a1b_sigmoid_np(scale * raw).astype(np.float32)


def _a1b_safe_auc(y, score):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, score))


def select_unmapped_prototype_whitelist(
    emb_train,
    y_train,
    meta_train,
    first_pass_train,
    class_indices=None,
    min_pos=4,
    min_neg=30,
    inner_splits=3,
    min_auc_gain=0.02,
    min_proto_auc=0.58,
    scale=7.0,
):
    """Select unmapped classes whose prototype signal passes inner grouped CV."""
    if class_indices is None:
        class_indices = get_no_signal_unmapped_positions()
    class_indices = np.asarray(class_indices, dtype=np.int32)
    if len(class_indices) == 0:
        return []

    filenames = meta_train["filename"].astype(str).to_numpy()
    unique_files = pd.unique(filenames)
    n_splits = int(min(inner_splits, len(unique_files)))
    if n_splits < 2:
        return []

    x = _a1b_l2_normalize(emb_train)
    groups = filenames
    splitter = GroupKFold(n_splits=n_splits)
    selected = []

    for c in class_indices:
        y = y_train[:, c].astype(np.uint8)
        if int(y.sum()) < min_pos or int(len(y) - y.sum()) < min_neg:
            continue

        proto_oof = np.full(len(y), np.nan, dtype=np.float32)
        for tr_idx, va_idx in splitter.split(x, y, groups=groups):
            scores = _a1b_proto_scores(
                x[tr_idx], y[tr_idx], x[va_idx],
                min_pos=max(2, min_pos // 2), min_neg=min_neg, scale=scale,
            )
            if scores is not None:
                proto_oof[va_idx] = scores

        valid = np.isfinite(proto_oof)
        if valid.sum() < max(20, min_pos + min_neg):
            continue
        proto_auc = _a1b_safe_auc(y[valid], proto_oof[valid])
        base_auc = _a1b_safe_auc(y[valid], _a1b_sigmoid_np(first_pass_train[valid, c]))
        if not np.isfinite(proto_auc) or not np.isfinite(base_auc):
            continue
        if proto_auc >= min_proto_auc and (proto_auc - base_auc) >= min_auc_gain:
            selected.append((int(c), float(proto_auc), float(base_auc), int(y.sum())))

    return selected


def apply_unmapped_prototype_whitelist_boost(
    probs,
    emb_train,
    y_train,
    meta_train,
    emb_valid,
    first_pass_train,
    min_pos=4,
    min_neg=30,
    inner_splits=3,
    min_auc_gain=0.02,
    min_proto_auc=0.58,
    blend=0.10,
    scale=7.0,
    q_gate=0.90,
    verbose=False,
):
    """Apply weak prototype boost only for inner-CV-whitelisted unmapped classes."""
    selected = select_unmapped_prototype_whitelist(
        emb_train=emb_train,
        y_train=y_train,
        meta_train=meta_train,
        first_pass_train=first_pass_train,
        min_pos=min_pos,
        min_neg=min_neg,
        inner_splits=inner_splits,
        min_auc_gain=min_auc_gain,
        min_proto_auc=min_proto_auc,
        scale=scale,
    )
    if not selected:
        if verbose:
            print("A1b unmapped whitelist: selected=0; no boost applied")
        return probs

    out = probs.copy()
    x_tr = _a1b_l2_normalize(emb_train)
    x_va = _a1b_l2_normalize(emb_valid)
    boosted_cells = 0

    for c, proto_auc, base_auc, pos in selected:
        proto_prob = _a1b_proto_scores(
            x_tr, y_train[:, c], x_va,
            min_pos=max(2, min_pos // 2), min_neg=min_neg, scale=scale,
        )
        if proto_prob is None:
            continue
        gate = proto_prob >= np.quantile(proto_prob, q_gate)
        stronger = proto_prob > out[:, c]
        mask = gate & stronger
        if not np.any(mask):
            continue
        before = out[:, c].copy()
        blended = (1.0 - blend) * out[:, c] + blend * proto_prob
        out[mask, c] = blended[mask]
        boosted_cells += int((out[:, c] > before + 1e-7).sum())

    if verbose:
        labels = []
        for c, proto_auc, base_auc, pos in selected[:10]:
            name = PRIMARY_LABELS[c] if 'PRIMARY_LABELS' in globals() else str(c)
            labels.append(f"{name}:proto={proto_auc:.3f}/base={base_auc:.3f}/pos={pos}")
        print(
            f"A1b unmapped whitelist: target={len(get_no_signal_unmapped_positions())}, "
            f"selected={len(selected)}, boosted_cells={boosted_cells}, "
            f"blend={blend:.2f}, q_gate={q_gate:.2f}"
        )
        if labels:
            print("  selected:", "; ".join(labels))
    return out.astype(np.float32)
# /A1B_UNMAPPED_WHITELIST_PROTO
'''


def insert_helper(text: str) -> str:
    for anchor in ["# ── Cell 5: Perch inference engine", "# -- Cell 5: Perch inference engine", "import concurrent.futures"]:
        idx = text.find(anchor)
        if idx >= 0:
            return text[:idx] + HELPER_CODE + "\n" + text[idx:]
    raise RuntimeError("Could not find insertion point for A1b helper")


def patch_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"A1b patch already present: {path}")
        return

    text = insert_helper(text)

    old = 'probs_va = adaptive_delta_smooth(probs_va, n_windows=N_WINDOWS, base_alpha=0.20)'
    new = old + '''
        probs_va = apply_unmapped_prototype_whitelist_boost(
            probs_va,
            emb_train=emb_tr_f,
            y_train=Y_tr_f,
            meta_train=meta_tr_f,
            emb_valid=emb_va_f,
            first_pass_train=first_pass_tr_f,
            min_pos=4,
            min_neg=30,
            inner_splits=3,
            min_auc_gain=0.02,
            min_proto_auc=0.58,
            blend=0.10,
            scale=7.0,
            q_gate=0.90,
            verbose=(fold == 1),
        )'''
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one adaptive_delta_smooth validation line, found {text.count(old)}")
    text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    print(f"A1b OOF-only patch applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    patch_script(args.script)


if __name__ == "__main__":
    main()
