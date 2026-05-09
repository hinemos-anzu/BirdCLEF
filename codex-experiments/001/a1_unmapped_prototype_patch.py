"""Patch V18 script with A1 unmapped-class embedding prototype boost.

This edits only the Kaggle working copy of
birdclef_v18_full_pipeline_oof_codex_ready_fixed.py.

A1 targets classes with no direct Perch mapping and no genus proxy. It learns
simple positive-vs-negative embedding prototypes on each OOF training fold and
adds a conservative validation-fold probability boost for those classes.

The patch intentionally modifies only the OOF/CV path first. Submission-path
changes should be added only if CV shows a meaningful improvement.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = "A1_UNMAPPED_PROTOTYPE_BOOST"

HELPER_CODE = r'''
# A1_UNMAPPED_PROTOTYPE_BOOST
def _a1_l2_normalize(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _a1_sigmoid_np(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def get_no_signal_unmapped_positions():
    """Classes with neither direct Perch mapping nor genus proxy."""
    proxy_keys = set(proxy_map.keys()) if 'proxy_map' in globals() else set()
    return np.array([int(i) for i in UNMAPPED_POS if int(i) not in proxy_keys], dtype=np.int32)


def apply_unmapped_prototype_boost(
    probs,
    emb_train,
    y_train,
    emb_valid,
    class_indices=None,
    min_pos=2,
    min_neg=20,
    blend=0.35,
    scale=7.0,
    q_gate=0.70,
    verbose=False,
):
    """
    Conservative OOF-only boost for no-signal unmapped classes.

    For each supported class, compute normalized positive and negative embedding
    centroids on the training fold. Validation scores are cosine(pos)-cos(neg),
    converted to probabilities and blended only in the high-confidence tail.
    """
    if class_indices is None:
        class_indices = get_no_signal_unmapped_positions()
    class_indices = np.asarray(class_indices, dtype=np.int32)
    if len(class_indices) == 0:
        return probs

    out = probs.copy()
    x_tr = _a1_l2_normalize(emb_train)
    x_va = _a1_l2_normalize(emb_valid)
    tuned = 0
    boosted_cells = 0

    for c in class_indices:
        y = y_train[:, c].astype(bool)
        pos = int(y.sum())
        neg = int((~y).sum())
        if pos < min_pos or neg < min_neg:
            continue

        pos_cent = x_tr[y].mean(axis=0, keepdims=True)
        neg_cent = x_tr[~y].mean(axis=0, keepdims=True)
        pos_cent = _a1_l2_normalize(pos_cent)[0]
        neg_cent = _a1_l2_normalize(neg_cent)[0]

        raw = x_va @ pos_cent - x_va @ neg_cent
        proto_prob = _a1_sigmoid_np(scale * raw).astype(np.float32)
        gate = proto_prob >= np.quantile(proto_prob, q_gate)
        if not np.any(gate):
            continue

        blended = (1.0 - blend) * out[:, c] + blend * proto_prob
        before = out[:, c].copy()
        out[gate, c] = np.maximum(out[gate, c], blended[gate])
        boosted_cells += int((out[:, c] > before + 1e-7).sum())
        tuned += 1

    if verbose:
        print(
            f"A1 unmapped prototype boost: target={len(class_indices)} classes, "
            f"tuned={tuned}, boosted_cells={boosted_cells}, "
            f"blend={blend:.2f}, q_gate={q_gate:.2f}"
        )
    return out.astype(np.float32)
# /A1_UNMAPPED_PROTOTYPE_BOOST
'''


def insert_helper(text: str) -> str:
    """Insert helper after unmapped/proxy setup using robust anchors."""
    anchors = [
        "# ── Cell 5: Perch inference engine",
        "# -- Cell 5: Perch inference engine",
        "import concurrent.futures",
    ]
    for anchor in anchors:
        idx = text.find(anchor)
        if idx >= 0:
            return text[:idx] + HELPER_CODE + "\n" + text[idx:]

    # Fallback: insert after the proxy target print loop body if present.
    pattern = r'(for idx, bc_idxs in list\(proxy_map\.items\(\)\)\[:8\]:[\s\S]*?print\(f"\s+\{label:12s\}.*?\n)'
    text2, count = re.subn(pattern, lambda m: m.group(1) + "\n" + HELPER_CODE + "\n", text, count=1)
    if count == 1:
        return text2

    raise RuntimeError("Could not find a robust insertion point for A1 helper")


def patch_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"A1 patch already present: {path}")
        return

    text = insert_helper(text)

    old = 'probs_va = adaptive_delta_smooth(probs_va, n_windows=N_WINDOWS, base_alpha=0.20)'
    new = old + '''
        probs_va = apply_unmapped_prototype_boost(
            probs_va,
            emb_train=emb_tr_f,
            y_train=Y_tr_f,
            emb_valid=emb_va_f,
            min_pos=2,
            min_neg=20,
            blend=0.35,
            scale=7.0,
            q_gate=0.70,
            verbose=(fold == 1),
        )'''
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one adaptive_delta_smooth validation line, found {text.count(old)}")
    text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    print(f"A1 OOF-only patch applied: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    patch_script(args.script)


if __name__ == "__main__":
    main()
