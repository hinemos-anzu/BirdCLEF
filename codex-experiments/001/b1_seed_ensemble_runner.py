"""B1 seed ensemble CV runner for BirdCLEF V18.

Runs V18 multiple times with different seeds in separate Python subprocesses.
This avoids TensorFlow/PyTorch/ONNX memory buildup inside one notebook process.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_REPO_DIR = Path("/kaggle/working/BirdCLEF")
DEFAULT_CACHE_DIR = Path("/kaggle/working/cache")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/codex_experiment_001/b1_seed_ensemble")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run V18 OOF over multiple seeds and average predictions")
    p.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--clean-oof-cache", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    vals: list[float] = []
    for c in range(y_true.shape[1]):
        y = y_true[:, c]
        if y.min() == y.max():
            continue
        vals.append(float(roc_auc_score(y, y_score[:, c])))
    return float(np.mean(vals)) if vals else float("nan")


def rank_cols(p: np.ndarray) -> np.ndarray:
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)


def blend_auc(proto: np.ndarray, sed: np.ndarray, y: np.ndarray, w_proto: float) -> float:
    rp = rank_cols(np.clip(proto, 1e-5, 1 - 1e-5))
    rs = rank_cols(np.clip(sed, 1e-5, 1 - 1e-5))
    return macro_auc(y, rp * w_proto + rs * (1.0 - w_proto))


def patch_script_for_seed(src_script: Path, dst_script: Path, seed: int, seed_dir: Path) -> None:
    text = src_script.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^(\s*)MODE\s*=\s*["\'](?:train|submit)["\'](.*)$', r'\1MODE = "train"\2', text, count=1)

    seed_code = f'''
# B1_SEED_PATCH
import random as _b1_random
_b1_seed = {seed}
np.random.seed(_b1_seed)
_b1_random.seed(_b1_seed)
try:
    tf.random.set_seed(_b1_seed)
except Exception:
    pass
try:
    torch.manual_seed(_b1_seed)
except Exception:
    pass
print(f"B1 seed patch active: {{_b1_seed}}")
# /B1_SEED_PATCH
'''
    anchor = 'print("Config ready")'
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError("Could not find Config ready anchor for seed patch")
    line_end = text.find("\n", idx)
    text = text[:line_end + 1] + seed_code + text[line_end + 1:]

    save_code = f'''
# B1_SAVE_ARTIFACTS
try:
    import numpy as _b1_np
    from pathlib import Path as _b1_Path
    _b1_dir = _b1_Path(r"{seed_dir}")
    _b1_dir.mkdir(parents=True, exist_ok=True)
    if "oof_proto_probs" in globals():
        _b1_np.save(_b1_dir / "oof_proto_probs.npy", oof_proto_probs)
    if "Y_FULL_aligned" in globals():
        _b1_np.save(_b1_dir / "Y_FULL_aligned.npy", Y_FULL_aligned)
    if "sed_preds_tr_aligned" in globals() and sed_preds_tr_aligned is not None:
        _b1_np.save(_b1_dir / "sed_preds_tr_aligned.npy", sed_preds_tr_aligned)
    print(f"B1 artifacts saved to {{_b1_dir}}")
except Exception as _b1_ex:
    print(f"B1 artifact save failed: {{_b1_ex}}")
    raise
# /B1_SAVE_ARTIFACTS
'''
    text = text + "\n" + save_code
    dst_script.write_text(text, encoding="utf-8")


def run_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    src_script = args.repo_dir / "birdclef_v18_full_pipeline_oof_codex_ready_fixed.py"
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    script_copy = seed_dir / "v18_seed_run.py"
    log_path = seed_dir / "run.log"

    patch_script_for_seed(src_script, script_copy, seed, seed_dir)

    oof_cache = args.cache_dir / "oof_proto_probs.npy"
    if args.clean_oof_cache and oof_cache.exists():
        oof_cache.unlink()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, str(script_copy)],
            cwd="/kaggle/working",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {proc.returncode}. See {log_path}")

    oof_path = seed_dir / "oof_proto_probs.npy"
    y_path = seed_dir / "Y_FULL_aligned.npy"
    if not oof_path.exists() or not y_path.exists():
        raise RuntimeError(f"Seed {seed}: missing saved OOF artifacts. See {log_path}")

    oof = np.load(oof_path)
    y = np.load(y_path)
    sed_path = seed_dir / "sed_preds_tr_aligned.npy"
    sed = np.load(sed_path) if sed_path.exists() else None

    metrics: dict[str, Any] = {
        "seed": seed,
        "proto_oof_auc": macro_auc(y, oof),
        "log_path": str(log_path),
        "oof_path": str(oof_path),
    }
    if sed is not None:
        metrics["blend_060_auc"] = blend_auc(oof, sed, y, 0.60)
        metrics["blend_065_auc"] = blend_auc(oof, sed, y, 0.65)
        metrics["blend_060_minus_065"] = metrics["blend_060_auc"] - metrics["blend_065_auc"]
    return metrics


def load_existing(seed_dir: Path, seed: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    oof = np.load(seed_dir / "oof_proto_probs.npy")
    y = np.load(seed_dir / "Y_FULL_aligned.npy")
    sed_path = seed_dir / "sed_preds_tr_aligned.npy"
    sed = np.load(sed_path) if sed_path.exists() else None
    metrics: dict[str, Any] = {
        "seed": seed,
        "proto_oof_auc": macro_auc(y, oof),
        "log_path": str(seed_dir / "run.log"),
        "oof_path": str(seed_dir / "oof_proto_probs.npy"),
    }
    if sed is not None:
        metrics["blend_060_auc"] = blend_auc(oof, sed, y, 0.60)
        metrics["blend_065_auc"] = blend_auc(oof, sed, y, 0.65)
        metrics["blend_060_minus_065"] = metrics["blend_060_auc"] - metrics["blend_065_auc"]
    return metrics, oof, y, sed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    oofs = []
    y_ref = None
    sed_ref = None

    for seed in seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        if args.skip_existing and (seed_dir / "oof_proto_probs.npy").exists():
            metrics, oof, y, sed = load_existing(seed_dir, seed)
        else:
            metrics = run_one_seed(args, seed)
            metrics, oof, y, sed = load_existing(seed_dir, seed)
        rows.append(metrics)
        oofs.append(oof)
        y_ref = y if y_ref is None else y_ref
        if sed_ref is None and sed is not None:
            sed_ref = sed
        print(f"completed seed={seed}: proto_oof_auc={metrics['proto_oof_auc']:.6f}")

    avg_oof = np.mean(np.stack(oofs, axis=0), axis=0).astype(np.float32)
    np.save(args.output_dir / "oof_proto_probs_seedavg.npy", avg_oof)

    summary = pd.DataFrame(rows)
    avg_row: dict[str, Any] = {
        "seed": "avg",
        "proto_oof_auc": macro_auc(y_ref, avg_oof),
        "log_path": "",
        "oof_path": str(args.output_dir / "oof_proto_probs_seedavg.npy"),
    }
    if sed_ref is not None:
        avg_row["blend_060_auc"] = blend_auc(avg_oof, sed_ref, y_ref, 0.60)
        avg_row["blend_065_auc"] = blend_auc(avg_oof, sed_ref, y_ref, 0.65)
        avg_row["blend_060_minus_065"] = avg_row["blend_060_auc"] - avg_row["blend_065_auc"]
    summary = pd.concat([summary, pd.DataFrame([avg_row])], ignore_index=True)

    summary_path = args.output_dir / "seed_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)
    (args.output_dir / "seed_ensemble_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {args.output_dir / 'oof_proto_probs_seedavg.npy'}")


if __name__ == "__main__":
    main()
