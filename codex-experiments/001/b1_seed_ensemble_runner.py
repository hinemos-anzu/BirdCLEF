"""B1 seed ensemble CV runner for BirdCLEF V18.

Runs the V18 script multiple times with different seeds, collects OOF metrics,
and builds an averaged ProtoSSM OOF prediction. This is for CV stability checks
before spending Kaggle LB submissions.

Typical Kaggle usage:
  python BirdCLEF/codex-experiments/001/b1_seed_ensemble_runner.py \
    --repo-dir /kaggle/working/BirdCLEF \
    --seeds 42,43,44 \
    --clean-oof-cache

Notes:
- The runner edits only the Kaggle working copy.
- It expects V18 to create /kaggle/working/cache/oof_proto_probs.npy.
- It preserves the Perch cache, but removes OOF cache between seeds.
- It does not submit anything.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
import shutil
from contextlib import redirect_stderr, redirect_stdout
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


def patch_script_for_seed(script_path: Path, seed: int) -> None:
    text = script_path.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^(\s*)MODE\s*=\s*["\'](?:train|submit)["\'](.*)$', r'\1MODE = "train"\2', text, count=1)

    marker = "# B1_SEED_PATCH"
    seed_code = f'''
{marker}
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
    if marker in text:
        text = re.sub(r'# B1_SEED_PATCH[\s\S]*?# /B1_SEED_PATCH\n', seed_code, text, count=1)
    else:
        anchor = 'print("Config ready")'
        idx = text.find(anchor)
        if idx < 0:
            raise RuntimeError("Could not find Config ready anchor for seed patch")
        line_end = text.find("\n", idx)
        text = text[:line_end + 1] + seed_code + text[line_end + 1:]

    script_path.write_text(text, encoding="utf-8")


def copy_artifact(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_one_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    script_path = args.repo_dir / "birdclef_v18_full_pipeline_oof_codex_ready_fixed.py"
    patch_script_for_seed(script_path, seed)

    oof_cache = args.cache_dir / "oof_proto_probs.npy"
    if args.clean_oof_cache and oof_cache.exists():
        oof_cache.unlink()

    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_path = seed_dir / "run.log"

    with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
        ns = runpy.run_path(str(script_path), run_name="__main__")

    oof = ns.get("oof_proto_probs")
    y = ns.get("Y_FULL_aligned")
    sed = ns.get("sed_preds_tr_aligned")
    if oof is None or y is None:
        raise RuntimeError(f"Seed {seed}: expected oof_proto_probs and Y_FULL_aligned after run")

    oof = np.asarray(oof, dtype=np.float32)
    y = np.asarray(y)
    np.save(seed_dir / "oof_proto_probs.npy", oof)
    np.save(seed_dir / "Y_FULL_aligned.npy", y)
    if sed is not None:
        sed = np.asarray(sed, dtype=np.float32)
        np.save(seed_dir / "sed_preds_tr_aligned.npy", sed)

    metrics: dict[str, Any] = {
        "seed": seed,
        "proto_oof_auc": macro_auc(y, oof),
        "log_path": str(log_path),
        "oof_path": str(seed_dir / "oof_proto_probs.npy"),
    }
    if sed is not None:
        metrics["blend_060_auc"] = blend_auc(oof, sed, y, 0.60)
        metrics["blend_065_auc"] = blend_auc(oof, sed, y, 0.65)
        metrics["blend_060_minus_065"] = metrics["blend_060_auc"] - metrics["blend_065_auc"]
    return metrics


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
            oof = np.load(seed_dir / "oof_proto_probs.npy")
            y = np.load(seed_dir / "Y_FULL_aligned.npy")
            sed_path = seed_dir / "sed_preds_tr_aligned.npy"
            sed = np.load(sed_path) if sed_path.exists() else None
            metrics = {"seed": seed, "proto_oof_auc": macro_auc(y, oof), "log_path": str(seed_dir / "run.log"), "oof_path": str(seed_dir / "oof_proto_probs.npy")}
            if sed is not None:
                metrics["blend_060_auc"] = blend_auc(oof, sed, y, 0.60)
                metrics["blend_065_auc"] = blend_auc(oof, sed, y, 0.65)
                metrics["blend_060_minus_065"] = metrics["blend_060_auc"] - metrics["blend_065_auc"]
        else:
            metrics = run_one_seed(args, seed)
            oof = np.load(metrics["oof_path"])
            y = np.load(seed_dir / "Y_FULL_aligned.npy")
            sed_path = seed_dir / "sed_preds_tr_aligned.npy"
            sed = np.load(sed_path) if sed_path.exists() else None

        rows.append(metrics)
        oofs.append(oof)
        y_ref = y if y_ref is None else y_ref
        if sed_ref is None and sed is not None:
            sed_ref = sed

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
