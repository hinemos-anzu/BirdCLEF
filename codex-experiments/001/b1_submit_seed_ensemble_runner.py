"""B1 submit seed ensemble runner for BirdCLEF V18.

Runs V18 in submit mode for multiple seeds in separate subprocesses, averages
submission_protossm.csv, then blends with submission_sed.csv using the current
LB-best fixed ratio w_proto=0.60 / w_sed=0.40.

This runner is designed for Kaggle submit notebooks. It does not call Kaggle's
submission API; it writes /kaggle/working/submission.csv.

Typical usage:
  python BirdCLEF/codex-experiments/001/b1_submit_seed_ensemble_runner.py \
    --repo-dir /kaggle/working/BirdCLEF \
    --seeds 42,43 \
    --w-proto 0.60
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

DEFAULT_REPO_DIR = Path("/kaggle/working/BirdCLEF")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/codex_experiment_001/b1_submit_seed_ensemble")
DEFAULT_FINAL_SUBMISSION = Path("/kaggle/working/submission.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create seed-ensemble V18 submission")
    p.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--final-submission", type=Path, default=DEFAULT_FINAL_SUBMISSION)
    p.add_argument("--seeds", type=str, default="42,43")
    p.add_argument("--w-proto", type=float, default=0.60)
    p.add_argument("--mode", choices=["rank", "mean"], default="rank")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def patch_script_for_submit_seed(src_script: Path, dst_script: Path, seed: int, seed_dir: Path, w_proto: float) -> None:
    text = src_script.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^(\s*)MODE\s*=\s*["\'](?:train|submit)["\'](.*)$', r'\1MODE = "submit"\2', text, count=1)

    seed_code = f'''
# B1_SUBMIT_SEED_PATCH
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
print(f"B1 submit seed patch active: {{_b1_seed}}")
# /B1_SUBMIT_SEED_PATCH
'''
    anchor = 'print("Config ready")'
    idx = text.find(anchor)
    if idx < 0:
        raise RuntimeError("Could not find Config ready anchor for seed patch")
    line_end = text.find("\n", idx)
    text = text[:line_end + 1] + seed_code + text[line_end + 1:]

    # Force the manual override to the current LB-best fixed blend ratio.
    text = re.sub(
        r'BLEND_W_PROTO\s*=\s*0\.65\s*#.*',
        f'BLEND_W_PROTO = {w_proto:.2f}   # B1 submit fixed: w_sed={1.0 - w_proto:.2f}',
        text,
        count=1,
    )

    save_code = f'''
# B1_SUBMIT_SAVE_ARTIFACTS
try:
    from pathlib import Path as _b1_Path
    import shutil as _b1_shutil
    _b1_dir = _b1_Path(r"{seed_dir}")
    _b1_dir.mkdir(parents=True, exist_ok=True)
    for _name in ["submission_protossm.csv", "submission_sed.csv", "submission.csv"]:
        _p = _b1_Path(_name)
        if _p.exists():
            _b1_shutil.copy2(_p, _b1_dir / _name)
    print(f"B1 submit artifacts saved to {{_b1_dir}}")
except Exception as _b1_ex:
    print(f"B1 submit artifact save failed: {{_b1_ex}}")
    raise
# /B1_SUBMIT_SAVE_ARTIFACTS
'''
    text = text + "\n" + save_code
    dst_script.write_text(text, encoding="utf-8")


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    src_script = args.repo_dir / "birdclef_v18_full_pipeline_oof_codex_ready_fixed.py"
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    script_copy = seed_dir / "v18_submit_seed_run.py"
    log_path = seed_dir / "run.log"

    if args.skip_existing and (seed_dir / "submission_protossm.csv").exists():
        return {"seed": seed, "status": "skipped", "log_path": str(log_path), "proto_path": str(seed_dir / "submission_protossm.csv")}

    patch_script_for_submit_seed(src_script, script_copy, seed, seed_dir, args.w_proto)
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

    proto_path = seed_dir / "submission_protossm.csv"
    if not proto_path.exists():
        raise RuntimeError(f"Seed {seed}: missing submission_protossm.csv. See {log_path}")
    return {"seed": seed, "status": "ok", "log_path": str(log_path), "proto_path": str(proto_path)}


def rank_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df[cols].rank(axis=0, pct=True).astype(np.float32)
    return out


def average_submissions(paths: list[Path], mode: str) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    base = dfs[0].copy()
    cols = [c for c in base.columns if c != "row_id"]
    for df in dfs[1:]:
        df2 = df.set_index("row_id").loc[base["row_id"]].reset_index()
        if list(df2["row_id"]) != list(base["row_id"]):
            raise ValueError("row_id alignment failed")
    if mode == "rank":
        mats = [rank_frame(df.set_index("row_id").loc[base["row_id"]].reset_index(), cols)[cols].to_numpy(np.float32) for df in dfs]
    else:
        mats = [df.set_index("row_id").loc[base["row_id"]].reset_index()[cols].to_numpy(np.float32) for df in dfs]
    base[cols] = np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)
    return base


def blend_proto_sed(proto: pd.DataFrame, sed_path: Path | None, w_proto: float, mode: str) -> pd.DataFrame:
    if sed_path is None or not sed_path.exists():
        print("SED submission not found; using proto ensemble only")
        return proto
    sed = pd.read_csv(sed_path)
    sed = sed.set_index("row_id").loc[proto["row_id"]].reset_index()
    cols = [c for c in proto.columns if c != "row_id"]
    out = proto.copy()
    if mode == "rank":
        p_proto = proto[cols].rank(axis=0, pct=True).to_numpy(np.float32)
        p_sed = sed[cols].rank(axis=0, pct=True).to_numpy(np.float32)
    else:
        p_proto = proto[cols].to_numpy(np.float32)
        p_sed = sed[cols].to_numpy(np.float32)
    out[cols] = (w_proto * p_proto + (1.0 - w_proto) * p_sed).astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        result = run_seed(args, seed)
        rows.append(result)
        print(f"completed seed={seed}: {result['status']}")

    proto_paths = [Path(r["proto_path"]) for r in rows]
    proto_ens = average_submissions(proto_paths, mode=args.mode)
    proto_ens_path = args.output_dir / "submission_protossm_seedavg.csv"
    proto_ens.to_csv(proto_ens_path, index=False)

    sed_path = None
    for seed in seeds:
        candidate = args.output_dir / f"seed_{seed}" / "submission_sed.csv"
        if candidate.exists():
            sed_path = candidate
            break

    final = blend_proto_sed(proto_ens, sed_path, w_proto=args.w_proto, mode=args.mode)
    args.final_submission.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.final_submission, index=False)

    summary = {
        "seeds": seeds,
        "w_proto": args.w_proto,
        "w_sed": 1.0 - args.w_proto,
        "mode": args.mode,
        "proto_ensemble_path": str(proto_ens_path),
        "sed_path": str(sed_path) if sed_path else None,
        "final_submission": str(args.final_submission),
        "runs": rows,
        "shape": list(final.shape),
    }
    summary_path = args.output_dir / "submit_seed_ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {proto_ens_path}")
    print(f"Wrote final submission: {args.final_submission}")


if __name__ == "__main__":
    main()
