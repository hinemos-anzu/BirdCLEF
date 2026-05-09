"""CLI runner for LB-scarce blend-ratio CV gating.

Run after the V18 notebook/script has generated OOF artifacts. The gate compares
a candidate blend weight against the current LB-best baseline without spending a
Kaggle submission.

Defaults:
  proto OOF: /kaggle/working/cache/oof_proto_probs.npy
  meta:      /kaggle/working/cache/perch_meta.parquet
  labels:    /kaggle/input/competitions/birdclef-2026/train_soundscapes_labels.csv
  sample:    /kaggle/input/competitions/birdclef-2026/sample_submission.csv

If SED train predictions only exist as a notebook variable, save them first:
  np.save('/kaggle/working/cache/sed_preds_tr_aligned.npy', sed_preds_tr_aligned)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_INPUT_ROOT = Path("/kaggle/input/competitions/birdclef-2026")
DEFAULT_CACHE_DIR = Path("/kaggle/working/cache")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/codex_experiment_001")
DEFAULT_KNOWN_LB = {1.00: 0.927, 0.50: 0.941, 0.60: 0.946}
N_WINDOWS = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV gate for BirdCLEF blend-ratio LB decisions")
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--proto-oof", type=Path, default=None)
    p.add_argument("--sed-oof", type=Path, default=None)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--candidate-w", type=float, default=0.65)
    p.add_argument("--baseline-w", type=float, default=0.60)
    p.add_argument("--min-lb-gain", type=float, default=0.0010)
    p.add_argument("--min-boot-prob", type=float, default=0.65)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w-grid", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    return p.parse_args()


def safe_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    vals: list[float] = []
    for j in range(y_true.shape[1]):
        yy = y_true[:, j]
        if yy.min() == yy.max():
            continue
        vals.append(float(roc_auc_score(yy, y_score[:, j])))
    return float(np.mean(vals)) if vals else float("nan")


def rank_cols(p: np.ndarray) -> np.ndarray:
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)


def blend_ranked(r_proto: np.ndarray, r_sed: np.ndarray, w_proto: float) -> np.ndarray:
    return (r_proto * float(w_proto) + r_sed * (1.0 - float(w_proto))).astype(np.float32)


def load_array(path: Path, expected_cols: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = next((k for k in ["preds", "scores", "oof", "sed_preds", "arr_0"] if k in data.files), data.files[0])
        arr = data[key].astype(np.float32)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        arr = df[[c for c in df.columns if c != "row_id"]].to_numpy(np.float32)
    else:
        raise ValueError(f"Unsupported array file: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array at {path}, got shape={arr.shape}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns at {path}, got shape={arr.shape}")
    return arr


def find_sed_path(cache_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for path in [
        cache_dir / "sed_preds_tr_aligned.npy",
        cache_dir / "sed_oof.npy",
        cache_dir / "oof_sed_probs.npy",
        cache_dir / "sed_preds_tr_aligned.csv",
        Path("/kaggle/working/sed_preds_tr_aligned.npy"),
        Path("/kaggle/working/sed_preds_tr_aligned.csv"),
    ]:
        if path.exists():
            return path
    raise FileNotFoundError(
        "SED predictions were not found. Pass --sed-oof, or save "
        "np.save('/kaggle/working/cache/sed_preds_tr_aligned.npy', sed_preds_tr_aligned)."
    )


def union_labels(values: pd.Series) -> list[str]:
    out: set[str] = set()
    for x in values:
        if pd.notna(x):
            out.update(t.strip() for t in str(x).split(";") if t.strip())
    return sorted(out)


def build_y_aligned(input_root: Path, meta: pd.DataFrame, primary_labels: list[str]) -> np.ndarray:
    labels_path = input_root / "train_soundscapes_labels.csv"
    labels = pd.read_csv(labels_path)
    grouped = (
        labels.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    grouped["end_sec"] = pd.to_timedelta(grouped["end"]).dt.total_seconds().astype(int)
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)

    label_to_idx = {label: i for i, label in enumerate(primary_labels)}
    y_by_row: dict[str, np.ndarray] = {}
    for row_id, lbls in zip(grouped["row_id"], grouped["label_list"]):
        yy = np.zeros(len(primary_labels), dtype=np.uint8)
        for label in lbls:
            idx = label_to_idx.get(label)
            if idx is not None:
                yy[idx] = 1
        y_by_row[str(row_id)] = yy

    if "row_id" not in meta.columns:
        if "filename" not in meta.columns:
            raise ValueError("meta must contain row_id or filename")
        file_counts = meta.groupby("filename").cumcount().to_numpy()
        meta = meta.copy()
        meta["row_id"] = (
            meta["filename"].astype(str).str.replace(".ogg", "", regex=False)
            + "_"
            + ((file_counts % N_WINDOWS + 1) * 5).astype(str)
        )

    missing = [r for r in meta["row_id"].astype(str).tolist() if r not in y_by_row]
    if missing:
        raise ValueError(f"Could not align {len(missing)} meta rows to labels. First missing row_id={missing[0]}")
    return np.stack([y_by_row[r] for r in meta["row_id"].astype(str)], axis=0)


def bootstrap_file_delta(
    r_proto: np.ndarray,
    r_sed: np.ndarray,
    y_true: np.ndarray,
    filenames: np.ndarray,
    candidate_w: float,
    baseline_w: float,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    files = np.array(pd.unique(filenames))
    file_to_idx = {f: np.where(filenames == f)[0] for f in files}
    cand = blend_ranked(r_proto, r_sed, candidate_w)
    base = blend_ranked(r_proto, r_sed, baseline_w)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([file_to_idx[f] for f in rng.choice(files, size=len(files), replace=True)])
        auc_c = safe_macro_auc(y_true[idx], cand[idx])
        auc_b = safe_macro_auc(y_true[idx], base[idx])
        if np.isfinite(auc_c) and np.isfinite(auc_b):
            deltas.append(auc_c - auc_b)
    return np.asarray(deltas, dtype=np.float32)


def calibrate_lb_proxy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cv_050 = float(df.loc[np.isclose(df["w_proto"], 0.50), "cv_auc"].iloc[0])
    cv_060 = float(df.loc[np.isclose(df["w_proto"], 0.60), "cv_auc"].iloc[0])
    lb_delta = DEFAULT_KNOWN_LB[0.60] - DEFAULT_KNOWN_LB[0.50]
    cv_delta = cv_060 - cv_050
    sign_agrees = np.sign(lb_delta) == np.sign(cv_delta)

    df = df.copy()
    if sign_agrees and abs(cv_delta) > 1e-8:
        raw_scale = lb_delta / cv_delta
        scale = float(np.clip(raw_scale, -2.0, 2.0))
        df["lb_proxy"] = DEFAULT_KNOWN_LB[0.60] + (df["cv_auc"] - cv_060) * scale
        df["lb_proxy_delta_vs_060"] = df["lb_proxy"] - DEFAULT_KNOWN_LB[0.60]
        proxy_usable = True
    else:
        raw_scale = float("nan")
        scale = 0.0
        df["lb_proxy"] = np.nan
        df["lb_proxy_delta_vs_060"] = np.nan
        proxy_usable = False

    return df, {
        "cv_anchor_delta_060_minus_050": float(cv_delta),
        "lb_anchor_delta_060_minus_050": float(lb_delta),
        "cv_lb_anchor_sign_agrees": bool(sign_agrees),
        "cv_to_lb_scale_raw": float(raw_scale),
        "cv_to_lb_scale_capped": float(scale),
        "lb_proxy_usable": bool(proxy_usable),
    }


def run_gate(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = pd.read_csv(args.input_root / "sample_submission.csv")
    primary_labels = sample.columns[1:].tolist()
    n_classes = len(primary_labels)

    proto_path = args.proto_oof or (args.cache_dir / "oof_proto_probs.npy")
    sed_path = find_sed_path(args.cache_dir, args.sed_oof)
    meta_path = args.meta or (args.cache_dir / "perch_meta.parquet")

    meta = pd.read_parquet(meta_path)
    proto = load_array(proto_path, expected_cols=n_classes)
    sed = load_array(sed_path, expected_cols=n_classes)
    y_true = build_y_aligned(args.input_root, meta, primary_labels)
    if len(meta) != len(proto) or len(meta) != len(sed) or len(meta) != len(y_true):
        raise ValueError(f"Length mismatch: meta={len(meta)}, proto={len(proto)}, sed={len(sed)}, y={len(y_true)}")

    w_grid = np.array([float(x) for x in args.w_grid.split(",")], dtype=np.float32)
    r_proto = rank_cols(np.clip(proto, 1e-5, 1.0 - 1e-5))
    r_sed = rank_cols(np.clip(sed, 1e-5, 1.0 - 1e-5))

    rows = []
    for w in w_grid:
        rows.append({"w_proto": float(w), "w_sed": float(1.0 - w), "cv_auc": safe_macro_auc(y_true, blend_ranked(r_proto, r_sed, float(w)))})
    df = pd.DataFrame(rows)
    base_cv = float(df.loc[np.isclose(df["w_proto"], args.baseline_w), "cv_auc"].iloc[0])
    df["cv_delta_vs_baseline"] = df["cv_auc"] - base_cv
    df, calibration = calibrate_lb_proxy(df)

    boot_delta = bootstrap_file_delta(
        r_proto,
        r_sed,
        y_true,
        meta["filename"].astype(str).to_numpy(),
        candidate_w=args.candidate_w,
        baseline_w=args.baseline_w,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    p_win = float((boot_delta > 0).mean()) if len(boot_delta) else float("nan")
    delta_mean = float(np.nanmean(boot_delta)) if len(boot_delta) else float("nan")
    ci_low, ci_high = np.nanpercentile(boot_delta, [5, 95]) if len(boot_delta) else (float("nan"), float("nan"))
    cand_row = df.loc[np.isclose(df["w_proto"], args.candidate_w)].iloc[0]

    if calibration["lb_proxy_usable"]:
        proxy_gain = float(cand_row["lb_proxy_delta_vs_060"])
        proxy_pass = proxy_gain >= args.min_lb_gain
    else:
        proxy_gain = float("nan")
        proxy_pass = True

    should_submit = bool(proxy_pass and (p_win >= args.min_boot_prob) and (delta_mean > 0.0))
    decision = {
        "proto_path": str(proto_path),
        "sed_path": str(sed_path),
        "meta_path": str(meta_path),
        "known_lb": DEFAULT_KNOWN_LB,
        "candidate_w": float(args.candidate_w),
        "baseline_w": float(args.baseline_w),
        **calibration,
        "candidate_lb_proxy_gain": proxy_gain,
        "bootstrap_mean_delta": delta_mean,
        "bootstrap_p_win": p_win,
        "bootstrap_ci90": [float(ci_low), float(ci_high)],
        "min_lb_gain": float(args.min_lb_gain),
        "min_boot_prob": float(args.min_boot_prob),
        "should_submit": should_submit,
    }
    return df, decision


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, decision = run_gate(args)
    table_path = args.output_dir / "blend_cv_gate_table.csv"
    decision_path = args.output_dir / "blend_cv_gate_decision.json"
    table.to_csv(table_path, index=False)
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Blend CV Gate: LB submissions are scarce")
    print("=" * 72)
    print(table.to_string(index=False, formatters={
        "w_proto": "{:.2f}".format,
        "w_sed": "{:.2f}".format,
        "cv_auc": "{:.6f}".format,
        "cv_delta_vs_baseline": "{:+.6f}".format,
        "lb_proxy": lambda x: "NA" if pd.isna(x) else f"{x:.6f}",
        "lb_proxy_delta_vs_060": lambda x: "NA" if pd.isna(x) else f"{x:+.6f}",
    }))
    print("\nDecision summary:")
    for key, value in decision.items():
        print(f"  {key}: {value}")
    if not decision["lb_proxy_usable"]:
        print("\nNote: CV and known LB anchors disagree in direction, so lb_proxy is disabled and bootstrap is the primary gate.")
    if decision["should_submit"]:
        print("\nDecision: Submit candidate is locally justified if this is the highest-priority test.")
    else:
        print("\nDecision: Do not spend LB yet. Keep baseline or move to the next improvement candidate.")
    print(f"\nWrote: {table_path}")
    print(f"Wrote: {decision_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
