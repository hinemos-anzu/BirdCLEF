# Drop-in cell: CV gate for LB-scarce blend-ratio decisions
# Paste/run this after Cell 12 has produced:
#   oof_proto_probs, sed_preds_tr_aligned, Y_FULL_aligned, meta_tr
# It does not submit anything. It prints a conservative local decision for
# w_proto=0.65 vs the current LB-best w_proto=0.60.

import numpy as np
import pandas as pd


def _safe_macro_auc(y_true, y_score):
    """Macro AUC over classes with both positive and negative labels."""
    vals = []
    for j in range(y_true.shape[1]):
        yy = y_true[:, j]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, y_score[:, j]))
    return float(np.mean(vals)) if vals else np.nan


def _rank_cols(p):
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)


def _blend_ranked(r_proto, r_sed, w_proto):
    return (r_proto * float(w_proto) + r_sed * (1.0 - float(w_proto))).astype(np.float32)


def _bootstrap_file_delta(
    r_proto,
    r_sed,
    y_true,
    filenames,
    candidate_w=0.65,
    baseline_w=0.60,
    n_boot=300,
    seed=42,
):
    """Bootstrap candidate-vs-baseline AUC deltas by soundscape file."""
    rng = np.random.default_rng(seed)
    files = np.array(pd.unique(filenames))
    file_to_idx = {f: np.where(filenames == f)[0] for f in files}
    cand = _blend_ranked(r_proto, r_sed, candidate_w)
    base = _blend_ranked(r_proto, r_sed, baseline_w)

    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(files, size=len(files), replace=True)
        idx = np.concatenate([file_to_idx[f] for f in sampled])
        auc_c = _safe_macro_auc(y_true[idx], cand[idx])
        auc_b = _safe_macro_auc(y_true[idx], base[idx])
        if np.isfinite(auc_c) and np.isfinite(auc_b):
            deltas.append(auc_c - auc_b)
    return np.asarray(deltas, dtype=np.float32)


def blend_cv_gate(
    oof_proto,
    oof_sed,
    y_true,
    meta,
    w_grid=None,
    candidate_w=0.65,
    baseline_w=0.60,
    known_lb=None,
    min_lb_gain=0.0010,
    min_boot_prob=0.65,
    n_boot=300,
):
    """
    Conservative gate for deciding whether a scarce LB submit is worth using.

    The SED train predictions are known to be leaky, so raw OOF AUC is not used
    as an absolute LB estimate. Instead this reports:
      1. raw rank-blend CV curve,
      2. candidate-vs-baseline file bootstrap delta,
      3. LB-calibrated proxy anchored on already observed LB points.
    """
    if w_grid is None:
        w_grid = np.round(np.arange(0.30, 0.81, 0.05), 2)
    if known_lb is None:
        known_lb = {1.00: 0.927, 0.50: 0.941, 0.60: 0.946}

    r_proto = _rank_cols(np.clip(oof_proto, 1e-5, 1.0 - 1e-5))
    r_sed = _rank_cols(np.clip(oof_sed, 1e-5, 1.0 - 1e-5))

    rows = []
    for w in w_grid:
        pred = _blend_ranked(r_proto, r_sed, w)
        rows.append({"w_proto": float(w), "w_sed": float(1.0 - w), "cv_auc": _safe_macro_auc(y_true, pred)})
    df = pd.DataFrame(rows)
    base_cv = float(df.loc[np.isclose(df["w_proto"], baseline_w), "cv_auc"].iloc[0])
    df["cv_delta_vs_060"] = df["cv_auc"] - base_cv

    # Calibrate CV deltas to observed LB deltas using the local 0.50 -> 0.60 move.
    cv_050 = float(df.loc[np.isclose(df["w_proto"], 0.50), "cv_auc"].iloc[0])
    cv_060 = base_cv
    lb_050 = float(known_lb[0.50])
    lb_060 = float(known_lb[0.60])
    raw_scale = (lb_060 - lb_050) / max(cv_060 - cv_050, 1e-8)
    # SED train predictions are leaky; cap positive amplification and keep sign.
    scale = float(np.clip(raw_scale, -2.0, 2.0))
    df["lb_proxy"] = lb_060 + (df["cv_auc"] - cv_060) * scale
    df["lb_proxy_delta_vs_060"] = df["lb_proxy"] - lb_060

    filenames = meta["filename"].astype(str).to_numpy()
    boot_delta = _bootstrap_file_delta(
        r_proto,
        r_sed,
        y_true,
        filenames,
        candidate_w=candidate_w,
        baseline_w=baseline_w,
        n_boot=n_boot,
    )
    p_win = float((boot_delta > 0).mean()) if len(boot_delta) else np.nan
    delta_mean = float(np.nanmean(boot_delta)) if len(boot_delta) else np.nan
    ci_low, ci_high = np.nanpercentile(boot_delta, [5, 95]) if len(boot_delta) else (np.nan, np.nan)

    cand_row = df.loc[np.isclose(df["w_proto"], candidate_w)].iloc[0]
    proxy_gain = float(cand_row["lb_proxy_delta_vs_060"])
    should_submit = bool((proxy_gain >= min_lb_gain) and (p_win >= min_boot_prob))

    print("\n" + "=" * 72)
    print("Blend CV Gate: LB回数節約用の仮判定")
    print("=" * 72)
    print(df.to_string(index=False, formatters={
        "w_proto": "{:.2f}".format,
        "w_sed": "{:.2f}".format,
        "cv_auc": "{:.6f}".format,
        "cv_delta_vs_060": "{:+.6f}".format,
        "lb_proxy": "{:.6f}".format,
        "lb_proxy_delta_vs_060": "{:+.6f}".format,
    }))
    print("\nCalibration anchors:", known_lb)
    print(f"CV->LB local scale from 0.50->0.60: raw={raw_scale:.3f}, capped={scale:.3f}")
    print(
        f"Bootstrap candidate {candidate_w:.2f} vs baseline {baseline_w:.2f}: "
        f"mean_delta={delta_mean:+.6f}, p(delta>0)={p_win:.3f}, "
        f"90% CI=[{ci_low:+.6f}, {ci_high:+.6f}]"
    )
    print(
        f"LB proxy gain for w_proto={candidate_w:.2f}: {proxy_gain:+.6f} "
        f"(threshold {min_lb_gain:+.6f})"
    )
    if should_submit:
        print("Decision: Submit candidate is locally justified. Use one LB submit if this is the current highest-priority test.")
    else:
        print("Decision: Do not spend LB yet. Keep w_proto=0.60 or move to the next improvement candidate.")
    print("=" * 72)
    return df, {
        "candidate_w": float(candidate_w),
        "baseline_w": float(baseline_w),
        "candidate_lb_proxy_gain": proxy_gain,
        "bootstrap_mean_delta": delta_mean,
        "bootstrap_p_win": p_win,
        "bootstrap_ci90": (float(ci_low), float(ci_high)),
        "should_submit": should_submit,
    }


# Execute the gate for Submit C by default.
blend_cv_table, blend_cv_decision = blend_cv_gate(
    oof_proto=oof_proto_probs,
    oof_sed=sed_preds_tr_aligned,
    y_true=Y_FULL_aligned,
    meta=meta_tr,
    candidate_w=0.65,
    baseline_w=0.60,
    n_boot=300,
)
