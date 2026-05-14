# BirdCLEF+ 2026 — v5: 0.947 Pipeline + BirdNET Gate3b-Only
#
# Change from v4:
#   Base blend is always 60/40 (ProtoSSM/SED) regardless of BirdNET availability.
#   BirdNET is used only as a conservative Gate 3b supplement (threshold 0.97, weight 0.05).
#   This restores the 0.947 base while retaining BirdNET as a safety net for
#   species that both ProtoSSM and SED miss with very high confidence.
#
# Attribution: Vyanktesh Dwivedi (base), Imaad Mahmood (ProtoSSM 0.946),
#              Tucker Arrants (SED), Stefan Kahl et al. (BirdNET),
#              Shadi Akiki (BirdNET Kaggle model)

# ── Cell 1: ProtoSSM pipeline ─────────────────────────────────────────────────
import subprocess, sys, os
from pathlib import Path
import random
import numpy as np
import torch

INPUT_ROOT = Path("/kaggle/input")

def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)

ONNX_WHL = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
if ONNX_WHL.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(ONNX_WHL)], check=True)
    print("ONNX Runtime installed")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorboard-2.20.0-*.whl"))], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                str(find_wheel("tensorflow-2.20.0-*.whl"))], check=True)
print("TF 2.20 installed")

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
    print("ONNX Runtime available")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(4)
print("Global random seed set to 4")

MODE = "submit"
assert MODE in {"train", "submit"}
print("MODE =", MODE)

import os, re, gc, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

tf.experimental.numpy.experimental_enable_numpy_behavior()
try: tf.config.set_visible_devices([], "GPU")
except: pass

_WALL_START = time.time()

BASE      = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
WORK_DIR  = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)

SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12

CFG = {
    "batch_files": 16,
    "oof_n_splits": 5   if MODE == "train" else 3,
    "dryrun_n_files": 20 if MODE == "train" else 0,
    "run_oof": MODE == "train",
    "verbose": MODE == "train",
    "proto_ssm_train": {
        "n_epochs":        80  if MODE == "train" else 40,
        "lr":              8e-4,
        "weight_decay":    1e-3,
        "val_ratio":       0.15,
        "patience":        20  if MODE == "train" else 8,
        "pos_weight_cap":  25.0,
        "distill_weight":  0.15,
        "proto_margin":    0.15,
        "label_smoothing": 0.03,
        "oof_n_splits":    5   if MODE == "train" else 3,
        "mixup_alpha":     0.4,
        "focal_gamma":     2.5,
        "swa_start_frac":  0.65,
        "swa_lr":          4e-4,
        "use_cosine_restart": True,
        "restart_period":  20,
    },
    "residual_ssm": {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.1, "correction_weight": 0.35,
        "n_epochs": 40  if MODE == "train" else 20,
        "lr": 8e-4,
        "patience": 12  if MODE == "train" else 6,
    },
    "mlp_params": {
        "hidden_layer_sizes": (256, 128), "activation": "relu",
        "max_iter": 500  if MODE == "train" else 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20  if MODE == "train" else 10,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
}
print("CFG loaded")

taxonomy          = pd.read_csv(BASE / "taxonomy.csv")
sample_sub        = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_fname(name):
    m = FNAME_RE.match(name)
    if not m: return {"site": "unknown", "hour_utc": -1}
    _, site, _, hms = m.groups()
    return {"site": site, "hour_utc": int(hms[:2])}

def union_labels(series):
    out = set()
    for x in series:
        if pd.notna(x):
            for t in str(x).split(";"):
                t = t.strip()
                if t: out.add(t)
    return sorted(out)

sc = (soundscape_labels
      .groupby(["filename", "start", "end"])["primary_label"]
      .apply(union_labels)
      .reset_index(name="label_list"))

sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
sc["row_id"]  = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

_meta = sc["filename"].apply(parse_fname).apply(pd.Series)
sc = pd.concat([sc, _meta], axis=1)

Y_SC = np.zeros((len(sc), N_CLASSES), dtype=np.uint8)
for i, lbls in enumerate(sc["label_list"]):
    for lbl in lbls:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

windows_per_file = sc.groupby("filename").size()
full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc["fully_labeled"] = sc["filename"].isin(full_files)

full_rows = (sc[sc["fully_labeled"]]
             .sort_values(["filename", "end_sec"])
             .reset_index(drop=False))
Y_FULL = Y_SC[full_rows["index"].to_numpy()]

print(f"Classes: {N_CLASSES} | Fully-labeled files: {len(full_files)}")
print(f"Full-file windows: {len(full_rows)} | Active classes: {int((Y_FULL.sum(0) > 0).sum())}")

birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn       = birdclassifier.signatures["serving_default"]

ONNX_PERCH_PATH = next(INPUT_ROOT.glob("**/perch_v2_no_dft*.onnx"),
                   next(INPUT_ROOT.glob("**/perch_v2*.onnx"), Path("")))
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print(f"Using ONNX Perch: {ONNX_PERCH_PATH.name}")
else:
    print("Using TF SavedModel Perch")

bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
             .reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)

mapping = (taxonomy
           .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")

import re as _re
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

proxy_map = {}
unmapped_df = (taxonomy[taxonomy["primary_label"]
               .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])].copy())

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci    = str(row["scientific_name"])
    genus  = sci.split()[0]
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}

print(f"Unmapped: {len(UNMAPPED_POS)} | Proxy: {len(proxy_map)} | No signal: {len(UNMAPPED_POS)-len(proxy_map)}")

temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    temperatures[ci] = 0.95 if cls in TEXTURE_TAXA else 1.10

import concurrent.futures

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:                      y = y[:FILE_SAMPLES]
    return y

def run_perch(paths, batch_files=16, verbose=True):
    paths  = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS
    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)
    wr  = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]
        for start in itr:
            batch_paths  = next_paths
            batch_n      = len(batch_paths)
            batch_audio  = [f.result() for f in future_audio]
            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]
            x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr
            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS
            if USE_ONNX:
                outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out    = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb    = out["embedding"].numpy().astype(np.float32)
            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb
            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)
            del x, logits, emb, batch_audio
            gc.collect()
    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                             "site": sites, "hour_utc": hours})
    return meta_df, scores, embs

print("Perch inference engine defined")

print(f"USE_ONNX = {USE_ONNX}")

EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]
CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"

def _find_external_cache():
    for d in EXTERNAL_CACHE_DIRS:
        meta = d / "perch_meta.parquet"
        npz  = d / "perch_arrays.npz"
        if meta.exists() and npz.exists():
            return meta, npz
    return None, None

SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]

def _pick_array(arr, candidates, shape_hint_cols):
    for k in candidates:
        if k in arr.files:
            return arr[k], k
    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k
    raise KeyError(f"None of {candidates} found in npz. Available keys: {arr.files}")

def _build_cache():
    print(f"Building Perch cache from {len(full_files)} training files…")
    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    train_paths = [p for p in train_paths if p.exists()]
    t0 = time.time()
    meta_built, sc_built, emb_built = run_perch(train_paths, batch_files=CFG["batch_files"], verbose=True)
    print(f"  Perch pass done in {time.time()-t0:.1f}s  scores={sc_built.shape} embs={emb_built.shape}")
    meta_built.to_parquet(CACHE_META_LOCAL)
    np.savez(CACHE_NPZ_LOCAL, scores=sc_built.astype(np.float32),
             embs=emb_built.astype(np.float32), primary_labels=np.array(PRIMARY_LABELS))
    print(f"  Cache saved to {WORK_DIR}")
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL

ext_meta, ext_npz = _find_external_cache()
if ext_meta is not None:
    CACHE_META, CACHE_NPZ = ext_meta, ext_npz
    print(f"Using external cache: {CACHE_META.parent}")
elif CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
    CACHE_META, CACHE_NPZ = CACHE_META_LOCAL, CACHE_NPZ_LOCAL
    print(f"Using local cache: {WORK_DIR}")
else:
    print("No cache found — building from scratch")
    CACHE_META, CACHE_NPZ = _build_cache()

meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)
sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)
sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)

if "primary_labels" in _arr.files:
    if _arr["primary_labels"].tolist() != PRIMARY_LABELS:
        print("  WARNING: cached primary_labels differ — scores columns may not align!")
    else:
        print("  primary_labels schema OK")

if "row_id" not in meta_tr.columns:
    if "end_sec" in meta_tr.columns:
        end_sec = meta_tr["end_sec"].astype(int)
    elif "window_idx" in meta_tr.columns:
        end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5
    else:
        end_sec = np.tile(np.arange(5, 65, 5), len(meta_tr) // N_WINDOWS)
    meta_tr["row_id"] = (meta_tr["filename"].str.replace(".ogg", "", regex=False)
                         + "_" + end_sec.astype(str))

row_id_to_index = full_rows.set_index("row_id")["index"]
missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)
if missing_rows:
    raise RuntimeError(f"Cache has {len(missing_rows)} row_ids not in labeled set.")

Y_FULL_aligned = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]
print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}")

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")

def smooth_predictions(probs, n_windows=12, alpha=0.3):
    N, C = probs.shape
    assert N % n_windows == 0
    view = probs.reshape(-1, n_windows, C).copy()
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)
    return ((1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)).reshape(N, C)

def build_prior_tables(sc_df, Y_labels):
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    sh_keys = sorted({(str(s), int(h)) for s, h in zip(sc_df["site"].dropna(), sc_df["hour_utc"].dropna())
                      if not pd.isna(s) and not pd.isna(h)})
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (sc_df["site"].astype(str).values == s) & (sc_df["hour_utc"].astype(int).values == h)
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i": sh_to_i,    "sh_p": sh_p,    "sh_n": sh_n,
    }

def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    eps = 1e-4; n = len(scores); out = scores.copy()
    p = np.tile(tables["global_p"], (n, 1))
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j = tables["hour_to_i"][h]; nh = tables["hour_n"][j]; w = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j = tables["site_to_i"][s]; ns = tables["site_n"][j]; w = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    if "sh_to_i" in tables:
        for i, (s, h) in enumerate(zip(sites, hours)):
            key = (str(s), int(h))
            if key in tables["sh_to_i"]:
                j = tables["sh_to_i"][key]; nsh = tables["sh_n"][j]; w = nsh / (nsh + 4.0)
                p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]
    p = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)

def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    N, C = probs.shape
    view      = probs.reshape(-1, n_windows, C)
    sorted_v  = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    return (view * np.power(top_k_mean, power)).reshape(N, C)

def rank_aware_scaling(probs, n_windows=12, power=0.4):
    N, C = probs.shape
    view     = probs.reshape(-1, n_windows, C)
    file_max = view.max(axis=1, keepdims=True)
    return (view * np.power(file_max, power)).reshape(N, C)

def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    N, C = probs.shape
    result = probs.copy(); view = probs.reshape(-1, n_windows, C); out = result.reshape(-1, n_windows, C)
    for t in range(n_windows):
        conf = view[:, t, :].max(axis=-1, keepdims=True); alpha = base_alpha * (1.0 - conf)
        if t == 0:             neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows-1: neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:                  neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return result

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
import torch.nn as nn
import torch.nn.functional as F

def build_class_freq_weights(Y, cap=10.0):
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq = pos_count / Y.shape[0]
    weights = np.clip(1.0 / (freq ** 0.5), 1.0, cap)
    return (weights / weights.mean()).astype(np.float32)

def build_sequential_features(scores_col, n_windows=12):
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std

def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    scaler = StandardScaler(); emb_s = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding: {emb.shape} → PCA: {Z.shape}  (variance retained: {pca.explained_variance_ratio_.sum():.2%})")
    class_weights = build_class_freq_weights(Y, cap=10.0)
    probe_models = {}; active = np.where(Y.sum(axis=0) >= min_pos)[0]; MAX_ROWS = 3000
    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y): continue
        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([Z, scores_raw[:, ci:ci+1], prev[:, None], next_[:, None], mean[:, None], max_[:, None], std[:, None]])
        n_pos = int(y.sum()); n_neg = len(y) - n_pos; pos_idx = np.where(y == 1)[0]
        w = float(class_weights[ci]); repeat = max(1, min(int(round(w * n_neg / max(n_pos, 1))), 8))
        if n_pos * repeat + len(y) > MAX_ROWS: repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))
        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=300,
                            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                            random_state=42, learning_rate_init=5e-4, alpha=0.005)
        clf.fit(X_bal, y_bal); probe_models[ci] = clf
    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend

class VectorizedMLPProbes(nn.Module):
    def __init__(self, probe_models):
        super().__init__(); self.valid_classes = sorted(probe_models.keys()); V = len(self.valid_classes)
        if V == 0: self.weights = nn.ParameterList(); self.biases = nn.ParameterList(); self.n_layers = 0; return
        sample = probe_models[self.valid_classes[0]]; self.n_layers = len(sample.coefs_)
        self.weights = nn.ParameterList(); self.biases = nn.ParameterList()
        for li in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[li] for c in self.valid_classes], axis=0)
            b = np.stack([probe_models[c].intercepts_[li] for c in self.valid_classes], axis=0)
            self.weights.append(nn.Parameter(torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(torch.tensor(b, dtype=torch.float32), requires_grad=False))
    def forward(self, x):
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1: h = torch.relu(h)
        return h.squeeze(-1)

def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    if len(probe_models) == 0: return scores_test.copy()
    Z_test = pca.transform(scaler.transform(emb_test)).astype(np.float32)
    valid_classes = sorted(probe_models.keys()); V = len(valid_classes); N = len(scores_test)
    raw = scores_test[:, valid_classes].T; n_files = N // N_WINDOWS; raw_view = raw.reshape(V, n_files, N_WINDOWS)
    prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(V, N)
    nxt  = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(V, N)
    mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1); mx = np.repeat(raw_view.max(axis=2), N_WINDOWS, axis=1)
    std  = np.repeat(raw_view.std(axis=2),  N_WINDOWS, axis=1)
    scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)
    Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))
    X_all = np.concatenate([Z_expanded.astype(np.float32), scalar_feats], axis=-1)
    vec_probe = VectorizedMLPProbes(probe_models).eval()
    with torch.no_grad(): preds = vec_probe(torch.tensor(X_all)).numpy()
    result = scores_test.copy()
    result[:, valid_classes] = (1.0 - alpha_blend) * scores_test[:, valid_classes] + alpha_blend * preds.T
    return result

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, threshold_grid=None, n_windows=12):
    if threshold_grid is None: threshold_grid = [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]
    n_samples, n_cls = oof_probs.shape; thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files = n_samples // n_windows
    file_oof = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y   = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    n_calibrated = 0
    for c in range(n_cls):
        y_true = file_y[:, c]; y_prob = file_oof[:, c]
        if y_true.sum() < 3: continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip"); ir.fit(y_prob, y_true); y_cal = ir.transform(y_prob)
        except: y_cal = y_prob
        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp=((pred==1)&(y_true==1)).sum(); fp=((pred==1)&(y_true==0)).sum(); fn=((pred==0)&(y_true==1)).sum()
            prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1=2*prec*rec/(prec+rec+1e-8)
            if f1 > best_f1: best_f1,best_t = f1,t
        thresholds[c] = best_t; n_calibrated += 1
    print(f"Calibrated {n_calibrated} classes | Mean threshold: {thresholds.mean():.3f} | Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds

def apply_per_class_thresholds(scores, thresholds):
    C = scores.shape[1]; scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]; above = scores[:, c] > t
        scaled[above, c]  = 0.5 + 0.5 * (scores[above, c]  - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__(); self.d_model=d_model; self.d_state=d_state
        self.in_proj=nn.Linear(d_model,2*d_model,bias=False)
        self.conv1d=nn.Conv1d(d_model,d_model,d_conv,padding=d_conv-1,groups=d_model)
        self.dt_proj=nn.Linear(d_model,d_model,bias=True)
        A=torch.arange(1,d_state+1,dtype=torch.float32).unsqueeze(0).expand(d_model,-1)
        self.A_log=nn.Parameter(torch.log(A)); self.D=nn.Parameter(torch.ones(d_model))
        self.B_proj=nn.Linear(d_model,d_state,bias=False); self.C_proj=nn.Linear(d_model,d_state,bias=False)
        self.out_proj=nn.Linear(d_model,d_model,bias=False)
    def forward(self,x):
        B_sz,T,D=x.shape; xz=self.in_proj(x); x_ssm,z=xz.chunk(2,dim=-1)
        x_conv=F.silu(self.conv1d(x_ssm.transpose(1,2))[:,:,:T].transpose(1,2))
        dt=F.softplus(self.dt_proj(x_conv)); A=-torch.exp(self.A_log)
        B=self.B_proj(x_conv); C=self.C_proj(x_conv)
        h=torch.zeros(B_sz,D,self.d_state,device=x.device); ys=[]
        for t in range(T):
            dA=torch.exp(A[None]*dt[:,t,:,None]); dB=dt[:,t,:,None]*B[:,t,None,:]
            h=h*dA+x[:,t,:,None]*dB; ys.append((h*C[:,t,None,:]).sum(-1))
        return torch.stack(ys,dim=1)+x*self.D[None,None,:]

class LightProtoSSM(nn.Module):
    def __init__(self,d_input=1536,d_model=128,d_state=16,n_classes=234,n_windows=12,
                 dropout=0.15,n_sites=20,meta_dim=16,use_cross_attn=True,cross_attn_heads=2):
        super().__init__(); self.n_classes=n_classes; self.n_windows=n_windows; self.use_cross_attn=use_cross_attn
        self.input_proj=nn.Sequential(nn.Linear(d_input,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
        self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
        self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
        self.meta_proj=nn.Linear(2*meta_dim,d_model)
        self.ssm_fwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
        self.ssm_bwd=nn.ModuleList([SelectiveSSM(d_model,d_state) for _ in range(2)])
        self.ssm_merge=nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(2)])
        self.ssm_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop=nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn=nn.ModuleList([nn.MultiheadAttention(d_model,cross_attn_heads,dropout=dropout,batch_first=True) for _ in range(2)])
            self.cross_norm=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes=nn.Parameter(torch.randn(n_classes,d_model)*0.02)
        self.proto_temp=nn.Parameter(torch.tensor(5.0))
        self.class_bias=nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha=nn.Parameter(torch.zeros(n_classes))
    def init_prototypes(self,emb_tensor,labels_tensor):
        with torch.no_grad():
            h=self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask=labels_tensor[:,c]>0.5
                if mask.sum()>0: self.prototypes.data[c]=F.normalize(h[mask].mean(0),dim=0)
    def forward(self,emb,perch_logits=None,site_ids=None,hours=None):
        B,T,_=emb.shape; h=self.input_proj(emb)+self.pos_enc[:,:T,:]
        if site_ids is not None and hours is not None:
            meta=self.meta_proj(torch.cat([self.site_emb(site_ids),self.hour_emb(hours)],dim=-1))
            h=h+meta[:,None,:]
        for i,(fwd,bwd,merge,norm) in enumerate(zip(self.ssm_fwd,self.ssm_bwd,self.ssm_merge,self.ssm_norm)):
            res=h; hf=fwd(h); hb=bwd(h.flip(1)).flip(1)
            h=self.drop(merge(torch.cat([hf,hb],dim=-1))); h=norm(h+res)
            if self.use_cross_attn:
                attn_out,_=self.cross_attn[i](h,h,h); h=self.cross_norm[i](h+attn_out)
        h_n=F.normalize(h,dim=-1); p_n=F.normalize(self.prototypes,dim=-1)
        sim=torch.matmul(h_n,p_n.T)*F.softplus(self.proto_temp)+self.class_bias[None,None,:]
        if perch_logits is not None:
            alpha=torch.sigmoid(self.fusion_alpha)[None,None,:]
            out=alpha*sim+(1-alpha)*perch_logits
        else: out=sim
        return out

class ResidualSSM(nn.Module):
    def __init__(self,d_input=1536,d_scores=234,d_model=64,d_state=8,n_classes=234,
                 n_windows=12,dropout=0.1,n_sites=20,meta_dim=8):
        super().__init__(); self.n_classes=n_classes
        self.input_proj=nn.Sequential(nn.Linear(d_input+d_scores,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(dropout))
        self.site_emb=nn.Embedding(n_sites,meta_dim); self.hour_emb=nn.Embedding(24,meta_dim)
        self.meta_proj=nn.Linear(2*meta_dim,d_model)
        self.pos_enc=nn.Parameter(torch.randn(1,n_windows,d_model)*0.02)
        self.ssm_fwd=SelectiveSSM(d_model,d_state); self.ssm_bwd=SelectiveSSM(d_model,d_state)
        self.ssm_merge=nn.Linear(2*d_model,d_model); self.ssm_norm=nn.LayerNorm(d_model); self.ssm_drop=nn.Dropout(dropout)
        self.output_head=nn.Linear(d_model,n_classes)
        nn.init.zeros_(self.output_head.weight); nn.init.zeros_(self.output_head.bias)
    def forward(self,emb,first_pass,site_ids=None,hours=None):
        B,T,_=emb.shape; x=torch.cat([emb,first_pass],dim=-1)
        h=self.input_proj(x)+self.pos_enc[:,:T,:]
        if site_ids is not None and hours is not None:
            meta=self.meta_proj(torch.cat([self.site_emb(site_ids.clamp(0,self.site_emb.num_embeddings-1)),
                                            self.hour_emb(hours.clamp(0,23))],dim=-1))
            h=h+meta.unsqueeze(1)
        res=h; hf=self.ssm_fwd(h); hb=self.ssm_bwd(h.flip(1)).flip(1)
        h=self.ssm_drop(self.ssm_merge(torch.cat([hf,hb],dim=-1))); h=self.ssm_norm(h+res)
        return self.output_head(h)

def train_light_proto_ssm(emb_full, scores_full, Y_full, meta_full, n_epochs=40, patience=8, lr=1e-3, n_sites=20, verbose=False):
    n_files=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(n_files,N_WINDOWS,-1)
    log_f=scores_full.reshape(n_files,N_WINDOWS,-1); lab_f=Y_full.reshape(n_files,N_WINDOWS,-1).astype(np.float32)
    fnames=meta_full["filename"].unique(); sites_u=sorted(meta_full["site"].unique())
    site2i={s:i+1 for i,s in enumerate(sites_u)}
    site_ids=np.array([min(site2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0],0),n_sites-1) for fn in fnames],dtype=np.int64)
    hour_ids=np.array([int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in fnames],dtype=np.int64)
    model=LightProtoSSM(n_classes=N_CLASSES,n_sites=n_sites,use_cross_attn=True,cross_attn_heads=2)
    model.init_prototypes(torch.tensor(emb_full,dtype=torch.float32),torch.tensor(Y_full,dtype=torch.float32))
    emb_t=torch.tensor(emb_f,dtype=torch.float32); log_t=torch.tensor(log_f,dtype=torch.float32)
    lab_t=torch.tensor(lab_f,dtype=torch.float32); site_t=torch.tensor(site_ids,dtype=torch.long)
    hour_t=torch.tensor(hour_ids,dtype=torch.long)
    pos_cnt=lab_t.sum(dim=(0,1)); total=lab_t.shape[0]*lab_t.shape[1]
    pos_weight=((total-pos_cnt)/(pos_cnt+1)).clamp(max=25.0)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
    best_loss,best_state,wait=float("inf"),None,0
    swa_model=torch.optim.swa_utils.AveragedModel(model); swa_start=int(n_epochs*0.65)
    swa_sched=torch.optim.swa_utils.SWALR(opt,swa_lr=4e-4)
    for ep in range(n_epochs):
        model.train()
        out=model(emb_t,log_t,site_ids=site_t,hours=hour_t)
        loss=F.binary_cross_entropy_with_logits(out,lab_t,pos_weight=pos_weight[None,None,:])+0.15*F.mse_loss(out,log_t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep>=swa_start: swa_model.update_parameters(model); swa_sched.step()
        else: sched.step()
        if loss.item()<best_loss:
            best_loss=loss.item(); best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    if ep>=swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0),swa_model); model=swa_model
    else: model.load_state_dict(best_state)
    model.eval(); return model,site2i

def run_tta_proto(proto_model, emb_files, sc_files, site_t, hour_t, shifts=[0,1,-1,2,-2]):
    proto_model.eval(); all_preds=[]
    emb_t=torch.tensor(emb_files,dtype=torch.float32); sc_t=torch.tensor(sc_files,dtype=torch.float32)
    for shift in shifts:
        e=torch.roll(emb_t,shift,dims=1) if shift else emb_t
        s=torch.roll(sc_t,shift,dims=1) if shift else sc_t
        with torch.no_grad():
            out=proto_model(e,s,site_ids=site_t,hours=hour_t).numpy()
        if shift: out=np.roll(out,-shift,axis=1)
        all_preds.append(out)
    return np.mean(all_preds,axis=0)

def train_residual_ssm(emb_full, first_pass_flat, Y_full, site_ids, hour_ids,
                        n_epochs=30, patience=8, lr=1e-3, correction_weight=0.30, verbose=False):
    n_files=len(emb_full)//N_WINDOWS; emb_f=emb_full.reshape(n_files,N_WINDOWS,-1)
    fp_f=first_pass_flat.reshape(n_files,N_WINDOWS,-1); lab_f=Y_full.reshape(n_files,N_WINDOWS,-1).astype(np.float32)
    fp_prob=1.0/(1.0+np.exp(-np.clip(fp_f,-30,30))); residuals=lab_f-fp_prob
    n_val=max(1,int(n_files*0.15)); rng=torch.Generator(); rng.manual_seed(42)
    perm=torch.randperm(n_files,generator=rng).numpy(); val_i=perm[:n_val]; train_i=perm[n_val:]
    emb_t=torch.tensor(emb_f,dtype=torch.float32); fp_t=torch.tensor(fp_f,dtype=torch.float32)
    res_t=torch.tensor(residuals,dtype=torch.float32)
    site_t=torch.tensor(site_ids,dtype=torch.long); hour_t=torch.tensor(hour_ids,dtype=torch.long)
    model=ResidualSSM(n_classes=N_CLASSES)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,epochs=n_epochs,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
    best_loss,best_state,wait=float("inf"),None,0
    for ep in range(n_epochs):
        model.train()
        corr=model(emb_t[train_i],fp_t[train_i],site_ids=site_t[train_i],hours=hour_t[train_i])
        loss=F.mse_loss(corr,res_t[train_i])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            val_corr=model(emb_t[val_i],fp_t[val_i],site_ids=site_t[val_i],hours=hour_t[val_i])
            val_loss=F.mse_loss(val_corr,res_t[val_i])
        if val_loss.item()<best_loss:
            best_loss=val_loss.item(); best_state={k:v.clone() for k,v in model.state_dict().items()}; wait=0
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(best_state); return model,correction_weight

print("Sequence Models defined")

test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0
if IS_DRY_RUN:
    n = CFG["dryrun_n_files"] or 20
    print(f"No hidden test — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")

meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
print(f"Test scores: {sc_te.shape}")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

t0 = time.time()
proto_model, site2i_tr = train_light_proto_ssm(
    emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
    n_epochs=40, patience=8, lr=1e-3, verbose=False)
print(f"ProtoSSM training: {time.time()-t0:.1f}s")

n_test_files  = len(sc_te) // N_WINDOWS
emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
n_sites_cap   = 20
test_site_ids = np.array([min(site2i_tr.get(meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0],0),n_sites_cap-1)
                           for fn in test_fnames], dtype=np.int64)
test_hour_ids = np.array([int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0])%24
                           for fn in test_fnames], dtype=np.int64)

proto_model.eval()
with torch.no_grad():
    proto_out = proto_model(
        torch.tensor(emb_te_f, dtype=torch.float32),
        torch.tensor(sc_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(sc_te, sites=meta_te["site"].to_numpy(),
                              hours=meta_te["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.4)

probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned, min_pos=5, pca_dim=64, alpha_blend=0.4)
sc_te_adjusted = apply_mlp_probes_vectorized(emb_te, sc_te_adjusted, probe_models, emb_scaler, emb_pca, alpha_blend)

ENSEMBLE_W      = 0.5
first_pass_flat = (ENSEMBLE_W * proto_scores_flat + (1.0 - ENSEMBLE_W) * sc_te_adjusted)

n_tr_files    = len(sc_tr) // N_WINDOWS
emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)

tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
tr_site_ids   = np.array([min(site2i_tr.get(meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0],0),n_sites_cap-1)
                           for fn in tr_fnames], dtype=np.int64)
tr_hour_ids   = np.array([int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0])%24
                           for fn in tr_fnames], dtype=np.int64)

proto_tr_out = run_tta_proto(proto_model, emb_tr_f, sc_tr_f,
    site_t=torch.tensor(tr_site_ids, dtype=torch.long),
    hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2])
proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

sc_tr_prior = apply_prior(sc_tr, sites=meta_tr["site"].to_numpy(),
                           hours=meta_tr["hour_utc"].to_numpy(), tables=prior_tables, lambda_prior=0.4)
sc_tr_mlp = apply_mlp_probes_vectorized(emb_tr, sc_tr_prior, probe_models, emb_scaler, emb_pca, alpha_blend)
first_pass_tr = (ENSEMBLE_W * proto_tr_flat + (1.0 - ENSEMBLE_W) * sc_tr_mlp)

train_probs_for_calib = sigmoid(first_pass_tr)
PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
    oof_probs=train_probs_for_calib, Y_FULL=Y_FULL_aligned,
    threshold_grid=[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70], n_windows=N_WINDOWS)

t0 = time.time()
res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr, first_pass_flat=first_pass_tr, Y_full=Y_FULL_aligned,
    site_ids=tr_site_ids, hour_ids=tr_hour_ids, n_epochs=30, patience=8, lr=1e-3,
    correction_weight=0.30, verbose=False)
print(f"ResidualSSM training: {time.time()-t0:.1f}s")

first_pass_te_f = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f,         dtype=torch.float32),
        torch.tensor(first_pass_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()
correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores    = first_pass_flat + correction_weight * correction_flat
final_scores    = final_scores / temperatures[None, :]
probs = sigmoid(final_scores)
probs = file_confidence_scale(probs, n_windows=N_WINDOWS, top_k=2, power=0.4)
probs = rank_aware_scaling(probs,    n_windows=N_WINDOWS, power=0.4)
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS, base_alpha=0.20)
probs = np.clip(probs, 0.0, 1.0)
probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
sub.to_csv("submission_protossm.csv", index=False)
print("ProtoSSM execution complete")
print(f"Total wall time so far: {(time.time() - _WALL_START)/60:.1f} min")
del emb_tr_f, sc_tr_f, proto_model, res_model
gc.collect()
print("Memory freed. Ready for SED cell.")

# ── Cell 2: Distilled SED ─────────────────────────────────────────────────────
import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80

def find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError("sed_fold0.onnx not found. Attach tuckerarrants/bc2026-distilled-sed-public.")
    return hits[0].parent

def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

def audio_to_mel(chunks):
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
                                            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def file_to_sed_chunks(path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if sr0 != SR: y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    if len(y) < n: y = np.pad(y, (0, n - len(y)))
    else:          y = y[:n]
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
    return chunks, ends

def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
IS_DRY_RUN = len(test_paths) == 0
if IS_DRY_RUN:
    dry_n = CFG["dryrun_n_files"] if "CFG" in dir() else 20
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:(dry_n or 20)]

sed_dir = find_sed_dir()
sed_fold_paths = sorted(sed_dir.glob("sed_fold*.onnx"),
                         key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")

sed_rows, sed_preds = [], []

for i, path in enumerate(test_paths, 1):
    chunks, ends = file_to_sed_chunks(path)
    mel = audio_to_mel(chunks)
    p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)

    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})
        clip_logits = outs[0]
        frame_max   = outs[1].max(axis=1)
        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)

    p_mean = p_sum / len(sed_sessions)

    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)

    stem = path.stem
    sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
    sed_preds.append(p_mean)

    if i == 1 or i % 50 == 0 or i == len(test_paths):
        print(f"SED: {i}/{len(test_paths)}")

sed_preds_arr = np.concatenate(sed_preds, axis=0)
sed_sub = pd.DataFrame(np.clip(sed_preds_arr, 0.0, 1.0), columns=PRIMARY_LABELS)
sed_sub.insert(0, "row_id", sed_rows)
sed_sub.to_csv("submission_sed.csv", index=False)
print(f"Distilled SED Processing Complete. Shape: {sed_sub.shape}")

# ── Cell 3: BirdNET v2.4 (third branch, graceful fallback) ───────────────────
import librosa as _librosa
from scipy.ndimage import gaussian_filter1d as _gf1d

BIRDNET_SR = 48_000
BIRDNET_CHUNK_SEC = 3
BIRDNET_CHUNK_SAMPLES = BIRDNET_SR * BIRDNET_CHUNK_SEC  # 144000

def _find_birdnet_model():
    for pat in ["**/birdnet_global_6k_v2.4_model_fp32*.tflite",
                "**/BirdNET_GLOBAL_6K_V2.4_Model_FP32*.tflite",
                "**/*birdnet*fp32*.tflite",
                "**/*birdnet*.tflite"]:
        hits = sorted(Path("/kaggle/input").rglob(pat))
        if hits:
            print(f"  Found BirdNET model: {hits[0].name}")
            return hits[0]
    return None

def _find_birdnet_labels():
    for pat in ["**/BirdNET_GLOBAL_6K_V2.4_Labels.txt",
                "**/birdnet*labels*.txt",
                "**/*birdnet*label*.txt"]:
        hits = sorted(Path("/kaggle/input").rglob(pat))
        if hits:
            print(f"  Found BirdNET labels: {hits[0].name}")
            return hits[0]
    return None

_bn_model_path  = _find_birdnet_model()
_bn_labels_path = _find_birdnet_labels()

if _bn_model_path is None:
    USE_BIRDNET = False
    print("BirdNET model not found — Gate 3b inactive")
    BN_TO_COMP = {}
    BN_PROXY   = {}
else:
    USE_BIRDNET = True
    try:
        from tflite_runtime.interpreter import Interpreter as _TFLiteInterp
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter as _TFLiteInterp

    _bn_interp = _TFLiteInterp(model_path=str(_bn_model_path), num_threads=4)
    _bn_interp.allocate_tensors()
    _bn_in  = _bn_interp.get_input_details()[0]
    _bn_out = _bn_interp.get_output_details()
    _bn_logit_idx = _bn_out[-1]["index"]
    print(f"  BirdNET input:  {_bn_in['shape']}")
    print(f"  BirdNET output: {_bn_out[-1]['shape']}")

    if _bn_labels_path:
        _bn_labels_raw = [l.strip() for l in _bn_labels_path.read_text().splitlines() if l.strip()]
    else:
        _bn_labels_raw = []
        print("  WARNING: BirdNET labels not found — species mapping will be empty")

    _bn_sci = [lbl.split("_", 1)[0].strip() for lbl in _bn_labels_raw]

    _tax_sci = taxonomy.set_index("scientific_name")["primary_label"].to_dict()
    BN_TO_COMP = {}
    for bn_i, sci in enumerate(_bn_sci):
        if sci in _tax_sci and _tax_sci[sci] in label_to_idx:
            BN_TO_COMP[bn_i] = label_to_idx[_tax_sci[sci]]

    _mapped_comp = set(BN_TO_COMP.values())
    BN_PROXY = {}
    for ci, primary in enumerate(PRIMARY_LABELS):
        if ci in _mapped_comp:
            continue
        row = taxonomy[taxonomy["primary_label"] == primary]
        if row.empty:
            continue
        genus = str(row.iloc[0]["scientific_name"]).split()[0]
        idxs = [i for i, s in enumerate(_bn_sci) if s.startswith(genus + " ")]
        if idxs:
            BN_PROXY[ci] = idxs

    n_mapped = len(set(BN_TO_COMP.values()) | set(BN_PROXY.keys()))
    print(f"  BirdNET→competition: {len(BN_TO_COMP)} direct + {len(BN_PROXY)} genus-proxy = {n_mapped}/{N_CLASSES} classes")

_N_BN_CHUNKS = 20
_win_to_chunks = []
for w in range(N_WINDOWS):
    ws, we = w * 5, (w + 1) * 5
    _win_to_chunks.append([j for j in range(_N_BN_CHUNKS)
                            if 3*j < we and 3*(j+1) > ws])

def run_birdnet(paths, verbose=True):
    if not USE_BIRDNET or not _bn_labels_raw:
        return None, None

    paths = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS
    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    wr = 0
    itr = tqdm(paths, desc="BirdNET") if verbose else paths

    for path in itr:
        y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        if sr0 != BIRDNET_SR:
            y = _librosa.resample(y, orig_sr=sr0, target_sr=BIRDNET_SR)
        tgt = 60 * BIRDNET_SR
        if len(y) < tgt: y = np.pad(y, (0, tgt - len(y)))
        else:             y = y[:tgt]

        chunks = y.reshape(_N_BN_CHUNKS, BIRDNET_CHUNK_SAMPLES)
        chunk_probs = np.zeros((_N_BN_CHUNKS, len(_bn_labels_raw)), dtype=np.float32)
        for j, chunk in enumerate(chunks):
            _bn_interp.set_tensor(_bn_in["index"], chunk[None, :].astype(np.float32))
            _bn_interp.invoke()
            logits = _bn_interp.get_tensor(_bn_logit_idx)[0]
            chunk_probs[j] = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        stem = path.stem
        for w, clist in enumerate(_win_to_chunks):
            wp = chunk_probs[clist].max(axis=0)
            r = wr + w
            row_ids[r]   = f"{stem}_{(w+1)*5}"
            filenames[r] = path.name
            for bn_i, ci in BN_TO_COMP.items():
                if wp[bn_i] > scores[r, ci]:
                    scores[r, ci] = wp[bn_i]
            for ci, bn_idxs in BN_PROXY.items():
                v = wp[bn_idxs].max()
                if v > scores[r, ci]:
                    scores[r, ci] = v
        wr += N_WINDOWS

    meta_df = pd.DataFrame({"row_id": row_ids[:wr], "filename": filenames[:wr]})
    return meta_df, scores[:wr]

print("BirdNET inference engine defined")

_test_paths_bn = sorted((BASE / "test_soundscapes").glob("*.ogg"))
if len(_test_paths_bn) == 0:
    _dry_n = 20
    _test_paths_bn = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:_dry_n]

if USE_BIRDNET:
    t0 = time.time()
    _meta_bn, _scores_bn = run_birdnet(_test_paths_bn, verbose=True)
    print(f"BirdNET inference: {time.time()-t0:.1f}s  shape={_scores_bn.shape}")

    _scores_bn_v = _scores_bn.reshape(len(_scores_bn)//N_WINDOWS, N_WINDOWS, N_CLASSES)
    for fi in range(len(_scores_bn_v)):
        _scores_bn_v[fi] = _gf1d(_scores_bn_v[fi], sigma=0.65, axis=0, mode="nearest")
    _scores_bn = _scores_bn_v.reshape(-1, N_CLASSES)

    _bn_sub = pd.DataFrame(np.clip(_scores_bn, 0.0, 1.0), columns=PRIMARY_LABELS)
    _bn_sub.insert(0, "row_id", _meta_bn["row_id"].values)
    _bn_sub.to_csv("submission_birdnet.csv", index=False)
    covered = (_scores_bn > 0.01).any(axis=0).sum()
    print(f"BirdNET saved. Coverage: {covered}/{N_CLASSES} classes")
else:
    _dummy = pd.read_csv("submission_protossm.csv")
    for c in PRIMARY_LABELS: _dummy[c] = 0.0
    _dummy.to_csv("submission_birdnet.csv", index=False)
    print("BirdNET unavailable — zero submission saved")

# ── Cell 4: Blend + post-processing (v5: 60/40 fixed, BirdNET = Gate 3b only) ─
import os
import numpy as np
import pandas as pd
from pathlib import Path

PROTOSSM_CSV = "submission_protossm.csv"
SED_CSV      = "submission_sed.csv"
OUT_CSV      = "submission.csv"
EPS = 1e-5

df_proto = pd.read_csv(PROTOSSM_CSV)
df_sed   = pd.read_csv(SED_CSV)

cols = [c for c in df_proto.columns if c != "row_id"]

df_sed = df_sed.set_index("row_id").loc[df_proto["row_id"]].reset_index()
p_proto = np.clip(df_proto[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
p_sed   = np.clip(df_sed[cols].to_numpy(np.float32),   EPS, 1.0 - EPS)

rank_proto = pd.DataFrame(p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
rank_sed   = pd.DataFrame(p_sed).rank(axis=0, pct=True).to_numpy(np.float32)

# ── BirdNET loaded for future use but Gate 3b disabled to match 0.947 exactly ─
rank_birdnet = None  # Gate 3b disabled: BirdNET scores not used in blend
print("BirdNET blend disabled — Gate 3b inactive (matching original 0.947)")

# ── Base blend: always 60/40 (original 0.947 setting) ─────────────────────────
print("Executing standard 2-way rank blend (60% Proto / 40% SED)...")
pred = (rank_proto * 0.60) + (rank_sed * 0.40)

row_ids  = df_proto["row_id"].astype(str).to_numpy()
file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

# ── Gate 1: Noise suppression ──────────────────────────────────────────────────
fake_only = (p_proto > 0.50) & (p_sed < 0.05)
pred = np.where(fake_only, (1.0 - 0.08) * pred + 0.08 * rank_proto, pred)

# ── Gate 2: Temporal continuity (fat-tailed t-distribution kernel) ─────────────
offs = np.arange(-3, 4, dtype=np.float32)
proto_kernel = (1.0 + (offs / 1.20) ** 2 / 2.0) ** (-1.5)
proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

pa_ctx = p_proto.copy()
for fid in pd.unique(file_ids):
    m  = file_ids == fid
    x  = p_proto[m]
    if len(x) > 1:
        xp = np.pad(x, ((3, 3), (0, 0)), mode="edge")
        pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(7))

xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
proto_cont = (xctx > 0.88) & (rank_proto > 0.75) & (p_sed < 0.12) & (~fake_only)
pred = np.where(proto_cont,
                (1.0 - 0.15) * pred + 0.15 * np.maximum(rank_proto, xctx),
                pred)

# ── Gate 3: SED spike preservation ────────────────────────────────────────────
sed_only = (rank_sed > 0.95) & (rank_proto < 0.80) & (~fake_only) & (~proto_cont)
pred = np.where(sed_only, (1.0 - 0.12) * pred + 0.12 * rank_sed, pred)

# ── Gate 3b: disabled in v5 (rank_birdnet = None) ────────────────────────────
# Re-enable by setting rank_birdnet above and tuning threshold/weight.

sub = df_proto.copy()
sub[cols] = pred.astype(np.float32)

# ── Gate 4: Sonotype mirroring ────────────────────────────────────────────────
MIRROR_PAIRS = (
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23")
)
col_to_idx = {l: i for i, l in enumerate(cols)}

mirror_count = 0
for group in MIRROR_PAIRS:
    valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
    if len(valid_idx) >= 2:
        group_max = sub[cols].iloc[:, valid_idx].max(axis=1).to_numpy(np.float32)
        for idx in valid_idx:
            sub.iloc[:, idx + 1] = group_max
        mirror_count += len(valid_idx)
print(f"Sonotype mirroring applied to {mirror_count} columns.")

# ── Gate 5: Adaptive rare-class thresholding ──────────────────────────────────
BASE_PATH = Path("/kaggle/input/competitions/birdclef-2026")
try:
    tax_df = pd.read_csv(BASE_PATH / "taxonomy.csv").set_index("primary_label")
    rare_classes = {"Amphibia", "Mammalia", "Reptilia"}
    rare_count = 0
    for ci, species in enumerate(cols):
        if species in tax_df.index and tax_df.loc[species, "class_name"] in rare_classes:
            col_idx = ci + 1
            vals = sub.iloc[:, col_idx].to_numpy(np.float32)
            thr = vals.mean() + 0.05
            sub.iloc[:, col_idx] = np.where(vals < thr, vals * 0.9, vals)
            rare_count += 1
    print(f"Adaptive thresholding applied to {rare_count} rare species.")
except Exception as e:
    print(f"Adaptive thresholding skipped: {e}")

# ── Dry-run alignment ──────────────────────────────────────────────────────────
test_paths = list(BASE_PATH.glob("test_soundscapes/*.ogg"))
IS_DRY_RUN = len(test_paths) == 0
if IS_DRY_RUN:
    print("Dry-run detected: Aligning rows with sample_submission.csv")
    sample_public = pd.read_csv(BASE_PATH / "sample_submission.csv")
    template = sub[cols].mean(axis=0).astype(np.float32)
    sub = sample_public.copy()
    for label in cols:
        sub[label] = template[label]

sub.to_csv(OUT_CSV, index=False)
print(f"Blend and post-processing complete. Saved {OUT_CSV} shape={sub.shape}")
print("Ready for submission!")
