# BirdCLEF 2026 V18 Full Pipeline OOF script
# Generated from user-provided Kaggle notebook text with notebook output logs removed.
# Intended for Codex/repo sync and Kaggle execution.

# ── Cell 0: Install ONNX Runtime + TF 2.20 ────────────────────────────
import subprocess, sys, os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")

def find_wheel(pattern):
    for p in INPUT_ROOT.rglob(pattern):
        return p
    raise FileNotFoundError(pattern)

# Try ONNX first (150x faster than TF SavedModel)
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
    print("ONNX Runtime available ✅")
except ImportError:
    _ONNX_AVAILABLE = False
    print("ONNX not available, falling back to TF")
# ── Cell 1: Mode switch ────────────────────────────────────────────────
MODE = "train"   # ← change to "train" for local CV
 
assert MODE in {"train", "submit"}
print("MODE =", MODE)
# ── Cell 2: Imports & config ───────────────────────────────────────────
import os, re, gc, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
 
import numpy as np
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
N_WINDOWS      = 12          # 12 × 5s = 60s per file
 
CFG = {
    # inference
    "batch_files": 16,

    # local CV
    "oof_n_splits": 5   if MODE == "train" else 3,

    # dry-run
    "dryrun_n_files": 20 if MODE == "train" else 0,

    # train-only flags
    "run_oof": MODE == "train",
    "verbose": MODE == "train",

    # V18 proto_ssm
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
print("✅ V18 CFG loaded")
print(f"  n_epochs={CFG['proto_ssm_train']['n_epochs']}  "
      f"patience={CFG['proto_ssm_train']['patience']}  "
      f"oof_n_splits={CFG['proto_ssm_train']['oof_n_splits']}  "
      f"mlp_max_iter={CFG['mlp_params']['max_iter']}")
 
print("Config ready")
print(f"  run_oof={CFG['run_oof']}  verbose={CFG['verbose']}  dryrun={CFG['dryrun_n_files']}")
# ── Cell 3: Data loading & label parsing ──────────────────────────────
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
# ── Cell 4: Load Perch model (ONNX preferred) ─────────────────────────
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn       = birdclassifier.signatures["serving_default"]

# ONNX session (150x faster)
ONNX_PERCH_PATH = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx")
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("Using ONNX Perch (150x faster)")
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
# ── Cell 4b: Genus proxy logits for unmapped species ──────────────────
import re as _re

# Find which species have no direct Perch mapping
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

# For each unmapped species, find genus-level matches in Perch vocab
proxy_map = {}   # label_idx -> list of bc_indices

unmapped_df = (taxonomy[taxonomy["primary_label"]
               .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])]
               .copy())

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci    = str(row["scientific_name"])
    genus  = sci.split()[0]
    
    # Find all Perch labels from the same genus
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

# Only use proxies for biologically meaningful taxa
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}

print(f"Unmapped species total:        {len(UNMAPPED_POS)}")
print(f"Species with genus proxy:      {len(proxy_map)}")
print(f"Species still without signal:  {len(UNMAPPED_POS) - len(proxy_map)}")
print("\nProxy targets:")
for idx, bc_idxs in list(proxy_map.items())[:8]:
    label = PRIMARY_LABELS[idx]
    cls   = CLASS_NAME_MAP.get(label, "?")
    print(f"  {label:12s} ({cls:10s}) ← {len(bc_idxs)} Perch genus matches")

# ── Cell 5: Perch inference engine (ONNX + multithreaded I/O) ─────────
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
        # Prefetch first batch
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths  = next_paths
            batch_n      = len(batch_paths)
            batch_audio  = [f.result() for f in future_audio]

            # Prefetch next batch immediately
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

            # ── ONNX or TF inference ───────────────────────────────────
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

print("✅ Perch inference engine (ONNX + multithreaded I/O) defined")
# ── Cell 6: Build-or-load Perch training cache ────────────────────────
# Strategy:
#   1. Try external mounted cache (multiple known locations)
#   2. Try local cache in /kaggle/working/cache
#   3. If neither exists, build it on the fly with run_perch()
# Built cache is always saved to /kaggle/working/cache for reuse in the
# same session. For cross-session reuse, publish /kaggle/working/cache as
# this notebook's output and mount it as input next time.

print(f"USE_ONNX = {USE_ONNX}  "
      f"(cache will be built with {'ONNX' if USE_ONNX else 'TF SavedModel'})")

# ── Candidate external cache locations (add your own paths here) ──────
EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
    # add more here if needed
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

# ── Robust npz key resolver ───────────────────────────────────────────
SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]

def _pick_array(arr, candidates, shape_hint_cols):
    """Try known key names first, then shape-based fallback."""
    for k in candidates:
        if k in arr.files:
            return arr[k], k
    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k
    raise KeyError(
        f"None of {candidates} found in npz. Available keys: {arr.files}"
    )

# ── Build cache from scratch if needed ────────────────────────────────
def _build_cache():
    print(f"Building Perch cache from {len(full_files)} fully-labeled "
          f"train_soundscape files…")
    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    missing = [p for p in train_paths if not p.exists()]
    if missing:
        print(f"  WARNING: {len(missing)} files listed but not on disk; skipping")
        train_paths = [p for p in train_paths if p.exists()]

    t0 = time.time()
    meta_built, sc_built, emb_built = run_perch(
        train_paths,
        batch_files=CFG["batch_files"],
        verbose=True,
    )
    print(f"  Perch pass finished in {time.time()-t0:.1f}s  "
          f"scores={sc_built.shape} embs={emb_built.shape}")

    # Save with explicit keys + schema fingerprint
    meta_built.to_parquet(CACHE_META_LOCAL)
    np.savez(
        CACHE_NPZ_LOCAL,
        scores=sc_built.astype(np.float32),
        embs=emb_built.astype(np.float32),
        primary_labels=np.array(PRIMARY_LABELS),
    )
    print(f"  Cache saved to {WORK_DIR}")
    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL

# ── Priority: external > local working > build ────────────────────────
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

# ── Load cache with robust key handling ───────────────────────────────
print("Loading Perch cache from:", CACHE_META.parent)
meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)
print("  npz keys      :", list(_arr.keys()))
print("  parquet cols  :", meta_tr.columns.tolist())

sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)
print(f"  scores ← '{sk}'  shape={sc_tr_raw.shape}")
print(f"  embs   ← '{ek}'  shape={emb_tr_raw.shape}")

sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)

# ── Schema validation: primary_labels, if present ─────────────────────
if "primary_labels" in _arr.files:
    cached_labels = _arr["primary_labels"].tolist()
    if cached_labels != PRIMARY_LABELS:
        print("  WARNING: cached primary_labels differ from current "
              "sample_submission — scores columns may not align!")
    else:
        print("  primary_labels schema OK")

# ── Rebuild row_id in parquet if missing ──────────────────────────────
if "row_id" not in meta_tr.columns:
    print("  row_id missing in parquet — reconstructing")
    if "end_sec" in meta_tr.columns:
        end_sec = meta_tr["end_sec"].astype(int)
    elif "window_idx" in meta_tr.columns:
        end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5
    else:
        # last resort: assume consecutive 12 windows per file in order
        n_files_cache = len(meta_tr) // N_WINDOWS
        end_sec = np.tile(np.arange(5, 65, 5), n_files_cache)
    meta_tr["row_id"] = (
        meta_tr["filename"].str.replace(".ogg", "", regex=False)
        + "_" + end_sec.astype(str)
    )

# ── Align Y_FULL to cache row order ───────────────────────────────────
row_id_to_index = full_rows.set_index("row_id")["index"]
missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)
if missing_rows:
    raise RuntimeError(
        f"Cache contains {len(missing_rows)} row_ids not present in current "
        f"fully-labeled set. Example: {list(missing_rows)[:3]}. "
        f"This usually means the cache was built against a different competition "
        f"data version — rebuild the cache by deleting {CACHE_META_LOCAL} and "
        f"{CACHE_NPZ_LOCAL}, then rerunning this cell."
    )

Y_FULL_aligned = Y_SC[row_id_to_index.loc[meta_tr["row_id"]].to_numpy()]

expected_rows = len(full_files) * N_WINDOWS
if len(meta_tr) != expected_rows:
    print(f"  NOTE: cache has {len(meta_tr)} rows, current full_files implies "
          f"{expected_rows}. Proceeding with cache's own coverage.")

print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  "
      f"Y_FULL_aligned: {Y_FULL_aligned.shape}")
# ── Cell 7: Metric helpers ─────────────────────────────────────────────
def macro_auc(y_true, y_score):
    """
    Exact replica of the competition metric:
    macro-averaged ROC-AUC, skipping classes with no positive labels.
    This is the ONLY number you should track locally.
    """
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
 
 
def honest_oof_auc(scores, Y, meta_df, n_splits=5, label="scores"):
    """
    GroupKFold by filename — files never split across folds.
    This is the only correct way to estimate LB performance locally.
    Leaking a file across train/val inflates AUC by ~0.01–0.03.
    """
    groups = meta_df["filename"].to_numpy()
    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros_like(scores, dtype=np.float32)
 
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores, groups=groups), 1):
        oof[va_idx] = scores[va_idx]
 
    auc = macro_auc(Y, oof)
    print(f"[{label}] honest OOF macro-AUC: {auc:.6f}")
    return auc, oof
# ── Cell 7b: Temporal smoothing helper ─────────────────────────────────
def smooth_predictions(probs, n_windows=12, alpha=0.3):
    """
    For each file's 12 windows, blend each window with its neighbors.
    
    new[t] = (1 - alpha) * old[t] + 0.5*alpha * (old[t-1] + old[t+1])
    
    alpha=0: no smoothing (your current baseline)
    alpha=0.3: moderate smoothing (good starting point)
    
    Shape: (n_files * 12, n_classes) → same shape output
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"
    
    # Reshape to (n_files, 12, 234) so we can work file-by-file
    view = probs.reshape(-1, n_windows, C).copy()
    
    # Shift left and right (with edge padding = repeat boundary)
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)  # t-1
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)  # t+1
    
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)
    
    return smoothed.reshape(N, C)


print("✅ Temporal smoothing helper defined")
# ── Cell 7c: Prior table builder ───────────────────────────────────────
def build_prior_tables(sc_df, Y_labels):
    """
    Build site-level and hour-level species frequency tables.
    
    These answer: "How often is species X observed at site S at hour H?"
    
    We use these as a soft prior: add them to raw Perch logits.
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)  # overall frequency
    
    # ── Site-level frequencies ──────────────────────────────────────────
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    
    for s in site_keys:
        i     = site_to_i[s]
        mask  = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)
    
    # ── Hour-level frequencies ──────────────────────────────────────────
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    
    for h in hour_keys:
        i     = hour_to_i[h]
        mask  = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)
    
    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """
    Add a scaled prior logit to the raw Perch scores.
    
    lambda_prior=0: no effect (your baseline)
    lambda_prior=0.4: moderate influence from location/time
    
    The prior is converted to a logit (log-odds) before adding.
    This is mathematically correct — you add logits, not probabilities.
    """
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()
    
    # Start from global average
    p = np.tile(tables["global_p"], (n, 1))  # (n, 234)
    
    # Override with hour-level estimate (if enough data)
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j   = tables["hour_to_i"][h]
            nh  = tables["hour_n"][j]
            w   = nh / (nh + 8.0)   # shrink toward global if little data
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    
    # Override with site-level estimate (if enough data)
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j   = tables["site_to_i"][s]
            ns  = tables["site_n"][j]
            w   = ns / (ns + 8.0)   # same shrinkage logic
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    
    # Convert prior probability to logit and add
    p      = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    out   += lambda_prior * logit_prior
    
    return out.astype(np.float32)


print("✅ Prior table functions defined")
# ── Cell 7d: File-level confidence scaling ─────────────────────────────
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """
    Scale each window's predictions by how confident the file is overall.
    
    Steps:
    1. For each file, find the top-k highest scores across all 12 windows
    2. Compute their mean → "file confidence"
    3. Multiply every window's scores by (file_confidence ** power)
    
    power=0: no effect (baseline)
    power=0.4: moderate suppression of uncertain files
    
    Why top-k and not max?
    Max is noisy (one lucky spike). Top-2 mean is more robust.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    
    view      = probs.reshape(-1, n_windows, C)       # (n_files, 12, 234)
    sorted_v  = np.sort(view, axis=1)                 # sort across time
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)  # (n_files, 1, 234)
    
    scale  = np.power(top_k_mean, power)              # (n_files, 1, 234)
    scaled = view * scale                             # broadcast across 12 windows
    
    return scaled.reshape(N, C)


print("✅ File-level confidence scaling defined")
# ── Cell 7e: Per-taxon temperature scaling ─────────────────────────────
# Build lookup: which species class are they?
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}   # continuous callers

# Build per-class temperature vector
temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    if cls in TEXTURE_TAXA:
        temperatures[ci] = 0.95   # frogs/insects: slightly sharper
    else:
        temperatures[ci] = 1.10   # birds: slightly softer

n_texture = (temperatures < 1.0).sum()
n_event   = (temperatures > 1.0).sum()
print(f"✅ Temperatures: {n_event} event species (T=1.10), {n_texture} texture species (T=0.95)")
# ── Cell 7f: UPGRADED MLP probe on PCA embeddings ─────────────────────
# CHANGE 1: Larger hidden layers (128,64), PCA 64-dim, max_iter=300
# Expected gain: +0.003–0.006 vs baseline (32,) hidden layer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def build_class_freq_weights(Y, cap=10.0):
    total     = Y.shape[0]
    pos_count = Y.sum(axis=0).astype(np.float32) + 1.0
    freq      = pos_count / total
    weights   = 1.0 / (freq ** 0.5)
    weights   = np.clip(weights, 1.0, cap)
    weights   = weights / weights.mean()
    return weights.astype(np.float32)


def build_sequential_features(scores_col, n_windows=12):
    N = len(scores_col)
    assert N % n_windows == 0
    x     = scores_col.reshape(-1, n_windows)
    prev  = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    next_ = np.concatenate([x[:, 1:], x[:, -1:]], axis=1)
    mean  = np.repeat(x.mean(axis=1), n_windows)
    max_  = np.repeat(x.max(axis=1),  n_windows)
    std   = np.repeat(x.std(axis=1),  n_windows)
    return prev.reshape(-1), next_.reshape(-1), mean, max_, std


def train_mlp_probes(emb, scores_raw, Y, min_pos=5, pca_dim=64, alpha_blend=0.4):
    """
    CHANGE 1: Upgraded MLP probe.
    - pca_dim: 32 → 64  (more embedding information)
    - hidden:  (32,) → (128, 64)  (more capacity)
    - max_iter: 100 → 300  (longer training)
    - min_pos: 8 → 5  (catches more rare species)
    """
    # Step 1: Compress embeddings
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding: {emb.shape} → PCA: {Z.shape}  "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})")

    class_weights = build_class_freq_weights(Y, cap=10.0)

    probe_models = {}
    active = np.where(Y.sum(axis=0) >= min_pos)[0]
    print(f"Training MLP probes for {len(active)} species (>= {min_pos} pos windows)...")

    MAX_ROWS = 3000   # slightly higher budget for (128,64) layers

    for ci in tqdm(active, desc="MLP probes"):
        y = Y[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            continue

        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X = np.hstack([
            Z,
            scores_raw[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])

        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        pos_idx = np.where(y == 1)[0]

        w      = float(class_weights[ci])
        repeat = max(1, int(round(w * n_neg / max(n_pos, 1))))
        repeat = min(repeat, 8)
        if n_pos * repeat + len(y) > MAX_ROWS:
            repeat = max(1, (MAX_ROWS - len(y)) // max(n_pos, 1))

        X_bal = np.vstack([X, np.tile(X[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(n_pos * repeat, dtype=y.dtype)])

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),   # CHANGE 1: was (32,)
            activation="relu",
            max_iter=300,                   # CHANGE 1: was 100
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,            # CHANGE 1: was 10
            random_state=42,
            learning_rate_init=5e-4,        # CHANGE 1: was 1e-3 (lower lr for deeper net)
            alpha=0.005,                    # CHANGE 1: was 0.01
        )
        clf.fit(X_bal, y_bal)
        probe_models[ci] = clf

    print(f"Trained {len(probe_models)} MLP probes")
    return probe_models, scaler, pca, alpha_blend


def apply_mlp_probes(emb_test, scores_test, probe_models, scaler, pca, alpha_blend=0.4):
    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)
    result = scores_test.copy()
    for ci, clf in probe_models.items():
        prev, next_, mean, max_, std = build_sequential_features(scores_test[:, ci])
        X_test = np.hstack([
            Z_test, scores_test[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])
        prob  = clf.predict_proba(X_test)[:, 1].astype(np.float32)
        logit = np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)
        result[:, ci] = (1 - alpha_blend) * scores_test[:, ci] + alpha_blend * logit
    return result

print("✅ CHANGE 1: Upgraded MLP probe (pca_dim=64, hidden=(128,64), max_iter=300, min_pos=5)")
# ── Cell 7f-2: Vectorized MLP probe inference ──────────────────────────
import torch
import torch.nn as nn

class VectorizedMLPProbes(nn.Module):
    """Stacks all per-class MLP weights into a single batched PyTorch model.
    Replaces the slow Python for-loop over probe_models at inference time."""
    def __init__(self, probe_models):
        super().__init__()
        self.valid_classes = sorted(probe_models.keys())
        V = len(self.valid_classes)
        if V == 0:
            self.weights = nn.ParameterList()
            self.biases  = nn.ParameterList()
            self.n_layers = 0
            return

        sample = probe_models[self.valid_classes[0]]
        self.n_layers = len(sample.coefs_)
        self.weights  = nn.ParameterList()
        self.biases   = nn.ParameterList()

        for layer_idx in range(self.n_layers):
            W = np.stack([probe_models[c].coefs_[layer_idx]
                          for c in self.valid_classes], axis=0)       # (V, in, out)
            b = np.stack([probe_models[c].intercepts_[layer_idx]
                          for c in self.valid_classes], axis=0)       # (V, out)
            self.weights.append(nn.Parameter(
                torch.tensor(W, dtype=torch.float32), requires_grad=False))
            self.biases.append(nn.Parameter(
                torch.tensor(b, dtype=torch.float32), requires_grad=False))

    def forward(self, x):
        # x: (V, N, in_dim)
        h = x
        for i in range(self.n_layers):
            h = torch.bmm(h, self.weights[i]) + self.biases[i].unsqueeze(1)
            if i < self.n_layers - 1:
                h = torch.relu(h)
        return h.squeeze(-1)   # (V, N)


def apply_mlp_probes_vectorized(emb_test, scores_test, probe_models,
                                 scaler, pca, alpha_blend=0.4):
    """
    Drop-in replacement for apply_mlp_probes().
    Uses batched PyTorch matrix multiply instead of a Python for-loop —
    ~10-50x faster at inference time.
    """
    if len(probe_models) == 0:
        return scores_test.copy()

    emb_s  = scaler.transform(emb_test)
    Z_test = pca.transform(emb_s).astype(np.float32)

    valid_classes = sorted(probe_models.keys())
    V = len(valid_classes)
    N = len(scores_test)

    # Build sequential features for all classes at once
    raw  = scores_test[:, valid_classes].T          # (V, N)
    n_files = N // N_WINDOWS
    raw_view = raw.reshape(V, n_files, N_WINDOWS)
    prev = np.concatenate([raw_view[:, :, :1], raw_view[:, :, :-1]], axis=2).reshape(V, N)
    nxt  = np.concatenate([raw_view[:, :, 1:], raw_view[:, :, -1:]], axis=2).reshape(V, N)
    mean = np.repeat(raw_view.mean(axis=2), N_WINDOWS, axis=1)
    mx   = np.repeat(raw_view.max(axis=2),  N_WINDOWS, axis=1)
    std  = np.repeat(raw_view.std(axis=2),  N_WINDOWS, axis=1)

    # scalar_feats: (V, N, 6)
    scalar_feats = np.stack([raw, prev, nxt, mean, mx, std], axis=-1).astype(np.float32)

    # Z_test: (N, D) → broadcast to (V, N, D)
    Z_expanded = np.broadcast_to(Z_test, (V, N, Z_test.shape[1]))

    # X_all: (V, N, D+6)
    X_all = np.concatenate(
        [Z_expanded.astype(np.float32), scalar_feats], axis=-1)

    vec_probe = VectorizedMLPProbes(probe_models)
    vec_probe.eval()
    with torch.no_grad():
        preds = vec_probe(torch.tensor(X_all)).numpy()   # (V, N)

    result = scores_test.copy()
    base_valid = scores_test[:, valid_classes]           # (N, V)
    result[:, valid_classes] = (
        (1.0 - alpha_blend) * base_valid +
        alpha_blend * preds.T
    )
    return result

print("✅ Vectorized MLP probe inference defined")
# ── Cell 7f-3: Isotonic Calibration + Per-Class Threshold Optimization ──
# CHANGE 2: Used by top notebooks (a.txt/d.txt), expected +0.004–0.008
# Trains isotonic regression per class on OOF scores to calibrate probs,
# then finds the best F1-threshold per species via grid search.
from sklearn.isotonic import IsotonicRegression

def calibrate_and_optimize_thresholds(oof_probs, Y_FULL, 
                                       threshold_grid=None, n_windows=12):
    """
    CHANGE 2: For each species:
    1. Fit isotonic regression on OOF scores (calibrates overconfident/underconfident classes)
    2. Grid-search F1-optimal threshold over calibrated probs
    Returns: thresholds array of shape (n_classes,)
    """
    if threshold_grid is None:
        threshold_grid = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    n_samples, n_cls = oof_probs.shape
    thresholds = np.full(n_cls, 0.5, dtype=np.float32)
    n_files    = n_samples // n_windows
    file_oof   = oof_probs.reshape(n_files, n_windows, n_cls).max(axis=1)
    file_y     = Y_FULL.reshape(n_files, n_windows, n_cls).max(axis=1)
    
    n_calibrated = 0
    for c in range(n_cls):
        y_true = file_y[:, c]
        y_prob = file_oof[:, c]
        if y_true.sum() < 3:
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(y_prob, y_true)
            y_cal = ir.transform(y_prob)
        except Exception:
            y_cal = y_prob
        
        best_f1, best_t = 0.0, 0.5
        for t in threshold_grid:
            pred = (y_cal >= t).astype(int)
            tp = ((pred==1) & (y_true==1)).sum()
            fp = ((pred==1) & (y_true==0)).sum()
            fn = ((pred==0) & (y_true==1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        n_calibrated += 1
    
    print(f"Calibrated {n_calibrated} classes")
    print(f"Mean threshold: {thresholds.mean():.3f}")
    print(f"Range: [{thresholds.min():.2f}, {thresholds.max():.2f}]")
    return thresholds


def apply_per_class_thresholds(scores, thresholds):
    """
    Sharpens probabilities around the per-class threshold:
    - above threshold → push toward 1
    - below threshold → push toward 0
    """
    C = scores.shape[1]
    assert C == len(thresholds)
    scaled = np.copy(scores)
    for c in range(C):
        t = thresholds[c]
        above = scores[:, c] > t
        scaled[ above, c] = 0.5 + 0.5 * (scores[ above, c] - t) / (1 - t + 1e-8)
        scaled[~above, c] = 0.5 * scores[~above, c] / (t + 1e-8)
    return np.clip(scaled, 0.0, 1.0)

print("✅ CHANGE 2: Isotonic calibration + per-class threshold optimization defined")
# ── Cell 7g: Rank-aware scaling ────────────────────────────────────────
def rank_aware_scaling(probs, n_windows=12, power=0.4):
    """
    CHANGE 6: Scale each window by the file's single peak confidence.

    How it works:
      1. For each file, find the MAX score across all 12 windows (per species)
      2. Raise it to power → scale factor
      3. Multiply every window's score by that scale factor

    Example for one species across 12 windows:
      Confident file:  max=0.90 → scale=0.90^0.4=0.96 → mild boost
      Uncertain file:  max=0.10 → scale=0.10^0.4=0.40 → strong suppression

    How this differs from Change 3 (file_confidence_scale):
      Change 3 uses top-2 MEAN → smoother, less aggressive
      Change 6 uses single MAX  → asks "does ANY window have strong evidence?"

    power=0.0 → no effect (baseline)
    power=0.4 → moderate suppression of uncertain files (recommended start)
    power=1.0 → multiply directly by file max (very aggressive)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    view     = probs.reshape(-1, n_windows, C)              # (n_files, 12, 234)
    file_max = view.max(axis=1, keepdims=True)              # (n_files, 1, 234)

    scale  = np.power(file_max, power)                      # (n_files, 1, 234)
    scaled = view * scale                                   # broadcast to all 12 windows

    return scaled.reshape(N, C)


print("✅ Rank-aware scaling defined")
# ── Cell 7h: Adaptive delta smoothing ─────────────────────────────────
def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """
    CHANGE 7: Smooth uncertain windows toward their neighbors,
    while leaving confident windows almost untouched.

    How it works:
      For each window t:
        conf  = max probability across all 234 species at window t
        alpha = base_alpha * (1 - conf)   ← KEY: adapts to confidence
        new[t] = (1 - alpha) * old[t] + alpha * avg(old[t-1], old[t+1])

    Why alpha adapts to confidence:
      Confident window (max=0.90):
        alpha = 0.20 * (1 - 0.90) = 0.02  → barely smoothed, peak preserved
      Uncertain window (max=0.10):
        alpha = 0.20 * (1 - 0.10) = 0.18  → smoothed more, noise reduced

    This is exactly why your Change 1 hurt (-0.005) but this one should help:
      Change 1 used fixed alpha=0.3 → diluted confident peaks equally
      Change 7 uses adaptive alpha  → protects confident peaks, smooths noise

    base_alpha=0.0  → no smoothing (baseline)
    base_alpha=0.20 → recommended starting point
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    result = probs.copy()
    view   = probs.reshape(-1, n_windows, C)    # (n_files, 12, 234) original
    out    = result.reshape(-1, n_windows, C)   # (n_files, 12, 234) to modify

    for t in range(n_windows):

        # Confidence at this window = max prob across all species
        # Shape: (n_files, 1) — one confidence value per file per window
        conf = view[:, t, :].max(axis=-1, keepdims=True)   # (n_files, 1)

        # Adaptive alpha — low confidence = more smoothing
        alpha = base_alpha * (1.0 - conf)                  # (n_files, 1)

        # Neighbor average with edge padding
        if t == 0:
            # First window: left neighbor = itself
            neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows - 1:
            # Last window: right neighbor = itself
            neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0

        # Blend: confident windows barely change, uncertain ones smooth more
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg

    return result


print("✅ Adaptive delta smoothing defined")
# ── Cell 7i: LightProtoSSM WITH Cross-Attention ────────────────────────
# CHANGE 4: Add cross-attention between SSM layers (matches top notebooks)
# Expected gain: +0.004–0.007 vs no cross-attention

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            d_model, -1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_sz, D, self.d_state)
        ys = []

        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """
    CHANGE 4: LightProtoSSM with cross-attention between SSM layers.
    """

    def __init__(
        self,
        d_input=1536,
        d_model=128,
        d_state=16,
        n_classes=234,
        n_windows=12,
        dropout=0.15,
        n_sites=20,
        meta_dim=16,
        use_cross_attn=True,
        cross_attn_heads=2,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(2)]
        )
        self.ssm_bwd = nn.ModuleList(
            [SelectiveSSM(d_model, d_state) for _ in range(2)]
        )
        self.ssm_merge = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for _ in range(2)]
        )
        self.ssm_norm = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(2)]
        )
        self.drop = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model,
                        num_heads=cross_attn_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(2)
                ]
            )
            self.cross_norm = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(2)]
            )

        self.prototypes = nn.Parameter(
            torch.randn(n_classes, d_model) * 0.02
        )
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(
                        h[mask].mean(0), dim=0
                    )

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat(
                    [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1
                )
            )
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(
            zip(
                self.ssm_fwd,
                self.ssm_bwd,
                self.ssm_merge,
                self.ssm_norm,
            )
        ):
            res = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)

            h = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h = norm(h + res)

            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)

        sim = (
            torch.matmul(h_n, p_n.T)
            * F.softplus(self.proto_temp)
            + self.class_bias[None, None, :]
        )

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(
    emb_full,
    scores_full,
    Y_full,
    meta_full,
    n_epochs=40,
    patience=8,
    lr=1e-3,
    n_sites=20,
    verbose=False,
):
    """Train LightProtoSSM with cross-attention."""

    n_files = len(emb_full) // N_WINDOWS
    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].unique())
    site2i = {s: i + 1 for i, s in enumerate(sites_u)}

    site_ids = np.array(
        [
            min(
                site2i.get(
                    meta_full.loc[
                        meta_full["filename"] == fn, "site"
                    ].iloc[0],
                    0,
                ),
                n_sites - 1,
            )
            for fn in fnames
        ],
        dtype=np.int64,
    )

    hour_ids = np.array(
        [
            int(
                meta_full.loc[
                    meta_full["filename"] == fn, "hour_utc"
                ].iloc[0]
            )
            % 24
            for fn in fnames
        ],
        dtype=np.int64,
    )

    model = LightProtoSSM(
        n_classes=N_CLASSES,
        n_sites=n_sites,
        use_cross_attn=True,
        cross_attn_heads=2,
    )

    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full, dtype=torch.float32),
    )

    print(f"LightProtoSSM params: {model.count_parameters():,}")

    emb_t = torch.tensor(emb_f, dtype=torch.float32)
    log_t = torch.tensor(log_f, dtype=torch.float32)
    lab_t = torch.tensor(lab_f, dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    pos_cnt = lab_t.sum(dim=(0, 1))
    total = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_loss, best_state, wait = float("inf"), None, 0

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    for ep in range(n_epochs):
        model.train()

        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)

        loss = (
            F.binary_cross_entropy_with_logits(
                out, lab_t, pos_weight=pos_weight[None, None, :]
            )
            + 0.15 * F.mse_loss(out, log_t)
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"  Early stop ep {ep+1}")
            break

    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)

    print(f"LightProtoSSM trained — best loss={best_loss:.4f}")
    return model, site2i


print("✅ CHANGE 4: LightProtoSSM with cross-attention (2 heads) defined")
# ── Cell 7i-2: TTA — Circular Shift Test-Time Augmentation ───────────
# CHANGE 3: Average ProtoSSM predictions across 5 time shifts
# Expected gain: +0.003–0.005 on public LB

def run_tta_proto(proto_model, emb_files, sc_files,
                  site_t, hour_t, shifts=[0, 1, -1, 2, -2]):
    """
    CHANGE 3: TTA by circular-shifting 12-window sequences.
    
    For each shift s:
      1. Roll embeddings and perch logits by s windows
      2. Run ProtoSSM → get predictions
      3. Roll predictions back by -s (undo shift)
    
    Finally average all predictions across shifts.
    
    Why this works:
      - ProtoSSM sees temporal context across all 12 windows
      - Different starting points expose different context patterns
      - Averaging over 5 views reduces temporal boundary artifacts
    """
    proto_model.eval()
    all_preds = []
    
    emb_t  = torch.tensor(emb_files, dtype=torch.float32)
    sc_t   = torch.tensor(sc_files,  dtype=torch.float32)
    
    for shift in shifts:
        if shift == 0:
            e_shifted = emb_t
            s_shifted = sc_t
        else:
            e_shifted = torch.roll(emb_t, shift, dims=1)
            s_shifted = torch.roll(sc_t,  shift, dims=1)
        
        with torch.no_grad():
            out = proto_model(
                e_shifted, s_shifted,
                site_ids=site_t, hours=hour_t
            ).numpy()   # (n_files, 12, 234)
        
        if shift != 0:
            out = np.roll(out, -shift, axis=1)  # undo shift
        
        all_preds.append(out)
    
    return np.mean(all_preds, axis=0)  # (n_files, 12, 234)

print("✅ CHANGE 3: TTA with 5 circular shifts defined")
# ── Cell 7j: Residual SSM (second-pass error correction) ──────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSSM(nn.Module):
    """
    Lightweight second-pass model that learns to correct
    systematic errors from the first-pass ensemble.
    
    Input:  embeddings + first-pass scores (concatenated)
    Output: additive correction to first-pass scores
    
    Key design: output head initialized to zero
    so corrections start small and only grow if helpful.
    ~25s training on 59 files.
    """
    def __init__(self, d_input=1536, d_scores=234,
                 d_model=64, d_state=8,
                 n_classes=234, n_windows=12,
                 dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(
            torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)
        # Zero init — corrections start at zero, only grow if helpful
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings-1)),
                 self.hour_emb(hours.clamp(0, 23))], dim=-1))
            h = h + meta.unsqueeze(1)

        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(
            torch.cat([h_f, h_b], dim=-1)))
        h   = self.ssm_norm(h + res)

        return self.output_head(h)   # (B, T, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def train_residual_ssm(emb_full, first_pass_flat, Y_full,
                       site_ids, hour_ids,
                       n_epochs=30, patience=8, lr=1e-3,
                       correction_weight=0.30,
                       verbose=False):
    """
    Train ResidualSSM to predict (Y - sigmoid(first_pass)).
    Returns corrected flat scores (n_rows, n_classes).
    ~20s on CPU.
    """
    n_files    = len(emb_full) // N_WINDOWS
    emb_f      = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f       = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f      = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    # Residual target = label - sigmoid(first_pass)
    fp_prob    = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals  = lab_f - fp_prob   # values in [-1, 1]

    print(f"Residuals: mean={residuals.mean():.4f}  "
          f"std={residuals.std():.4f}  "
          f"abs_mean={np.abs(residuals).mean():.4f}")

    # Train / val split (file level, no shuffle leakage)
    n_val    = max(1, int(n_files * 0.15))
    rng      = torch.Generator(); rng.manual_seed(42)
    perm     = torch.randperm(n_files, generator=rng).numpy()
    val_i    = perm[:n_val];  train_i = perm[n_val:]

    emb_t    = torch.tensor(emb_f,    dtype=torch.float32)
    fp_t     = torch.tensor(fp_f,     dtype=torch.float32)
    res_t    = torch.tensor(residuals, dtype=torch.float32)
    site_t   = torch.tensor(site_ids, dtype=torch.long)
    hour_t   = torch.tensor(hour_ids, dtype=torch.long)

    model    = ResidualSSM(n_classes=N_CLASSES)
    print(f"ResidualSSM params: {model.count_parameters():,}")

    opt      = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i],
                     site_ids=site_t[train_i],
                     hours   =hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i],
                             site_ids=site_t[val_i],
                             hours   =hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss  = val_loss.item()
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose: print(f"  Early stop ep {ep+1}")
            break

    model.load_state_dict(best_state)
    print(f"ResidualSSM trained — best val MSE={best_loss:.6f}")

    # Apply correction to ALL training data (for verification)
    model.eval()
    with torch.no_grad():
        all_corr = model(emb_t, fp_t,
                         site_ids=site_t,
                         hours   =hour_t).numpy()
    print(f"Correction magnitude: "
          f"mean_abs={np.abs(all_corr).mean():.4f}  "
          f"max={np.abs(all_corr).max():.4f}")

    return model, correction_weight


print("✅ ResidualSSM defined (~439K params, ~20s training)")
# ── Cell 8: OOF evaluation (train mode only) ──────────────────────────
baseline_auc = None
oof_raw      = None
 
if CFG["run_oof"]:
    print("Running honest OOF evaluation on training data…")
    baseline_auc, oof_raw = honest_oof_auc(
        sc_tr, Y_FULL_aligned, meta_tr,
        n_splits=CFG["oof_n_splits"],
        label="raw Perch"
    )
    print(f"\nBaseline OOF AUC: {baseline_auc:.6f}  ← your starting point")
else:
    print("Submit mode: skipping OOF evaluation")

# ── Cell 8b: Full Pipeline OOF ─────────────────────────────────────────

def run_pipeline_oof(emb_full, sc_full, Y_full, meta_full, n_splits=5):
    """
    Proper full-pipeline OOF.
    Trains ProtoSSM + MLP on K-1 folds, predicts on held-out fold.
    ~3-4 min total on CPU. Use this instead of the raw-Perch OOF.
    """
    file_meta = (
        meta_full.drop_duplicates("filename")
        .reset_index(drop=True)
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        # Safety checks: GroupKFold must split by filename without leakage.
        assert tr_fnames.isdisjoint(va_fnames), (
            "GroupKFold leakage: train/val filenames overlap"
        )
        assert tr_mask.sum() + va_mask.sum() == len(meta_full), (
            "Train/val masks do not cover all rows"
        )
        assert tr_mask.sum() > 0 and va_mask.sum() > 0, (
            "Empty train or validation fold"
        )

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        # ── Train ProtoSSM on train fold ───────────────────────────────
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            n_epochs=40,
            patience=8,
            lr=1e-3,
            verbose=False,
        )

        # ── ProtoSSM predict on val fold ───────────────────────────────
        n_va = len(emb_va_f) // N_WINDOWS

        va_fn_list = (
            meta_va_f.drop_duplicates("filename")["filename"].tolist()
        )

        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        meta_va_f.loc[
                            meta_va_f["filename"] == fn, "site"
                        ].iloc[0],
                        0,
                    ),
                    19,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        va_hour_ids = np.array(
            [
                int(
                    meta_va_f.loc[
                        meta_va_f["filename"] == fn, "hour_utc"
                    ].iloc[0]
                )
                % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        proto_model.eval()
        with torch.no_grad():
            proto_va = proto_model(
                torch.tensor(
                    emb_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                torch.tensor(
                    sc_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours=torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        # ── Train MLP on train fold ────────────────────────────────────
        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=5,
            pca_dim=64,
            alpha_blend=0.4,
        )

        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_f,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )

        # ── Ensemble + sigmoid ─────────────────────────────────────────
        first_pass = 0.5 * proto_va + 0.5 * sc_va_mlp
        probs_va = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  AUC={fold_auc:.6f}"
        )

    overall = macro_auc(Y_full, oof_probs)
    print(f"\nFull pipeline OOF AUC: {overall:.6f}")
    return overall, oof_probs


if CFG["run_oof"]:
    pipeline_auc, oof_pipeline = run_pipeline_oof(
        emb_tr,
        sc_tr,
        Y_FULL_aligned,
        meta_tr,
        n_splits=5,
    )
    if baseline_auc is not None:
        print(f"raw vs full pipeline delta: {pipeline_auc - baseline_auc:+.6f}")

# ── Cell 9: Test inference ─────────────────────────────────────────────
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
 
if not test_paths:
    n = CFG["dryrun_n_files"] or 20
    print(f"No hidden test — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")
 
meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
print(f"Test scores: {sc_te.shape}")
# ── Cell 10: Full pipeline with ProtoSSM + ResidualSSM ─────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ── Step A: Train LightProtoSSM ────────────────────────────────────────
t0 = time.time()
proto_model, site2i_tr = train_light_proto_ssm(
    emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
    n_epochs=40, patience=8, lr=1e-3, verbose=False)
print(f"ProtoSSM training: {time.time()-t0:.1f}s")

# ── Step B: Run ProtoSSM on TEST ───────────────────────────────────────
n_test_files  = len(sc_te) // N_WINDOWS
emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
n_sites_cap   = 20
test_site_ids = np.array([
    min(site2i_tr.get(
        meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in test_fnames], dtype=np.int64)
test_hour_ids = np.array([
    int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0]) % 24
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

# ── Step C: Prior tables ───────────────────────────────────────────────
prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)

# ── Step D: MLP probes ─────────────────────────────────────────────────
probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned,
    min_pos=5, pca_dim=64, alpha_blend=0.4,
)
sc_te_adjusted = apply_mlp_probes_vectorized(
    emb_te, sc_te_adjusted,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)

# ── Step E: First-pass ensemble (ProtoSSM + MLP) ───────────────────────
ENSEMBLE_W      = 0.5
first_pass_flat = (ENSEMBLE_W * proto_scores_flat
                   + (1.0 - ENSEMBLE_W) * sc_te_adjusted)

# ── Step F: ResidualSSM (second-pass correction) ───────────────────────
# Build training-data first-pass scores for residual training
n_tr_files    = len(sc_tr) // N_WINDOWS
emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)

tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
tr_site_ids   = np.array([
    min(site2i_tr.get(
        meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in tr_fnames], dtype=np.int64)
tr_hour_ids   = np.array([
    int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in tr_fnames], dtype=np.int64)


# Get ProtoSSM scores on training data
# CORRECT — using emb_tr_f, sc_tr_f, tr_site_ids (train data)
proto_tr_out = run_tta_proto(
    proto_model, emb_tr_f, sc_tr_f,
    site_t=torch.tensor(tr_site_ids, dtype=torch.long),
    hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2],
)

proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

# Get MLP scores on training data
sc_tr_prior   = apply_prior(
    sc_tr,
    sites=meta_tr["site"].to_numpy(),
    hours=meta_tr["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)
sc_tr_mlp = apply_mlp_probes_vectorized(
    emb_tr, sc_tr_prior,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)
first_pass_tr = (ENSEMBLE_W * proto_tr_flat
                 + (1.0 - ENSEMBLE_W) * sc_tr_mlp)

train_probs_for_calib = sigmoid(first_pass_tr)
PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
    oof_probs=train_probs_for_calib,
    Y_FULL=Y_FULL_aligned,
    threshold_grid=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    n_windows=N_WINDOWS,
)


# Train ResidualSSM on training errors
t0 = time.time()
res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr,
    first_pass_flat=first_pass_tr,
    Y_full=Y_FULL_aligned,
    site_ids=tr_site_ids,
    hour_ids=tr_hour_ids,
    n_epochs=30,
    patience=8,
    lr=1e-3,
    correction_weight=0.30,
    verbose=False,
)
print(f"ResidualSSM training: {time.time()-t0:.1f}s")

# Apply ResidualSSM correction to TEST scores
first_pass_te_f  = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f,         dtype=torch.float32),
        torch.tensor(first_pass_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()

correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores    = (first_pass_flat
                   + correction_weight * correction_flat)

print(f"Correction applied — "
      f"mean_abs={np.abs(correction_flat).mean():.4f}  "
      f"score range [{final_scores.min():.3f}, {final_scores.max():.3f}]")

# ── Step G: Temperature scaling ────────────────────────────────────────
final_scores = final_scores / temperatures[None, :]

# ── Step H: Sigmoid → probabilities ───────────────────────────────────
probs = sigmoid(final_scores)

# ── Step I: Post-processing pipeline ──────────────────────────────────
probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                               top_k=2,       power=0.4)
probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,
                               power=0.4)
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,
                               base_alpha=0.20)
probs = np.clip(probs, 0.0, 1.0)

probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

# ── Step J: Build submission ───────────────────────────────────────────
sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
assert list(sub.columns) == ["row_id"] + PRIMARY_LABELS
assert len(sub) == len(test_paths) * N_WINDOWS
assert not sub.isna().any().any()
sub.to_csv("submission.csv", index=False)

print(f"\nsubmission.csv saved — shape {sub.shape}")
print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")

# ProtoSSM単体提出を別名でも保存（後段のブレンドセルで参照する）
sub.to_csv("submission_protossm.csv", index=False)
print("submission_protossm.csv も保存しました（SEDブレンド用）")

# ── Cell 11: Distilled SED Branch ─────────────────────────────────────
# Tucker Arrants 公開 ONNX folds を使い mel-spectrogram ベースの予測を生成する。
# テストデータへの推論 → submission_sed.csv
# 訓練データへの推論 → sed_preds_tr_aligned  （Cell 12 のブレンド比率最適化用）
#
# 注意: SED モデルは競技訓練データで蒸留済みのため、訓練データへの適用は
#       厳密な意味でのリークを含む。ただし目的はブレンド比率のキャリブレーション
#       のみであり、最終スコアの絶対値ではなく比率の相対比較に使用する。

import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16_000
TOP_DB_SED = 80

def _find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError("sed_fold0.onnx が見つかりません。")
    return hits[0].parent

def _make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads  = 4
    so.inter_op_num_threads  = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=so, providers=["CPUExecutionProvider"]
    )

def audio_to_mel(chunks):
    """(n_chunks, WINDOW_SAMPLES) → (n_chunks, 1, N_MELS, T) float32"""
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)

def file_to_sed_chunks(path):
    """60 秒音声 → (12, WINDOW_SAMPLES) chunks と秒数ラベル"""
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
    return chunks, ends

def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)

def run_sed_on_paths(audio_paths, sed_sessions_list, desc="SED"):
    """
    SED ONNX folds を指定パスに適用し (row_ids, probs) を返す。
    probs: (n_paths * N_WINDOWS, N_CLASSES) float32
    """
    all_rows, all_preds = [], []
    for path in tqdm(audio_paths, desc=desc):
        chunks, ends = file_to_sed_chunks(path)
        mel   = audio_to_mel(chunks)
        p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)
        for sess in sed_sessions_list:
            outs   = sess.run(None, {sess.get_inputs()[0].name: mel})
            p_sum += 0.5 * sigmoid_sed(outs[0]) + 0.5 * sigmoid_sed(outs[1].max(axis=1))
        p_mean = (p_sum / len(sed_sessions_list)).astype(np.float32)
        if len(p_mean) > 1:
            p_mean = gaussian_filter1d(
                p_mean, sigma=0.65, axis=0, mode="nearest"
            ).astype(np.float32)
        stem = path.stem
        all_rows.extend([f"{stem}_{int(t)}" for t in ends])
        all_preds.append(p_mean)
    preds_arr = np.clip(np.concatenate(all_preds, axis=0), 0.0, 1.0)
    return np.array(all_rows), preds_arr

# ---- SED セッション初期化 -----------------------------------------------
_SED_AVAILABLE = False
sed_sessions   = []
try:
    _sed_dir        = _find_sed_dir()
    _sed_fold_paths = sorted(
        _sed_dir.glob("sed_fold*.onnx"),
        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)),
    )
    sed_sessions    = [_make_sed_session(p) for p in _sed_fold_paths]
    _SED_AVAILABLE  = True
    print(f"SED: {len(sed_sessions)} folds  dir={_sed_dir}")
except FileNotFoundError as _e:
    print(f"SED ONNX が見つかりません: {_e}")
    print("  → SED ブランチをスキップ。ProtoSSM 単体提出を使用します。")

# ---- テストデータへの SED 推論 -------------------------------------------
if _SED_AVAILABLE:
    _test_paths_sed = sorted((BASE / "test_soundscapes").glob("*.ogg"))
    if not _test_paths_sed:
        _n = CFG["dryrun_n_files"] or 20
        _test_paths_sed = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:_n]
        print(f"Dry-run: SED を訓練音声 {len(_test_paths_sed)} ファイルで実行")

    _sed_rows_te, _sed_preds_te = run_sed_on_paths(
        _test_paths_sed, sed_sessions, desc="SED (test)"
    )
    _sed_sub = pd.DataFrame(
        np.clip(_sed_preds_te, 0.0, 1.0), columns=PRIMARY_LABELS
    )
    _sed_sub.insert(0, "row_id", _sed_rows_te)
    _sed_sub.to_csv("submission_sed.csv", index=False)
    print(f"submission_sed.csv 保存完了  shape={_sed_sub.shape}")

    # ---- 訓練データへの SED 推論（ブレンド比率最適化用） --------------------
    _train_paths_sed = sorted(
        (BASE / "train_soundscapes").glob("*.ogg")
    )
    _train_paths_sed = [p for p in _train_paths_sed if p.name in set(full_files)]

    _sed_rows_tr, _sed_preds_tr_raw = run_sed_on_paths(
        _train_paths_sed, sed_sessions, desc="SED (train-OOF)"
    )
    _sed_row_to_idx = {rid: i for i, rid in enumerate(_sed_rows_tr)}
    _tr_row_ids     = meta_tr["row_id"].to_numpy()

    _missing = set(_tr_row_ids) - set(_sed_row_to_idx)
    if _missing:
        raise RuntimeError(
            f"SED予測にないrow_id: {len(_missing)}件 — "
            "full_files に含まれない訓練ファイルが存在します。"
        )
    _aligned_idx          = np.array([_sed_row_to_idx[r] for r in _tr_row_ids])
    sed_preds_tr_aligned  = _sed_preds_tr_raw[_aligned_idx]  # (708, 234)

    _sed_tr_auc = macro_auc(Y_FULL_aligned, sed_preds_tr_aligned)
    print(f"SED (train) AUC: {_sed_tr_auc:.6f}  shape={sed_preds_tr_aligned.shape}")
    print("Cell 11 完了")
else:
    sed_preds_tr_aligned = None
    print("SED 利用不可 — Cell 12 はグリッド探索をスキップします。")

# ── Cell 12: S1 OOF評価 + ブレンド比率最適化 ──────────────────────────────
# 設計方針:
#   ① ProtoSSM フルパイプライン OOF（Prior + ResidualSSM + 後処理込み）
#   ② SED 訓練データ予測（Cell 11 で取得済み）
#   ③ 両者のランク変換後に grid search + 任意で Optuna 精密探索
#   ④ 最適ウェイトを BLEND_W_PROTO / BLEND_W_SED に格納
#
# 挿入位置: Cell 10（ProtoSSM実行）の後、Cell 13（最終ブレンド）の前

# ---------- 12-A: OOF 設定 ------------------------------------------------
_OOF_N_SPLITS  = 3      # 59 ファイル → 3-fold が現実的（各fold ~20 val files）
_OOF_N_EPOCHS  = 40     # ProtoSSM fold 学習エポック
_OOF_PATIENCE  = 8
_RES_N_EPOCHS  = 20     # ResidualSSM fold 学習エポック（軽量化）
_RES_PATIENCE  = 6
_N_SITES_CAP   = 20
_EPS           = 1e-5

# ---------- 12-B: OOF キャッシュ ------------------------------------------
_OOF_CACHE = WORK_DIR / "oof_proto_probs.npy"

def _get_site_hour_ids_for_fold(meta_df, fn_list, site2i,
                                 n_sites_cap=_N_SITES_CAP):
    """ファイルリストからサイト/時刻IDを取得するユーティリティ。"""
    site_ids = np.array([
        min(site2i.get(
            meta_df.loc[meta_df["filename"] == fn, "site"].iloc[0], 0),
            n_sites_cap - 1)
        for fn in fn_list
    ], dtype=np.int64)
    hour_ids = np.array([
        int(meta_df.loc[meta_df["filename"] == fn, "hour_utc"].iloc[0]) % 24
        for fn in fn_list
    ], dtype=np.int64)
    return site_ids, hour_ids

def run_full_pipeline_oof_v2(
    emb_full, sc_full, Y_full, meta_full,
    n_splits=3, n_epochs=40, patience=8,
    res_n_epochs=20, res_patience=6,
):
    """
    GroupKFold OOF で ProtoSSM + MLP + Prior + ResidualSSM +
    後処理（温度スケーリング・FileConfidenceScale・RankAwareScale・
    AdaptiveDeltaSmooth）まで含んだ完全パイプラインを実行する。

    既存 run_pipeline_oof との差分:
      - apply_prior                  追加
      - train_residual_ssm           追加
      - temperature scaling          追加
      - file_confidence_scale        追加
      - rank_aware_scaling           追加
      - adaptive_delta_smooth        追加

    Returns
    -------
    oof_probs : np.ndarray (n_windows, N_CLASSES) float32
    overall_auc : float
    """
    file_meta = meta_full.drop_duplicates("filename").reset_index(drop=True)
    gkf       = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    # Prior テーブルは全訓練データから構築（fold 外で一度だけ）
    _prior = build_prior_tables(sc, Y_SC)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        t_fold = time.time()
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f  = emb_full[tr_mask];  sc_tr_f  = sc_full[tr_mask]
        Y_tr_f    = Y_full[tr_mask];    meta_tr_f = meta_full[tr_mask].reset_index(drop=True)
        emb_va_f  = emb_full[va_mask];  sc_va_f  = sc_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        n_va = len(emb_va_f) // N_WINDOWS
        n_tr = len(emb_tr_f) // N_WINDOWS

        # ---- ProtoSSM 訓練 ------------------------------------------------
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f, sc_tr_f, Y_tr_f, meta_tr_f,
            n_epochs=n_epochs, patience=patience, lr=1e-3, verbose=False,
        )

        # ---- 検証: ProtoSSM 予測 ------------------------------------------
        va_fn_list = meta_va_f.drop_duplicates("filename")["filename"].tolist()
        va_site_ids, va_hour_ids = _get_site_hour_ids_for_fold(
            meta_va_f, va_fn_list, site2i)

        proto_model.eval()
        with torch.no_grad():
            proto_va = proto_model(
                torch.tensor(emb_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                torch.tensor(sc_va_f.reshape(n_va, N_WINDOWS, -1),  dtype=torch.float32),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours   =torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        # ---- 検証: Prior + MLP Probes ------------------------------------
        sc_va_prior = apply_prior(
            sc_va_f,
            sites=meta_va_f["site"].to_numpy(),
            hours=meta_va_f["hour_utc"].to_numpy(),
            tables=_prior, lambda_prior=0.4,
        )
        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb=emb_tr_f, scores_raw=sc_tr_f, Y=Y_tr_f,
            min_pos=5, pca_dim=64, alpha_blend=0.4,
        )
        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f, sc_va_prior,
            probe_models, emb_scaler, emb_pca, alpha_blend,
        )
        first_pass_va = 0.5 * proto_va + 0.5 * sc_va_mlp

        # ---- 訓練: ResidualSSM 用の first_pass を生成 --------------------
        tr_fn_list = meta_tr_f.drop_duplicates("filename")["filename"].tolist()
        tr_site_ids_f, tr_hour_ids_f = _get_site_hour_ids_for_fold(
            meta_tr_f, tr_fn_list, site2i)

        # 訓練側は TTA あり（元パイプラインと同設計）
        proto_tr_tta = run_tta_proto(
            proto_model,
            emb_tr_f.reshape(n_tr, N_WINDOWS, -1),
            sc_tr_f.reshape(n_tr, N_WINDOWS, -1),
            site_t=torch.tensor(tr_site_ids_f, dtype=torch.long),
            hour_t=torch.tensor(tr_hour_ids_f, dtype=torch.long),
            shifts=[0, 1, -1, 2, -2],
        )
        sc_tr_prior_f = apply_prior(
            sc_tr_f,
            sites=meta_tr_f["site"].to_numpy(),
            hours=meta_tr_f["hour_utc"].to_numpy(),
            tables=_prior, lambda_prior=0.4,
        )
        sc_tr_mlp_f = apply_mlp_probes_vectorized(
            emb_tr_f, sc_tr_prior_f,
            probe_models, emb_scaler, emb_pca, alpha_blend,
        )
        first_pass_tr_f = (
            0.5 * proto_tr_tta.reshape(-1, N_CLASSES) + 0.5 * sc_tr_mlp_f
        )

        # ---- ResidualSSM 訓練 & 検証適用 ---------------------------------
        res_model, correction_weight = train_residual_ssm(
            emb_full=emb_tr_f,
            first_pass_flat=first_pass_tr_f,
            Y_full=Y_tr_f,
            site_ids=tr_site_ids_f,
            hour_ids=tr_hour_ids_f,
            n_epochs=res_n_epochs, patience=res_patience,
            lr=1e-3, correction_weight=0.30, verbose=False,
        )

        res_model.eval()
        with torch.no_grad():
            va_correction = res_model(
                torch.tensor(emb_va_f.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                torch.tensor(first_pass_va.reshape(n_va, N_WINDOWS, -1), dtype=torch.float32),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours   =torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        # ---- 後処理（元パイプライン Step G-I と同一） --------------------
        final_va = first_pass_va + correction_weight * va_correction
        final_va = final_va / temperatures[None, :]          # 温度スケーリング
        probs_va = sigmoid(final_va)
        probs_va = file_confidence_scale(probs_va, n_windows=N_WINDOWS, top_k=2, power=0.4)
        probs_va = rank_aware_scaling(probs_va, n_windows=N_WINDOWS, power=0.4)
        probs_va = adaptive_delta_smooth(probs_va, n_windows=N_WINDOWS, base_alpha=0.20)
        probs_va = np.clip(probs_va, 0.0, 1.0).astype(np.float32)

        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(f"  Fold {fold}/{n_splits} | val={len(va_fnames):2d}files "
              f"| AUC={fold_auc:.6f} | {time.time()-t_fold:.1f}s")

        del proto_model, probe_models, res_model
        gc.collect()

    overall_auc = macro_auc(Y_full, oof_probs)
    print(f"\n[ProtoSSM OOF v2] 全体 AUC: {overall_auc:.6f}")
    return oof_probs, overall_auc

# ---- OOF 実行（キャッシュ優先） -----------------------------------------
if _OOF_CACHE.exists():
    oof_proto_probs = np.load(str(_OOF_CACHE))
    _proto_oof_auc  = macro_auc(Y_FULL_aligned, oof_proto_probs)
    print(f"OOF キャッシュを読み込みました: {_OOF_CACHE}")
    print(f"[ProtoSSM OOF v2] AUC = {_proto_oof_auc:.6f}")
else:
    print("="*60)
    print("S1 Step 1: ProtoSSM フルパイプライン OOF 予測を実行中…")
    print(f"  {_OOF_N_SPLITS}-fold GroupKFold | "
          f"n_epochs={_OOF_N_EPOCHS} | patience={_OOF_PATIENCE}")
    print("="*60)
    _t0 = time.time()
    oof_proto_probs, _proto_oof_auc = run_full_pipeline_oof_v2(
        emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
        n_splits=_OOF_N_SPLITS,
        n_epochs=_OOF_N_EPOCHS, patience=_OOF_PATIENCE,
        res_n_epochs=_RES_N_EPOCHS, res_patience=_RES_PATIENCE,
    )
    np.save(str(_OOF_CACHE), oof_proto_probs)
    print(f"OOF 完了: {time.time()-_t0:.1f}s  → キャッシュ保存: {_OOF_CACHE}")

# ---------- 12-C: ブレンド比率最適化 ---------------------------------------
def _to_rank(p):
    """(N, C) 確率配列 → パーセンタイルランク配列（列方向）"""
    return pd.DataFrame(p).rank(axis=0, pct=True).to_numpy(np.float32)

def _blend_auc(rank_a, rank_b, w_a, Y_true):
    return macro_auc(Y_true, rank_a * w_a + rank_b * (1.0 - w_a))

def optimize_blend_ratio(
    oof_proto,          # (n_windows, N_CLASSES) ProtoSSM OOF probs
    oof_sed,            # (n_windows, N_CLASSES) SED probs on train
    Y_true,             # ground truth
    w_grid=None,        # グリッド探索の候補値
    use_optuna=True,
    n_optuna_trials=60,
):
    """
    ランクベース 2-way ブレンドの最適ウェイトを OOF で探索する。

    グリッド探索で大域を確認 → Optuna で精密化（任意）。
    BLEND_W_PROTO / BLEND_W_SED をグローバルに格納して返す。
    """
    if w_grid is None:
        w_grid = np.round(np.arange(0.30, 0.81, 0.05), 2)

    # ランク配列を事前計算（繰り返し呼び出しを高速化）
    r_proto = _to_rank(np.clip(oof_proto, _EPS, 1.0 - _EPS))
    r_sed   = _to_rank(np.clip(oof_sed,   _EPS, 1.0 - _EPS))

    # ---- グリッド探索 -------------------------------------------------------
    print("\n--- グリッド探索 (w_proto: 0.30 → 0.80) ---")
    _ref_60_40 = _blend_auc(r_proto, r_sed, 0.60, Y_true)
    print(f"  [参考] ProtoSSM 単体   : {macro_auc(Y_true, oof_proto):.6f}")
    print(f"  [参考] SED 単体        : {macro_auc(Y_true, oof_sed):.6f}")
    print(f"  [参考] 固定 60/40      : {_ref_60_40:.6f}  ← 現在の設定")
    print()

    _grid = []
    for w in w_grid:
        auc    = _blend_auc(r_proto, r_sed, w, Y_true)
        marker = " ←" if abs(w - 0.60) < 0.01 else ""
        print(f"  w_proto={w:.2f} | AUC={auc:.6f}{marker}")
        _grid.append((w, auc))

    _best_w_grid, _best_auc_grid = max(_grid, key=lambda x: x[1])
    print(f"\n[グリッド] 最良: w_proto={_best_w_grid:.2f} → AUC={_best_auc_grid:.6f} "
          f"(Δ vs 60/40 = {_best_auc_grid - _ref_60_40:+.6f})")

    _best_w = _best_w_grid

    # ---- Optuna 精密探索（任意） -------------------------------------------
    if use_optuna:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                w = trial.suggest_float("w_proto", 0.30, 0.80)
                return _blend_auc(r_proto, r_sed, w, Y_true)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.enqueue_trial({"w_proto": _best_w_grid})  # グリッド最良から開始
            study.optimize(_objective, n_trials=n_optuna_trials,
                           show_progress_bar=True)

            _best_w     = study.best_params["w_proto"]
            _best_auc_o = study.best_value
            print(f"\n[Optuna] 最良: w_proto={_best_w:.4f} → AUC={_best_auc_o:.6f} "
                  f"(Δ vs グリッド = {_best_auc_o - _best_auc_grid:+.6f})")

        except ImportError:
            print("\nOptuna 未インストール → グリッド結果を採用します。")
            print("  インストール: pip install optuna -q")

    return float(_best_w), float(1.0 - _best_w)

# ---- ブレンド比率を決定 ---------------------------------------------------
if _SED_AVAILABLE and sed_preds_tr_aligned is not None:
    print("\n" + "="*60)
    print("S1 Step 2: ブレンド比率最適化")
    print("="*60)
    BLEND_W_PROTO, BLEND_W_SED = optimize_blend_ratio(
        oof_proto=oof_proto_probs,
        oof_sed=sed_preds_tr_aligned,
        Y_true=Y_FULL_aligned,
        use_optuna=True,
        n_optuna_trials=60,
    )
    print(f"\n採用ブレンド比率: w_proto={BLEND_W_PROTO:.4f} / w_sed={BLEND_W_SED:.4f}")
    print("  ※ 旧: w_proto=0.60 / w_sed=0.40 (固定値)")
else:
    BLEND_W_PROTO = 1.0
    BLEND_W_SED   = 0.0
    print("SED 利用不可 — ProtoSSM 単体 (w_proto=1.00) を使用します。")

print("\nCell 12 完了")

# ── Cell 13: 最適ブレンド比率で最終 submission を再生成 ─────────────────────
# submission_protossm.csv  (Cell 10 で生成済み)
# submission_sed.csv       (Cell 11 で生成済み)
# → ランク平均ブレンド → 後処理 → submission.csv 上書き保存
#
# 後処理ロジックはオリジナルノートブックの "Rank Blend & Fat-Tail" セルと同一。

print("="*60)
print(f"S1 Step 3: 最適ブレンドで submission.csv を再生成")
print(f"  w_proto={BLEND_W_PROTO:.4f}  w_sed={BLEND_W_SED:.4f}")
print("="*60)

if not _SED_AVAILABLE:
    # SED が使えない場合は ProtoSSM 単体提出をそのまま使用
    import shutil
    shutil.copy("submission_protossm.csv", "submission.csv")
    print("SED 利用不可 → submission_protossm.csv を submission.csv にコピーしました。")
else:
    _EPS_BLEND = 1e-5

    _df_proto = pd.read_csv("submission_protossm.csv")
    _df_sed   = pd.read_csv("submission_sed.csv")
    _cols     = [c for c in _df_proto.columns if c != "row_id"]

    # row_id 順序を ProtoSSM に揃える
    _df_sed = _df_sed.set_index("row_id").loc[_df_proto["row_id"]].reset_index()

    _p_proto = np.clip(_df_proto[_cols].to_numpy(np.float32), _EPS_BLEND, 1.0 - _EPS_BLEND)
    _p_sed   = np.clip(_df_sed  [_cols].to_numpy(np.float32), _EPS_BLEND, 1.0 - _EPS_BLEND)

    _r_proto = pd.DataFrame(_p_proto).rank(axis=0, pct=True).to_numpy(np.float32)
    _r_sed   = pd.DataFrame(_p_sed  ).rank(axis=0, pct=True).to_numpy(np.float32)

    # ---- 最適比率でランクブレンド -----------------------------------------
    _pred = _r_proto * BLEND_W_PROTO + _r_sed * BLEND_W_SED

    _row_ids  = _df_proto["row_id"].astype(str).to_numpy()
    _file_ids = np.array(["_".join(r.split("_")[:-1]) for r in _row_ids])

    # ---- ノイズ抑制: ProtoSSM > 0.50 かつ SED < 0.05 ----------------------
    _fake_only = (_p_proto > 0.50) & (_p_sed < 0.05)
    _pred = np.where(_fake_only, (1.0 - 0.08) * _pred + 0.08 * _r_proto, _pred)

    # ---- 継続性ゲート: fat-tail t分布カーネル（35秒窓） --------------------
    _offs          = np.arange(-3, 4, dtype=np.float32)
    _proto_kernel  = (1.0 + (_offs / 1.20) ** 2 / 2.0) ** (-1.5)
    _proto_kernel  = (_proto_kernel / _proto_kernel.sum()).astype(np.float32)
    _pa_ctx        = _p_proto.copy()
    for _fid in pd.unique(_file_ids):
        _m = _file_ids == _fid
        _x = _p_proto[_m]
        if len(_x) > 1:
            _xp = np.pad(_x, ((3, 3), (0, 0)), mode="edge")
            _pa_ctx[_m] = sum(_proto_kernel[i] * _xp[i:i + len(_x)] for i in range(7))
    _xctx       = pd.DataFrame(_pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)
    _proto_cont = ((_xctx > 0.88) & (_r_proto > 0.75)
                   & (_p_sed < 0.12) & (~_fake_only))
    _pred = np.where(
        _proto_cont,
        (1.0 - 0.15) * _pred + 0.15 * np.maximum(_r_proto, _xctx),
        _pred,
    )

    # ---- SED スパイク保存: SED 高信頼 + ProtoSSM 低信頼 -----------------
    _sed_only = ((_r_sed > 0.95) & (_r_proto < 0.80)
                 & (~_fake_only) & (~_proto_cont))
    _pred = np.where(
        _sed_only, (1.0 - 0.12) * _pred + 0.12 * _r_sed, _pred
    )

    # ---- 最終 DataFrame 構築 -----------------------------------------------
    _sub = _df_proto.copy()
    _sub[_cols] = _pred.astype(np.float32)

    # ---- ソノタイプミラーリング --------------------------------------------
    _MIRROR_PAIRS = (
        ("47158son15", "47158son16"),
        ("47158son09", "47158son12"),
        ("47158son02", "47158son14"),
        ("47158son13", "47158son21", "47158son22", "47158son23"),
    )
    _col_to_idx = {lbl: i for i, lbl in enumerate(_cols)}
    _mirror_count = 0
    for _grp in _MIRROR_PAIRS:
        _vidx = [_col_to_idx[s] for s in _grp if s in _col_to_idx]
        if len(_vidx) >= 2:
            _gmax = _sub[_cols].iloc[:, _vidx].max(axis=1).to_numpy(np.float32)
            for _idx in _vidx:
                _sub.iloc[:, _idx + 1] = _gmax
            _mirror_count += len(_vidx)
    print(f"ソノタイプミラーリング: {_mirror_count} 列に適用")

    # ---- レア種の低信頼スコアを抑制 ----------------------------------------
    try:
        _tax_df      = pd.read_csv(BASE / "taxonomy.csv").set_index("primary_label")
        _rare_cls    = {"Amphibia", "Mammalia", "Reptilia"}
        _rare_count  = 0
        for _ci, _sp in enumerate(_cols):
            if _sp in _tax_df.index and _tax_df.loc[_sp, "class_name"] in _rare_cls:
                _vals = _sub.iloc[:, _ci + 1].to_numpy(np.float32)
                _thr  = _vals.mean() + 0.05
                _sub.iloc[:, _ci + 1] = np.where(_vals < _thr, _vals * 0.9, _vals)
                _rare_count += 1
        print(f"レア種抑制: {_rare_count} クラスに適用")
    except Exception as _ex:
        print(f"レア種抑制スキップ: {_ex}")

    # ---- ドライラン対応 -----------------------------------------------------
    _test_check = list((BASE / "test_soundscapes").glob("*.ogg"))
    if not _test_check:
        _sample_pub = pd.read_csv(BASE / "sample_submission.csv")
        _tmpl       = _sub[_cols].mean(axis=0).astype(np.float32)
        _sub        = _sample_pub.copy()
        for _lbl in _cols:
            _sub[_lbl] = _tmpl[_lbl]
        print("ドライラン: sample_submission の行順に揃えました。")

    _sub.to_csv("submission.csv", index=False)
    print(f"\nsubmission.csv 保存完了  shape={_sub.shape}")
    print(f"  w_proto={BLEND_W_PROTO:.4f} / w_sed={BLEND_W_SED:.4f}  "
          f"(旧固定値: 0.60 / 0.40)")

print(f"\nS1 OOF Blend Optimization 完了")
print(f"総経過時間: {(time.time() - _WALL_START)/60:.1f} min")

 
 
