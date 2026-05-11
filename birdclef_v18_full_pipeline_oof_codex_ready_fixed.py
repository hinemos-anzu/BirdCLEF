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
        "n_epochs":        80  if MODE == "train" else 60,
        "lr":              8e-4,
        "weight_decay":    1e-3,
        "val_ratio":       0.15,
        "patience":        20  if MODE == "train" else 12,
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

# ── Partial-file detection (files with 1 ≤ labeled windows < 12) ──────────
partial_files = sorted(
    windows_per_file[
        (windows_per_file > 0) & (windows_per_file < N_WINDOWS)
    ].index.tolist()
)
print(f"Partial files (≥1 labeled window): {len(partial_files)} "
      f"— will supplement MLP probe training")
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
PROXY_TAXA = {"Amphibia", "Insecta", "Aves", "Mammalia", "Reptilia"}
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

# ── Supplement cache with partial-file embeddings (for MLP training) ────────
# Partial files contribute labeled windows without affecting ProtoSSM/ResidualSSM
# (those still train on full files only).
PARTIAL_CACHE_META_P = WORK_DIR / "perch_meta_partial.parquet"
PARTIAL_CACHE_NPZ_P  = WORK_DIR / "perch_arrays_partial.npz"

def _load_or_build_partial_cache(files_list):
    paths = [BASE / "train_soundscapes" / fn for fn in files_list
             if (BASE / "train_soundscapes" / fn).exists()]
    if not paths:
        return None, None, None
    if PARTIAL_CACHE_META_P.exists() and PARTIAL_CACHE_NPZ_P.exists():
        print(f"  Loading partial-file Perch cache ({len(paths)} files)")
        meta_p = pd.read_parquet(PARTIAL_CACHE_META_P)
        _ap    = np.load(PARTIAL_CACHE_NPZ_P)
        return meta_p, _ap["scores"].astype(np.float32), _ap["embs"].astype(np.float32)
    print(f"  Building partial-file Perch cache from {len(paths)} files…")
    meta_p, sc_p, emb_p = run_perch(paths, CFG["batch_files"], verbose=True)
    meta_p.to_parquet(PARTIAL_CACHE_META_P)
    np.savez(PARTIAL_CACHE_NPZ_P,
             scores=sc_p.astype(np.float32), embs=emb_p.astype(np.float32))
    print(f"  Partial cache saved  scores={sc_p.shape}")
    return meta_p, sc_p, emb_p

if partial_files:
    t_part = time.time()
    meta_part, sc_part, emb_part = _load_or_build_partial_cache(partial_files)

    if meta_part is not None:
        # Ensure row_id column exists
        if "row_id" not in meta_part.columns:
            n_pf = len(meta_part) // N_WINDOWS
            meta_part["row_id"] = (
                meta_part["filename"].str.replace(".ogg", "", regex=False)
                + "_" + np.tile(np.arange(5, 65, 5), n_pf).astype(str)
            )

        # Build label lookup for partial-file windows
        partial_sc = sc[sc["filename"].isin(set(partial_files))]
        part_rid_map = {row["row_id"]: row["label_list"]
                        for _, row in partial_sc.iterrows()}

        # Label matrix + validity mask for partial windows
        Y_part    = np.zeros((len(meta_part), N_CLASSES), dtype=np.uint8)
        mask_part = np.zeros(len(meta_part), dtype=bool)
        for i, rid in enumerate(meta_part["row_id"].values):
            if rid in part_rid_map:
                for lbl in part_rid_map[rid]:
                    if lbl in label_to_idx:
                        Y_part[i, label_to_idx[lbl]] = 1
                mask_part[i] = True

        # Extended arrays: full files first, partial files appended
        meta_tr_ext = pd.concat([meta_tr, meta_part], ignore_index=True)
        sc_tr_ext   = np.concatenate([sc_tr,  sc_part],  axis=0)
        emb_tr_ext  = np.concatenate([emb_tr, emb_part], axis=0)
        Y_tr_ext    = np.concatenate([Y_FULL_aligned, Y_part], axis=0)
        mask_tr_ext = np.concatenate(
            [np.ones(len(Y_FULL_aligned), dtype=bool), mask_part], axis=0
        )
        n_labeled = int(mask_tr_ext.sum())
        print(f"Extended training: {len(sc_tr_ext)} windows | "
              f"{n_labeled} labeled | "
              f"Active classes: {int((Y_tr_ext[mask_tr_ext].sum(0) > 0).sum())}  "
              f"({time.time()-t_part:.1f}s)")
    else:
        meta_tr_ext = meta_tr; sc_tr_ext = sc_tr; emb_tr_ext = emb_tr
        Y_tr_ext = Y_FULL_aligned
        mask_tr_ext = np.ones(len(Y_FULL_aligned), dtype=bool)
else:
    meta_tr_ext = meta_tr; sc_tr_ext = sc_tr; emb_tr_ext = emb_tr
    Y_tr_ext = Y_FULL_aligned
    mask_tr_ext = np.ones(len(Y_FULL_aligned), dtype=bool)
    print("No partial files detected; using full-file data only")
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
    3-tier hierarchical prior: global → site+hour independent → site×hour joint.
    Joint bucket uses tighter shrinkage (4.0 vs 8.0) to reflect finer granularity.
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)

    # ── Site-level frequencies ──────────────────────────────────────────
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    for s in site_keys:
        i = site_to_i[s]
        mask = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)

    # ── Hour-level frequencies ──────────────────────────────────────────
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    for h in hour_keys:
        i = hour_to_i[h]
        mask = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)

    # ── Joint site×hour bucket (tighter shrinkage=4) ────────────────────
    valid = sc_df["site"].notna() & sc_df["hour_utc"].notna()
    sh_keys = sorted({
        (str(s), int(h))
        for s, h in zip(sc_df.loc[valid, "site"], sc_df.loc[valid, "hour_utc"])
    })
    sh_to_i = {k: i for i, k in enumerate(sh_keys)}
    sh_p    = np.zeros((len(sh_keys), Y_labels.shape[1]), dtype=np.float32)
    sh_n    = np.zeros(len(sh_keys), dtype=np.float32)
    for (s, h) in sh_keys:
        i = sh_to_i[(s, h)]
        mask = (sc_df["site"].astype(str).values == s) & (sc_df["hour_utc"].astype(int).values == h)
        sh_n[i] = mask.sum()
        sh_p[i] = Y_labels[mask].mean(axis=0)

    print(f"Prior tables: {len(site_keys)} sites | {len(hour_keys)} hours | "
          f"{len(sh_keys)} site×hour buckets")
    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
        "sh_to_i":   sh_to_i,   "sh_p":   sh_p,   "sh_n":   sh_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """
    3-tier prior fusion: global → hour/site (shrinkage=8) → site×hour (shrinkage=4).
    Converts final probability estimate to logit and adds to raw scores.
    """
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()

    # Tier 1: global baseline
    p = np.tile(tables["global_p"], (n, 1))

    # Tier 2a: hour-level refinement
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j  = tables["hour_to_i"][h]
            nh = tables["hour_n"][j]
            w  = nh / (nh + 8.0)
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]

    # Tier 2b: site-level refinement (builds on hour estimate)
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j  = tables["site_to_i"][s]
            ns = tables["site_n"][j]
            w  = ns / (ns + 8.0)
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]

    # Tier 3: joint site×hour (finer shrinkage=4, builds on tier-2 estimate)
    for i, (s, h) in enumerate(zip(sites, hours)):
        key = (str(s), int(h))
        if key in tables["sh_to_i"]:
            j   = tables["sh_to_i"][key]
            nsh = tables["sh_n"][j]
            w   = nsh / (nsh + 4.0)
            p[i] = w * tables["sh_p"][j] + (1 - w) * p[i]

    p   = np.clip(p, eps, 1 - eps)
    out += lambda_prior * (np.log(p) - np.log1p(-p))
    return out.astype(np.float32)


print("✅ Prior tables: 3-tier joint site×hour prior defined")
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


def train_mlp_probes(emb, scores_raw, Y, label_mask=None, min_pos=5, pca_dim=128, alpha_blend=0.4):
    """
    Train per-class MLP probes on PCA-compressed embeddings + sequential features.

    label_mask: bool array (n_rows,) — True where labels are valid.
                Allows mixing full files (all windows labeled) with partial files
                (only some windows labeled). PCA/scaler are fit on ALL rows so
                sequential context is preserved; only labeled rows are used for
                MLP fitting.  If None, all rows are used (backward compatible).
    pca_dim: 128 (was 64) — retains ~90%+ variance for rare-species discrimination.
    """
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb)
    pca    = PCA(n_components=min(pca_dim, emb_s.shape[1] - 1))
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"Embedding: {emb.shape} → PCA: {Z.shape}  "
          f"(variance retained: {pca.explained_variance_ratio_.sum():.2%})")

    # Determine which rows are labeled (support partial files)
    if label_mask is not None:
        train_idx = np.where(label_mask)[0]
    else:
        train_idx = np.arange(len(Y))

    Y_train       = Y[train_idx]
    class_weights = build_class_freq_weights(Y_train, cap=10.0)

    probe_models = {}
    active = np.where(Y_train.sum(axis=0) >= min_pos)[0]
    print(f"Training MLP probes for {len(active)} species (>= {min_pos} pos windows, "
          f"{len(train_idx)}/{len(Y)} labeled rows)...")

    MAX_ROWS = 3000

    for ci in tqdm(active, desc="MLP probes"):
        # Build sequential features on the FULL array to preserve per-file context
        prev, next_, mean, max_, std = build_sequential_features(scores_raw[:, ci])
        X_full = np.hstack([
            Z,
            scores_raw[:, ci:ci+1],
            prev[:, None], next_[:, None],
            mean[:, None], max_[:, None], std[:, None],
        ])

        # Filter to labeled rows for actual training
        X = X_full[train_idx]
        y = Y_train[:, ci]

        if y.sum() == 0 or y.sum() == len(y):
            continue

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
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            learning_rate_init=5e-4,
            alpha=0.005,
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

print("✅ MLP probe: pca_dim=128, hidden=(128,64), label_mask support, max_iter=300, min_pos=5")
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
            pca_dim=128,
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
    n_epochs=CFG["proto_ssm_train"]["n_epochs"],
    patience=CFG["proto_ssm_train"]["patience"],
    lr=1e-3, verbose=False)
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

proto_out = run_tta_proto(
    proto_model, emb_te_f, sc_te_f,
    site_t=torch.tensor(test_site_ids, dtype=torch.long),
    hour_t=torch.tensor(test_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2],
)
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)
print("TTA applied to test (5 circular shifts)")

# ── Step C: Prior tables ───────────────────────────────────────────────
prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)

# ── Step D: MLP probes (trained on full + partial files with label mask) ──
probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb=emb_tr_ext, scores_raw=sc_tr_ext, Y=Y_tr_ext,
    label_mask=mask_tr_ext,
    min_pos=5, pca_dim=128, alpha_blend=0.4,
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

 
 
