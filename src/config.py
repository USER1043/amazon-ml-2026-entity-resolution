"""Central config. Every value can be overridden with an environment variable ER_<NAME>."""
import os

def _e(name, default, cast=str):
    v = os.environ.get("ER_" + name)
    return cast(v) if v is not None else default

DATA_DIR = _e("DATA_DIR", "/kaggle/input/amazon-ml-2026/dataset")   # folder containing train/ and test/
WORK = _e("WORK", "/kaggle/working/er")        # small, persistent outputs (models, normalized data, candidates)
TMP = _e("TMP", "/tmp/er")                     # large scratch (train features, ~4 GB). Falls back to WORK if not writable
OUT_DIR = _e("OUT_DIR", "/kaggle/working/output")

BASE_MODEL = _e("BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")   # Apache-2.0, 22M params
FT_PAIRS = _e("FT_PAIRS", 800_000, int)        # bi-encoder fine-tune pairs (1 per S1)
FT_BATCH = _e("FT_BATCH", 384, int)
FT_LR = _e("FT_LR", 1e-4, float)
FT_EPOCHS = _e("FT_EPOCHS", 1, int)
MAX_LEN = _e("MAX_LEN", 64, int)
ENC_BATCH = _e("ENC_BATCH", 2048, int)

TOPK = _e("TOPK", 12, int)                    # dense neighbours kept per query
KEY_BUCKET_MAX = _e("KEY_BUCKET_MAX", 25, int) # skip exact-key buckets with more S1 than this
SEED_FRAC = _e("SEED_FRAC", 0.12, float)       # fraction of train S1 held out: reranker training + local eval
N_PROC = _e("N_PROC", os.cpu_count() or 4, int)
CHUNK_Q = _e("CHUNK_Q", 250_000, int)        # local 16 GB: 400k -> 250k (600k was OOM-killed in pair_features)          # queries per feature chunk
XGB_ROUNDS = _e("XGB_ROUNDS", 2000, int)
DEVICE = _e("DEVICE", "cuda")
# Unseen-country simulation (src/loco_train.py): this train country's labels are hidden from the learned
# normalization maps, the bi-encoder fine-tune and the reranker; only used to score it. "" = off.
HOLDOUT_COUNTRY = _e("HOLDOUT_COUNTRY", "")
# 1 = learn normalization maps without the seed S1 pairs (seeds are the local eval set; learning maps
# from them inflates local scores: seed Indic-token coverage 100% vs 96.4% on test)
MAPS_EXCLUDE_SEEDS = _e("MAPS_EXCLUDE_SEEDS", 0, int)
# parquet (s1_id, q_id) of confident TEST pairs (entity ids) added to map learning (transductive maps)
EXTRA_PAIRS = _e("EXTRA_PAIRS", "")
# stage-1 final model: "full" = one model on all rows (needs a 16 GB GPU, e.g. Kaggle T4),
# "folds" = average of the two fold models (8 GB laptop GPU)
STAGE1_FINAL = _e("STAGE1_FINAL", "folds")
# 1 = score only the stage-0 survivors (src/stage0.py): they are the final candidate set / candidate_pairs.tsv
STAGE0 = _e("STAGE0", 1, int)

for d in (WORK, OUT_DIR):
    os.makedirs(d, exist_ok=True)
try:
    os.makedirs(TMP, exist_ok=True)
except OSError:
    TMP = os.path.join(WORK, "tmp"); os.makedirs(TMP, exist_ok=True)

def p(*a):
    return os.path.join(WORK, *a)

def t(*a):
    return os.path.join(TMP, *a)
