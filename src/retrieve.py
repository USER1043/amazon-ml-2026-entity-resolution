import os, sys, time
import numpy as np
import polars as pl
import torch
from . import config as C
from .biencoder import Encoder, train as ft_train

MODEL_DIR = C.p("model_biencoder")


def seeds_mask(n_s1):
    f = C.p("seeds.npy")
    if os.path.exists(f):
        return np.load(f)
    m = np.random.default_rng(42).random(n_s1) < C.SEED_FRAC
    np.save(f, m)
    return m


def finetune(s1, q, seeds):
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print("bi-encoder already fine-tuned, skipping", flush=True)
        return
    m = q.filter(pl.col("label") >= 0)
    if C.HOLDOUT_COUNTRY:
        m = m.filter(pl.col("country") != C.HOLDOUT_COUNTRY)
    m = m.select("text", "label")
    m = m.filter(~pl.col("label").is_in(pl.Series(np.nonzero(seeds)[0].astype(np.int32)).implode()))
    m = m.sample(fraction=1.0, shuffle=True, seed=1).unique("label", keep="first")
    m = m.head(C.FT_PAIRS)
    st = s1["text"].gather(m["label"]).to_list()
    print(f"fine-tuning bi-encoder on {m.height:,} pairs", flush=True)
    ft_train(m["text"].to_list(), st, MODEL_DIR)


def key_pairs(qd, sd):
    out = []
    for col, bit in (("key_addr", 2), ("key_name", 4), ("key_nn", 8), ("key_n2", 16)):
        sk = sd.filter(pl.col(col) != "").select(col, pl.col("idx").alias("s1_idx"))
        big = sk.group_by(col).len().filter(pl.col("len") > 10)
        sk = sk.join(big.select(col), on=col, how="anti")
        j = (qd.filter(pl.col(col) != "").select(col, pl.col("idx").alias("q_idx"))
               .join(sk, on=col).select("q_idx", "s1_idx", pl.lit(bit, pl.Int8).alias("via")))
        out.append(j)
    return pl.concat(out).group_by("q_idx", "s1_idx").agg(pl.col("via").sum().cast(pl.Int8))


def pair_cos(Q, q_pos, S_all, s_idx, step=500_000):
    out = np.empty(len(q_pos), np.float32)
    for i in range(0, len(q_pos), step):
        a = Q[q_pos[i:i + step]].astype(np.float32)
        b = S_all[s_idx[i:i + step]].astype(np.float32)
        out[i:i + step] = np.einsum("ij,ij->i", a, b)
    return out


def add_keys(df):
    return df.with_columns(key_nn=pl.when(pl.col("num1") != "").then(
        pl.concat_str([pl.col("num1"), pl.lit("|"), pl.col("core").str.split(" ").list.first().str.slice(0, 4)])
    ).otherwise(pl.lit("")))


def dense_topk(Q, S, k, rows=None):
    """rows: optional index array into S; the selected rows are uploaded to the device in chunks
    (local 16 GB box: avoids a ~1 GB host copy of S[rows]). Indices are returned as int32."""
    dev = "cuda" if torch.cuda.is_available() and C.DEVICE == "cuda" else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    n_s = S.shape[0] if rows is None else len(rows)
    St = torch.empty((n_s, S.shape[1]), dtype=dt, device=dev)
    for i in range(0, n_s, 200_000):
        blk = S[i:i + 200_000] if rows is None else S[rows[i:i + 200_000]]
        St[i:i + len(blk)] = torch.from_numpy(np.ascontiguousarray(blk)).to(dev, dt)
    k = min(k, n_s)
    chunk = max(256, int(3e9 // max(1, n_s * 2)))
    vals = np.empty((Q.shape[0], k), np.float32)
    inds = np.empty((Q.shape[0], k), np.int32)
    for i in range(0, Q.shape[0], chunk):
        qc = torch.from_numpy(np.ascontiguousarray(Q[i:i + chunk])).to(dev, dt)   # Q may be a read-only memmap
        v, ix = (qc @ St.T).topk(k, dim=1)
        vals[i:i + chunk] = v.float().cpu().numpy()
        inds[i:i + chunk] = ix.cpu().numpy()
    del St
    return vals, inds


def encode_to_disk(enc, texts, path, step=1_000_000):
    """Local 16 GB box: query embeddings go to a disk-backed .npy (open_memmap), encoded 1M texts at a
    time, instead of one in-RAM array (6.2M US queries x 384 fp16 = 4.7 GB)."""
    done = path + ".done"
    if os.path.exists(done) and os.path.exists(path):
        Q = np.load(path, mmap_mode="r")
        if Q.shape == (len(texts), enc.dim):
            print(f"    reusing encoded {path}", flush=True)
            return Q
    Q = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), enc.dim))
    for i in range(0, len(texts), step):
        Q[i:i + step] = enc.encode(texts[i:i + step])
        print(f"    encoded {min(i + step, len(texts)):,}/{len(texts):,} (to disk)", flush=True)
    Q.flush()
    del Q
    open(done, "w").close()               # marker: the .npy is complete and can be reused after a crash
    return np.load(path, mmap_mode="r")


def run(split):
    t0 = time.time()
    os.makedirs(C.p("cand"), exist_ok=True)
    cols = ["idx", "country", "text", "key_addr", "key_name", "key_n2", "num1", "core"]
    s1 = add_keys(pl.read_parquet(C.p("norm", f"{split}_s1.parquet"), columns=cols))
    q_path = C.p("norm", f"{split}_q.parquet")
    lab_cols = ["label"] if split == "train" else []
    # Local 16 GB box: keep only small columns for all queries; text/keys are loaded per country below.
    q = pl.read_parquet(q_path, columns=["idx", "country", *lab_cols])
    if split == "train":
        seeds = seeds_mask(s1.height)
        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            finetune(s1, add_keys(pl.read_parquet(q_path, columns=cols + lab_cols)), seeds)
        seed_idx = pl.Series(np.nonzero(seeds)[0].astype(np.int32))
        print(f"train queries to retrieve: {q.height:,} (seed S1: {int(seeds.sum()):,})", flush=True)
    enc = Encoder(MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else C.BASE_MODEL)

    emb_f = C.p(f"emb_{split}_s1.npy")
    if os.path.exists(emb_f):
        S_all = np.load(emb_f)
    else:
        print(f"encoding {s1.height:,} S1", flush=True)
        S_all = enc.encode(s1["text"].to_list())
        np.save(emb_f, S_all)
    parts = []
    for c in s1["country"].unique().sort().to_list():
        part_f = C.p("cand", f"{split}_part_{c}.parquet")
        if os.path.exists(part_f):
            parts.append(part_f); print(f"[{c}] cached part", flush=True)
            continue
        sc = s1.filter(pl.col("country") == c)
        s_ids = sc["idx"].to_numpy()
        qc = add_keys(pl.scan_parquet(q_path).filter(pl.col("country") == c).select(cols).collect()).sort("idx")
        if qc.height == 0:
            continue
        q_ids = qc["idx"].to_numpy()
        print(f"[{c}] S1={len(s_ids):,} queries={len(q_ids):,}  encoding... {time.time()-t0:.0f}s", flush=True)
        Q = encode_to_disk(enc, qc["text"].to_list(), C.t(f"q_emb_{split}_{c}.npy"))
        v, ix = dense_topk(Q, S_all, C.TOPK, rows=s_ids)
        k = v.shape[1]
        dense = pl.DataFrame({"q_idx": np.repeat(q_ids, k).astype(np.int32),
                              "s1_idx": s_ids[ix.ravel()].astype(np.int32),
                              "cos": v.ravel(), "via": np.ones(v.size, np.int8)})
        del v, ix
        kp = key_pairs(qc, sc)
        kp_new = kp.join(dense.select("q_idx", "s1_idx"), on=["q_idx", "s1_idx"], how="anti")
        dense = (dense.join(kp.rename({"via": "kv"}), on=["q_idx", "s1_idx"], how="left")
                      .with_columns(via=(pl.col("via") + pl.col("kv").fill_null(0)).cast(pl.Int8)).drop("kv"))
        if kp_new.height:
            cosv = pair_cos(Q, np.searchsorted(q_ids, kp_new["q_idx"].to_numpy()), S_all, kp_new["s1_idx"].to_numpy())
            kp_new = (kp_new.with_columns(cos=pl.Series(cosv, dtype=pl.Float32))
                            .sort("cos", descending=True).group_by("q_idx", maintain_order=True).head(4)
                            .select("q_idx", "s1_idx", "cos", "via"))
            dense = pl.concat([dense, kp_new])
        print(f"[{c}] pairs={dense.height:,} (key-only extra {kp_new.height:,})  {time.time()-t0:.0f}s", flush=True)
        dense.write_parquet(part_f)
        parts.append(part_f)
        del Q, kp, kp_new, dense, qc, sc
        os.remove(C.t(f"q_emb_{split}_{c}.npy"))
    del S_all, s1
    cand = pl.concat([pl.read_parquet(f) for f in parts])
    cand = cand.with_columns(rank_q=pl.col("cos").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16))

    if split == "train":
        lab = q.select(pl.col("idx").alias("q_idx"), "label", "country")
        ls = lab.filter(pl.col("label").is_in(seed_idx.implode()))
        hit = cand.join(ls, on="q_idx").filter(pl.col("label") == pl.col("s1_idx"))
        n_pos = ls.height
        print(f"TRAIN (held-out seed S1) blocking recall: {hit.height/n_pos:.4f} | dense-only "
              f"{hit.filter((pl.col('via') & 1) == 1).height/n_pos:.4f} | top1 {hit.filter(pl.col('rank_q') == 1).height/n_pos:.4f}"
              f" | pairs/query {cand.height/max(1, cand['q_idx'].n_unique()):.1f}", flush=True)
        print(hit.group_by("country").len().join(ls.group_by("country").len(), on="country")
                 .with_columns(recall=pl.col("len") / pl.col("len_right")), flush=True)
    cand.write_parquet(C.p("cand", f"{split}.parquet"))
    print(f"{split}: {cand.height:,} candidate pairs, {cand['q_idx'].n_unique():,} queries  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "train")
