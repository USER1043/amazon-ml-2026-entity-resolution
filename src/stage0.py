"""Stage 0: learned blocking cascade -> the final candidate set (candidate_pairs.tsv).

Retrieval (bi-encoder top-k + exact-key blocks) is recall-oriented: ~60-80 candidates per S1. Stage 0 is a
small LightGBM that sees ONLY retrieval-time signals (cosine, ranks and gaps of the pair inside its query's
and its S1's candidate lists, which blocking key produced it, query source / missing address / Indic
script). No string comparison is computed, so it costs ~1 us per pair. Pairs with p0 < tau are dropped
before the expensive pair features; the survivors are exactly what the matching model (stage 1 + 2)
scores, i.e. candidate_pairs.tsv.

python -m src.stage0 fit        # train on the reranker sample (TMP/feat/train), 2-fold OOF by S1,
                                # pick tau, report recall / pairs kept / local F0.5 per tau, save model
python -m src.stage0 apply      # score TMP/test_ctx_<country>.parquet -> cand/test_stage0.parquet
"""
import glob, json, os, sys, time
import numpy as np
import polars as pl
import lightgbm as lgb
from . import config as C
from . import features as F
from .model import argmax_q, select, f05

# retrieval-only context (computed by features.cand_context_to_parquet) + 3 query attributes
CTX = ["cos", "rank_q", "cos_max_q", "n_q", "rank_s1", "cos_max_s1", "n_s1", "gap_q", "dq", "margin_q",
       "gap_s1", "ds1", "via_dense", "via_addr", "via_name", "via_nn", "via_n2"]
QCOLS = {"src": "q_src", "addr_empty": "q_addr_empty", "is_indic": "q_is_indic"}
FEATS = CTX + list(QCOLS.values())
RECALL = float(os.environ.get("ER_STAGE0_RECALL", 0.999))   # share of true candidate pairs tau must keep
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
              num_threads=C.N_PROC, seed=0)
ROUNDS = 400
TAUS = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)


def q_attrs(split):
    """Query attributes aligned with idx (row i = q_idx i), for positional gathers."""
    return pl.read_parquet(C.p("norm", f"{split}_q.parquet"), columns=["idx"] + list(QCOLS)).sort("idx")


def add_q_attrs(df, qa):
    ix = df["q_idx"]
    return df.with_columns(**{new: qa[old].gather(ix) for old, new in QCOLS.items()})


def matrix(df):
    return df.select(FEATS).to_numpy().astype(np.float32)


def fit():
    t0 = time.time()
    files = sorted(glob.glob(C.t("feat", "train", "*.parquet")))
    d = pl.read_parquet(files, columns=["q_idx", "s1_idx", "label"] + FEATS)
    y = d["label"].to_numpy()
    fold = (d["s1_idx"].hash(seed=0) % 2).to_numpy()   # S1-disjoint folds
    X = matrix(d)
    print(f"stage0 train rows {len(y):,}  pos {int(y.sum()):,}  {time.time()-t0:.0f}s", flush=True)
    oof = np.zeros(len(y), np.float32)
    iters = []
    for f in (0, 1):
        tr, va = fold != f, fold == f
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        b = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
        iters.append(b.best_iteration)
        print(f"  fold {f}: best iter {b.best_iteration}  {time.time()-t0:.0f}s", flush=True)

    # blocking recall ceiling = true pairs present in the candidates / all true pairs of the seed S1
    seeds = np.load(C.p("seeds.npy"))
    seed_idx = np.nonzero(seeds)[0]
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
    n_true = int(qlab.filter(pl.col("label").is_in(pl.Series(seed_idx.astype(np.int32)).implode())).height)
    is_seed = seeds[d["s1_idx"].to_numpy()]
    pos_seed = (y == 1) & is_seed

    # local F0.5 of the current matching model restricted to stage-0 survivors (stage-2 context is
    # recomputed on survivors in the real run; here its OOF scores are reused -> close approximation)
    cfg = json.load(open(C.p("decision.json")))
    x = cfg.get("fill", 0.1)
    oofp = pl.read_parquet(C.p("oof_pred.parquet"), columns=["q_idx", "s1_idx", "p"])
    p0 = d.select("q_idx", "s1_idx").with_columns(p0=pl.Series(oof))
    oofp = oofp.join(p0, on=["q_idx", "s1_idx"], how="left").with_columns(pl.col("p0").fill_null(1.0))

    def decide(sc):
        a = argmax_q(sc)
        pred = select(a, cfg["t"], cfg["mode"])
        add = (a.filter(pl.col("p") >= x).join(pred.select("s1_idx").unique(), on="s1_idx", how="anti")
                .sort(["s1_idx", "p"], descending=[False, True]).group_by("s1_idx").first().select("q_idx", "s1_idx"))
        return pl.concat([pred, add])

    base = f05(decide(oofp), qlab, seed_idx)[0]
    print(f"seed S1 {len(seed_idx):,}; true pairs {n_true:,}; in retrieval candidates {int(pos_seed.sum()):,} "
          f"(recall {pos_seed.sum()/n_true:.4f}); F0.5 without stage 0 {base:.5f}", flush=True)
    table = []
    for tau in TAUS:
        k = oof >= tau
        row = dict(tau=tau, keep_rows=float(k.mean()), keep_pos=float(k[y == 1].sum() / y.sum()),
                   seed_recall=float((k & pos_seed).sum() / n_true),
                   f05=f05(decide(oofp.filter(pl.col("p0") >= tau)), qlab, seed_idx)[0])
        table.append(row)
        print("  tau {tau:<7} rows kept {keep_rows:.3%}  true pairs kept {keep_pos:.4%}  seed recall "
              "{seed_recall:.4f}  F0.5 {f05:.5f}".format(**row), flush=True)
    ok = [r for r in table if r["keep_pos"] >= RECALL]
    tau = max(r["tau"] for r in ok) if ok else min(TAUS)
    print(f"chosen tau {tau} (largest keeping >= {RECALL:.2%} of true candidate pairs)", flush=True)

    full = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    full.save_model(C.p("stage0.txt"))
    json.dump({"tau": tau, "features": FEATS, "recall_target": RECALL, "f05_base": base, "table": table},
              open(C.p("stage0.json"), "w"), indent=1)
    print(f"saved stage0.txt / stage0.json  {time.time()-t0:.0f}s", flush=True)


def apply(split="test", rows=8_000_000):
    """Score every retrieval candidate of `split`; keep p0 >= tau -> cand/{split}_stage0.parquet."""
    t0 = time.time()
    cfg = json.load(open(C.p("stage0.json")))
    b = lgb.Booster(model_file=C.p("stage0.txt"))
    qa = q_attrs(split)
    parts, n_all, hist = [], 0, np.zeros(len(TAUS), np.int64)
    ctx_files = []
    for cf in F.cand_countries(split):   # per-country retrieval context (also reused by predict.py)
        country = os.path.basename(cf)[len(f"{split}_part_"):-len(".parquet")]
        ctx_f = C.t(f"{split}_ctx_{country}.parquet")
        if not os.path.exists(ctx_f):
            F.cand_context_to_parquet(cf, ctx_f)
        ctx_files.append(ctx_f)
    for f in ctx_files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        for lo in range(0, n, rows):
            df = add_q_attrs(pl.scan_parquet(f).slice(lo, rows).select(["q_idx", "s1_idx"] + CTX).collect(), qa)
            p0 = b.predict(matrix(df)).astype(np.float32)
            hist += np.array([(p0 >= t).sum() for t in TAUS])
            parts.append(df.select("q_idx", "s1_idx").with_columns(p0=pl.Series(p0)).filter(pl.col("p0") >= cfg["tau"]))
            n_all += df.height
            del df, p0
        print(f"  {os.path.basename(f)}: {n:,} candidates scored  {time.time()-t0:.0f}s", flush=True)
    out = pl.concat(parts)
    out.write_parquet(C.p("cand", f"{split}_stage0.parquet"))
    n_s1 = pl.scan_parquet(C.p("norm", f"{split}_s1.parquet")).select(pl.len()).collect().item()
    per = out.group_by("s1_idx").len()["len"]
    print(f"retrieval candidates {n_all:,} ({n_all/n_s1:.1f} per S1) -> stage 0 (tau {cfg['tau']}) "
          f"{out.height:,} ({out.height/n_s1:.2f} per S1; median {per.median():.0f}, p99 {per.quantile(0.99):.0f}, "
          f"max {per.max()}; S1 with no candidate {1 - per.len()/n_s1:.2%})", flush=True)
    for t, h in zip(TAUS, hist):
        print(f"  tau {t:<7} -> {h/n_s1:.2f} candidates per S1", flush=True)
    print(f"-> cand/{split}_stage0.parquet  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"fit": fit, "apply": apply}[sys.argv[1]]()
