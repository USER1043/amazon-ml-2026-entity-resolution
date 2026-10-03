"""Stage 3: cross-encoder re-scoring of the stage-2 shortlist + a small stacker.

The stage-1/2 XGBoost models see hand-made string similarities. The cross-encoder reads the two normalized
records jointly ("query text [SEP] S1 text") with full token attention, so it can weigh e.g. a differing
house number against a transliterated name. It starts from our fine-tuned bi-encoder (MiniLM-L6,
Apache-2.0, 22M params) with a linear head, and only scores pairs that reach stage 2 (p1 >= P1_FLOOR among
stage-0 survivors: ~4.5 per S1), so it stays cheap.

python -m src.crossenc train        # fine-tune on candidates of NON-evaluation queries (no seed S1, no
                                    # query of the local evaluation sample) -> model_crossenc/
python -m src.crossenc score train  # CE score for every row of oof_pred.parquet  -> ce_train.parquet
python -m src.crossenc score test   # CE score for every row of test_scores.parquet -> ce_test.parquet
python -m src.crossenc stack        # 2-fold stacker (stage-2 p + CE context) -> local F0.5, test scores,
                                    # OUT_DIR/matching_results_ce.tsv
"""
import json, os, sys, time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModel
from . import config as C

MAXLEN = 128
BATCH = 128
LR = float(os.environ.get("ER_CE_LR", 5e-5))
EPOCHS = int(os.environ.get("ER_CE_EPOCHS", 1))
SEED = int(os.environ.get("ER_CE_SEED", 0))               # query sample seed
NAME = os.environ.get("ER_CE_NAME", "")                  # "" = v1; e.g. "_v2" -> model_crossenc_v2, ce_*_v2 ...
SNAME = NAME + "".join(x for x in os.environ.get("ER_CE_STACK_ALSO", "").split(",") if x)   # stacker outputs
PSEUDO = os.environ.get("ER_CE_PSEUDO", "")              # test scores parquet -> add confident test pseudo-pairs
N_PSEUDO = int(os.environ.get("ER_CE_N_PSEUDO", 2_500_000))
INIT = os.environ.get("ER_CE_INIT", "")
BASE = os.environ.get("ER_CE_BASE", "")                  # other base model (HF id), e.g. cross-encoder/ms-marco-MiniLM-L12-v2 (Apache-2.0)                  # continue from this CE dir (weights + head) instead of the bi-encoder
N_POS_Q = int(os.environ.get("ER_CE_POS_Q", 450_000))   # matched training queries
N_DIS_Q = int(os.environ.get("ER_CE_DIS_Q", 120_000))   # distractor training queries
TOPR = 5                                                  # candidates per training query (by bi-encoder rank)
OUT = lambda: C.p("model_crossenc" + NAME)


class CE(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.m = AutoModel.from_pretrained(path)
        self.head = nn.Linear(self.m.config.hidden_size, 1)

    def forward(self, ids, mask, tt):
        h = self.m(input_ids=ids, attention_mask=mask, token_type_ids=tt).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head(((h * m).sum(1) / m.sum(1).clamp(min=1)).float()).squeeze(-1)


def texts(split, who, idx):
    """Normalized record text for idx (np array), in the given order."""
    t = pl.read_parquet(C.p("norm", f"{split}_{who}.parquet"), columns=["idx", "text"]).sort("idx")["text"]
    return t.gather(pl.Series(idx)).to_list()


def train():
    t0 = time.time()
    dev = "cuda"
    seeds = np.load(C.p("seeds.npy"))
    evalq = pl.Series(np.load(C.p("train_included.npy")).astype(np.int32))
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"]).rename({"idx": "q_idx"})
    qlab = qlab.filter(~pl.col("q_idx").is_in(evalq.implode()))
    seed_idx = pl.Series(np.nonzero(seeds)[0].astype(np.int32)).implode()
    rng = np.random.default_rng(SEED)
    pos = qlab.filter((pl.col("label") >= 0) & ~pl.col("label").is_in(seed_idx))["q_idx"].to_numpy()
    dis = qlab.filter(pl.col("label") < 0)["q_idx"].to_numpy()
    pick = np.concatenate([rng.choice(pos, min(N_POS_Q, len(pos)), replace=False),
                           rng.choice(dis, min(N_DIS_Q, len(dis)), replace=False)])
    pick = pl.Series(np.sort(pick).astype(np.int32))
    d = (pl.scan_parquet(C.p("cand", "train.parquet")).select("q_idx", "s1_idx", "rank_q")
           .filter(pl.col("q_idx").is_in(pick.implode()) & ~pl.col("s1_idx").is_in(seed_idx)).collect()
           .join(qlab, on="q_idx", how="left")
           .with_columns(y=(pl.col("label") == pl.col("s1_idx")).cast(pl.Float32))
           .filter((pl.col("rank_q") <= TOPR) | (pl.col("y") == 1)))
    d = d.sample(fraction=1.0, shuffle=True, seed=1 + SEED)
    print(f"CE train pairs {d.height:,}  pos rate {d['y'].mean():.3f}  queries {pick.len():,}  {time.time()-t0:.0f}s", flush=True)
    qt, st = texts("train", "q", d["q_idx"].to_numpy()), texts("train", "s1", d["s1_idx"].to_numpy())
    yv = d["y"].to_numpy()
    if PSEUDO:
        # test-time adaptation: confident pairs of the current TEST scores (pseudo.select_pseudo: argmax with
        # p >= 0.98 and margin >= 0.3 -> 1; other candidates of those queries and queries whose best p <= 0.02
        # -> 0). Teaches the CE the test vocabulary (e.g. the unseen country) on top of the real train pairs.
        from .pseudo import select_pseudo
        sc = pl.read_parquet(PSEUDO, columns=["q_idx", "s1_idx", "p"])
        qs = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "src"]).rename({"idx": "q_idx", "src": "q_src"})
        ps = select_pseudo(sc, qs, 0.98, 0.3, 0.02)
        # the shortlist yields far more confident matches than confident non-matches: cap positives at 2x the
        # negatives, or the CE drifts towards "match" on test text (false merges cost most under F0.5)
        neg = ps.filter(pl.col("label") == 0)
        pos = ps.filter(pl.col("label") == 1)
        pos = pos.sample(min(pos.height, 2 * neg.height), seed=2 + SEED)
        ps = pl.concat([pos, neg])
        ps = ps.sample(min(N_PSEUDO, ps.height), seed=2 + SEED)
        print(f"pseudo test pairs {ps.height:,}  pos rate {ps['label'].cast(pl.Float32).mean():.3f}", flush=True)
        qt += texts("test", "q", ps["q_idx"].to_numpy()); st += texts("test", "s1", ps["s1_idx"].to_numpy())
        yv = np.concatenate([yv, ps["label"].cast(pl.Float32).to_numpy()])
        perm0 = np.random.default_rng(9).permutation(len(yv))
        qt, st, yv = [qt[j] for j in perm0], [st[j] for j in perm0], yv[perm0]
    y = torch.tensor(yv)

    base = INIT or BASE or C.p("model_biencoder")
    tok = AutoTokenizer.from_pretrained(base)
    model = CE(base)
    if INIT:
        model.head.load_state_dict(torch.load(os.path.join(INIT, "head.pt")))
    model = model.to(dev).train()
    n = len(qt); spe = n // BATCH; steps = spe * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda")
    run = 0.0
    perm = np.arange(n)
    for i in range(steps):
        if i % spe == 0 and i:
            perm = np.random.default_rng(i).permutation(n)   # new epoch: reshuffle
        ix = perm[(i % spe) * BATCH:(i % spe + 1) * BATCH]
        sl = ix
        tk = tok([qt[j] for j in sl], [st[j] for j in sl], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lo = model(tk["input_ids"].to(dev), tk["attention_mask"].to(dev), tk["token_type_ids"].to(dev))
        loss = Fn.binary_cross_entropy_with_logits(lo.float(), y[torch.from_numpy(sl)].to(dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if (i + 1) % 1000 == 0 or i + 1 == steps:
            print(f"  step {i+1}/{steps} loss {run:.4f}  {time.time()-t0:.0f}s", flush=True)
    os.makedirs(OUT(), exist_ok=True)
    model.m.save_pretrained(OUT()); tok.save_pretrained(OUT())
    torch.save(model.head.state_dict(), os.path.join(OUT(), "head.pt"))
    print(f"saved {OUT()}  {time.time()-t0:.0f}s", flush=True)


@torch.inference_mode()
def score(split):
    t0 = time.time()
    dev = "cuda"
    src = C.p("oof_pred.parquet") if split == "train" else C.p("test_scores.parquet")
    d = pl.read_parquet(src, columns=["q_idx", "s1_idx"])
    tok = AutoTokenizer.from_pretrained(OUT())
    model = CE(OUT()); model.head.load_state_dict(torch.load(os.path.join(OUT(), "head.pt")))
    model = model.to(dev).eval()
    model.m.half()   # encoder in fp16; the pooled vector is cast to fp32 for the (fp32) head
    out = np.empty(d.height, np.float32)
    step = 2_000_000   # text lookups per block (RAM)
    for b0 in range(0, d.height, step):
        blk = d.slice(b0, step)
        qt, st = texts(split, "q", blk["q_idx"].to_numpy()), texts(split, "s1", blk["s1_idx"].to_numpy())
        order = np.argsort(np.fromiter((len(a) + len(b) for a, b in zip(qt, st)), np.int32, len(qt)), kind="stable")
        res = np.empty(len(qt), np.float32)
        B = 1024
        for i in range(0, len(qt), B):
            ix = order[i:i + B]
            tk = tok([qt[j] for j in ix], [st[j] for j in ix], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            res[ix] = model(tk["input_ids"].to(dev), tk["attention_mask"].to(dev), tk["token_type_ids"].to(dev)).float().cpu().numpy()
        out[b0:b0 + len(qt)] = res
        print(f"  scored {b0 + len(qt):,}/{d.height:,}  {time.time()-t0:.0f}s", flush=True)
    d.with_columns(ce=pl.Series(out)).write_parquet(C.p(f"ce_{split}{NAME}.parquet"))
    print(f"-> ce_{split}{NAME}.parquet  {time.time()-t0:.0f}s", flush=True)


def ce_context(d):
    """d: q_idx, s1_idx, p (stage-2), ce (logit). Per-pair + per-query / per-S1 context of both scores."""
    d = d.with_columns(pce=(1 / (1 + (-pl.col("ce")).exp())).cast(pl.Float32))
    d = d.with_columns(
        ce_rank_q=pl.col("ce").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32),
        ce_max_q=pl.col("ce").max().over("q_idx"), ce_2_q=pl.col("ce").top_k(2).min().over("q_idx"),
        p_rank_q=pl.col("p").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32),
        p_max_q=pl.col("p").max().over("q_idx"), p_2_q=pl.col("p").top_k(2).min().over("q_idx"),
        n_q=pl.len().over("q_idx").cast(pl.Float32),
        ce_rank_s1=pl.col("ce").rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32),
        ce_max_s1=pl.col("ce").max().over("s1_idx"), n_s1=pl.len().over("s1_idx").cast(pl.Float32),
        hi_s1=(pl.col("pce") > 0.5).sum().over("s1_idx").cast(pl.Float32),
        psum_s1=pl.col("pce").sum().over("s1_idx"),
    )
    return d.with_columns(
        ce_other_q=pl.when(pl.col("ce_rank_q") == 1).then(pl.col("ce_2_q")).otherwise(pl.col("ce_max_q")),
        p_other_q=pl.when(pl.col("p_rank_q") == 1).then(pl.col("p_2_q")).otherwise(pl.col("p_max_q")),
        ce_gap_s1=pl.col("ce") - pl.col("ce_max_s1"),
    ).drop("ce_2_q", "p_2_q")


ALSO = [x for x in os.environ.get("ER_CE_STACK_ALSO", "").split(",") if x]   # extra CE runs (names) as stacker inputs


def add_also(d, split):
    """Adds ce<name> (logit) and its rank within the record for every extra CE run in ER_CE_STACK_ALSO."""
    for nm in ALSO:
        e = pl.read_parquet(C.p(f"ce_{split}{nm}.parquet")).rename({"ce": f"ce{nm}"})
        d = d.join(e, on=["q_idx", "s1_idx"], how="left").with_columns(
            pl.col(f"ce{nm}").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32).alias(f"ce{nm}_rank_q"),
            (pl.col(f"ce{nm}") - pl.col(f"ce{nm}").max().over("q_idx")).alias(f"ce{nm}_gap_q"))
    return d


SFEATS = ["p", "ce", "pce", "ce_rank_q", "ce_max_q", "p_rank_q", "p_max_q", "n_q", "ce_rank_s1", "ce_max_s1",
          "n_s1", "hi_s1", "psum_s1", "ce_other_q", "p_other_q", "ce_gap_s1"]


def stack(local_only=False):
    import xgboost as xgb
    from .model import argmax_q, select, f05, xgb_params
    t0 = time.time()
    seeds = np.nonzero(np.load(C.p("seeds.npy")))[0]
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
    tr = pl.read_parquet(C.p("oof_pred.parquet"), columns=["q_idx", "s1_idx", "p", "label"])
    tr = ce_context(tr.join(pl.read_parquet(C.p(f"ce_train{NAME}.parquet")), on=["q_idx", "s1_idx"]))
    # folds by the query's home S1 (true S1, else its best-scored S1) so a query never straddles folds
    home = tr.sort("p", descending=True).group_by("q_idx").agg(pl.col("s1_idx").first().alias("top"), pl.col("label").first())
    home = home.with_columns(h=pl.when(pl.col("label") >= 0).then(pl.col("label")).otherwise(pl.col("top")))
    tr = tr.join(home.select("q_idx", fold=(pl.col("h").hash(seed=3) % 2).cast(pl.Int8)), on="q_idx")
    tr = add_also(tr, "train")
    feats = SFEATS + [f"ce{nm}{x}" for nm in ALSO for x in ("", "_rank_q", "_gap_q")]
    y = (tr["label"] == tr["s1_idx"]).cast(pl.Float32).to_numpy()
    X = tr.select(feats).to_numpy().astype(np.float32)
    fold = tr["fold"].to_numpy()
    es = np.random.default_rng(5).random(len(y)) < 0.1
    params = dict(xgb_params(6, 0.05))
    oof = np.zeros(len(y), np.float32)
    models = []
    for f in (0, 1):
        a, b = (fold != f) & ~es, (fold != f) & es
        dtr, des = xgb.DMatrix(X[a], y[a]), xgb.DMatrix(X[b], y[b])
        m = xgb.train(params, dtr, 3000, evals=[(des, "es")], early_stopping_rounds=80, verbose_eval=False)
        oof[fold == f] = m.predict(xgb.DMatrix(X[fold == f]), iteration_range=(0, m.best_iteration + 1))
        models.append(m)
        print(f"  stacker fold {f}: {m.best_iteration + 1} trees  {time.time()-t0:.0f}s", flush=True)

    def decide(sc, t, x):
        a = argmax_q(sc)
        pred = select(a, t, "ef")
        if x is not None:
            add = (a.filter(pl.col("p") >= x).join(pred.select("s1_idx").unique(), on="s1_idx", how="anti")
                    .sort(["s1_idx", "p"], descending=[False, True]).group_by("s1_idx").first().select("q_idx", "s1_idx"))
            pred = pl.concat([pred, add])
        return pred

    base = tr.select("q_idx", "s1_idx", "p")
    print(f"baseline (stage 2, t 0.15, fill 0.1): F0.5 {f05(decide(base, 0.15, 0.1), qlab, seeds)[0]:.5f}", flush=True)
    new = tr.select("q_idx", "s1_idx", p=pl.Series(oof))
    res = []
    for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        for x in (None, 0.05, 0.1, 0.2):
            f, pr, rc = f05(decide(new, t, x), qlab, seeds)
            res.append((f, t, x)); print(f"  CE stack t={t} fill={x}: F0.5 {f:.5f}  P {pr:.4f} R {rc:.4f}", flush=True)
    best = max(res, key=lambda r: r[0])
    print(f"BEST CE stack: F0.5 {best[0]:.5f} at t={best[1]} fill={best[2]}", flush=True)
    s1c = pl.read_parquet(C.p("norm", "train_s1.parquet"), columns=["idx", "country"])
    for c in s1c["country"].unique().to_list():
        ev = np.intersect1d(seeds, s1c.filter(pl.col("country") == c)["idx"].to_numpy())
        print(f"  {c}: base {f05(decide(base, 0.15, 0.1), qlab, ev)[0]:.5f} -> CE {f05(decide(new, best[1], best[2]), qlab, ev)[0]:.5f}", flush=True)
    for i, m in enumerate(models):
        m.save_model(C.p(f"xgb_ce_stack{SNAME}_fold{i}.json"))
    json.dump({"t": best[1], "fill": best[2], "f05": best[0], "features": feats}, open(C.p(f"ce_stack{SNAME}.json"), "w"), indent=1)

    if local_only:   # local F0.5 only (test not scored yet)
        return
    te = pl.read_parquet(C.p("test_scores.parquet"))
    te = ce_context(te.join(pl.read_parquet(C.p(f"ce_test{NAME}.parquet")), on=["q_idx", "s1_idx"]))
    te = add_also(te, "test")
    Xt = xgb.DMatrix(te.select(feats).to_numpy().astype(np.float32))
    pt = np.mean([m.predict(Xt, iteration_range=(0, m.best_iteration + 1)) for m in models], axis=0)
    sc = te.select("q_idx", "s1_idx", p=pl.Series(pt, dtype=pl.Float32))
    sc.write_parquet(C.p(f"test_scores_ce{SNAME}.parquet"))
    pred = decide(sc, best[1], best[2])
    s1_ids = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "entity_id"])
    q_ids = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "entity_id"]).sort("idx")["entity_id"]
    g = (pred.with_columns(qid=q_ids.gather(pred["q_idx"])).group_by("s1_idx")
             .agg(pl.col("qid").unique().sort().str.join(",").alias("matched_entity_ids")))
    out = (s1_ids.join(g, left_on="idx", right_on="s1_idx", how="left")
                 .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null("")))
    path = os.path.join(C.OUT_DIR, f"matching_results_ce{SNAME}.tsv")
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"-> {path}: pairs {pred.height:,}, non-empty {int((out['matched_entity_ids'] != '').sum()):,}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"train": lambda: train(), "score": lambda: score(sys.argv[2]), "stack": lambda: stack(), "stack_local_only": lambda: stack(True)}[sys.argv[1]]()
