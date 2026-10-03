"""Fast self-training for an unlabelled test country (default France). Run after predict.py.

python -m src.france_selftrain [--country France] [--max-q 300000] [--rounds 150]

1. pseudo-labels from test_scores.parquet (stage-2 p) with pseudo.select_pseudo
2. stage-1 features for up to --max-q pseudo-labelled queries of that country
3. continue boosting each stage-1 fold model (+rounds trees, low eta) on those rows mixed 1:1 with
   real-labelled train rows (anchors the model to known-correct data)
4. re-score only that country with the adapted models + the unchanged stage 2 and decision rule,
   and write OUT_DIR/matching_results_<country>_selftrain.tsv (other countries copied from predict.py)
Unvalidated by construction (the country has no labels): report shows how many assignments changed.
"""
import argparse, glob, json, os, time
import numpy as np
import polars as pl
import xgboost as xgb
from . import config as C
from . import features as F
from .model import S2_RAW, P1_FLOOR, xgb_params, stage2_frame, argmax_q, select
from .pseudo import select_pseudo
from .train import _FoldAvg, _Parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="France")
    ap.add_argument("--max-q", type=int, default=300_000)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--tau-hi", type=float, default=0.98)
    ap.add_argument("--delta", type=float, default=0.3)
    args = ap.parse_args()
    t0 = time.time()
    cty = args.country
    s1c = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "country"])
    cs1 = s1c.filter(pl.col("country") == cty)["idx"]
    scores = pl.read_parquet(C.p("test_scores.parquet")).filter(pl.col("s1_idx").is_in(cs1.implode()))
    q_src = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "src"]).rename({"idx": "q_idx", "src": "q_src"})
    ps = select_pseudo(scores, q_src, args.tau_hi, args.delta, 0.02)
    pos_q = ps.filter(pl.col("label") == 1)["q_idx"].unique()
    neg_only_q = ps.group_by("q_idx").agg(pl.col("label").max()).filter(pl.col("label") == 0)["q_idx"]
    print(f"[{cty}] scored queries {scores['q_idx'].n_unique():,}; pseudo positive queries {pos_q.len():,}; "
          f"distractor queries {neg_only_q.len():,}  {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    n_pos = min(pos_q.len(), int(args.max_q * 0.85))
    n_neg = min(neg_only_q.len(), args.max_q - n_pos)
    pick = pl.concat([pos_q.sample(n_pos, seed=1), neg_only_q.sample(n_neg, seed=2)]).sort()

    # features for the picked queries: every candidate of the query (context from the full country table)
    ctx = pl.read_parquet(C.t(f"test_ctx_{cty}.parquet")).filter(pl.col("q_idx").is_in(pick.implode()))
    s1, q, idf = F.load_tables("test")
    F._IDF = idf
    fn1 = json.load(open(C.p("features_stage1.json")))
    lab = ps.select("q_idx", "s1_idx", "label")
    fdir = C.t("feat", f"pseudo_{cty}")
    os.makedirs(fdir, exist_ok=True)
    for f in glob.glob(os.path.join(fdir, "*.parquet")):
        os.remove(f)
    n_rows = 0
    for k, df in enumerate(F.iter_chunks("test", ctx.sort("q_idx", "cos", descending=[False, True]), s1, q, idf)):
        # candidates below the stage-1 floor never entered test_scores: they are negatives too
        df = df.join(lab, on=["q_idx", "s1_idx"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        df.select(fn1 + ["label"]).write_parquet(os.path.join(fdir, f"part_{k:04d}.parquet"))
        n_rows += df.height
    pos_rate = pl.scan_parquet(os.path.join(fdir, "*.parquet")).select(pl.col("label").mean()).collect().item()
    print(f"[{cty}] pseudo feature rows {n_rows:,} (pos rate {pos_rate:.4f})  {time.time()-t0:.0f}s", flush=True)
    del ctx

    # 1:1 anchor rows from the real-labelled train features
    tfiles = sorted(glob.glob(C.t("feat", "train", "*.parquet")))
    n_tr = pl.scan_parquet(tfiles).select(pl.len()).collect().item()
    take = np.zeros(n_tr, bool)
    take[rng.choice(n_tr, min(n_rows, n_tr), replace=False)] = True
    ytr = pl.scan_parquet(tfiles).select(pl.col("label").cast(pl.Float32)).collect()["label"].to_numpy()
    pfiles = sorted(glob.glob(os.path.join(fdir, "*.parquet")))
    ypl = pl.scan_parquet(pfiles).select(pl.col("label").cast(pl.Float32)).collect()["label"].to_numpy()

    class Both(xgb.DataIter):
        def __init__(self):
            self.a = _Parts(tfiles, fn1, take, ytr)
            self.b = _Parts(pfiles, fn1, np.ones(len(ypl), bool), ypl)
            self.stage = 0
            super().__init__()

        def next(self, input_data):
            if self.stage == 0:
                if self.a.next(input_data):
                    return True
                self.stage = 1
            return self.b.next(input_data)

        def reset(self):
            self.a.reset(); self.b.reset(); self.stage = 0

    params = dict(xgb_params(8, 0.03))
    d = xgb.QuantileDMatrix(Both(), max_bin=params.get("max_bin", 256))
    spec = json.load(open(C.p("xgb_stage1_folds.json")))
    new_paths = []
    for pth, it in zip(spec["models"], spec["iters"]):
        base = xgb.Booster(); base.load_model(pth)
        base = base[: int(it)]                               # continue from the best iteration
        b = xgb.train(params, d, args.rounds, xgb_model=base)
        out = pth.replace(".json", f"_st_{cty}.json")
        b.save_model(out)
        new_paths.append(out)
        print(f"  adapted {os.path.basename(pth)} -> +{args.rounds} trees  {time.time()-t0:.0f}s", flush=True)
    del d
    st_spec = C.p(f"xgb_stage1_folds_st_{cty}.json")
    json.dump({"models": new_paths, "iters": [int(i) + args.rounds for i in spec["iters"]]}, open(st_spec, "w"))

    # re-score the whole country with adapted stage 1 + original stage 2 / decision
    b1 = _FoldAvg.load(st_spec)
    ctx_f = C.t(f"test_ctx_{cty}.parquet")
    qarr = pl.read_parquet(ctx_f, columns=["q_idx"])["q_idx"].to_numpy()
    uq = np.unique(qarr)
    parts = []
    for i in range(0, len(uq), 200_000):
        lo = int(np.searchsorted(qarr, uq[i], "left"))
        hi = int(np.searchsorted(qarr, uq[min(i + 200_000, len(uq)) - 1], "right"))
        df = F.pair_features(pl.scan_parquet(ctx_f).slice(lo, hi - lo).collect(), s1, q)
        p1 = b1.predict(xgb.DMatrix(df.select(fn1).to_numpy().astype(np.float32)))
        parts.append(df.select("q_idx", "s1_idx", *S2_RAW).with_columns(p1=pl.Series(p1, dtype=pl.Float32))
                       .filter(pl.col("p1") >= P1_FLOOR))
        print(f"  rescored {min(i + 200_000, len(uq)):,}/{len(uq):,} {cty} queries  {time.time()-t0:.0f}s", flush=True)
    del s1, q
    cfg = json.load(open(C.p("decision.json")))
    fn2 = json.load(open(C.p("features_stage2.json")))
    b2 = xgb.Booster(); b2.load_model(C.p("xgb_stage2.json"))
    d2 = stage2_frame(pl.concat(parts))
    p = b2.predict(xgb.DMatrix(d2.select(fn2).to_numpy().astype(np.float32)))
    new_sc = d2.select("q_idx", "s1_idx", p=pl.Series(p, dtype=pl.Float32))
    new_pred = select(argmax_q(new_sc), cfg["t"], cfg["mode"])
    old_pred = select(argmax_q(scores), cfg["t"], cfg["mode"])
    common = new_pred.join(old_pred, on=["q_idx", "s1_idx"]).height
    print(f"[{cty}] assignments old {old_pred.height:,} -> new {new_pred.height:,}; unchanged {common:,} "
          f"({common / max(old_pred.height, 1):.2%}); S1 with a match old {old_pred['s1_idx'].n_unique():,} "
          f"new {new_pred['s1_idx'].n_unique():,}", flush=True)

    # write the combined submission: this country from new_pred, the rest from the existing file
    s1_ids = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "entity_id", "country"])
    q_ids = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "entity_id"]).sort("idx")["entity_id"]
    g = (new_pred.with_columns(qid=q_ids.gather(new_pred["q_idx"]))
                 .group_by("s1_idx").agg(pl.col("qid").unique().sort().str.join(",").alias("m_new")))
    newrows = (s1_ids.filter(pl.col("country") == cty).join(g, left_on="idx", right_on="s1_idx", how="left")
                     .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("m_new").fill_null("")))
    base = pl.read_csv(os.path.join(C.OUT_DIR, "matching_results.tsv"), separator="\t", quote_char=None,
                       schema={"source1_entity_id": pl.String, "matched_entity_ids": pl.String}).with_columns(
                       pl.col("matched_entity_ids").fill_null(""))
    out = (base.join(newrows, on="source1_entity_id", how="left", maintain_order="left")
               .with_columns(matched_entity_ids=pl.coalesce("m_new", "matched_entity_ids")).drop("m_new"))
    path = os.path.join(C.OUT_DIR, f"matching_results_{cty}_selftrain.tsv")
    out.write_csv(path, separator="\t", quote_style="never")
    ch = (out["matched_entity_ids"] != base["matched_entity_ids"]).sum()
    print(f"-> {path}: {out.height:,} rows; {ch:,} S1 rows changed (all in {cty})  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
