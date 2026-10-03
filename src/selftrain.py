"""Test-time self-training of stage 1 for every test country (targets the train->test shift).
Run after predict.py.

python -m src.selftrain [--max-q-per-country 120000] [--rounds 150]

1. pseudo-labels per country from test_scores.parquet (stage-2 p) via pseudo.select_pseudo
2. stage-1 features for up to --max-q-per-country pseudo-labelled queries per country
3. continue boosting each stage-1 fold model (+rounds trees, eta 0.03) on those rows mixed 1:1 with
   real-labelled train rows (anchor)
4. re-score ALL test candidates with the adapted stage 1 + original stage 2 / decision; write
   OUT_DIR/matching_results_selftrain.tsv and test_scores_selftrain.parquet
Unvalidated locally (test has no labels): per-country change stats are printed.
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
    ap.add_argument("--max-q-per-country", type=int, default=120_000)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--tau-hi", type=float, default=0.98)
    ap.add_argument("--delta", type=float, default=0.3)
    args = ap.parse_args()
    t0 = time.time()
    s1c = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "country"])
    scores = pl.read_parquet(C.p("test_scores.parquet"))
    q_src = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "src"]).rename({"idx": "q_idx", "src": "q_src"})
    ps = select_pseudo(scores, q_src, args.tau_hi, args.delta, 0.02)
    ps = ps.join(s1c.rename({"idx": "s1_idx"}), on="s1_idx")
    countries = s1c["country"].unique().sort().to_list()

    s1, q, idf = F.load_tables("test")
    F._IDF = idf
    fn1 = json.load(open(C.p("features_stage1.json")))
    fdir = C.t("feat", "pseudo_all")
    os.makedirs(fdir, exist_ok=True)
    for f in glob.glob(os.path.join(fdir, "*.parquet")):
        os.remove(f)
    n_rows, k = 0, 0
    for cty in countries:
        pc = ps.filter(pl.col("country") == cty)
        pos_q = pc.filter(pl.col("label") == 1)["q_idx"].unique()
        neg_q = pc.group_by("q_idx").agg(pl.col("label").max()).filter(pl.col("label") == 0)["q_idx"]
        n_pos = min(pos_q.len(), int(args.max_q_per_country * 0.85))
        n_neg = min(neg_q.len(), args.max_q_per_country - n_pos)
        pick = pl.concat([pos_q.sample(n_pos, seed=1), neg_q.sample(n_neg, seed=2)]).sort()
        ctx = (pl.read_parquet(C.t(f"test_ctx_{cty}.parquet")).filter(pl.col("q_idx").is_in(pick.implode()))
                 .sort("q_idx", "cos", descending=[False, True]))
        lab = pc.select("q_idx", "s1_idx", "label")
        for df in F.iter_chunks("test", ctx, s1, q, idf):
            # candidates below the stage-1 floor never entered test_scores: negatives as well
            df = df.join(lab, on=["q_idx", "s1_idx"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
            df.select(fn1 + ["label"]).write_parquet(os.path.join(fdir, f"part_{k:04d}.parquet"))
            n_rows += df.height
            k += 1
        print(f"[{cty}] pseudo queries {pick.len():,} (pos {n_pos:,}, distractor {n_neg:,}); rows so far {n_rows:,}  "
              f"{time.time()-t0:.0f}s", flush=True)
        del ctx

    tfiles = sorted(glob.glob(C.t("feat", "train", "*.parquet")))
    n_tr = pl.scan_parquet(tfiles).select(pl.len()).collect().item()
    take = np.zeros(n_tr, bool)
    take[np.random.default_rng(0).choice(n_tr, min(n_rows, n_tr), replace=False)] = True
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
    print(f"adaptation matrix: {int(take.sum()):,} train anchor + {len(ypl):,} pseudo rows (pos rate {ypl.mean():.4f})", flush=True)
    spec = json.load(open(C.p("xgb_stage1_folds.json")))
    new_paths = []
    for pth, it in zip(spec["models"], spec["iters"]):
        base = xgb.Booster(); base.load_model(pth)
        b = xgb.train(params, d, args.rounds, xgb_model=base[: int(it)])
        out = pth.replace(".json", "_st.json")
        b.save_model(out)
        new_paths.append(out)
        print(f"  adapted {os.path.basename(pth)}: +{args.rounds} trees  {time.time()-t0:.0f}s", flush=True)
    del d
    st_spec = C.p("xgb_stage1_folds_st.json")
    json.dump({"models": new_paths, "iters": [int(i) + args.rounds for i in spec["iters"]]}, open(st_spec, "w"))

    b1 = _FoldAvg.load(st_spec)
    parts = []
    for cty in countries:
        ctx_f = C.t(f"test_ctx_{cty}.parquet")
        qarr = pl.read_parquet(ctx_f, columns=["q_idx"])["q_idx"].to_numpy()
        uq = np.unique(qarr)
        for i in range(0, len(uq), 200_000):
            lo = int(np.searchsorted(qarr, uq[i], "left"))
            hi = int(np.searchsorted(qarr, uq[min(i + 200_000, len(uq)) - 1], "right"))
            df = F.pair_features(pl.scan_parquet(ctx_f).slice(lo, hi - lo).collect(), s1, q)
            p1 = b1.predict(xgb.DMatrix(df.select(fn1).to_numpy().astype(np.float32)))
            parts.append(df.select("q_idx", "s1_idx", *S2_RAW).with_columns(p1=pl.Series(p1, dtype=pl.Float32))
                           .filter(pl.col("p1") >= P1_FLOOR))
            del df
        print(f"  [{cty}] rescored {len(uq):,} queries  {time.time()-t0:.0f}s", flush=True)
    del s1, q
    cfg = json.load(open(C.p("decision.json")))
    fn2 = json.load(open(C.p("features_stage2.json")))
    b2 = xgb.Booster(); b2.load_model(C.p("xgb_stage2.json"))
    d2 = stage2_frame(pl.concat(parts))
    del parts
    p = b2.predict(xgb.DMatrix(d2.select(fn2).to_numpy().astype(np.float32)))
    new_sc = d2.select("q_idx", "s1_idx", p=pl.Series(p, dtype=pl.Float32))
    new_sc.write_parquet(C.p("test_scores_selftrain.parquet"))
    new_pred = select(argmax_q(new_sc), cfg["t"], cfg["mode"]).join(s1c.rename({"idx": "s1_idx"}), on="s1_idx")
    old_pred = select(argmax_q(scores), cfg["t"], cfg["mode"]).join(s1c.rename({"idx": "s1_idx"}), on="s1_idx")
    for cty in countries:
        o, n = old_pred.filter(pl.col("country") == cty), new_pred.filter(pl.col("country") == cty)
        same = o.join(n, on=["q_idx", "s1_idx"]).height
        print(f"[{cty}] assignments old {o.height:,} -> new {n.height:,}; unchanged {same:,} ({same / max(o.height, 1):.2%})", flush=True)

    s1_ids = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "entity_id"])
    q_ids = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "entity_id"]).sort("idx")["entity_id"]
    g = (new_pred.with_columns(qid=q_ids.gather(new_pred["q_idx"]))
                 .group_by("s1_idx").agg(pl.col("qid").unique().sort().str.join(",").alias("matched_entity_ids")))
    out = (s1_ids.join(g, left_on="idx", right_on="s1_idx", how="left")
                 .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null("")))
    path = os.path.join(C.OUT_DIR, "matching_results_selftrain.tsv")
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"-> {path}: {out.height:,} rows, non-empty {int((out['matched_entity_ids'] != '').sum()):,}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
