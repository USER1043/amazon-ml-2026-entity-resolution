"""How optimistic is the local F0.5? (run after train.py)

python -m src.honest_eval [--frac 0.1]

The reranker is trained/evaluated on `train_included` queries only (records tied to seed S1s).
Records that truly belong to OTHER S1s but have a seed S1 among their candidates are never
scored, so a wrong assignment of such a record to a seed S1 (a false positive on the leaderboard)
is invisible locally. On a random sample of seed S1s this scores the same S1s two ways with the
SAME models (stage1 fold-average + stage2 + saved decision):
  (a) only train_included queries  (what train.py reports)
  (b) every query that has a sampled seed S1 among its candidates (what the leaderboard sees)
(a) - (b) = optimism caused by the missing queries.
"""
import argparse, json, os, time
import numpy as np
import polars as pl
import xgboost as xgb
from . import config as C
from . import features as F
from .model import S2_RAW, P1_FLOOR, stage2_frame, argmax_q, select, f05
from .train import _FoldAvg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.1)
    args = ap.parse_args()
    t0 = time.time()
    seeds = np.nonzero(np.load(C.p("seeds.npy")))[0].astype(np.int32)
    samp = np.sort(np.random.default_rng(7).choice(seeds, int(len(seeds) * args.frac), replace=False))
    samp_s = pl.Series("s1_idx", samp)
    keep = pl.Series("q_idx", np.load(C.p("train_included.npy")).astype(np.int32))
    touch = (pl.scan_parquet(C.p("cand", "train.parquet")).select("q_idx", "s1_idx")
               .join(pl.DataFrame(samp_s).lazy(), on="s1_idx", how="semi")
               .select("q_idx").unique().collect(engine="streaming")["q_idx"].sort())
    print(f"sampled seed S1 {len(samp):,}; queries touching them {touch.len():,} "
          f"(of which in train_included {int(touch.is_in(keep.implode()).sum()):,})  {time.time()-t0:.0f}s", flush=True)

    ctx = []
    for f in F.cand_countries("train"):
        o = C.t("honest_ctx_" + os.path.basename(f))
        F.cand_context_to_parquet(f, o, keep=touch)
        ctx.append(o)
    cand = pl.read_parquet(ctx).sort("q_idx", "cos", descending=[False, True], maintain_order=True)
    print(f"pairs {cand.height:,}  {time.time()-t0:.0f}s", flush=True)

    s1, q, idf = F.load_tables("train")
    fn1 = json.load(open(C.p("features_stage1.json")))
    fn2 = json.load(open(C.p("features_stage2.json")))
    cfg = json.load(open(C.p("decision.json")))
    b1 = _FoldAvg.load(C.p("xgb_stage1_folds.json"))
    parts = []
    for k, df in enumerate(F.iter_chunks("train", cand, s1, q, idf)):
        p1 = b1.predict(xgb.DMatrix(df.select(fn1).to_numpy().astype(np.float32)))
        parts.append(df.select("q_idx", "s1_idx", *S2_RAW).with_columns(p1=pl.Series(p1, dtype=pl.Float32))
                       .filter(pl.col("p1") >= P1_FLOOR))
        print(f"  chunk {k}: {df.height:,} pairs scored  {time.time()-t0:.0f}s", flush=True)
    del s1, q, cand
    d = pl.concat(parts)
    b2 = xgb.Booster(); b2.load_model(C.p("xgb_stage2.json"))
    d2 = stage2_frame(d)
    p = b2.predict(xgb.DMatrix(d2.select(fn2).to_numpy().astype(np.float32))) if cfg["use_stage2"] else d2["p1"].to_numpy()
    scores = d2.select("q_idx", "s1_idx", p=pl.Series(p, dtype=pl.Float32))

    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
    s1c = pl.read_parquet(C.p("norm", "train_s1.parquet"), columns=["idx", "country"])
    res = {}
    for tag, sc in (("a_train_included_only", scores.filter(pl.col("q_idx").is_in(keep.implode()))),
                    ("b_all_queries", scores)):
        pred = select(argmax_q(sc), cfg["t"], cfg["mode"])
        f, pr, rc = f05(pred, qlab, samp)
        res[tag] = {"f05": f, "P": pr, "R": rc}
        for c in s1c["country"].unique().sort().to_list():
            ev = np.intersect1d(samp, s1c.filter(pl.col("country") == c)["idx"].to_numpy())
            res[tag][c] = f05(pred, qlab, ev)[0]
        print(f"  ({tag}) F0.5 {f:.4f}  P {pr:.4f}  R {rc:.4f}  "
              + "  ".join(f"{c} {res[tag][c]:.4f}" for c in s1c['country'].unique().sort().to_list()), flush=True)
    gap = res["a_train_included_only"]["f05"] - res["b_all_queries"]["f05"]
    print(f"optimism from unscored queries: {gap:+.4f}  {time.time()-t0:.0f}s", flush=True)
    json.dump(res | {"optimism": gap, "n_s1": int(len(samp))}, open(C.p("honest_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
