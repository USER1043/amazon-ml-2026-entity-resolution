"""Unseen-country simulation for the reranker (Step 1 of the France plan).

ER_HOLDOUT_COUNTRY=US ER_WORK=<loco work dir> python -m src.loco_train [--pseudo PATH --pseudo-weight 0.5]

Run after `prep`, `retrieve train` and `features train` in the same ER_WORK with the same
ER_HOLDOUT_COUNTRY (so maps and bi-encoder never saw the held-out country's labels).
Trains stage1/stage2 on the other countries only, then scores the held-out country's seed S1 with
its real labels, reporting:
  * F0.5 at the decision tuned on the seen countries (what France gets today),
  * the best F0.5 reachable on the held-out country (oracle t/mode: the headroom of re-tuning),
  * the same for the seen countries (in-domain reference).
--pseudo adds pseudo-labelled held-out rows (q_idx, s1_idx, label) to training (self-training, Step 3.4).
"""
import argparse, glob, json, time
import numpy as np
import polars as pl
from . import config as C
from .model import S2_RAW, P1_FLOOR, xgb_params, stage2_frame, s2_features, argmax_q, select, f05
from .train import oof_and_final, tune, _predict


def _matrix(lf, cols, rows=None):
    """Column-by-column float32 matrix (keeps peak RAM ~1x the matrix on a 16 GB box)."""
    n = lf.select(pl.len()).collect().item() if rows is None else int(rows.sum())
    X = np.empty((n, len(cols)), np.float32)
    for j, c in enumerate(cols):
        v = lf.select(pl.col(c).cast(pl.Float32)).collect()[c].to_numpy()
        X[:, j] = v if rows is None else v[rows]
    return X


def f05_grid(pdf, qlab, ev, tag):
    a = argmax_q(pdf)
    res = {(mode, float(t)): f05(select(a, t, mode), qlab, ev)[0]
           for mode in ("argmax", "ef") for t in np.round(np.arange(0.05, 0.951, 0.05), 2)}
    (bm, bt), bf = max(res.items(), key=lambda kv: kv[1])
    print(f"  [{tag}] oracle best F0.5 {bf:.4f} at t={bt} mode={bm}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo", default=None, help="parquet q_idx, s1_idx, label for held-out rows to add")
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    args = ap.parse_args()
    H = C.HOLDOUT_COUNTRY
    assert H, "set ER_HOLDOUT_COUNTRY"
    t0 = time.time()
    lf = pl.scan_parquet(sorted(glob.glob(C.t("feat", "train", "*.parquet"))))
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label", "country"])
    s1c = pl.read_parquet(C.p("norm", "train_s1.parquet"), columns=["idx", "country"])
    seeds = np.load(C.p("seeds.npy"))
    ids = lf.select("q_idx", "s1_idx", "label", *[c for c in S2_RAW if c not in ("q_idx", "s1_idx")]).collect()
    qc = qlab.select(pl.col("idx").alias("q_idx"), pl.col("country").alias("q_country"))
    ids = ids.join(qc, on="q_idx", how="left")
    is_h = (ids["q_country"] == H).to_numpy()
    fn1 = [c for c in lf.collect_schema().names() if c not in ("q_idx", "s1_idx", "label", "fold", "es")]
    print(f"rows {ids.height:,}: seen {int((~is_h).sum()):,}  held-out {H} {int(is_h.sum()):,}  {time.time()-t0:.0f}s", flush=True)

    # folds by home S1 (as train.py), early-stopping split
    rng = np.random.default_rng(123)
    fold_s1 = rng.integers(0, 2, s1c.height).astype(np.int8)
    home = ids.join(qlab.select(pl.col("idx").alias("q_idx"), pl.col("label").alias("true")), on="q_idx")
    home = home.with_columns(h=pl.when(pl.col("true") >= 0).then(pl.col("true")).otherwise(pl.col("s1_idx")))
    fold = fold_s1[home["h"].to_numpy()]
    es = rng.random(ids.height) < 0.08
    y = ids["label"].to_numpy().astype(np.float32)
    w = np.ones(ids.height, np.float32)

    train_rows = ~is_h
    if args.pseudo:
        ps = pl.read_parquet(args.pseudo).select("q_idx", "s1_idx", pl.col("label").cast(pl.Float32).alias("pl"))
        j = ids.select("q_idx", "s1_idx").with_row_index("r").join(ps, on=["q_idx", "s1_idx"])
        r = j["r"].to_numpy()
        y[r] = j["pl"].to_numpy(); w[r] = args.pseudo_weight; train_rows[r] = True
        print(f"  + {len(r):,} pseudo-labelled {H} rows (weight {args.pseudo_weight})", flush=True)

    X = _matrix(lf, fn1)
    tr = train_rows
    # stage 1 on seen (+pseudo) rows only; score everything with the final model
    import xgboost as xgb
    Xt, yt, ft, et, wt = X[tr], y[tr], fold[tr], es[tr], w[tr]
    oof = np.zeros(tr.sum(), np.float32)
    iters = []
    params = xgb_params(8, 0.08)
    for f in (0, 1):
        m_tr, m_es = (ft != f) & ~et, (ft != f) & et
        dtr = xgb.QuantileDMatrix(Xt[m_tr], yt[m_tr], weight=wt[m_tr])
        des = xgb.QuantileDMatrix(Xt[m_es], yt[m_es], weight=wt[m_es], ref=dtr)
        b = xgb.train(params, dtr, C.XGB_ROUNDS, evals=[(des, "es")], early_stopping_rounds=60, verbose_eval=500)
        oof[ft == f] = _predict(b, Xt[ft == f]); iters.append(b.best_iteration + 1)
        del dtr, des, b
    b1 = xgb.train(params, xgb.QuantileDMatrix(Xt, yt, weight=wt), int(np.mean(iters) * 1.1) + 1)
    p1 = _predict(b1, X, b1.num_boosted_rounds())
    p1[np.nonzero(tr)[0]] = oof                      # seen rows keep out-of-fold scores
    del X, Xt
    print(f"  stage1 done {time.time()-t0:.0f}s", flush=True)

    seed_idx = np.nonzero(seeds)[0]
    s1_country = s1c["country"].to_numpy()
    ev_h = seed_idx[s1_country[seed_idx] == H]
    ev_seen = seed_idx[s1_country[seed_idx] != H]
    qlab2 = qlab.select("idx", "label")
    base = ids.select("q_idx", "s1_idx").with_columns(p=pl.Series(p1))
    best_seen = tune(base.filter(pl.Series(~is_h)), qlab2, ev_seen, "stage1 seen")
    grid = f05_grid(base.filter(pl.Series(is_h)), qlab2, ev_h, f"stage1 held-out {H}")
    f_h = grid[(best_seen[4], best_seen[3])]
    print(f"  => {H} F0.5 at seen-tuned decision (t={best_seen[3]}, {best_seen[4]}): {f_h:.4f}", flush=True)
    json.dump({"H": H, "seen_f05": best_seen[0], "heldout_f05_at_seen_decision": f_h,
               "heldout_oracle": max(grid.values()), "pseudo": args.pseudo, "pseudo_weight": args.pseudo_weight},
              open(C.p(f"loco_{H}{'_pseudo' if args.pseudo else ''}.json"), "w"), indent=1)
    base.filter(pl.Series(is_h)).write_parquet(C.p(f"loco_{H}_scores.parquet"))   # input to pseudo.py
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
