"""Stage 4: train the reranker (XGBoost, GPU) with 2-fold out-of-fold predictions, a stacked
stage-2 model on context features, and tune the decision rule on local macro F0.5.

python -m src.train
"""
import os, glob, json, time
import numpy as np
import polars as pl
import xgboost as xgb
from . import config as C
from .model import S2_RAW, P1_FLOOR, xgb_params, stage2_frame, s2_features, argmax_q, select, f05


def _fit(X, y, tr, es, params, rounds):
    dtr = xgb.QuantileDMatrix(X[tr & ~es], y[tr & ~es])
    des = xgb.QuantileDMatrix(X[tr & es], y[tr & es], ref=dtr)
    bst = xgb.train(params, dtr, rounds, evals=[(des, "es")], early_stopping_rounds=60, verbose_eval=250)
    del dtr, des
    return bst


def _predict(bst, X, it=None):
    it = bst.best_iteration + 1 if it is None else it
    out = np.empty(X.shape[0], np.float32)
    step = 5_000_000
    for i in range(0, X.shape[0], step):
        out[i:i + step] = bst.predict(xgb.DMatrix(X[i:i + step]), iteration_range=(0, it))
    return out


def oof_and_final(X, y, fold, es, params, rounds, name):
    oof = np.zeros(len(y), np.float32)
    iters = []
    for f in (0, 1):
        t0 = time.time()
        bst = _fit(X, y, fold != f, es, params, rounds)
        te = fold == f
        oof[te] = _predict(bst, X[te])
        iters.append(bst.best_iteration + 1)
        print(f"  {name} fold {f}: best_iter {iters[-1]}  {time.time()-t0:.0f}s", flush=True)
        del bst
    n_final = int(np.mean(iters) * 1.1) + 1
    t0 = time.time()
    dall = xgb.QuantileDMatrix(X, y)
    final = xgb.train(params, dall, n_final)
    del dall
    final.save_model(C.p(f"xgb_{name}.json"))
    print(f"  {name} final model: {n_final} rounds  {time.time()-t0:.0f}s", flush=True)
    return oof, final


# ----------------------------------------------------------------------------- stage 1 from parquet parts
# Local 16 GB box: the 20.6M x 88 float32 matrix (7.2 GB) plus the per-fold row copies was OOM-killed.
# Stage 1 therefore streams the feature parts: XGBoost builds its quantised matrix from batches
# (DataIter), and predictions are made part by part. Same rows, same order, same model.
WIN = 1_000_000   # rows read per batch (~0.35 GB float32 at 88 features)


def _windows(files):
    """(file, row offset in file, global row offset, n rows) windows of <= WIN rows, in global order."""
    out, g = [], 0
    for f in files:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        for o in range(0, n, WIN):
            out.append((f, o, g + o, min(WIN, n - o)))
        g += n
    return out


def _read(f, o, n, cols):
    return pl.scan_parquet(f).slice(o, n).select(cols).collect().to_numpy().astype(np.float32, copy=False)


class _Parts(xgb.DataIter):
    def __init__(self, files, cols, mask, y):
        self.win, self.cols, self.mask, self.y = _windows(files), cols, mask, y
        self.i = 0
        super().__init__()

    def next(self, input_data):
        while self.i < len(self.win):
            f, o, g, n = self.win[self.i]
            self.i += 1
            m = self.mask[g:g + n]
            if not m.any():
                continue
            X = _read(f, o, n, self.cols)
            input_data(data=X[m], label=self.y[g:g + n][m])
            return True
        return False

    def reset(self):
        self.i = 0


def _predict_parts(bst, files, cols, mask, it):
    out = []
    for f, o, g, n in _windows(files):
        m = mask[g:g + n]
        if m.any():
            X = _read(f, o, n, cols)
            out.append(bst.predict(xgb.DMatrix(X[m]), iteration_range=(0, it)))
            del X
    return np.concatenate(out).astype(np.float32)


def oof_and_final_parts(files, cols, y, fold, es, params, rounds, name):
    """Stage 1 on the local box. The final model is the pair of fold models (predictions averaged):
    one model on all 20.6M rows fits neither the 8 GB GPU nor the RAM left for CPU training.
    Returns out-of-fold predictions and a _FoldAvg booster saved as xgb_{name}_folds.json."""
    mb = params.get("max_bin", 256)
    oof, iters = _folds_parts(files, cols, y, fold, es, params, rounds, name, mb)
    if C.STAGE1_FINAL == "full":                 # original recipe: one model on all rows (16 GB GPU)
        n_final = int(np.mean(iters) * 1.1) + 1
        t0 = time.time()
        dall = xgb.QuantileDMatrix(_Parts(files, cols, np.ones(len(y), bool), y), max_bin=mb)
        final = xgb.train(params, dall, n_final)
        del dall
        final.save_model(C.p(f"xgb_{name}.json"))
        if os.path.exists(C.p(f"xgb_{name}_folds.json")):
            os.remove(C.p(f"xgb_{name}_folds.json"))
        print(f"  {name} final model (full data): {n_final} rounds  {time.time()-t0:.0f}s", flush=True)
        return oof, final
    paths = [C.p(f"xgb_{name}_fold{f}.json") for f in (0, 1)]
    json.dump({"models": paths, "iters": [int(i) for i in iters]}, open(C.p(f"xgb_{name}_folds.json"), "w"))
    print(f"  {name} final = average of fold models {[int(i) for i in iters]}", flush=True)
    return oof, _FoldAvg.load(C.p(f"xgb_{name}_folds.json"))


class _FoldAvg:
    """Average of fold boosters, each at its own best iteration. Duck-types what train/predict use."""
    def __init__(self, boosters, iters):
        self.b, self.it = boosters, iters

    @classmethod
    def load(cls, spec_path):
        spec = json.load(open(spec_path))
        bs = []
        for pth in spec["models"]:
            b = xgb.Booster(); b.load_model(pth); bs.append(b)
        return cls(bs, spec["iters"])

    def predict(self, dm, iteration_range=None):
        return np.mean([b.predict(dm, iteration_range=(0, it)) for b, it in zip(self.b, self.it)], axis=0)

    def get_score(self, importance_type="gain"):
        return self.b[0].get_score(importance_type=importance_type)


def _folds_parts(files, cols, y, fold, es, params, rounds, name, mb):
    oof = np.zeros(len(y), np.float32)
    iters = []
    for f in (0, 1):
        fc = C.p(f"oof_{name}_fold{f}.npz")
        mf = C.p(f"xgb_{name}_fold{f}.json")
        if os.path.exists(fc) and os.path.exists(mf):   # this fold finished in an earlier (stopped) run
            z = np.load(fc)
            oof[fold == f] = z["p"]
            iters.append(int(z["it"]))
            print(f"  {name} fold {f}: reusing cached fold, best_iter {iters[-1]}", flush=True)
            continue
        t0 = time.time()
        tr = fold != f
        dtr = xgb.QuantileDMatrix(_Parts(files, cols, tr & ~es, y), max_bin=mb)
        des = xgb.QuantileDMatrix(_Parts(files, cols, tr & es, y), ref=dtr, max_bin=mb)
        bst = xgb.train(params, dtr, rounds, evals=[(des, "es")], early_stopping_rounds=60, verbose_eval=250)
        del dtr, des
        oof[fold == f] = _predict_parts(bst, files, cols, fold == f, bst.best_iteration + 1)
        iters.append(bst.best_iteration + 1)
        bst.save_model(mf)
        np.savez(fc, p=oof[fold == f], it=iters[-1])
        print(f"  {name} fold {f}: best_iter {iters[-1]}  {time.time()-t0:.0f}s", flush=True)
        del bst
    return oof, iters


def tune(pdf, qlab, seed_idx, tag):
    a = argmax_q(pdf)
    best = None
    for mode in ("argmax", "ef"):
        for t in np.round(np.arange(0.10, 0.951, 0.05), 2):
            f, pr, rc = f05(select(a, t, mode), qlab, seed_idx)
            if best is None or f > best[0]:
                best = (f, pr, rc, float(t), mode)
    print(f"  [{tag}] best local F0.5 = {best[0]:.4f} (P {best[1]:.4f} R {best[2]:.4f}) at t={best[3]} mode={best[4]}", flush=True)
    for mode in ("argmax", "ef"):
        row = []
        for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            row.append(f"t{t}:{f05(select(a, t, mode), qlab, seed_idx)[0]:.4f}")
        print(f"    {mode}: " + "  ".join(row), flush=True)
    return best


def main():
    t0 = time.time()
    files = sorted(glob.glob(C.t("feat", "train", "*.parquet")))
    lf = pl.scan_parquet(files)
    feats = lf.select(list(dict.fromkeys(["q_idx", "s1_idx", "label", "rank_q", *S2_RAW]))).collect()
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
    s1c = pl.read_parquet(C.p("norm", "train_s1.parquet"), columns=["idx", "country"])
    seeds = np.load(C.p("seeds.npy"))
    seed_idx = np.nonzero(seeds)[0]
    print(f"features: {feats.shape}  pos rate {feats['label'].mean():.4f}  {time.time()-t0:.0f}s", flush=True)

    # folds by 'home' S1 (true S1, or top-1 candidate for unmatched records) so twins stay with their S1
    rng = np.random.default_rng(123)
    fold_s1 = rng.integers(0, 2, s1c.height).astype(np.int8)
    top1 = feats.filter(pl.col("rank_q") == 1).group_by("q_idx").agg(pl.col("s1_idx").first().alias("top"))
    qf = (qlab.rename({"idx": "q_idx"}).join(top1, on="q_idx", how="inner")
              .with_columns(home=pl.when(pl.col("label") >= 0).then(pl.col("label")).otherwise(pl.col("top"))))
    qf = qf.with_columns(fold=pl.Series(fold_s1[qf["home"].to_numpy()]),
                         es=pl.Series(rng.random(qf.height) < 0.08))
    feats = feats.join(qf.select("q_idx", "fold", "es"), on="q_idx", how="left")
    fn1 = [c for c in lf.collect_schema().names() if c not in ("q_idx", "s1_idx", "label", "fold", "es")]
    json.dump(fn1, open(C.p("features_stage1.json"), "w"))
    base = feats.select("q_idx", "s1_idx", "label", "fold", "es", *S2_RAW)
    del feats
    y = base["label"].to_numpy().astype(np.float32)
    fold = base["fold"].to_numpy()
    es = base["es"].to_numpy()

    print("stage 1 ...", flush=True)
    oof1, b1 = oof_and_final_parts(files, fn1, y, fold, es, xgb_params(8, 0.08), C.XGB_ROUNDS, "stage1")
    imp = sorted(b1.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:25]
    print("  top gain features:", ", ".join(f"{fn1[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k}" for k, _ in imp), flush=True)
    base = base.with_columns(p1=pl.Series(oof1))
    best1 = tune(base.select("q_idx", "s1_idx", pl.col("p1").alias("p")), qlab, seed_idx, "stage1")

    print("stage 2 ...", flush=True)
    d2 = stage2_frame(base.filter(pl.col("p1") >= P1_FLOOR))
    fn2 = [c for c in s2_features(d2) if c not in ("es",)]
    json.dump(fn2, open(C.p("features_stage2.json"), "w"))
    X2 = d2.select(fn2).to_numpy().astype(np.float32, copy=False)
    oof2, b2 = oof_and_final(X2, d2["label"].to_numpy().astype(np.float32), d2["fold"].to_numpy(),
                             d2["es"].to_numpy(), xgb_params(6, 0.05), 1500, "stage2")
    best2 = tune(d2.select("q_idx", "s1_idx", p=pl.Series(oof2)), qlab, seed_idx, "stage2")

    use2 = best2[0] >= best1[0]
    best = best2 if use2 else best1
    cfg = {"use_stage2": bool(use2), "t": best[3], "mode": best[4], "local_f05": best[0],
           "stage1_f05": best1[0], "stage2_f05": best2[0],
           # fill rule (used by predict.py): an S1 left empty gets its single best record if p >= fill.
           # 0.1 = best on local F0.5 (+0.0012) and confirmed on the leaderboard (0.9737 -> 0.9739)
           "fill": 0.1}
    json.dump(cfg, open(C.p("decision.json"), "w"), indent=1)
    print("DECISION:", cfg, flush=True)

    # per-country breakdown of the chosen setting
    pdf = (d2.select("q_idx", "s1_idx", p=pl.Series(oof2)) if use2
           else base.select("q_idx", "s1_idx", pl.col("p1").alias("p")))
    pred = select(argmax_q(pdf), best[3], best[4])
    for c in s1c["country"].unique().to_list():
        ev = np.intersect1d(seed_idx, s1c.filter(pl.col("country") == c)["idx"].to_numpy())
        f, pr, rc = f05(pred, qlab, ev)
        print(f"  {c}: F0.5 {f:.4f}  P {pr:.4f}  R {rc:.4f}  (S1 evaluated {len(ev):,})", flush=True)
    # save OOF for error analysis
    pdf.join(qlab.rename({"idx": "q_idx"}), on="q_idx").write_parquet(C.p("oof_pred.parquet"))
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
