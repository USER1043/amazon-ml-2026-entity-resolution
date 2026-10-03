import os, gc, glob, json, time
import numpy as np
import polars as pl
import xgboost as xgb
from . import config as C
from . import features as F
from .model import S2_RAW, P1_FLOOR, stage2_frame, argmax_q, select


def main():
    t0 = time.time()
    fn1 = json.load(open(C.p("features_stage1.json")))
    fn2 = json.load(open(C.p("features_stage2.json")))
    cfg = json.load(open(C.p("decision.json")))
    part_dir = C.p("pred_parts")
    os.makedirs(part_dir, exist_ok=True)
    # Stage-0 cascade: the matching model only ever sees the survivors (cand/test_stage0.parquet)
    keep = pl.read_parquet(C.p("cand", "test_stage0.parquet"), columns=["q_idx", "s1_idx"]) if C.STAGE0 else None

    # Local 16 GB box: context tables and chunk bounds per country (windows never cross countries),
    # instead of one ~133M-row context table.
    jobs = []
    for cf in F.cand_countries("test"):
        country = os.path.basename(cf)[len("test_part_"):-len(".parquet")]
        ctx_f = C.t(f"test_ctx_{country}.parquet")
        if not os.path.exists(ctx_f):
            F.cand_context_to_parquet(cf, ctx_f)
            gc.collect()
            print(f"[{country}] context table written  {time.time()-t0:.0f}s", flush=True)
        qarr = pl.read_parquet(ctx_f, columns=["q_idx"])["q_idx"].to_numpy()
        uq = np.unique(qarr)
        # Chunk k = queries uq[k*step:(k+1)*step] (names must stay stable across resumes). A chunk not yet
        # written is scored as SPLIT halves (part_<c>_<k>_<j>): 400k-query chunks were OOM-killed at
        # chunk 15 on the 16 GB box, and 14 finished 400k parts are kept.
        step, split = 400_000, 2
        for k, i in enumerate(range(0, len(uq), step)):
            whole = os.path.join(part_dir, f"part_{country}_{k:04d}.parquet")
            if os.path.exists(whole):
                jobs.append((ctx_f, 0, 0, whole))
                continue
            sub = int(np.ceil(min(step, len(uq) - i) / split))
            for j, a in enumerate(range(i, min(i + step, len(uq)), sub)):
                b = min(a + sub, i + step, len(uq))
                lo = int(np.searchsorted(qarr, uq[a], "left"))
                hi = int(np.searchsorted(qarr, uq[b - 1], "right"))
                jobs.append((ctx_f, lo, hi, os.path.join(part_dir, f"part_{country}_{k:04d}_{j}.parquet")))
        del qarr, uq
    print(f"test: {len(jobs)} chunks  {time.time()-t0:.0f}s", flush=True)

    todo = [j for j in jobs if not os.path.exists(j[3])]
    if todo:
        s1, q, idf = F.load_tables("test")
        F._IDF = idf
        from .train import _FoldAvg
        folds = C.p("xgb_stage1_folds.json")   # local run: stage 1 = average of the two fold models
        if os.path.exists(folds):
            b1 = _FoldAvg.load(folds)
        else:
            b1 = xgb.Booster(); b1.load_model(C.p("xgb_stage1.json"))
        for k, (ctx_f, lo, hi, out_f) in enumerate(todo):
            sub = pl.scan_parquet(ctx_f).slice(lo, hi - lo).collect()
            if keep is not None:
                sub = sub.join(keep, on=["q_idx", "s1_idx"], how="semi", maintain_order="left")
            df = F.pair_features(sub, s1, q)
            p1 = b1.predict(xgb.DMatrix(df.select(fn1).to_numpy().astype(np.float32)))
            (df.select("q_idx", "s1_idx", *S2_RAW).with_columns(p1=pl.Series(p1, dtype=pl.Float32))
               .filter(pl.col("p1") >= P1_FLOOR)
               .write_parquet(out_f))
            print(f"  chunk {k+1}/{len(todo)} ({os.path.basename(out_f)}): {df.height:,} pairs scored  {time.time()-t0:.0f}s", flush=True)
            del sub, df, p1; gc.collect()
        del s1, q; gc.collect()

    d = pl.read_parquet(sorted(glob.glob(os.path.join(part_dir, "part_*.parquet"))))
    if keep is not None:   # idempotent; also makes parts scored before stage 0 existed safe to reuse
        d = d.join(keep, on=["q_idx", "s1_idx"], how="semi")
    if cfg["use_stage2"]:
        b2 = xgb.Booster(); b2.load_model(C.p("xgb_stage2.json"))
        d = stage2_frame(d)
        p = b2.predict(xgb.DMatrix(d.select(fn2).to_numpy().astype(np.float32)))
        d = d.select("q_idx", "s1_idx", p=pl.Series(p, dtype=pl.Float32))
    else:
        d = d.select("q_idx", "s1_idx", pl.col("p1").alias("p"))
    d.write_parquet(C.p("test_scores.parquet"))
    a = argmax_q(d)
    pred = select(a, cfg["t"], cfg["mode"])
    if cfg.get("fill") is not None:
        # fill rule: an S1 left empty by the decision gets its single best argmax record if p >= fill
        # (local +0.0012 F0.5; LB 0.9737 -> 0.9739). Picks only scored pairs -> matches stay within candidates.
        add = (a.filter(pl.col("p") >= cfg["fill"]).join(pred.select("s1_idx").unique(), on="s1_idx", how="anti")
                .sort(["s1_idx", "p"], descending=[False, True]).group_by("s1_idx").first().select("q_idx", "s1_idx"))
        pred = pl.concat([pred, add])
    del d, a; gc.collect()
    print(f"assigned {pred.height:,} records to {pred['s1_idx'].n_unique():,} S1  (cfg {cfg})", flush=True)

    s1_ids = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "entity_id"]).sort("idx")
    q_ids = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "entity_id"]).sort("idx")["entity_id"]
    os.makedirs(C.OUT_DIR, exist_ok=True)

    def write_lists(pairs, col, path, n_parts=8):
        """One row per S1 (empty list allowed). Written in S1 hash partitions so the ~133M
        candidate pairs are never grouped in one go (local 16 GB box)."""
        rows = nonempty = 0
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for part in range(n_parts):
                pp = pairs.filter(pl.col("s1_idx") % n_parts == part)
                g = (pp.with_columns(qid=q_ids.gather(pp["q_idx"]))
                       .group_by("s1_idx").agg(pl.col("qid").unique().sort().str.join(",").alias(col)))
                out = (s1_ids.filter(pl.col("idx") % n_parts == part)
                             .join(g, left_on="idx", right_on="s1_idx", how="left")
                             .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))
                rows += out.height
                nonempty += int((out[col] != "").sum())
                f.write(out.write_csv(separator="\t", quote_style="never", include_header=False))
                del pp, g, out
        return rows, nonempty

    rows, ne = write_lists(pred, "matched_entity_ids", os.path.join(C.OUT_DIR, "matching_results.tsv"))
    print(f"matching_results.tsv: {rows:,} rows, non-empty {ne:,}  {time.time()-t0:.0f}s", flush=True)
    cand = keep if keep is not None else pl.read_parquet(C.p("cand", "test.parquet"), columns=["q_idx", "s1_idx"])
    write_lists(cand, "candidate_entity_ids", os.path.join(C.OUT_DIR, "candidate_pairs.tsv"))
    print(f"candidate_pairs.tsv written  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
