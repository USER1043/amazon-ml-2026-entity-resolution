"""Ensemble two rerankers' stage-2 scores (same pipeline family => identical q_idx / s1_idx), then decide.

Local (both OOF files present):
  python -m src.ensemble --oof-a data_tm/oof_pred.parquet --oof-b <friend>/oof_pred.parquet
      -> F0.5 grid over weight w (p = w*pA + (1-w)*pB), threshold t, fill x; prints the best
Test:
  python -m src.ensemble --test-a data_tm/test_scores.parquet --test-b <friend>/test_scores.parquet \
                         --w 0.5 --t 0.15 --fill 0.1 --out output_tm/matching_results_ens.tsv

A pair scored by only one model keeps that model's score times the other model's 'absent' value
(`--absent`, default 0: treated as 'not a match' by the missing model, i.e. averaging with 0).
"""
import argparse, itertools, json, os
import numpy as np
import polars as pl
from . import config as C
from .model import argmax_q, select, f05


def combine(a, b, w, absent):
    a = a.select("q_idx", "s1_idx", pl.col("p").alias("pa"))
    b = b.select("q_idx", "s1_idx", pl.col("p").alias("pb"))
    m = a.join(b, on=["q_idx", "s1_idx"], how="full", coalesce=True)
    return m.select("q_idx", "s1_idx",
                    p=(w * pl.col("pa").fill_null(absent) + (1 - w) * pl.col("pb").fill_null(absent)).cast(pl.Float32))


def decide(sc, t, mode, x):
    a = argmax_q(sc)
    pred = select(a, t, mode)
    if x is not None:
        add = (a.filter(pl.col("p") >= x).join(pred.select("s1_idx").unique(), on="s1_idx", how="anti")
                .sort(["s1_idx", "p"], descending=[False, True]).group_by("s1_idx").first().select("q_idx", "s1_idx"))
        pred = pl.concat([pred, add])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-a"); ap.add_argument("--oof-b")
    ap.add_argument("--test-a"); ap.add_argument("--test-b")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--t", type=float, default=0.15)
    ap.add_argument("--fill", type=float, default=0.1)
    ap.add_argument("--mode", default="ef")
    ap.add_argument("--absent", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.oof_a and args.oof_b:
        seeds = np.nonzero(np.load(C.p("seeds.npy")))[0]
        qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
        A, B = pl.read_parquet(args.oof_a), pl.read_parquet(args.oof_b)
        # score only S1 evaluated by BOTH (same seed set expected)
        print(f"OOF rows A {A.height:,}  B {B.height:,}  shared pairs "
              f"{A.join(B, on=['q_idx', 's1_idx'], how='semi').height:,}")
        res = []
        for w, t, x in itertools.product((1.0, 0.0, 0.3, 0.5, 0.7), (0.1, 0.15, 0.25), (None, 0.1)):
            f = f05(decide(combine(A, B, w, args.absent), t, args.mode, x), qlab, seeds)[0]
            res.append((f, w, t, x))
            print(f"  w={w:.1f} t={t:.2f} fill={x}  F0.5 {f:.5f}", flush=True)
        best = max(res)
        print(f"BEST: F0.5 {best[0]:.5f} at w={best[1]} t={best[2]} fill={best[3]}  "
              f"(A alone best {max(r for r in res if r[1] == 1.0)[0]:.5f}, B alone best {max(r for r in res if r[1] == 0.0)[0]:.5f})")

    if args.test_a and args.test_b and args.out:
        sc = combine(pl.read_parquet(args.test_a), pl.read_parquet(args.test_b), args.w, args.absent)
        pred = decide(sc, args.t, args.mode, None if args.fill < 0 else args.fill)
        s1_ids = pl.read_parquet(C.p("norm", "test_s1.parquet"), columns=["idx", "entity_id"])
        q_ids = pl.read_parquet(C.p("norm", "test_q.parquet"), columns=["idx", "entity_id"]).sort("idx")["entity_id"]
        g = (pred.with_columns(qid=q_ids.gather(pred["q_idx"])).group_by("s1_idx")
                 .agg(pl.col("qid").unique().sort().str.join(",").alias("matched_entity_ids")))
        out = (s1_ids.join(g, left_on="idx", right_on="s1_idx", how="left")
                     .select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null("")))
        out.write_csv(args.out, separator="\t", quote_style="never")
        print(f"-> {args.out}: pairs {pred.height:,}, non-empty S1 {(out['matched_entity_ids'] != '').sum():,}")


if __name__ == "__main__":
    main()
