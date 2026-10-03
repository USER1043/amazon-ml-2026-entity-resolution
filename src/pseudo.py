"""Pseudo-labels for an unlabelled country (Step 2 of the France plan). Model-agnostic: works on any
pair scores (q_idx, s1_idx, p).

Rules (each S2/S3 record matches at most one S1):
  positive   argmax pair of a query with p >= tau_hi and margin (p1 - p2) >= delta,
             unless its S1 is over-subscribed (> MAX_PER_SRC confident records from one source;
             train max is 5 S2 / 6 S3 per S1)
  negative   every other candidate of a positive query, and every candidate of a query whose best
             p <= tau_lo (confident distractor)
  everything else is left unlabelled.

python -m src.pseudo --scores <q_idx,s1_idx,p parquet> [--grid]              # proxy: precision grid
python -m src.pseudo --scores ... --tau-hi 0.98 --delta 0.3 --out pseudo.parquet
With labels available (train proxy) it reports positive precision / negative purity / coverage.
"""
import argparse
import numpy as np
import polars as pl
from . import config as C

MAX_PER_SRC = 6


def select_pseudo(scores, q_src, tau_hi, delta, tau_lo):
    """scores: q_idx, s1_idx, p. q_src: q_idx, q_src. -> q_idx, s1_idx, label (Int8)."""
    s = scores.sort(["q_idx", "p"], descending=[False, True]).with_columns(
        r=pl.int_range(pl.len()).over("q_idx"),
        p1=pl.col("p").max().over("q_idx"),
        p2=pl.col("p").top_k(2).min().over("q_idx"),
        n=pl.len().over("q_idx"))
    s = s.with_columns(margin=pl.when(pl.col("n") > 1).then(pl.col("p1") - pl.col("p2")).otherwise(pl.col("p1")))
    pos = (s.filter((pl.col("r") == 0) & (pl.col("p") >= tau_hi) & (pl.col("margin") >= delta))
             .join(q_src, on="q_idx", how="left"))
    over = pos.group_by("s1_idx", "q_src").len().filter(pl.col("len") > MAX_PER_SRC).select("s1_idx")
    pos = pos.join(over, on="s1_idx", how="anti")
    conf_q = pos.select("q_idx")
    neg_other = s.join(conf_q, on="q_idx", how="semi").filter(pl.col("r") > 0)
    neg_dis = s.filter(pl.col("p1") <= tau_lo)
    return pl.concat([
        pos.select("q_idx", "s1_idx", label=pl.lit(1, pl.Int8)),
        neg_other.select("q_idx", "s1_idx", label=pl.lit(0, pl.Int8)),
        neg_dis.select("q_idx", "s1_idx", label=pl.lit(0, pl.Int8)),
    ]).unique(["q_idx", "s1_idx"])


def quality(ps, qlab):
    """Against real labels (proxy country). qlab: idx, label (true S1 idx or -1)."""
    j = ps.join(qlab.rename({"idx": "q_idx", "label": "true"}), on="q_idx", how="left")
    pos = j.filter(pl.col("label") == 1)
    neg = j.filter(pl.col("label") == 0)
    prec = (pos["true"] == pos["s1_idx"]).mean() if pos.height else float("nan")
    purity = (neg["true"] != neg["s1_idx"]).mean() if neg.height else float("nan")
    n_true = qlab.filter(pl.col("label") >= 0).join(ps.select(pl.col("q_idx").alias("idx")).unique(), on="idx",
                                                    how="semi").height
    return dict(pos=pos.height, pos_precision=prec, neg=neg.height, neg_purity=purity,
                queries=ps["q_idx"].n_unique(), matched_queries_touched=n_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tau-hi", type=float, default=0.98)
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--tau-lo", type=float, default=0.02)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scores = pl.read_parquet(args.scores).select("q_idx", "s1_idx", "p")
    qn = pl.read_parquet(C.p("norm", f"{args.split}_q.parquet"),
                         columns=["idx", "src"] + (["label"] if args.split == "train" else []))
    q_src = qn.select(pl.col("idx").alias("q_idx"), pl.col("src").alias("q_src"))
    qlab = qn.select("idx", "label") if "label" in qn.columns else None
    n_q = scores["q_idx"].n_unique()
    if args.grid:
        assert qlab is not None, "--grid needs labels (train proxy)"
        print(f"{'tau_hi':>6} {'delta':>5} {'pos':>9} {'pos_prec':>8} {'share_q':>7} {'neg':>9} {'neg_pur':>7}")
        for tau in (0.9, 0.95, 0.98, 0.99, 0.995):
            for d in (0.0, 0.2, 0.5):
                q = quality(select_pseudo(scores, q_src, tau, d, args.tau_lo), qlab)
                print(f"{tau:6.3f} {d:5.2f} {q['pos']:9,} {q['pos_precision']:8.4f} {q['pos']/n_q:7.3f} "
                      f"{q['neg']:9,} {q['neg_purity']:7.4f}", flush=True)
        return
    ps = select_pseudo(scores, q_src, args.tau_hi, args.delta, args.tau_lo)
    print(f"pseudo rows {ps.height:,}: positives {int((ps['label'] == 1).sum()):,} "
          f"({(ps['label'] == 1).sum() / n_q:.3f} of queries), negatives {int((ps['label'] == 0).sum()):,}")
    if qlab is not None:
        print("quality vs real labels:", quality(ps, qlab))
    if args.out:
        ps.write_parquet(args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
