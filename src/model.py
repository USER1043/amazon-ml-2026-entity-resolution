"""Reranker helpers shared by train.py and predict.py: stage-2 context features, decision rule, F0.5."""
import numpy as np
import polars as pl

S2_RAW = ["cos", "gap_q", "rank_q", "n_tset", "n_wjac", "n_ratio", "cov_sq", "a_tset", "a_wjac",
          "m_num1_eq", "m_s1num_in_q", "num_lev", "m_logdiff", "m_q_none", "l_eq", "l_conflict",
          "q_addr_empty", "q_src"]
P1_FLOOR = 0.01   # pairs below this stage-1 prob are dropped before stage 2 (cannot be selected anyway)


def xgb_params(depth=8, eta=0.08):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    return {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "device": dev, "max_depth": depth, "eta": eta, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "lambda": 2.0, "max_bin": 256}


def stage2_frame(df):
    """df: q_idx, s1_idx, p1, S2_RAW columns (rows with p1 >= P1_FLOOR). Adds context features."""
    hi = (pl.col("p1") > 0.5).cast(pl.Float32)
    df = df.with_columns(hi=hi)
    df = df.with_columns(
        p_max_q=pl.col("p1").max().over("q_idx"),
        p_2_q=pl.col("p1").top_k(2).min().over("q_idx"),
        p_sum_q=pl.col("p1").sum().over("q_idx"),
        p_rank_q=pl.col("p1").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32),
        p_max_s1=pl.col("p1").max().over("s1_idx"),
        p_2_s1=pl.col("p1").top_k(2).min().over("s1_idx"),
        p_sum_s1=pl.col("p1").sum().over("s1_idx"),
        p_rank_s1=pl.col("p1").rank("ordinal", descending=True).over("s1_idx").cast(pl.Float32),
        hi_s1=pl.col("hi").sum().over("s1_idx") - pl.col("hi"),
        hi_s1_src=pl.col("hi").sum().over("s1_idx", "q_src") - pl.col("hi"),
        hi_numeq_s1=(pl.col("hi") * pl.col("m_num1_eq")).sum().over("s1_idx") - pl.col("hi") * pl.col("m_num1_eq"),
        hi_leq_s1=(pl.col("hi") * pl.col("l_eq")).sum().over("s1_idx") - pl.col("hi") * pl.col("l_eq"),
        n_s1_p=pl.len().over("s1_idx").cast(pl.Float32),
    )
    df = df.with_columns(
        p_other_q=pl.when(pl.col("p_rank_q") == 1).then(pl.col("p_2_q")).otherwise(pl.col("p_max_q")),
        p_other_s1=pl.when(pl.col("p_rank_s1") == 1).then(pl.col("p_2_s1")).otherwise(pl.col("p_max_s1")),
        hi_numeq_frac=pl.col("hi_numeq_s1") / pl.col("hi_s1").clip(lower_bound=1),
        hi_leq_frac=pl.col("hi_leq_s1") / pl.col("hi_s1").clip(lower_bound=1),
        logit_p1=(pl.col("p1").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p1").clip(1e-6, 1 - 1e-6))).log(),
    ).drop("p_2_q", "p_2_s1", "hi")
    return df


def s2_features(df):
    return [c for c in df.columns if c not in ("q_idx", "s1_idx", "label", "p2", "fold")]


def argmax_q(df, pcol="p"):
    """Best S1 per query (each S2/S3 record belongs to at most one S1)."""
    return (df.sort(["q_idx", pcol], descending=[False, True]).group_by("q_idx", maintain_order=True).first()
              .select("q_idx", "s1_idx", pl.col(pcol).alias("p")))


def select(a, t, mode="argmax"):
    """a: output of argmax_q. Keep pairs with p >= t; mode 'ef' additionally keeps, per S1, the top-j
    assigned records that maximise expected F0.5 (or none if an empty list scores better)."""
    a = a.filter(pl.col("p") >= t)
    if mode == "ef" and a.height:
        a = a.sort(["s1_idx", "p"], descending=[False, True]).with_columns(
            j=pl.int_range(1, pl.len() + 1).over("s1_idx"),
            cum=pl.col("p").cum_sum().over("s1_idx"),
            T=pl.col("p").sum().over("s1_idx"),
            e0=(1 - pl.col("p")).clip(1e-6, 1.0).log().sum().over("s1_idx").exp(),
        ).with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("T") + pl.col("j")))
        a = a.with_columns(best=pl.col("ef").max().over("s1_idx"))
        a = a.with_columns(jb=pl.col("j").filter(pl.col("ef") == pl.col("best")).min().over("s1_idx"))
        a = a.filter((pl.col("j") <= pl.col("jb")) & (pl.col("best") > pl.col("e0")))
    return a.select("q_idx", "s1_idx")


def decide(df, t, mode="argmax", pcol="p"):
    return select(argmax_q(df, pcol), t, mode)


def f05(pred, qlab, eval_s1):
    """Macro F0.5 over eval_s1 (numpy array of S1 idx), exactly as the leaderboard defines it."""
    ev = pl.DataFrame({"s1_idx": eval_s1.astype(np.int32)})
    truth = (qlab.filter(pl.col("label") >= 0).group_by("label").len()
                 .rename({"label": "s1_idx", "len": "T"}).with_columns(pl.col("s1_idx").cast(pl.Int32)))
    pr = pred.join(qlab.rename({"idx": "q_idx"}), on="q_idx", how="left")
    pc = pr.group_by("s1_idx").agg(P=pl.len(), TP=(pl.col("label") == pl.col("s1_idx")).sum())
    m = (ev.join(truth, on="s1_idx", how="left").join(pc, on="s1_idx", how="left")
           .fill_null(0))
    f = np.where((m["T"] == 0) & (m["P"] == 0), 1.0,
                 1.25 * m["TP"].to_numpy() / np.maximum(0.25 * m["T"].to_numpy() + m["P"].to_numpy(), 1e-9))
    prec = m["TP"].sum() / max(1, m["P"].sum())
    rec = m["TP"].sum() / max(1, m["T"].sum())
    return float(f.mean()), float(prec), float(rec)
