"""Stage 3: pair features.

python -m src.features train     # writes TMP/feat/train/part_*.parquet for the reranker sample
Test features are generated on the fly inside predict.py (too large to store).
"""
import os, sys, glob, time, math
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler, Levenshtein
from . import config as C

REC_COLS = ["idx", "src", "core", "legal", "is_domain", "is_indic", "addr_n", "nums", "num1", "atoks", "addr_empty"]


# ----------------------------------------------------------------------------- tables
def _idf_map(s1, col):
    n = s1.height
    df = (s1.select(pl.col(col).list.unique()).explode(col).drop_nulls()
            .group_by(col).len())
    idf = (np.log((n + 1) / (df["len"].to_numpy() + 1)) + 1.0).astype(np.float32)
    return dict(zip(df[col].to_list(), idf.tolist())), float(math.log(n + 1) + 1.0)


def _prep_rec(df, nidf, ndef, aidf, adef):
    df = df.with_columns(
        ctoks=pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)),
        ltoks=pl.col("legal").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)),
        ns=pl.col("core").str.replace_all(" ", ""),
    ).with_columns(
        ini=pl.col("ctoks").list.eval(pl.element().str.slice(0, 1)).list.join(""),
        nidf=pl.col("ctoks").list.eval(pl.element().replace_strict(nidf, default=ndef, return_dtype=pl.Float32)).list.sum(),
        aidf=pl.col("atoks").list.eval(pl.element().replace_strict(aidf, default=adef, return_dtype=pl.Float32)).list.sum(),
        n1int=pl.col("num1").str.slice(-9).cast(pl.Int64, strict=False),
    )
    return df


def load_tables(split):
    s1 = pl.read_parquet(C.p("norm", f"{split}_s1.parquet"), columns=REC_COLS).sort("idx")
    q = pl.read_parquet(C.p("norm", f"{split}_q.parquet"),
                        columns=REC_COLS + (["label"] if split == "train" else [])).sort("idx")
    s1_tmp = s1.with_columns(ctoks=pl.col("core").str.split(" "))
    nidf, ndef = _idf_map(s1_tmp, "ctoks")
    aidf, adef = _idf_map(s1, "atoks")
    s1 = _prep_rec(s1, nidf, ndef, aidf, adef)
    q = _prep_rec(q, nidf, ndef, aidf, adef)
    # v4: name uniqueness in the S1 index (how many S1 share this exact core name); matters most when the
    # record has no address. Left joins keep row order, so s1/q stay aligned with idx for positional gathers.
    dups = s1.group_by("core").agg(pl.len().cast(pl.Float32).alias("core_dups"))
    s1 = s1.join(dups, on="core", how="left", maintain_order="left")
    q = q.join(dups, on="core", how="left", maintain_order="left").with_columns(pl.col("core_dups").fill_null(0))
    return s1, q, (nidf, ndef, aidf, adef)


# ----------------------------------------------------------------------------- context (cos only)
def add_cos_context(cand):
    cand = cand.with_columns(
        cos_max_q=pl.col("cos").max().over("q_idx"),
        cos_2_q=pl.col("cos").top_k(2).min().over("q_idx"),
        n_q=pl.len().over("q_idx").cast(pl.Int16),
        rank_s1=pl.col("cos").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int32),
        cos_max_s1=pl.col("cos").max().over("s1_idx"),
        cos_2_s1=pl.col("cos").top_k(2).min().over("s1_idx"),
        n_s1=pl.len().over("s1_idx").cast(pl.Int32),
    )
    return cand.with_columns(
        gap_q=pl.col("cos") - pl.col("cos_max_q"),
        dq=pl.when(pl.col("rank_q") == 1).then(pl.col("cos") - pl.col("cos_2_q")).otherwise(pl.col("cos") - pl.col("cos_max_q")),
        margin_q=pl.col("cos_max_q") - pl.col("cos_2_q"),
        gap_s1=pl.col("cos") - pl.col("cos_max_s1"),
        ds1=pl.when(pl.col("rank_s1") == 1).then(pl.col("cos") - pl.col("cos_2_s1")).otherwise(pl.col("cos") - pl.col("cos_max_s1")),
        via_dense=(pl.col("via") & 1).cast(pl.Int8), via_addr=((pl.col("via") & 2) > 0).cast(pl.Int8),
        via_name=((pl.col("via") & 4) > 0).cast(pl.Int8), via_nn=((pl.col("via") & 8) > 0).cast(pl.Int8),
        via_n2=((pl.col("via") & 16) > 0).cast(pl.Int8),
    ).drop("cos_2_q", "cos_2_s1", "via")


def cand_countries(split):
    """Per-country candidate parts written by retrieve.py (context windows never cross countries)."""
    return sorted(glob.glob(C.p("cand", f"{split}_part_*.parquet")))


def load_cand_part(path):
    """Candidates of ONE country with rank_q and cos context. Local 16 GB machine: computing context
    per country gives identical values to the all-country table at a fraction of the RAM."""
    cand = pl.read_parquet(path)
    cand = cand.with_columns(rank_q=pl.col("cos").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16))
    return add_cos_context(cand).sort("q_idx", "cos", descending=[False, True])


def _derived():
    """The per-row columns of add_cos_context (same expressions), given the aggregates."""
    return [
        (pl.col("cos") - pl.col("cos_max_q")).alias("gap_q"),
        pl.when(pl.col("rank_q") == 1).then(pl.col("cos") - pl.col("cos_2_q")).otherwise(pl.col("cos") - pl.col("cos_max_q")).alias("dq"),
        (pl.col("cos_max_q") - pl.col("cos_2_q")).alias("margin_q"),
        (pl.col("cos") - pl.col("cos_max_s1")).alias("gap_s1"),
        pl.when(pl.col("rank_s1") == 1).then(pl.col("cos") - pl.col("cos_2_s1")).otherwise(pl.col("cos") - pl.col("cos_max_s1")).alias("ds1"),
        (pl.col("via") & 1).cast(pl.Int8).alias("via_dense"), ((pl.col("via") & 2) > 0).cast(pl.Int8).alias("via_addr"),
        ((pl.col("via") & 4) > 0).cast(pl.Int8).alias("via_name"), ((pl.col("via") & 8) > 0).cast(pl.Int8).alias("via_nn"),
        ((pl.col("via") & 16) > 0).cast(pl.Int8).alias("via_n2"),
    ]


def cand_context_to_parquet(path, out, keep=None):
    """Same result as load_cand_part(path) [optionally filtered to q_idx in keep], written to `out`,
    without ever holding the wide ~20-column table (local 16 GB box: 81M-row US train OOM-killed).
    Ranks use the identical window expressions on a narrow frame in the original row order; per-query
    and per-S1 aggregates are group_bys; the wide columns are added in a streaming, order-keeping join."""
    b = pl.read_parquet(path, columns=["q_idx", "s1_idx", "cos", "via"])
    b = b.with_columns(
        rank_q=pl.col("cos").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16),
        rank_s1=pl.col("cos").rank("ordinal", descending=True).over("s1_idx").cast(pl.Int32))
    q_agg = b.group_by("q_idx").agg(cos_max_q=pl.col("cos").max(), cos_2_q=pl.col("cos").top_k(2).min(),
                                    n_q=pl.len().cast(pl.Int16))
    s_agg = b.group_by("s1_idx").agg(cos_max_s1=pl.col("cos").max(), cos_2_s1=pl.col("cos").top_k(2).min(),
                                     n_s1=pl.len().cast(pl.Int32))
    if keep is not None:
        b = b.filter(pl.col("q_idx").is_in(keep.implode()))
        q_agg = q_agg.filter(pl.col("q_idx").is_in(keep.implode()))
    b = b.sort("q_idx", "cos", descending=[False, True], maintain_order=True)
    cols = ["q_idx", "s1_idx", "cos", "via", "rank_q", "cos_max_q", "cos_2_q", "n_q", "rank_s1",
            "cos_max_s1", "cos_2_s1", "n_s1"]
    (b.lazy().join(q_agg.lazy(), on="q_idx", how="left", maintain_order="left")
      .join(s_agg.lazy(), on="s1_idx", how="left", maintain_order="left")
      .select(cols).with_columns(_derived()).drop("cos_2_q", "cos_2_s1", "via")
      .sink_parquet(out))


def load_cand(split):
    return pl.concat([load_cand_part(f) for f in cand_countries(split)]).sort("q_idx", "cos", descending=[False, True])


def train_included(cand, qlab, seeds):
    """Queries used for reranker training + local evaluation (see README)."""
    seed_idx = pl.Series(np.nonzero(seeds)[0].astype(np.int32)).implode()
    a = qlab.filter(pl.col("label").is_in(seed_idx))["idx"]
    b = cand.filter((pl.col("rank_q") == 1) & pl.col("s1_idx").is_in(seed_idx))["q_idx"]
    dis = qlab.filter(pl.col("label") < 0)["idx"]
    c = cand.filter((pl.col("rank_q") <= 2) & pl.col("s1_idx").is_in(seed_idx) & pl.col("q_idx").is_in(dis.implode()))["q_idx"]
    return pl.concat([a, b.alias("idx"), c.alias("idx")]).unique().sort()


# ----------------------------------------------------------------------------- pair features
def _cp(a, b, scorer, dtype=np.float32):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=dtype).astype(np.float32)


def pair_features(sub, s1, q):
    """sub: candidate rows (q_idx, s1_idx, cos + context). Returns float32 feature frame."""
    qi = sub["q_idx"].to_numpy()
    si = sub["s1_idx"].to_numpy()
    Q = q[qi]
    S = s1[si]
    pf = pl.concat([sub, Q.select(pl.all().name.prefix("q_")).drop("q_idx"),
                    S.select(pl.all().name.prefix("s_")).drop("s_idx")], how="horizontal")
    pf = pf.with_row_index("rid")

    qc, sc = pf["q_core"].to_list(), pf["s_core"].to_list()
    qn, sn = pf["q_ns"].to_list(), pf["s_ns"].to_list()
    qa, sa = pf["q_addr_n"].to_list(), pf["s_addr_n"].to_list()
    q1, s1n = pf["q_num1"].to_list(), pf["s_num1"].to_list()
    rf = {
        "n_ratio": _cp(qc, sc, fuzz.ratio), "n_tsort": _cp(qc, sc, fuzz.token_sort_ratio),
        "n_tset": _cp(qc, sc, fuzz.token_set_ratio), "n_partial": _cp(qc, sc, fuzz.partial_ratio),
        "n_jw": _cp(qc, sc, JaroWinkler.normalized_similarity), "n_ns_ratio": _cp(qn, sn, fuzz.ratio),
        "n_ns_partial": _cp(qn, sn, fuzz.partial_ratio),
        "a_ratio": _cp(qa, sa, fuzz.ratio), "a_tset": _cp(qa, sa, fuzz.token_set_ratio),
        "a_tsort": _cp(qa, sa, fuzz.token_sort_ratio), "a_partial": _cp(qa, sa, fuzz.partial_ratio),
        "num_lev": _cp(q1, s1n, Levenshtein.distance),
    }
    del qc, sc, qn, sn, qa, sa, q1, s1n

    # substring coverage (handles concatenated / domain names)
    def cover(ns_col, tok_col):
        e = (pf.select("rid", ns_col, tok_col).explode(tok_col)
               .filter(pl.col(tok_col).str.len_chars() >= 3)
               .with_columns(h=pl.col(ns_col).str.contains(pl.col(tok_col), literal=True))
               .group_by("rid").agg(pl.col("h").mean()))
        return pf.select("rid").join(e, on="rid", how="left")["h"].fill_null(0).cast(pl.Float32)

    cov_sq = cover("q_ns", "s_ctoks")
    cov_qs = cover("s_ns", "q_ctoks")

    nint = pl.col("q_ctoks").list.set_intersection("s_ctoks")
    aint = pl.col("q_atoks").list.set_intersection("s_atoks")
    mint = pl.col("q_nums").list.set_intersection("s_nums")
    nidf, ndef, aidf, adef = _IDF
    out = pf.select(
        "q_idx", "s1_idx", "cos", "rank_q", "cos_max_q", "n_q", "rank_s1", "cos_max_s1", "n_s1",
        "gap_q", "dq", "margin_q", "gap_s1", "ds1", "via_dense", "via_addr", "via_name", "via_nn", "via_n2",
        q_src=pl.col("q_src"), q_is_domain=pl.col("q_is_domain"), s_is_domain=pl.col("s_is_domain"),
        q_is_indic=pl.col("q_is_indic"), q_addr_empty=pl.col("q_addr_empty"),
        n_eq=(pl.col("q_core") == pl.col("s_core")),
        n_sorted_eq=(pl.col("q_ctoks").list.sort() == pl.col("s_ctoks").list.sort()),
        n_inter=nint.list.len(),
        n_lq=pl.col("q_ctoks").list.len(), n_ls=pl.col("s_ctoks").list.len(),
        n_widf=nint.list.eval(pl.element().replace_strict(nidf, default=ndef, return_dtype=pl.Float32)).list.sum(),
        n_idf_q=pl.col("q_nidf"), n_idf_s=pl.col("s_nidf"),
        n_acr=((pl.col("q_ns") == pl.col("s_ini")) & (pl.col("s_ctoks").list.len() >= 2)) |
              ((pl.col("s_ns") == pl.col("q_ini")) & (pl.col("q_ctoks").list.len() >= 2)),
        n_chars_diff=(pl.col("q_ns").str.len_chars().cast(pl.Int32) - pl.col("s_ns").str.len_chars().cast(pl.Int32)).abs(),
        l_eq=(pl.col("q_legal") == pl.col("s_legal")),
        l_q_empty=(pl.col("q_ltoks").list.len() == 0), l_s_empty=(pl.col("s_ltoks").list.len() == 0),
        l_inter=pl.col("q_ltoks").list.set_intersection("s_ltoks").list.len(),
        a_inter=aint.list.len(), a_lq=pl.col("q_atoks").list.len(), a_ls=pl.col("s_atoks").list.len(),
        a_widf=aint.list.eval(pl.element().replace_strict(aidf, default=adef, return_dtype=pl.Float32)).list.sum(),
        a_idf_q=pl.col("q_aidf"), a_idf_s=pl.col("s_aidf"),
        m_inter=mint.list.len(), m_lq=pl.col("q_nums").list.len(), m_ls=pl.col("s_nums").list.len(),
        m_num1_eq=((pl.col("q_num1") == pl.col("s_num1")) & (pl.col("q_num1") != "")),
        m_s1num_in_q=pl.col("q_nums").list.contains(pl.col("s_num1")) & (pl.col("s_num1") != ""),
        m_qnum_in_s=pl.col("s_nums").list.contains(pl.col("q_num1")) & (pl.col("q_num1") != ""),
        m_q_none=(pl.col("q_num1") == ""), m_s_none=(pl.col("s_num1") == ""),
        m_len_eq=(pl.col("q_num1").str.len_chars() == pl.col("s_num1").str.len_chars()),
        m_logdiff=((pl.col("q_n1int") - pl.col("s_n1int")).abs().cast(pl.Float64) + 1).log(),
        # v4 house-number noise: truncation (1 vs 127), dropped/extra digit (1740 vs 17400), same tail digits,
        # and the second address number
        m_prefix=((pl.col("q_num1") != "") & (pl.col("s_num1") != "") & (pl.col("q_num1") != pl.col("s_num1")) &
                  (pl.col("q_num1").str.starts_with(pl.col("s_num1")) | pl.col("s_num1").str.starts_with(pl.col("q_num1")))),
        m_len_diff=(pl.col("q_num1").str.len_chars().cast(pl.Int32) - pl.col("s_num1").str.len_chars().cast(pl.Int32)).abs(),
        m_last2_eq=((pl.col("q_num1").str.len_chars() >= 2) & (pl.col("q_num1").str.slice(-2) == pl.col("s_num1").str.slice(-2))),
        m_num2_eq=(pl.col("q_nums").list.get(1, null_on_oob=True) == pl.col("s_nums").list.get(1, null_on_oob=True)).fill_null(False),
        s_core_dups=pl.col("s_core_dups"), q_core_dups=pl.col("q_core_dups"),
    )
    out = out.with_columns(
        n_jac=pl.col("n_inter") / (pl.col("n_lq") + pl.col("n_ls") - pl.col("n_inter")).clip(lower_bound=1),
        n_cont_q=pl.col("n_inter") / pl.col("n_lq").clip(lower_bound=1),
        n_cont_s=pl.col("n_inter") / pl.col("n_ls").clip(lower_bound=1),
        n_wjac=pl.col("n_widf") / (pl.col("n_idf_q") + pl.col("n_idf_s") - pl.col("n_widf")).clip(lower_bound=1e-3),
        l_conflict=(~pl.col("l_q_empty")) & (~pl.col("l_s_empty")) & (pl.col("l_inter") == 0),
        a_jac=pl.col("a_inter") / (pl.col("a_lq") + pl.col("a_ls") - pl.col("a_inter")).clip(lower_bound=1),
        a_cont_s=pl.col("a_inter") / pl.col("a_ls").clip(lower_bound=1),
        a_wjac=pl.col("a_widf") / (pl.col("a_idf_q") + pl.col("a_idf_s") - pl.col("a_widf")).clip(lower_bound=1e-3),
        a_wcont_s=pl.col("a_widf") / pl.col("a_idf_s").clip(lower_bound=1e-3),
        m_jac=pl.col("m_inter") / (pl.col("m_lq") + pl.col("m_ls") - pl.col("m_inter")).clip(lower_bound=1),
        m_cont_s=pl.col("m_inter") / pl.col("m_ls").clip(lower_bound=1),
        n_q_extra=pl.col("n_lq") - pl.col("n_inter"), n_s_miss=pl.col("n_ls") - pl.col("n_inter"),
        n_idf_q_extra=pl.col("n_idf_q") - pl.col("n_widf"), n_idf_s_miss=pl.col("n_idf_s") - pl.col("n_widf"),
        a_idf_q_extra=pl.col("a_idf_q") - pl.col("a_widf"), a_idf_s_miss=pl.col("a_idf_s") - pl.col("a_widf"),
        m_q_extra=pl.col("m_lq") - pl.col("m_inter"), m_s_miss=pl.col("m_ls") - pl.col("m_inter"),
        cov_sq=cov_sq, cov_qs=cov_qs,
        **{k: pl.Series(v) for k, v in rf.items()},
    )
    out = out.with_columns(m_edit1=(pl.col("num_lev") == 1) & ~pl.col("m_q_none") & ~pl.col("m_s_none"),
                           m_indel1=(pl.col("num_lev") == 1) & (pl.col("m_len_diff") == 1))
    out = out.with_columns(pl.when(pl.col("m_q_none") | pl.col("m_s_none"))
                           .then(-1.0).otherwise(pl.col("num_lev")).alias("num_lev"))
    ids = out.select("q_idx", "s1_idx")
    feats = out.drop("q_idx", "s1_idx").select(pl.all().cast(pl.Float32))
    return pl.concat([ids, feats], how="horizontal")


_IDF = None


def iter_chunks(split, cand, s1, q, idf, q_keep=None):
    """Yields feature frames for chunks of queries. cand must be sorted by q_idx."""
    global _IDF
    _IDF = idf
    if q_keep is not None:
        cand = cand.filter(pl.col("q_idx").is_in(q_keep.implode()))
    qarr = cand["q_idx"].to_numpy()
    uq = np.unique(qarr)
    for i in range(0, len(uq), C.CHUNK_Q):
        lo = np.searchsorted(qarr, uq[i], "left")
        hi = np.searchsorted(qarr, uq[min(i + C.CHUNK_Q, len(uq)) - 1], "right")
        yield pair_features(cand.slice(lo, hi - lo), s1, q)


def feature_names(df):
    return [c for c in df.columns if c not in ("q_idx", "s1_idx", "label")]


def main_train():
    t0 = time.time()
    seeds = np.load(C.p("seeds.npy"))
    # train_included only needs query labels + rank_q (stored by retrieve.py in cand/train.parquet);
    # the big lookup tables are loaded after the context pass so the two memory peaks never overlap
    qlab = pl.read_parquet(C.p("norm", "train_q.parquet"), columns=["idx", "label"])
    small = pl.read_parquet(C.p("cand", "train.parquet"), columns=["q_idx", "s1_idx", "rank_q"])
    keep = train_included(small, qlab, seeds)
    del small, qlab
    np.save(C.p("train_included.npy"), keep.to_numpy())
    ctx = []
    for f in cand_countries("train"):
        o = C.t("ctx_train_" + os.path.basename(f))
        cand_context_to_parquet(f, o, keep=keep)
        ctx.append(o)
        print(f"  context {os.path.basename(f)} done  {time.time()-t0:.0f}s", flush=True)
    s1, q, idf = load_tables("train")
    cand = pl.read_parquet(ctx).sort("q_idx", "cos", descending=[False, True], maintain_order=True)
    print(f"train: {cand.height:,} pairs total; reranker sample queries {keep.len():,}  {time.time()-t0:.0f}s", flush=True)
    out = C.t("feat", "train")
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*.parquet")):
        os.remove(f)
    lab = q.select(pl.col("idx").alias("q_idx"), "label")
    n = 0
    for k, df in enumerate(iter_chunks("train", cand, s1, q, idf)):
        df = df.join(lab, on="q_idx", how="left").with_columns(
            label=(pl.col("label") == pl.col("s1_idx")).cast(pl.Int8))
        df.write_parquet(os.path.join(out, f"part_{k:04d}.parquet"))
        n += df.height
        print(f"  chunk {k}: {df.height:,} rows (total {n:,}) pos rate {df['label'].mean():.4f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main_train()
