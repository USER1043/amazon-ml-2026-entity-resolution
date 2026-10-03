"""Stage 1: read TSVs, learn normalization maps from train pairs, normalize all records -> parquet.

python -m src.prep
"""
import os, json, time, collections
import numpy as np
import polars as pl
from multiprocessing import Pool
from . import config as C
from . import normalize as N

IND = r"[ऀ-෿]"


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       encoding="utf8-lossy", truncate_ragged_lines=True)


def load_split(split):
    d = os.path.join(C.DATA_DIR, split)
    s1 = read_tsv(f"{d}/{split}_source1.tsv").with_columns(src=pl.lit(1, pl.Int8))
    qs = []
    for k in (2, 3):
        qs.append(read_tsv(f"{d}/{split}_source{k}.tsv").with_columns(src=pl.lit(k, pl.Int8)))
    q = pl.concat(qs)
    return s1, q


def load_pairs():
    gt = read_tsv(os.path.join(C.DATA_DIR, "train", "train_ground_truth.tsv"))
    return (gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"})
              .with_columns(pl.col("m").fill_null("").str.split(","))
              .explode("m").filter(pl.col("m").str.len_chars() > 0)
              .rename({"m": "q_id"}))


# ----------------------------------------------------------------------------- learned maps
def _best_map(df, a, b, min_cnt, min_share):
    tot = df.group_by(a).agg(pl.col("n").sum().alias("tot"))
    best = (df.sort("n", descending=True).group_by(a).first().join(tot, on=a)
              .filter((pl.col("n") >= min_cnt) & (pl.col("n") / pl.col("tot") >= min_share)))
    return dict(zip(best[a].to_list(), best[b].to_list()))


def learn_indic(s1, q, pairs):
    qi = q.filter(pl.col("business_name").str.contains(IND) |
                  pl.col("business_address").fill_null("").str.contains(IND))
    d = (pairs.join(qi.select(pl.col("entity_id").alias("q_id"), pl.col("business_name").alias("qn"),
                              pl.col("business_address").alias("qa")), on="q_id")
              .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("sn"),
                              pl.col("business_address").alias("sa")), on="s1_id"))
    # names: word-by-word translations -> positional alignment
    x = (d.filter(pl.col("qn").str.contains(IND))
          .with_columns(qt=pl.col("qn").str.extract_all(r"\S+"),
                        st=pl.col("sn").str.to_lowercase().str.extract_all(r"\S+"))
          .filter(pl.col("qt").list.len() == pl.col("st").list.len())
          .select("qt", "st").explode(["qt", "st"])
          .filter(pl.col("qt").str.contains(IND))
          .group_by("qt", "st").agg(n=pl.len()))
    tok = _best_map(x, "qt", "st", 2, 0.4)
    # addresses: whole comma components (mostly state / city names)
    y = (d.filter(pl.col("qa").fill_null("").str.contains(IND))
          .with_columns(qc=pl.col("qa").str.split(",").list.eval(pl.element().str.strip_chars()),
                        sc=pl.col("sa").str.to_lowercase().str.split(",").list.eval(pl.element().str.strip_chars()))
          .with_columns(ic=pl.col("qc").list.eval(pl.element().filter(pl.element().str.contains(IND))),
                        lc=pl.col("qc").list.eval(pl.element().filter(~pl.element().str.contains(IND)).str.to_lowercase()))
          .filter(pl.col("ic").list.len() == 1)
          .with_columns(ic=pl.col("ic").list.first(), cand=pl.col("sc").list.set_difference("lc"))
          .select("ic", "cand").explode("cand").drop_nulls()
          .group_by("ic", "cand").agg(n=pl.len()))
    comp = _best_map(y, "ic", "cand", 3, 0.3)
    return tok, comp


def _addr_alpha(args):
    maps, addrs = args
    N.set_maps(*maps)
    return [N.norm_addr(a, use_map=False)[3] for a in addrs]


def learn_addr_map(s1, q, pairs, maps, n=1_500_000):
    d = pairs.sample(min(n, pairs.height), seed=7)
    d = (d.join(q.select(pl.col("entity_id").alias("q_id"), pl.col("business_address").alias("qa")), on="q_id")
          .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_address").alias("sa")), on="s1_id"))
    qa, sa = d["qa"].to_list(), d["sa"].to_list()
    step = 100_000
    with Pool(C.N_PROC) as pool:
        qa_t = sum(pool.map(_addr_alpha, [(maps, qa[i:i + step]) for i in range(0, len(qa), step)]), [])
        sa_t = sum(pool.map(_addr_alpha, [(maps, sa[i:i + step]) for i in range(0, len(sa), step)]), [])
    cnt, tot = collections.Counter(), collections.Counter()
    for a, b in zip(qa_t, sa_t):
        A, B = set(a), set(b)
        da, db = A - B, B - A
        if len(da) == 1 and len(db) == 1:
            x, y = next(iter(da)), next(iter(db))
            cnt[(x, y)] += 1; tot[x] += 1
    m = {}
    for (x, y), c in cnt.items():
        if c >= 30 and c / tot[x] >= 0.5 and x != y and not x.isdigit() and not y.isdigit():
            m[x] = y
    # resolve chains a->b->c
    for k in list(m):
        seen = {k}
        while m[k] in m and m[m[k]] not in seen:
            seen.add(m[k]); m[k] = m[m[k]]
    return m


# ----------------------------------------------------------------------------- normalization
COLS = ["core", "legal", "is_domain", "is_indic", "addr_n", "nums", "num1", "atoks", "key_addr", "key_n2", "addr_empty"]


def _norm_chunk(args):
    maps, names, addrs = args
    N.set_maps(*maps)
    rows = [N.norm_record(a, b) for a, b in zip(names, addrs)]
    cols = list(zip(*rows)) if rows else [[] for _ in COLS]
    return pl.DataFrame({
        "core": list(cols[0]), "legal": list(cols[1]),
        "is_domain": pl.Series(cols[2], dtype=pl.Int8), "is_indic": pl.Series(cols[3], dtype=pl.Int8),
        "addr_n": list(cols[4]), "nums": pl.Series(list(cols[5]), dtype=pl.List(pl.String)),
        "num1": list(cols[6]), "atoks": pl.Series(list(cols[7]), dtype=pl.List(pl.String)),
        "key_addr": list(cols[8]), "key_n2": list(cols[9]), "addr_empty": pl.Series(cols[10], dtype=pl.Int8),
    })


def normalize_df(df, maps):
    names = df["business_name"].fill_null("").to_list()
    addrs = df["business_address"].to_list()
    step = 100_000
    jobs = [(maps, names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
    with Pool(C.N_PROC) as pool:
        parts = list(pool.imap(_norm_chunk, jobs))
    out = pl.concat(parts)
    out = pl.concat([df.select("entity_id", "country", "src"), out], how="horizontal")
    name = pl.concat_str([pl.col("core"), pl.col("legal")], separator=" ").str.strip_chars()
    return out.with_columns(
        pl.col("country").fill_null("UNK"),
        text=pl.concat_str([name, pl.col("addr_n")], separator=" , "),
        key_name=pl.col("core").str.split(" ").list.unique().list.sort().list.join(" "),
    ).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))


def normalize_to_parquet(df, maps, out, lab=None, rows=2_000_000):
    """normalize_df in row slices written to disk (local 16 GB box: the one-shot version of 10M records
    was OOM-killed). idx is offset per slice, so the result equals normalize_df(df) row for row."""
    parts = []
    for i in range(0, df.height, rows):
        part = normalize_df(df.slice(i, rows), maps).with_columns(pl.col("idx") + i)
        if lab is not None:   # label per slice: no global join/sort over all rows (that was OOM-killed)
            part = (part.join(lab, on="entity_id", how="left")
                        .with_columns(pl.col("label").fill_null(-1).cast(pl.Int32)).sort("idx"))
        f = f"{out}.part{i // rows:03d}"
        part.write_parquet(f)
        parts.append(f)
        del part
    pl.scan_parquet(parts).sink_parquet(out)       # slices are already in idx order: streaming concat
    for f in parts:
        os.remove(f)


def main():
    t0 = time.time()
    os.makedirs(C.p("norm"), exist_ok=True)
    s1, q = load_split("train")
    pairs = load_pairs()
    print(f"train s1={s1.height:,} q={q.height:,} pairs={pairs.height:,}", flush=True)
    learn = pairs
    if C.HOLDOUT_COUNTRY:
        learn = pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id") \
                     .filter(pl.col("country") != C.HOLDOUT_COUNTRY).drop("country")
        print(f"HOLDOUT_COUNTRY={C.HOLDOUT_COUNTRY}: maps learned from {learn.height:,} pairs", flush=True)
    if C.MAPS_EXCLUDE_SEEDS:
        # same mask as retrieve.seeds_mask (rng 42 over S1 rows in file order = S1 idx)
        seeds = np.random.default_rng(42).random(s1.height) < C.SEED_FRAC
        if not os.path.exists(C.p("seeds.npy")):
            np.save(C.p("seeds.npy"), seeds)
        seed_ids = s1.filter(pl.Series(seeds))["entity_id"]
        learn = learn.filter(~pl.col("s1_id").is_in(seed_ids.implode()))
        print(f"MAPS_EXCLUDE_SEEDS: maps learned without {int(seeds.sum()):,} seed S1 -> {learn.height:,} pairs", flush=True)
    ls1, lq = s1, q
    if C.EXTRA_PAIRS:
        # confident TEST pairs; test ids prefixed so they can never collide with train ids
        s1t, qt = load_split("test")
        tag = lambda df: df.with_columns(("T|" + pl.col("entity_id")).alias("entity_id"))
        ls1, lq = pl.concat([s1, tag(s1t)]), pl.concat([q, tag(qt)])
        xp = pl.read_parquet(C.EXTRA_PAIRS).select(("T|" + pl.col("s1_id")).alias("s1_id"), ("T|" + pl.col("q_id")).alias("q_id"))
        learn = pl.concat([learn.select("s1_id", "q_id"), xp])
        print(f"EXTRA_PAIRS: +{xp.height:,} test pairs for map learning -> {learn.height:,}", flush=True)
        del s1t, qt
    tok, comp = learn_indic(ls1, lq, learn)
    print(f"indic token map={len(tok):,} comp map={len(comp):,}  {time.time()-t0:.0f}s", flush=True)
    amap = learn_addr_map(ls1, lq, learn, (tok, comp, {}))
    del ls1, lq
    print(f"addr token map={len(amap):,}  e.g. {list(amap.items())[:25]}", flush=True)
    maps = (tok, comp, amap)
    json.dump({"indic_tok": tok, "indic_comp": comp, "addr_map": amap},
              open(C.p("norm", "maps.json"), "w"), ensure_ascii=False)

    s1n = normalize_df(s1, maps)
    s1n.write_parquet(C.p("norm", "train_s1.parquet"))
    lab = (pairs.join(s1n.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("label")), on="s1_id")
                .select(pl.col("q_id").alias("entity_id"), "label"))
    del s1n, s1
    normalize_to_parquet(q, maps, C.p("norm", "train_q.parquet"), lab=lab)
    n_m = pl.scan_parquet(C.p("norm", "train_q.parquet")).select((pl.col("label") >= 0).sum()).collect().item()
    print(f"train normalized {time.time()-t0:.0f}s; matched q={n_m:,}", flush=True)
    del q, lab

    s1, q = load_split("test")
    normalize_df(s1, maps).write_parquet(C.p("norm", "test_s1.parquet"))
    del s1
    normalize_to_parquet(q, maps, C.p("norm", "test_q.parquet"))
    print(f"test normalized {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
