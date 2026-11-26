# Interview Prep: Business Entity Resolution (Amazon ML Challenge 2026)

Final result: local macro F0.5 **0.9901** on held-out entities, public leaderboard **0.9835**.

## 1. The 60-second pitch

> We had to link noisy business records from two sources (S2, S3) to a master list (S1). That is 10M records
> against 2M entities, so you cannot compare everything. We built a **cascade**. Cheap models cut the candidates
> down first. Progressively more expensive models then score only what survives. A fine-tuned MiniLM bi-encoder
> plus exact-key blocks retrieve ~77 candidates per entity. A LightGBM "stage 0" trims that to ~11 without losing
> true matches. Two XGBoost rankers on ~95 hand-built features score the survivors. Four small transformer
> cross-encoders refine a ~4.5-pair shortlist. Finally we pick, per entity, the set of records that maximises
> *expected F0.5*.

## 2. The problem

- **Task:** for each S1 entity, list the S2/S3 records that are the same business. Metric: macro **F0.5**
  (precision weighted more than recall), singletons included.
- **Scale:** train 2.21M S1 and 10.3M S2/S3 records. Test 1.73M S1 and 9.97M S2/S3 records.
- **Structure we exploit:**
  - Every true pair shares the country label.
  - Each S2/S3 record matches at most one S1.
  - ~26% of S2/S3 records are distractors that match nothing.
  - 5.6% of S1 are singletons.
- **Noise:**
  - Only 14.7% of true pairs have an identical normalized name.
  - Legal suffixes vary (Pvt Ltd / Private Limited / Inc / LLC).
  - Names can be domain-style, or contain d/b/a clauses.
  - Names can be Indic-script translations.
  - Addresses are abbreviated, reordered or missing (~3% empty), and house numbers have prefixes and padding.
- **Hard negatives ("twins"):** a different business with the same name at a nearby address, such as chain
  stores or the same building. These cause most errors.
- **Distribution shift:** test adds an unseen country (France) and has ~1.9x more distractors per S1. So country
  is never a model feature.

## 3. Pipeline at a glance

```
normalize (learned maps)
  -> retrieve: bi-encoder top-12 + 4 exact-key blocks      ~77 candidates/S1
  -> stage 0: LightGBM on retrieval-only signals, tau=0.001 ~11 candidates/S1
  -> stage 1: XGBoost, ~95 pair features
  -> stage 2: XGBoost on context of stage-1 probabilities   ~4.5 pairs/S1 shortlist
  -> stage 3: 4 cross-encoders + XGBoost stacker
  -> per-record argmax -> per-S1 expected-F0.5 subset -> fill rule -> matching_results.tsv
```

**The idea in one line:** spend expensive models only where cheap ones are uncertain.

## 4. Stage by stage

### 4.1 Normalization (`prep.py`, `normalize.py`)
- All maps are **learned from train pairs**, not hand-written:
  - An Indic-script to Latin name-token map (~1.3k tokens, from word-aligned true pairs).
  - An Indic address-component map.
  - An address token canonicalization map (abbreviations, state full vs abbreviated, typos, from pairs that
    differ by one token).
- Names are split into core name / legal form / domain handling.
- Addresses are split into cleaned text, numbers (leading zeros stripped), first house number and blocking keys.

### 4.2 Retrieval, recall-oriented (`biencoder.py`, `retrieve.py`)
- **Dense route:**
  - Model: `all-MiniLM-L6-v2` (22M parameters), fine-tuned with **in-batch negatives**.
  - Training data: 800k non-seed (record, S1) pairs, batch 384, lr 1e-4, 1 epoch, max length 64.
  - Each S2/S3 record ("name , address") retrieves its **top-12 S1 inside the same country label**.
  - The search is a GPU matrix product, run per country.
- **Exact-key routes (4):**
  - house number + street token
  - sorted name tokens
  - house number + first name token
  - first two numbers
  - Buckets with more than 25 S1 are skipped (they are too generic to be informative).
- **Why both routes:** semantic search catches rewrites. Exact keys catch cases where embeddings blur
  near-identical strings with different numbers.
- **Result:** ~77 candidates per S1 and blocking recall **0.9896** on held-out entities. The remaining ~1% of
  true pairs are never retrieved, and this is the ceiling for everything downstream.

### 4.3 Stage 0, learned blocking (`stage0.py`)
- **Problem:** 133M pairs is too many for expensive string features.
- **Model:** LightGBM (63 leaves, lr 0.1, ≤400 rounds, early stopping) on **20 retrieval-only signals**. No
  string comparison is computed, so it costs ~1 µs per pair.
  - 17 context features: `cos`, rank/gap/margin of the pair in its record's list and in its S1's list, list
    sizes (`n_q`, `n_s1`), and which route produced it (`via_dense`, `via_addr`, `via_name`, `via_nn`, `via_n2`).
  - 3 query attributes: `q_src`, `q_addr_empty`, `q_is_indic`.
- **Threshold:** τ is the **largest value that keeps ≥99.9% of true candidate pairs out-of-fold**. It is
  recall-targeted, not budget-targeted. Result: τ = 0.001, 11.3 candidates per S1, 99.94% of true pairs kept,
  and local F0.5 changes only 0.98689 to 0.98685.
- **Why it beats a plain top-k cut:** a top-k cut cost −0.0005 to −0.0016, versus about −0.00004 for stage 0.
  It knows when a candidate is the clear winner in its list and when it is not.
- Training is 2-fold, grouped by S1, so there is no leakage.

### 4.4 Stage 1, pair features and XGBoost (`features.py`, `model.py`)
~95 features, in four groups:
- **Retrieval context:** cosine, ranks, gaps, route bits.
- **Name:** rapidfuzz ratio / partial / token-set / Jaro-Winkler, IDF-weighted token overlap, missing/extra
  token mass, acronym and concatenation coverage (`abccorp` vs `abc corp`), legal-form equality/conflict,
  domain-name flag, **name uniqueness** (how many S1 share the core name, which helps against twins).
- **Address:** token-set / partial ratios, IDF-weighted overlap, empty flags.
- **Numbers:** first house number equal, containment, Levenshtein and log difference, plus noise patterns
  (prefix match, edit-1, one inserted/deleted digit, last-two-digits, second number equal).

Model: XGBoost (depth 8, GPU hist), 2-fold out-of-fold grouped by S1, trained on 20.6M pairs. The training
sample deliberately includes hard twins.

### 4.5 Stage 2, context stacking
XGBoost on 18 raw features plus the **context of stage-1 probabilities**:
- max, second best, sum and rank per record and per S1
- number of confident siblings
- siblings sharing house number / legal form

This lets the model reason about the competition between pairs. For example, if another S1 is much more
likely for this record, this pair should be lowered.

### 4.6 Stage 3, cross-encoders (`crossenc.py`)
- **Architecture:** MiniLM-L6 weights from the bi-encoder plus a linear head. Input is
  `record text [SEP] S1 text` (max 128 tokens), so the model attends across both texts. A bi-encoder cannot do
  this, because it embeds each text separately.
- **Training:** BCE on candidates (true S1 + top-5 retrieved negatives). Only the ~4.5 pairs per S1 shortlist is
  scored.
- **Models:**
  - v1 (2.6M pairs), v2 (5.2M fresh pairs).
  - v3: continues from v2 on train pairs plus 1.6M **confident test pseudo-pairs**.
  - L12 (`ms-marco-MiniLM-L12-v2`, 33M parameters), and L12b (a second self-training round).
- **Pseudo-labels:**
  - argmax p ≥ 0.98 with margin ≥ 0.3 → match
  - p ≤ 0.02 → non-match
  - capped at 2 positives per negative so the model does not drift toward "match"
- **Stacker:** 2-fold XGBoost over the stage-2 probability + 4 cross-encoder logits + their per-record / per-S1
  context.
- **Why it mattered:** it was the biggest gain, 0.9739 to 0.9835 on the leaderboard (+0.0092 in total). It
  gained more on the leaderboard than locally (+0.0032), which suggests it transfers better to the shifted
  test set than the hand features do.

### 4.7 Decision rule
1. **Per record:** keep only the argmax S1, since each record matches at most one S1.
2. **Per S1:** sort its assigned records by probability and keep the top-j that maximises
   **expected F0.5**, E[F] ≈ 1.25·Σp / (0.25·Σp_all + j). The empty set is also an option, for likely singletons.
3. A probability floor `t` is tuned on held-out data (flat between 0.1 and 0.4).
4. **Fill rule:** an S1 left empty gets its single best record if p ≥ 0.1–0.2 (+0.0012 local, confirmed on the
   leaderboard).

## 5. Validation and honesty

- 12% of train S1 ("seeds", 264,524 entities) are held out. Their records stay in the search index, so retrieval
  faces the full distractor load.
- Seed pairs are not used to fit the bi-encoder or cross-encoders. Ranker scores on seeds are 2-fold
  out-of-fold, grouped by S1.
- Macro F0.5 is computed exactly like the leaderboard.
- **Known caveat:** the normalization maps in the final run were learned from all train pairs, so local scores
  are slightly optimistic for seeds. `ER_MAPS_EXCLUDE_SEEDS=1` removes this, but we did not re-run with it.

## 6. Results and ablation table

| Configuration | Local F0.5 | Public LB |
|---|---|---|
| Baseline (bi-encoder + XGBoost) | 0.9851 | 0.97186 |
| + number-noise / uniqueness features + fill rule | 0.98689 | 0.9739 |
| + stage 0 (77 → 11 candidates) | 0.98685 | — |
| + cross-encoder v2 stack | 0.98979 | 0.97874 |
| + L12 and test-adapted v3 | 0.99009 | 0.983 |
| **+ L12b, stack of 4 (final)** | **0.99007** | **0.983514** |

## 7. Error analysis

- **Wrong merges:** 70% are distractors (their real entity is absent from S1) attached to a same-name S1 such as
  chain stores or twins in the same building. 30% are records of a different S1 with a near-identical name.
- **Missed matches:** heavy name rewrites (DBA vs legal name, unknown transliterated tokens), empty or
  landmark-only addresses, heavily rewritten house numbers, and the ~1% never retrieved.
- **Local vs leaderboard gap (0.987 vs 0.974 pre-cross-encoder):**
  - A France-only probe implies France ≈ 0.958 versus US+India ≈ 0.974.
  - ~1.9x more distractors on test explains ~0.0014.
  - Evaluation-sample bias explains ~0.0024.
- **Tried, little or no help:** cluster/transitive post-processing (+0.0002), special handling of unseen Indic
  tokens, plain top-k cuts, majority vote of three models.

## 8. Compute and constraints

- One laptop: 16 GB RAM, RTX 4060 (8 GB), 16 threads. The pipeline takes ~3.5 h plus ~4.5 h for cross-encoders.
- Memory-safe design: per-country processing, streamed parquet, chunked features, XGBoost external-memory batches.
- Largest model is 33M parameters, all Apache-2.0/MIT components. No external data or APIs.

## 9. Likely interview questions and answers

**Why a cascade instead of one big model?**
Cost. Comparing 10M records against 2M entities is impossible, so each stage reduces the candidate set for the
next. The expensive cross-encoder only sees ~4.5 pairs per entity.

**Why a bi-encoder for retrieval and a cross-encoder for ranking?**
A bi-encoder embeds each text once, so you can do nearest-neighbour search. A cross-encoder reads both texts
together and is more accurate, but it needs a forward pass per pair, so it is only affordable on a shortlist.

**Why add exact-key blocks to dense retrieval?**
They are complementary. Dense search catches semantic rewrites. Exact keys catch the exact house number and
street cases that embeddings blur.

**How did you choose the stage-0 threshold?**
By a recall target (≥99.9% of true candidate pairs out-of-fold), not by a size budget. It is a safety-first
choice, and it still shrank the set 6.8x.

**Why F0.5, and how did you optimise for it?**
Precision is weighted more heavily. The expected-F0.5 subset rule per S1 uses calibrated probabilities to decide
how many records to keep, including none. This beats a single global threshold.

**How did you avoid leakage?**
Seeds are held out by entity but remain in the index. Folds are grouped by S1. Cross-encoders never train on
seed or evaluation queries. The one exception, the normalization maps, is disclosed above.

**How did you handle the unseen country?**
Country is never a feature. It only restricts retrieval to the same label. Nothing is hard-coded per country, so
France goes through the same code. Test-time self-training on confident predictions adapts the cross-encoders.

**What is the risk of pseudo-labelling?**
Confirmation bias and drift toward "match". We mitigated it with strict confidence and margin thresholds, labelled
confident non-matches as well, capped positives at 2 per negative, and trained on train pairs alongside. It is
transductive and disclosed. No test labels are used.

**What would you do next?**
Improve the retrieval ceiling (~1% of true pairs never retrieved), try a larger or multilingual cross-encoder,
and run more self-training rounds, which were still improving when time ran out.

## 10. Numbers to memorise

| Item | Value |
|---|---|
| Bi-encoder | MiniLM-L6, 22M parameters, top-k = 12 |
| Candidates per S1 | 77 → 11.3 after stage 0 |
| Stage-0 τ | 0.001, keeps 99.94% of true pairs |
| Stage-1 features | ~95 |
| Shortlist | ~4.5 pairs per S1 |
| Cross-encoders | 4 (L6 v2, L6 v3, L12, L12b) |
| Blocking recall | 0.9896 |
| Local F0.5 | 0.9901 |
| Leaderboard | 0.983514 |
