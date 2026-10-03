# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** LearningMachine
**Team Members:** Prajan Karthik V, Nehan G R M, Riteesh T M
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A multi-stage blocking + ranking cascade. A fine-tuned MiniLM bi-encoder and four exact-key blocks retrieve
~77 candidates per Source 1 (S1) entity; a learned **stage-0 blocking model** that uses only retrieval-time
signals cuts this to **11.3 candidates per S1** (6.8× smaller) while keeping 99.94% of the true candidate pairs.
Two stacked XGBoost rankers on ~95 hand-built pair features and three fine-tuned **cross-encoders** (MiniLM-L6
and MiniLM-L12, ≤ 33M parameters, Apache-2.0; two of them adapted to the test set with confident
pseudo-labels) then score the survivors; each S2/S3 record is assigned to at most one S1 and every
S1 keeps the subset of records that maximises its expected F0.5. Local macro F0.5 on held-out entities
**0.9901**; public leaderboard **0.9835**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.** Train 2.21M S1 / 10.3M S2+S3 records; test 1.73M S1 / 9.97M S2+S3. All-pairs comparison is
  impossible, so everything is built around blocking within a country label.
- **Structure we exploit.** Every true pair shares the country label; each S2/S3 record matches **at most
  one** S1 (so we assign by per-record argmax); ~26% of S2/S3 records match nothing (distractors); 5.6% of S1
  are singletons; non-singleton S1 have ~3.7 matches.
- **Name noise.** Only 14.7% of true pairs have identical normalized names. Legal suffixes (Pvt Ltd / Private
  Limited / Inc / LLC / P.C.), domain-style names (`telephonelegacy.com`), d/b/a and f/k/a clauses, word order,
  typos, and **Indic-script names** (Devanagari / Bengali word-by-word translations of the English name).
- **Address noise.** Abbreviations (Rd/Road, St/Street), state full vs abbreviated, reordered or missing
  components, house numbers with prefixes and padding (`00143`, `C-##23`, `H.NO 69`), number words
  (10th/Tenth), landmark references, ~3% empty addresses.
- **Hard negatives.** The dominant errors are "twins": a different business with the same or a very similar
  name at a nearby address (chains, same building), and distractor records that look like a real S1.
- **Shift.** The test set adds an unseen country (France) and has ~1.9× more distractors per S1 than train.
  Country is treated as an open set: it is never a model feature or a hard-coded filter.

### 2.2 Solution Strategy

**Approach Type:** Hybrid — learned blocking cascade + gradient-boosted ranker + transformer cross-encoder,
with an F0.5-aware set-selection rule.
**Core Innovation:** (1) a *learned* stage-0 blocking model on retrieval-only signals that shrinks the
candidate set 6.8× at no measurable accuracy cost; (2) cross-encoders restricted to a ~4.2-pairs-per-S1
shortlist, stacked with the feature ranker and adapted to the test distribution with confident pseudo-labels
(+0.0092 on the leaderboard in total); (3) decisions per S1 that
maximise expected F0.5 instead of a flat threshold.

Pipeline:

```
normalize (learned maps) → retrieve: bi-encoder top-12 + 4 exact-key blocks  (~77 cand/S1)
  → stage 0: LightGBM on retrieval signals, τ = 0.001                          (~11 cand/S1 = candidate_pairs.tsv)
  → stage 1: XGBoost, ~95 pair features
  → stage 2: XGBoost on per-record / per-S1 context of stage-1 probabilities  (shortlist ~4.5 pairs/S1)
  → stage 3: 4 cross-encoders (L6 v2, L6 v3, L12, L12b; the last three test-adapted) + XGBoost stacker
  → per-record argmax → per-S1 expected-F0.5 subset → fill rule → matching_results.tsv
```

**Validation protocol.** 12% of train S1 entities (264,524 "seeds") are held out; their records stay in
the full search index, so retrieval faces the full distractor load. No seed S1 pair is used to fit the
bi-encoder or the cross-encoders, and every ranker score on seeds is 2-fold out-of-fold (folds grouped by S1).
Macro F0.5 is computed exactly like the leaderboard (singletons included). One exception: in the final run the
normalization maps (§3) were learned from *all* train pairs, seeds included, so local scores are slightly
optimistic for the seed entities (`ER_MAPS_EXCLUDE_SEEDS=1` removes this; we did not re-run with it).

---

## 3. Candidate Generation / Blocking Strategy

**Normalization (`prep.py`, `normalize.py`).** Learned from train pairs only: an Indic-script → Latin name
token map (1.3k tokens, from word-aligned true pairs), an Indic address-component map (state names), and an
address token canonicalisation map (abbreviations, state full↔abbrev, frequent typos, from pairs that differ
by one token). Names are split into core / legal form (canonical set) / domain handling; addresses into
cleaned text, numbers (leading zeros stripped), first house number and blocking keys.

**Blocking keys used (`retrieve.py`).**
- *Dense:* all-MiniLM-L6-v2 fine-tuned with in-batch negatives on 800k non-seed (record, S1) pairs; the
  normalized "name , address" of every S2/S3 record retrieves its **top-12 S1 within the same country label**
  (GPU matrix product, per country).
- *Exact keys:* (house number + street token), (sorted name tokens), (house number + first name token),
  (first two numbers); buckets with more than 25 S1 are skipped.
- Union: **~77 candidates per S1** on test (133.2M pairs), blocking recall on held-out entities **0.9896**.

**Stage-0 learned blocking (`stage0.py`).** A LightGBM (≈100 trees) on 20 *retrieval-only* signals: cosine,
rank and gap of the pair within its record's list and within its S1's list, list sizes, which key produced the
pair, query source, empty-address and Indic-script flags. No string comparison is computed (~1 µs per pair).
Trained 2-fold by S1 on the ranker training sample; the threshold is the largest τ that keeps ≥ 99.9% of the
true candidate pairs out-of-fold (τ = 0.001).

| τ | cand. per test S1 | true cand. pairs kept | held-out recall | local F0.5 (pre-CE model) |
|---|---|---|---|---|
| none | 76.9 | 100% | 0.9896 | 0.98689 |
| 0.0005 | 13.4 | 99.977% | 0.9894 | 0.98688 |
| **0.001 (used)** | **11.3** | **99.94%** | **0.9893** | **0.98685** |
| 0.005 | 7.8 | 99.77% | 0.9883 | 0.98670 |
| 0.02 | 6.3 | 99.30% | 0.9855 | 0.98631 |

- **Candidate pairs generated:** 19,531,226 for 1,732,544 test S1 (11.27 per S1; median 9, p99 51); 12 S1
  have no candidate. `candidate_pairs.tsv` is exactly this set: every later model scores only these pairs,
  and every match is a subset of it.
- **How we ensured true matches were not lost:** two complementary retrieval routes (semantic + exact keys);
  recall measured on held-out entities after every change; the stage-0 threshold is set by a recall target
  (≥ 99.9% of retrievable true pairs), not by a size budget; only 854 held-out true pairs (0.09%) are
  dropped by the stage-1 probability floor before the final rankers.

---

## 4. Model Architecture and Feature Engineering

**Features used (`features.py`, ~95).**
- *Retrieval context:* cosine, rank / gap / margin per record and per S1, list sizes, key-route bits.
- *Name:* rapidfuzz ratio / partial / token-set / Jaro-Winkler, IDF-weighted token overlap and missing /
  extra token mass, acronym and concatenation coverage (`abccorp` vs `abc corp`), legal-form equality /
  conflict, domain-name flag, name uniqueness in the S1 index (how many S1 share the core name).
- *Address:* token-set / partial ratios, IDF-weighted overlap, empty-address flags.
- *Numbers:* first house number equal, S1 numbers contained in record, Levenshtein and log difference of
  numbers, missing numbers, and house-number noise patterns (prefix match, edit distance 1, one inserted /
  deleted digit, last-two-digit match, second number equal).

**Model type.**
1. *Stage 1:* XGBoost (depth 8, GPU hist) on all features, 2-fold OOF grouped by S1; trained on 20.6M pairs
   of the ranker sample (records of held-out S1, records whose top-1 is a held-out S1, and unmatched records
   near them — the hard twins).
2. *Stage 2:* XGBoost on 18 raw features + context of stage-1 probabilities (max / second / sum / rank per
   record and per S1, number of confident siblings, siblings sharing house number / legal form).
3. *Stage 3 — cross-encoder (`crossenc.py`):* the fine-tuned bi-encoder's MiniLM-L6 weights + a linear head,
   reading "record text [SEP] S1 text" (max 128 tokens). Trained with BCE on candidates of non-evaluation
   queries (true S1 + top-5 retrieved negatives): v1 on 2.6M pairs (1 epoch, lr 5e-5), then v2 continues on
   5.2M fresh pairs (lr 3e-5); **v3** continues from v2 on 1.7M train pairs + 1.6M confident *test*
   pseudo-pairs (argmax p ≥ 0.98 with margin ≥ 0.3 → match; other candidates of those records and records whose
   best p ≤ 0.02 → non-match; capped at 2 positives per negative so the model does not drift towards
   "match"), lr 2e-5. A second architecture, **L12** (cross-encoder/ms-marco-MiniLM-L12-v2, Apache-2.0, 33M
   parameters, already pretrained on text-pair relevance), is fine-tuned on 2.6M fresh train pairs + 1M balanced
   test pseudo-pairs labelled by v3; **L12b** continues L12 for a second self-training round on 0.86M train +
   0.7M test pseudo-pairs labelled by the L12+v2+v3 stack. Each scores only the stage-2 shortlist (1.2M held-out / 7.3M test pairs).
   A 2-fold XGBoost stacker combines the stage-2 probability with the four cross-encoder logits and their
   per-record / per-S1 context (rank, max, runner-up, number of confident siblings). No held-out S1 or
   evaluation query is ever used to train a cross-encoder (the normalization maps are the only component fit
   on all train pairs, see §2.2).

**Threshold selection method.** Per S2/S3 record keep only its argmax S1. Per S1, sort its assigned records by
probability and keep the top-j that maximises expected F0.5 (E[F] ≈ 1.25·Σp / (0.25·Σp_all + j)), or none if
the probability that the S1 is a singleton makes an empty list better; a floor t is tuned on held-out F0.5
(flat between 0.1 and 0.4). **Fill rule:** an S1 left empty gets its single best record if p ≥ 0.1–0.2
(tuned locally; +0.0012 local, confirmed on the leaderboard).

---

## 5. Results & Error Analysis

| Configuration | Local F0.5 (held-out) | Public LB |
|---|---|---|
| Teammate baseline (bi-encoder + XGBoost) | 0.9851 | 0.97186 |
| + v4 number-noise + name-uniqueness features, fill rule (final local pipeline) | 0.98689 | 0.9739 |
| + stage-0 cascade (candidates 77 → 11 per S1) | 0.98685 | — |
| + cross-encoder v1 stack | 0.98907 | — |
| + cross-encoder v2 stack | 0.98979 (US 0.9896, India 0.9901) | 0.97874 |
| + v3 test-adapted cross-encoder | 0.98987 | — |
| + L12 cross-encoder, stacked with v2 + v3 | 0.99009 (US 0.9899, India 0.9904) | 0.983 |
| **+ L12b (second test self-training round), stack of all four (final)** | **0.99007** | **0.983514** |

- **Where the remaining loss is (pre-CE breakdown of 0.013 local loss):** 0.0046 retrievable true matches
  scored too low (recall), 0.0029 S1 with a wrong merge, 0.0023 true matches never retrieved, 0.0022
  non-singletons predicted empty, 0.0010 singletons given a match. The cross-encoder raised precision
  (0.9961 → 0.9970) and recall (0.9699 → 0.9733) together.
- **Common false positives (wrong merges):** 70% are distractor records (their true entity is absent from S1)
  attached to a same-name S1 — chain stores, "twin" businesses in the same building differing only in unit or
  house number; 30% are records of a *different* S1 with a near-identical name.
- **Common false negatives (missed matches):** heavy name rewrites (DBA / trade name vs legal name,
  transliterated Indic names with unknown tokens), empty or landmark-only addresses, and house numbers
  rewritten beyond simple noise; plus the ~1% of true pairs no retrieval route surfaces.
- **Measured but not helpful:** cluster / transitive post-processing (+0.0002), treating unseen Indic tokens
  specially (they are distractor vocabulary), plain top-k candidate cuts (−0.0005 to −0.0016 vs stage 0's
  −0.00004), a three-model majority vote.
- **Local vs leaderboard gap.** Local 0.987 vs LB 0.974 before the cross-encoder. A France-only probe
  (0.191 on the public board) implies France ≈ 0.958 and US+India ≈ 0.974 on test; ~1.9× more distractors on
  test explains ~0.0014; evaluation-sample bias ~0.0024. The cross-encoders gained far more on the leaderboard
  (+0.0092) than locally (+0.0032), i.e. they transfer better than the hand features, and test-set
  adaptation with pseudo-labels (invisible to the local metric) adds on top. On France, test adaptation moved
  the share of S1 predicted as singletons from 4.6% to 5.2%, towards the ~5.6% train rate.

---

## 6. Conclusion

Blocking quality and ranking quality were attacked separately: a learned, recall-targeted stage-0 filter makes
the candidate set 6.8× smaller than the retrieval union at essentially zero cost, which in turn makes a
transformer cross-encoder affordable on the final shortlist; cross-encoders, adapted to the test set with balanced pseudo-labels, were the largest gain
(0.9739 → 0.9835 LB). Lessons: measure every stage on held-out entities with the full distractor load, decide per S1
with the metric's own expected value, and spend expensive models only where the cheap ones are uncertain.

---

## 7. Other Relevant Information

- **Exact final configuration.** Submission #9 (public 0.983514) = `./run_local.sh` with default settings
  (normalization maps learned from train pairs only; no test pairs used for maps) → stage-0 candidates
  (τ = 0.001) → stage 1 + 2 XGBoost → `ER_CE_SELFTRAIN=1 ER_CE_L12=1 ./run_ce.sh` (cross-encoders v1 → v2 → v3
  → L12 → L12b, stacker over v2, v3, L12, L12b). `output/matching_results.tsv` is byte-identical to that upload
  and `output/candidate_pairs.tsv` is the stage-0 set it was scored on.
- **Test-time adaptation (transductive, disclosed).** Cross-encoders v3, L12 and L12b are additionally trained on
  pseudo-labelled pairs from the *provided test files*, labelled by our own earlier model's confident
  predictions (argmax p ≥ 0.98 with margin ≥ 0.3 → match; p ≤ 0.02 → non-match; at most 2 matches per
  non-match). No test labels, external data, lookups or APIs are involved. The earlier Kaggle submission #5
  also learned normalization maps from confident test pairs (`ER_EXTRA_PAIRS`); the final pipeline does not.
- **Country handling.** Country is used only to restrict retrieval to the same label (every true pair shares it);
  it is never a model feature, and no rule names a specific country, so France (absent from train) goes through
  exactly the same code.
- **Compute.** One laptop: 16 GB RAM, RTX 4060 (8 GB), 16 CPU threads. Full pipeline ≈ 3.5 h (prep → predict) +
  ≈ 4.5 h of cross-encoder training and scoring. Memory-safe design: per-country processing, streamed parquet,
  chunked feature computation, XGBoost external-memory batches.
- **Model limits.** Largest model 33M parameters (MIT / Apache-2.0 components only), far below the 8B limit.
- **Limitations and next steps.** ~1% of true pairs are never retrieved (the recall ceiling); a larger or
  multilingual cross-encoder and further self-training rounds were still improving the score when time ran out;
  the local-to-leaderboard gap (≈ 0.007 at the end) is mostly the unseen country and the denser test distractors.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md`):

```
src/config.py            settings (env ER_<NAME>)
src/normalize.py         name / address normalization
src/prep.py              TSV → parquet, learned maps                  python -m src.prep
src/biencoder.py         MiniLM fine-tune / encoder
src/retrieve.py          dense top-12 + exact-key blocks              python -m src.retrieve train|test
src/features.py          pair features (ranker sample)                python -m src.features train
src/model.py, train.py   stage 1 + 2 XGBoost, decision grid           python -m src.train
src/stage0.py            learned blocking cascade                     python -m src.stage0 fit|apply
src/predict.py           test scoring, both TSVs                      python -m src.predict
src/crossenc.py          cross-encoder + stacker                      python -m src.crossenc train|score|stack
run_local.sh             prep → … → stage0 → predict → validate
run_ce.sh                cross-encoder stage → final matching_results.tsv
```

`output/candidate_pairs.tsv` = stage-0 survivors (from `predict.py`); `output/matching_results.tsv` = the
cross-encoder stack output (`ER_CE_SELFTRAIN=1 ER_CE_L12=1 ./run_ce.sh`, file `matching_results_ce_L12b_L12_v2_v3.tsv`).

### B. Submission history (public leaderboard)

| # | Date | Submission | Public F0.5 |
|---|---|---|---|
| 1 | 26 Sep | v1: char-TF-IDF name blocking + hand-weighted score (local 0.645) | low (rank ~1430) |
| 2 | 26 Sep | Teammate pipeline: bi-encoder + key blocks, 2-stage XGBoost | 0.971858 |
| 3 | 27 Sep | Teammate variant (decision tweak) | 0.972192 |
| 4 | 27 Sep | Diagnostic: France rows only (others empty) | 0.191 |
| 5 | 27 Sep | v3: leak-free + transductive normalization maps, full-data stage 1 (Kaggle) | 0.9737 |
| 6 | 27 Sep | v4: number-noise + name-uniqueness features, fill rule (local run) | 0.9739 |
| 7 | 27 Sep | Stage-0 candidates (11/S1) + cross-encoder v2 stack | 0.978743 |
| 8 | 27 Sep | + L12 cross-encoder + test-adapted v3, stacked | 0.983 |
| 9 | 27 Sep | + L12b: second self-training round (labels from #8), stack of 4 cross-encoders (final) | **0.983514** |

### C. Licences and fair play

Neural models: all-MiniLM-L6-v2 (Apache-2.0, 22M parameters) and cross-encoder/ms-marco-MiniLM-L12-v2
(Apache-2.0, 33M parameters), and our fine-tunes of them. XGBoost
(Apache-2.0), LightGBM (MIT), rapidfuzz (MIT), polars (MIT), PyTorch (BSD), transformers (Apache-2.0).
No external data, APIs, geocoding or lookups; every map and model is learned from the provided training data
(and, for transductive steps, from the model's own confident predictions on the provided test data).
