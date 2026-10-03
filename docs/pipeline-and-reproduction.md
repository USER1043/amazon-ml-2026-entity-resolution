# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline (all offline, trained only on the provided data):

```
TSVs ─► prep (learned normalization) ─► retrieve (fine-tuned bi-encoder top-12 + exact-key blocks, ~77 cand/S1)
     ─► stage 0 (LightGBM on retrieval-only signals: ~11 cand/S1  = candidate_pairs.tsv)
     ─► stage 1 XGBoost on ~95 pair features ─► stage 2 XGBoost on per-record / per-S1 context
     ─► stage 3 cross-encoders (MiniLM-L6 v2, test-adapted v3, MiniLM-L12) + stacker ─► per-record argmax, expected-F0.5 selection per S1, fill rule
     ─► matching_results.tsv
```

Local macro F0.5 is computed on 12% held-out train S1 entities ("seeds") exactly like the leaderboard
(singletons included). No seed S1 — and no query of the local evaluation sample — is used to fit the
bi-encoder, the cross-encoder or any model that scores it (2-fold out-of-fold predictions elsewhere).

## Environment

Python 3.14, CUDA GPU (developed on a 16 GB RAM laptop with an 8 GB RTX 4060; also runs on Kaggle T4).
`pip install -r requirements.txt`. Two public base models are downloaded from the Hugging Face hub on first
use: `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22M parameters; bi-encoder and cross-encoders
v1–v3) and `cross-encoder/ms-marco-MiniLM-L12-v2` (Apache-2.0, 33M parameters; cross-encoders L12, L12b).
The scripts use `$PY` (default: `python3`, or `../.venv/bin/python` if present).

## Reproduce end to end

```bash
export ER_DATA_DIR=/path/to/dataset          # folder containing train/ and test/
./run_local.sh                               # prep → retrieve → features → train → stage0 → predict → validate
ER_CE_SELFTRAIN=1 ER_CE_L12=1 ./run_ce.sh   # stage 3: cross-encoders v1 → v2 → v3 (test-adapted) → L12 → L12b,
                                             #          stacker → final matching_results.tsv (public LB 0.983514)
```

Every stage caches under `$ER_WORK` (default `../data_tm`) and can be resumed: `./run_local.sh --from stage0`.
Outputs land in `$ER_OUT_DIR` (default `../output_tm`):
- `candidate_pairs.tsv` — written by `predict` (the stage-0 set; = `output/candidate_pairs.tsv`);
- `matching_results.tsv` — overwritten by `run_ce.sh` with the final stack (= `output/matching_results.tsv`,
  public LB 0.983514; intermediate files `matching_results_ce_<runs>.tsv` are kept).

Seeds, sampling and folds are fixed, so a re-run reproduces every stage; GPU training of the bi-encoder,
XGBoost and the cross-encoders is not bit-exact, so a fresh run matches the submitted files up to small
numerical differences, not byte for byte.

| Stage | Command | Device | Time (laptop) |
|---|---|---|---|
| prep | `python -m src.prep` | CPU | ~15 min |
| retrieve train / test | `python -m src.retrieve train` / `test` | GPU | ~45 + 40 min |
| features | `python -m src.features train` | CPU | ~30 min |
| train (stage 1 + 2) | `python -m src.train` | GPU | ~40 min |
| stage 0 | `python -m src.stage0 fit` / `apply` | CPU | 3 + 1 min |
| predict | `python -m src.predict` | CPU + GPU | ~25 min |
| cross-encoders | `python -m src.crossenc train` / `score train` / `score test` / `stack` (see `run_ce.sh`) | GPU | ≈ 4 h for v1+v2+v3+L12 |

Settings are environment variables `ER_<NAME>` (see `src/config.py`, `src/stage0.py`, `src/crossenc.py`).

## Modules (`src/`)

| File | Role |
|---|---|
| `config.py` | paths and settings (env-overridable) |
| `normalize.py`, `prep.py` | TSV reading (tab, no quoting, strings only); learned maps from train pairs: Indic-script name tokens → Latin, Indic address components, address token canonicalization; name core / legal form / domain-name splitting; address numbers, tokens, blocking keys |
| `biencoder.py`, `retrieve.py` | MiniLM-L6 fine-tuned with in-batch negatives on non-seed pairs; top-12 dense neighbours per S2/S3 record within the same country label + 4 exact-key blocks; per-country, memory-mapped encodings |
| `stage0.py` | **blocking cascade**: LightGBM on 20 retrieval-only signals (cosine, ranks and gaps per record and per S1, list sizes, which key produced the pair, query source / empty address / Indic script); threshold = largest τ keeping ≥ 99.9% of true candidate pairs (OOF) → the candidate set |
| `features.py` | ~95 pair features: fuzzy name scores, IDF token overlap, legal-form agreement, acronym / concatenation coverage, address overlap, house-number agreement and noise (prefix, edit-1, indel-1, last-2 digits), name uniqueness in the S1 index |
| `model.py`, `train.py` | 2-fold (grouped by S1) XGBoost stage 1, stacked stage 2 on context of stage-1 probabilities, decision grid (threshold × expected-F0.5 selection) |
| `predict.py` | scores the stage-0 survivors, stage 2, decision + fill rule; writes both TSVs (every test S1 row, empty lists allowed) |
| `crossenc.py` | stage 3: cross-encoder on "record [SEP] S1 record" for the stage-2 shortlist (~4.5 pairs per S1) + 2-fold XGBoost stacker on both scores and their per-record / per-S1 context |
| `pseudo.py`, `selftrain.py`, `france_selftrain.py`, `honest_eval.py`, `loco_train.py`, `ensemble.py` | experiments (test-time pseudo-labels, leave-one-country-out, evaluation-bias check, score blending) — see the methodology document |

## Licences / fair play

all-MiniLM-L6-v2 (Apache-2.0, 22M params) is the only neural model (bi-encoder and cross-encoder are
fine-tunes of it). XGBoost (Apache-2.0), LightGBM (MIT), rapidfuzz (MIT), polars (MIT), PyTorch (BSD),
transformers (Apache-2.0), unidecode (GPL-2.0, transliteration library). No external data, APIs,
geocoding or lookups; every map and model is learned from the provided train data (plus, optionally,
the model's own confident test predictions).
