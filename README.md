# Business Entity Resolution - Amazon ML Challenge 2026

Team **LearningMachine** (Prajan Karthik V, Nehan G R M, Riteesh T M).
Public leaderboard macro **F0.5 = 0.9835**.

The task: given business records from three sources, decide which Source 2 / Source 3 records refer to the same
real-world entity as each Source 1 record. Names vary (`Pvt` vs `Private`, Indic scripts, acronyms) and addresses
are often incomplete (e.g. no street name), so no single field is enough. The dataset is ~2.2M S1 and ~10M
S2/S3 records, so all-pairs comparison is impossible.

## Approach in one picture
```
records -> normalise -> blocking (MiniLM bi-encoder top-12 + exact keys, ~77 cand/S1)
        -> stage 0 learned filter (~11 cand/S1)
        -> XGBoost on ~95 pair features (2 stages)
        -> cross-encoders (MiniLM L6/L12) + stacker
        -> per-record argmax + expected-F0.5 selection per S1
```

## Read next
- [`docs/solution-writeup.md`](docs/solution-writeup.md) - full methodology, blocking, features, results, error analysis
- [`docs/ideas-and-lessons.md`](docs/ideas-and-lessons.md) - what worked, what didn't, what I'd try next
- [`docs/pipeline-and-reproduction.md`](docs/pipeline-and-reproduction.md) - module map and how to run it

## Not included
The challenge dataset, trained models and prediction files are not in this repo. The dataset is mostly available on Kaggle (Amazon ML Challenge 2026); download it and point `ER_DATA_DIR` at it
to reproduce. Only public base models are used (all-MiniLM-L6-v2, ms-marco-MiniLM-L12-v2),
fine-tuned on the provided training data only; no external data or APIs.

## License
MIT, see [LICENSE](LICENSE).
