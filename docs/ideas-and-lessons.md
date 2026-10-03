# Ideas, experiments and lessons

What moved the score, what didn't, and what I'd try next. Scores are macro F0.5 on a 12% held-out slice of
train S1 entities (local) or the public leaderboard (LB).

## What moved the score
| Step | Local | Public LB |
|---|---|---|
| Name-only TF-IDF blocking + hand-set weights (my first attempt) | 0.645 | - (rank ~1430) |
| Name + address combined blocking, LightGBM matcher | 0.965 | - |
| Fine-tuned MiniLM bi-encoder + exact-key blocks, 2-stage XGBoost | 0.985 | 0.9719 |
| Leak-free normalisation maps + fill rule | 0.9869 | 0.9739 |
| Stage-0 blocking cascade (77 -> 11 candidates per S1) + cross-encoder stack | 0.9898 | 0.9787 |
| 12-layer cross-encoder + test-time self-training | 0.9901 | **0.9835** |

## Lessons
- **Blocking recall is the ceiling.** Name-only blocking recalled 0.61 of true pairs; adding address signal
  took it to 0.97. Everything downstream is capped by it.
- **Addresses carry what names miss.** Legal-form variants (Pvt / Private), Indic-script names and missing
  street names make names unreliable on their own.
- **Fewer candidates is a feature.** A learned stage-0 filter cut candidates 6.8x while keeping 99.94% of true
  pairs, and made the cross-encoder affordable.
- **Watch for evaluation leakage.** Features such as in-degree of an S1 were biased on a candidate subset and
  inflated dev scores; only a full-train holdout was trustworthy.
- **Local vs leaderboard gap.** Local 0.990 vs LB 0.984: the unseen country (France) is weaker, but it is not
  the main cause (France ~0.958, others ~0.974 at the time of probing).

## Tried and ruled out (measured)
- Cluster post-processing for duplicate S2/S3 records: +0.0002.
- Unknown Indic tokens: distractor vocabulary that never matches true pairs.
- Distractor-density shift between train and test: -0.0014.
- Optimism from "blind spots" in the evaluation sample: 0.0024.

## Ideas not yet tried
- Multilingual embeddings (e.g. multilingual-e5-small, MIT) for Indic-script names.
- Duplicate-cluster consistency features (an S1 often has several S2/S3 duplicates).
- Pseudo-labelling the unseen country earlier in the pipeline.
- Blending independently trained pipelines.
