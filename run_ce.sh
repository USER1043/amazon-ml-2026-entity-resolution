#!/usr/bin/env bash
# Stage 3 (after ./run_local.sh): cross-encoder re-scoring of the stage-2 shortlist + stacker.
# Exact recipe of the submitted model:
#   v1  fine-tune from the bi-encoder: 570k non-evaluation queries (2.6M pairs), lr 5e-5, 1 epoch
#   v2  continue from v1 on a fresh sample: 1.14M queries (5.2M pairs), lr 3e-5 -> score, stack  (LB 0.978743)
#   v3  (ER_CE_SELFTRAIN=1) continue from v2 with 1.7M train pairs + 1.6M confident TEST pseudo-pairs
#       (2:1 pos:neg, labels from v2's own test scores), lr 2e-5 -> score, stack
#   L12 (ER_CE_L12=1, needs v3) second architecture: cross-encoder/ms-marco-MiniLM-L12-v2 (Apache-2.0, 33M) on
#       2.6M fresh train pairs + 1M balanced test pseudo-pairs from v3, lr 5e-5 -> score; stacker uses L12 + v2 + v3 (LB 0.983)
#   L12b continues L12 on 0.86M train + 0.7M test pseudo-pairs labelled by that stack -> stack of all four (LB 0.983514)
set -euo pipefail
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")"
CODE=$(cd .. && pwd)
export ER_DATA_DIR=${ER_DATA_DIR:-$CODE/../6ab10eb3b23ba_student_resource/student_resource/dataset}
export ER_WORK=${ER_WORK:-$CODE/data_tm}
export ER_TMP=${ER_TMP:-$CODE/data_tm/tmp}
export ER_OUT_DIR=${ER_OUT_DIR:-$CODE/output_tm}
PY=${PY:-$( [[ -x "$CODE/.venv/bin/python" ]] && echo "$CODE/.venv/bin/python" || command -v python3 )}   # override with PY=...
[[ -d "$ER_DATA_DIR/train" && -d "$ER_DATA_DIR/test" ]] || { echo "set ER_DATA_DIR to the dataset folder containing train/ and test/ (now: $ER_DATA_DIR)" >&2; exit 2; }
ce() { echo "=== crossenc $*  (ER_CE_NAME=${ER_CE_NAME:-})  $(date +%T)"; $PY -W ignore -m src.crossenc "$@"; }

ER_CE_NAME= ce train
export ER_CE_NAME=_v2 ER_CE_INIT=$ER_WORK/model_crossenc ER_CE_SEED=1 ER_CE_POS_Q=900000 ER_CE_DIS_Q=240000 ER_CE_LR=3e-5
ce train; ce score train; ce score test; ce stack
FINAL=_v2
if [[ "${ER_CE_SELFTRAIN:-0}" == 1 ]]; then
    export ER_CE_NAME=_v3 ER_CE_INIT=$ER_WORK/model_crossenc_v2 ER_CE_SEED=2 ER_CE_POS_Q=300000 ER_CE_DIS_Q=80000 \
           ER_CE_LR=2e-5 ER_CE_PSEUDO=$ER_WORK/test_scores_ce_v2.parquet ER_CE_N_PSEUDO=2000000
    ce train; ce score train; ce score test; ce stack
    FINAL=_v3
    if [[ "${ER_CE_L12:-0}" == 1 ]]; then
        export ER_CE_NAME=_L12 ER_CE_INIT= ER_CE_BASE=cross-encoder/ms-marco-MiniLM-L12-v2 ER_CE_SEED=3 ER_CE_POS_Q=450000 \
               ER_CE_DIS_Q=120000 ER_CE_LR=5e-5 ER_CE_PSEUDO=$ER_WORK/test_scores_ce_v3.parquet ER_CE_N_PSEUDO=1000000
        ce train; ce score train; ce score test
        ER_CE_STACK_ALSO=_v2,_v3 ce stack
        FINAL=_L12_v2_v3
        # L12b: 2nd self-training round from L12 (labels from the L12+v2+v3 stack), 0.86M train + 0.7M test pairs
        export ER_CE_NAME=_L12b ER_CE_INIT=$ER_WORK/model_crossenc_L12 ER_CE_BASE= ER_CE_SEED=4 ER_CE_POS_Q=150000 \
               ER_CE_DIS_Q=40000 ER_CE_LR=3e-5 ER_CE_PSEUDO=$ER_WORK/test_scores_ce_L12_v2_v3.parquet ER_CE_N_PSEUDO=700000
        ce train; ce score train; ce score test
        ER_CE_STACK_ALSO=_L12,_v2,_v3 ce stack
        FINAL=_L12b_L12_v2_v3          # final submission (public LB 0.983514)
    fi
fi
cp "$ER_OUT_DIR/matching_results_ce$FINAL.tsv" "$ER_OUT_DIR/matching_results.tsv"
python3 utils/validate_submission.py --matching "$ER_OUT_DIR/matching_results.tsv" \
    --candidate "$ER_OUT_DIR/candidate_pairs.tsv" --test-dir "$ER_DATA_DIR/test"
