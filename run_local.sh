#!/usr/bin/env bash
# Local run of the teammate pipeline (Kaggle notebook -> 16 GB / RTX 4060 laptop).
#   ./run_local.sh                    # all stages
#   ./run_local.sh --from features    # resume (every stage caches under $ER_WORK)
# Heavy: uses all CPU cores + the GPU. Sleep is inhibited for the whole run.
set -euo pipefail
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")"
CODE=$(cd .. && pwd)
export ER_DATA_DIR=${ER_DATA_DIR:-$CODE/../6ab10eb3b23ba_student_resource/student_resource/dataset}
export ER_WORK=${ER_WORK:-$CODE/data_tm}
export ER_TMP=${ER_TMP:-$CODE/data_tm/tmp}
export ER_OUT_DIR=${ER_OUT_DIR:-$CODE/output_tm}
export ER_N_PROC=${ER_N_PROC:-$(nproc)}
PY=${PY:-$( [[ -x "$CODE/.venv/bin/python" ]] && echo "$CODE/.venv/bin/python" || command -v python3 )}   # override with PY=...
[[ -d "$ER_DATA_DIR/train" && -d "$ER_DATA_DIR/test" ]] || { echo "set ER_DATA_DIR to the dataset folder containing train/ and test/ (now: $ER_DATA_DIR)" >&2; exit 2; }

if [[ -z "${ER_INHIBITED:-}" ]] && command -v systemd-inhibit >/dev/null; then
    export ER_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --who=ber-teammate --why="teammate ER pipeline" "$SELF" "$@"
fi

STAGES=(prep retrieve_train retrieve_test features train stage0 predict validate)
FROM=${STAGES[0]}; [[ "${1:-}" == "--from" ]] && FROM=$2
started=0
for s in "${STAGES[@]}"; do
    [[ $s == "$FROM" ]] && started=1
    (( started )) || continue
    echo "=== $s  $(date +%T)"
    case $s in
        prep)           cmd=("$PY" -W ignore -m src.prep) ;;
        retrieve_train) cmd=("$PY" -W ignore -m src.retrieve train) ;;
        retrieve_test)  cmd=("$PY" -W ignore -m src.retrieve test) ;;
        features)       cmd=("$PY" -W ignore -m src.features train) ;;
        train)          cmd=("$PY" -W ignore -m src.train) ;;
        stage0)         cmd=(bash -c "'$PY' -W ignore -m src.stage0 fit && '$PY' -W ignore -m src.stage0 apply") ;;
        predict)        cmd=("$PY" -W ignore -m src.predict) ;;
        validate)       cmd=(python3 utils/validate_submission.py --matching "$ER_OUT_DIR/matching_results.tsv"
                             --candidate "$ER_OUT_DIR/candidate_pairs.tsv" --test-dir "$ER_DATA_DIR/test") ;;
    esac
    /usr/bin/time -f "=== $s done: %es wall, peak %M KB" "${cmd[@]}"
done
(( started )) || { echo "unknown stage $FROM (stages: ${STAGES[*]})" >&2; exit 2; }
