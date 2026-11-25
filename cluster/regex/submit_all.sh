#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant: submit train (array) -> eval (array, task i waits for train task i) -> analyze.
#
#   source /scratch/$USER/intros/cluster/env.sh
#   bash $REPO/cluster/regex/submit_all.sh
#   MODEL=Qwen/Qwen3-8B REGEX_TAG=regex-qwen3-8b bash $REPO/cluster/regex/submit_all.sh
#   LEVELS=1,2,3,4 ARRAY= REGEX_TAG=regex-qwen3-32b-mixed bash $REPO/cluster/regex/submit_all.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"
export REGEX_TAG="${REGEX_TAG:-regex-qwen3-32b}"
cd "$REPO"
mkdir -p logs
ARRAY="${ARRAY-1-4}"
ARR=(); [ -n "$ARRAY" ] && ARR=(--array="$ARRAY")
COMMON=(--parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" --export=ALL)

TRAIN=$(sbatch "${COMMON[@]}" "${ARR[@]}" cluster/regex/10_train.sbatch)
DEP="afterok:$TRAIN"; [ -n "$ARRAY" ] && DEP="aftercorr:$TRAIN"
EVAL=$(sbatch "${COMMON[@]}" "${ARR[@]}" --dependency="$DEP" cluster/regex/20_eval.sbatch)
ANALYZE=$(sbatch "${COMMON[@]}" --dependency="afterany:$EVAL" cluster/regex/30_analyze.sbatch)
echo "train=$TRAIN eval=$EVAL analyze=$ANALYZE  (tag=$REGEX_TAG model=$MODEL)"
echo "watch with: squeue --me ; results in $RUNS_ROOT/${REGEX_TAG}_analysis"
