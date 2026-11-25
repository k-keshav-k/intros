#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 70 — re-evaluate existing checkpoints with the report prefill (no training).
#
# Late in training the adapter often answers the report prompt with "Yes"/"No" instead of a
# regex.  `--prefill` (regexintro.evaluate / regexintro.conflict) starts each report answer with
# prompts.REPORT_PREFILL so the first token cannot be a decision.
#
# Builds view run dirs  $RUNS_ROOT/${SRC_TAG}-prefill_L<n>  holding copies of spec.json and
# train_config.json and symlinks to the source run's checkpoints up to MAX_STEP (source runs are
# only read), then submits:
#   eval (array, --prefill --every EVERY) -> analyze, diagnose
#                                         -> conflict (array, aftercorr, --prefill --step MAX_STEP)
#                                            -> conflict analyze
# Outputs: $RUNS_ROOT/${SRC_TAG}-prefill_{analysis,diagnosis,conflict_analysis}
#
#   bash $REPO/cluster/regex/70_prefill.sh
#   SRC_TAG=regex-qwen3-32b LEVELS="1 2 3 4" MAX_STEP=4000 EVERY=400 bash $REPO/cluster/regex/70_prefill.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="${REPO:-/scratch/$USER/intros}"
source "$REPO/cluster/env.sh"
SRC_TAG="${SRC_TAG:-regex-qwen3-32b}"
export REGEX_TAG="${SRC_TAG}-prefill"
LEVELS="${LEVELS:-1 2 3 4}"
MAX_STEP="${MAX_STEP:-4000}"
EVERY="${EVERY:-400}"
cd "$REPO"
mkdir -p logs

for L in $LEVELS; do
  SRC="$RUNS_ROOT/${SRC_TAG}_L$L"
  DST="$RUNS_ROOT/${REGEX_TAG}_L$L"
  [ -f "$SRC/spec.json" ] && [ -f "$SRC/train_config.json" ] || { echo "STOP: $SRC is not a run" >&2; exit 1; }
  [ -f "$SRC/checkpoints/step_$(printf %06d "$MAX_STEP")/adapter.safetensors" ] \
    || { echo "STOP: $SRC has no checkpoint at step $MAX_STEP" >&2; exit 1; }
  mkdir -p "$DST/checkpoints"
  cp "$SRC/spec.json" "$SRC/train_config.json" "$DST/"
  n=0
  for d in "$SRC"/checkpoints/step_*; do
    s=$((10#${d##*_}))
    if [ "$s" -le "$MAX_STEP" ] && [ -f "$d/adapter.safetensors" ]; then
      ln -sfn "$d" "$DST/checkpoints/$(basename "$d")"
      n=$((n + 1))
    fi
  done
  echo "view $DST: $n checkpoints (<= step $MAX_STEP) from $SRC"
done

ARR=$(echo $LEVELS | tr ' ' ',')
export EVAL_ARGS="--prefill --every $EVERY"
export CONFLICT_ARGS="--prefill --step $MAX_STEP"
COMMON=(--parsable --account="$TORCH_ACCOUNT" --chdir="$REPO" --export=ALL)
EVAL=$(sbatch "${COMMON[@]}" --array="$ARR" cluster/regex/20_eval.sbatch)
ANALYZE=$(sbatch "${COMMON[@]}" --dependency="afterany:$EVAL" cluster/regex/30_analyze.sbatch)
DIAG=$(sbatch "${COMMON[@]}" --dependency="afterany:$EVAL" cluster/regex/35_diagnose.sbatch)
CONF=$(sbatch "${COMMON[@]}" --array="$ARR" --dependency="aftercorr:$EVAL" cluster/regex/40_conflict.sbatch)
CAN=$(sbatch "${COMMON[@]}" --dependency="afterany:$CONF" cluster/regex/41_conflict_analyze.sbatch)
echo "tag=$REGEX_TAG eval=$EVAL analyze=$ANALYZE diagnose=$DIAG conflict=$CONF conflict_analyze=$CAN"
echo "EVAL_ARGS='$EVAL_ARGS' CONFLICT_ARGS='$CONFLICT_ARGS' OVERLAY=$OVERLAY REPO=$REPO"
