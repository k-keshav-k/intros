#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 50 — from-scratch small GPT-2 sandbox (regexintro/tiny): pretrain, finetune, probe.
# Mirrors cluster/50_tiny.sbatch (boolintro/tiny).  One array task per seed:
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 cluster/regex/50_tiny.sbatch
#
# Reports for only some levels in pretraining (separate run name via TINY_TAG; set variables
# in the environment, since --export splits on commas):
#   TINY_TAG=formula-L12reports PRE_ARGS="--report-families L1,L2" \
#     sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=0-2 --export=ALL cluster/regex/50_tiny.sbatch
#
# Runs land in $RUNS_ROOT/regex-tiny/${TINY_TAG:-formula}_s<seed>.  Each task runs
#   pretrain (levels L1-L4) -> finetune --train embed -> finetune --train all --replay 0.5 -> probe.
# Extra flags: PRE_ARGS (pretrain), FT_ARGS (both finetunes).  Every stage is skipped when
# already DONE, so a rerun after a timeout only redoes the unfinished stage.
# Analysis (CPU, or locally after copying the runs back):
#   python -m regexintro.tiny.analyze --runs $RUNS_ROOT/regex-tiny/formula_s* --out $RUNS_ROOT/regex-tiny_analysis
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=8:00:00
#SBATCH --output=logs/rx-tiny-%A_%a.out
#SBATCH --error=logs/rx-tiny-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR="$RUNS_ROOT/regex-tiny/${TINY_TAG:-formula}_s${SEED}"
echo "host=$(hostname) seed=$SEED run_dir=$RUN_DIR"
nvidia-smi || true
in_container "
  python -m regexintro.tiny.train pretrain --run-dir $RUN_DIR --seed $SEED ${PRE_ARGS:-}
  python -m regexintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train embed ${FT_ARGS:-}
  python -m regexintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train all --replay 0.5 --lr 3e-4 ${FT_ARGS:-}
  python -m regexintro.tiny.probe --run-dir $RUN_DIR
"
