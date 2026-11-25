#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 10 — train one multi-character LoRA adapter per difficulty level.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 cluster/regex/10_train.sbatch
# Mixed run (25 characters per level):  --export=ALL,LEVELS=1,2,3,4
# Smaller model:  --constraint="l40s|h200" --export=ALL,MODEL=Qwen/Qwen3-8B,REGEX_TAG=regex-qwen3-8b
# Extra flags for regexintro.train go in TRAIN_ARGS.  Resumable: resubmit the same command.
# Runs land in $RUNS_ROOT/${REGEX_TAG}_L<level> (default tag: regex-qwen3-32b).
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=24:00:00
#SBATCH --output=logs/rx-train-%A_%a.out
#SBATCH --error=logs/rx-train-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
export TAG="${REGEX_TAG:-regex-qwen3-32b}"
resolve_levels
nvidia-smi || true
in_container "
  python -m regexintro.train --run-dir $RUN_DIR --model $MODEL --levels $LEVELS ${TRAIN_ARGS:-}
"
