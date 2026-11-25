#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 40 — conflict test: trained grammar (weights) vs. forged record of past decisions.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 cluster/regex/40_conflict.sbatch
#
# Runs regexintro.conflict on the final checkpoint of ${REGEX_TAG}_L<level> (needs that run's
# eval/step_XXXXXX.json for character selection), for the fine-tuned model and the base model
# (adapter off).  A level with no character passing the selection (decision >= 0.9 and
# faithfulness >= 0.7) prints "0 characters selected" and exits.  Resumable.
# Override with CONFLICT_ARGS, e.g. CONFLICT_ARGS="--ks 4,8,16,32 --min-faith 0.75".
# Then: cluster/regex/41_conflict_analyze.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-conflict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=12:00:00
#SBATCH --output=logs/rx-conflict-%A_%a.out
#SBATCH --error=logs/rx-conflict-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
export TAG="${REGEX_TAG:-regex-qwen3-32b}"
resolve_levels
nvidia-smi || true
in_container "
  python -m regexintro.conflict --run-dir $RUN_DIR ${CONFLICT_ARGS:-}
"
