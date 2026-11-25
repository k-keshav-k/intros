#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 20 — evaluate checkpoints of one run: decisions, self-reports, controls.
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --array=1-4 cluster/regex/20_eval.sbatch
#
# Defaults: base model (step 0) + every 400th checkpoint + the final one, 24 reports per
# character, observer at the final checkpoint.  Override with EVAL_ARGS.  Resumable.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=16:00:00
#SBATCH --output=logs/rx-eval-%A_%a.out
#SBATCH --error=logs/rx-eval-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
export TAG="${REGEX_TAG:-regex-qwen3-32b}"
resolve_levels
nvidia-smi || true
in_container "
  python -m regexintro.evaluate --run-dir $RUN_DIR ${EVAL_ARGS:---every 400}
  python -m regexintro.analyze --runs $RUN_DIR --out $RUN_DIR/analysis
"
