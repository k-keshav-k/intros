#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 60 — explicit-rule check (Shashwat): can the BASE model follow each hidden rule
# when it is stated in the prompt (as a regex, or as the English description)?
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/regex/60_explicit.sbatch
#
# Needs only each run's spec.json (written when training starts), so it can run while the
# main experiment trains.  One job covers every ${REGEX_TAG}_L<level> run.  No adapter is loaded.
# Output: $RUNS_ROOT/${REGEX_TAG}_explicit/{report.md,summary.csv,per_character.csv}
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-explicit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --time=02:00:00
#SBATCH --output=logs/rx-explicit-%j.out
#SBATCH --error=logs/rx-explicit-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
TAG="${REGEX_TAG:-regex-qwen3-32b}"
RUNS=$(ls -d "$RUNS_ROOT/${TAG}"_L* 2>/dev/null | while read -r d; do [ -f "$d/spec.json" ] && echo "$d"; done | tr '\n' ' ')
[ -n "$RUNS" ] || { echo "no runs with spec.json under $RUNS_ROOT/${TAG}_L*" >&2; exit 1; }
echo "runs: $RUNS"
nvidia-smi || true
in_container "
  python -m regexintro.explicit --runs $RUNS --out $RUNS_ROOT/${TAG}_explicit ${EXPLICIT_ARGS:-}
"
