#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 41 — combined conflict-test analysis (CPU only).  Submit after 40:
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --dependency=afterany:<jobid> cluster/regex/41_conflict_analyze.sbatch
#
# Uses every ${REGEX_TAG}_L<level> run that has conflict results (CONFLICT_LEVELS, default
# "1 2 3 4").  Output: $RUNS_ROOT/${REGEX_TAG}_conflict_analysis/
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-conflict-an
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rx-conflict-an-%j.out
#SBATCH --error=logs/rx-conflict-an-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
TAG="${REGEX_TAG:-regex-qwen3-32b}"
RUNS=""
for L in ${CONFLICT_LEVELS:-1 2 3 4}; do
  R="$RUNS_ROOT/${TAG}_L$L"
  if ls "$R"/conflict/step_*/results.jsonl >/dev/null 2>&1 && [ -s "$(ls "$R"/conflict/step_*/results.jsonl | tail -1)" ]; then
    RUNS="$RUNS $R"
  fi
done
[ -n "$RUNS" ] || { echo "no conflict results under $RUNS_ROOT/${TAG}_L*" >&2; exit 1; }
echo "runs:$RUNS"
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  cd $REPO
  python -m regexintro.conflict_analyze --runs $RUNS --out $RUNS_ROOT/${TAG}_conflict_analysis
"
