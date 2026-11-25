#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 35 — diagnosis of saved self-reports (CPU only, no GPU).  Submit after eval:
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" --dependency=afterany:<eval jobid> cluster/regex/35_diagnose.sbatch
#
# Report kinds (bare strings, ".*", right template / letters), unparsed-report categories, and
# error-string faithfulness against its shuffled baseline, for every checkpoint of every
# ${REGEX_TAG}_L<level> run.  Output: $RUNS_ROOT/${REGEX_TAG}_diagnosis/
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-diagnose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rx-diagnose-%j.out
#SBATCH --error=logs/rx-diagnose-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
TAG="${REGEX_TAG:-regex-qwen3-32b}"
RUNS=$(ls -d "$RUNS_ROOT/${TAG}"_L* 2>/dev/null | tr '\n' ' ')
[ -n "$RUNS" ] || { echo "no runs matching $RUNS_ROOT/${TAG}_L*" >&2; exit 1; }
echo "runs: $RUNS"
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  source /ext3/env.sh
  cd $REPO
  python -m regexintro.diagnose --runs $RUNS --out $RUNS_ROOT/${TAG}_diagnosis --workers ${SLURM_CPUS_PER_TASK:-8}
"
