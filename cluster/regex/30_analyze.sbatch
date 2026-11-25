#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 30 — combine all runs of a tag into one report and figures (CPU only).
#
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/regex/30_analyze.sbatch
# Output: $RUNS_ROOT/${REGEX_TAG}_analysis/{report.md,summary.csv,*.png}
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-analyze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rx-analyze-%j.out
#SBATCH --error=logs/rx-analyze-%j.err
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
  python -m regexintro.analyze --runs $RUNS --out $RUNS_ROOT/${TAG}_analysis
"
