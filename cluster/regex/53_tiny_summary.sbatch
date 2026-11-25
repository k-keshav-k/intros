#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 53 — analysis and summary figures for the 52 runs (CPU only, no GPU).
# Submit after 52, from the same worktree and with the same REPO / OVERLAY:
#
#   cd /scratch/$USER/intros_tiny2
#   REPO=$PWD OVERLAY=/scratch/$USER/intros_overlay-15GB-500K.ext3 \
#     sbatch --account="$TORCH_ACCOUNT" --chdir="$PWD" --dependency=afterany:<52 jobid> cluster/regex/53_tiny_summary.sbatch
#
# Runs  regexintro.tiny.analyze       over $RUNS_ROOT/regex-tiny/{formula,attribute}-all_s*
#       regexintro.tiny.summary_plots (fig0-fig4; fig4 uses the Qwen diagnosis
#                                      $RUNS_ROOT/${REGEX_TAG:-regex-qwen3-32b}_diagnosis/per_character.csv
#                                      and is skipped when it is absent)
# Output: $RUNS_ROOT/regex-tiny-worlds_analysis/ (summary.csv, report.md, probe_all.csv, *.png)
#         and its summary_figs/.  Override the run pattern with TINY_RUNS="<glob> <glob>".
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-tiny-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rx-tiny-summary-%j.out
#SBATCH --error=logs/rx-tiny-summary-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$REPO/cluster/env.sh"
cd "$REPO"
[ -f "$OVERLAY" ] || { echo "ERROR: overlay not found: $OVERLAY" >&2; exit 1; }
TINY="$RUNS_ROOT/regex-tiny"
PATTERNS="${TINY_RUNS:-$TINY/formula-all_s* $TINY/attribute-all_s*}"
RUNS=""
for d in $PATTERNS; do
  [ -f "$d/spec.json" ] && RUNS="$RUNS $d"
done
[ -n "$RUNS" ] || { echo "no runs with a spec.json matching: $PATTERNS" >&2; exit 1; }
OUT="$RUNS_ROOT/regex-tiny-worlds_analysis"
DIAG="$RUNS_ROOT/${REGEX_TAG:-regex-qwen3-32b}_diagnosis"
echo "repo=$REPO commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "runs:$RUNS"
srun apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c "
  set -euo pipefail
  source /ext3/env.sh
  export MPLBACKEND=Agg OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-4} PYTHONUNBUFFERED=1
  cd $REPO
  python -m regexintro.tiny.analyze --runs $RUNS --out $OUT
  python -m regexintro.tiny.summary_plots --analysis $OUT --diagnosis $DIAG --tiny-runs $TINY \
      --fig0-runs 'formula-all_s*' --out $OUT/summary_figs
"
