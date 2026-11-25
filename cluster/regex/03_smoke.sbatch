#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, smoke test: unit tests + a tiny end-to-end run with Qwen3-0.6B.
#
#   source /scratch/$USER/intros/cluster/env.sh
#   sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/regex/03_smoke.sbatch
#
# Trains 60 steps on 10 level-1 characters, evaluates base + 2 checkpoints with the
# observer, and runs the analysis.  Checks the plumbing (incl. that "Yes"/"No" are single
# tokens for the real tokenizer), not the science.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=00:45:00
#SBATCH --output=logs/rx-smoke-%j.out
#SBATCH --error=logs/rx-smoke-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
SMOKE_DIR="$RUNS_ROOT/regex_smoke_$SLURM_JOB_ID"
nvidia-smi || true
in_container "
  python -m pytest -q tests
  python -m regexintro.train --run-dir $SMOKE_DIR/L1 --model $SMOKE_MODEL --levels 1 \
      --n-characters 10 --steps 60 --batch-size 16 --micro-batch 16 --warmup 10 --lr 2e-4 \
      --save-every 30 --log-every 10
  python -m regexintro.evaluate --run-dir $SMOKE_DIR/L1 --n-reports 4 --observer-k 8 --observer-n 2
  python -m regexintro.analyze --runs $SMOKE_DIR/L1 --out $SMOKE_DIR/analysis
  head -c 1500 $SMOKE_DIR/L1/eval/step_000060_reports.jsonl
"
echo "smoke OK: $SMOKE_DIR"
