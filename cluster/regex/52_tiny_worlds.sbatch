#!/bin/bash
# ---------------------------------------------------------------------------
# Regex variant, 52 — from-scratch small GPT-2, both report worlds, all six rule families
# (L1-L4 + maj + mux).  Mirrors cluster/50_tiny.sbatch run with WORLD=formula and
# WORLD=attribute (boolintro.tiny's default families).  One array task per (world, seed):
#
#   task i -> world = WORLDS[i / N_SEEDS], seed = i % N_SEEDS
#   defaults WORLDS="formula attribute", N_SEEDS=3  ->  --array=0-5
#
# Run from a separate worktree so the jobs already queued from the main clone keep their code.
# A worktree has the committed cluster/env.sh, whose OVERLAY default is not the intros
# overlay, so pass OVERLAY (and REPO) in the environment:
#
#   cd /scratch/$USER/intros_tiny2
#   REPO=$PWD OVERLAY=/scratch/$USER/intros_overlay-15GB-500K.ext3 \
#     sbatch --account="$TORCH_ACCOUNT" --chdir="$PWD" --array=0-5 cluster/regex/52_tiny_worlds.sbatch
#
# Runs land in $RUNS_ROOT/regex-tiny/<world>-all_s<seed>.  Each task runs
#   pretrain (--report-format <world> --families $FAMILIES) -> finetune --train embed
#   -> finetune --train all --replay 0.5 --lr 3e-4 -> probe.
# Extra flags: PRE_ARGS (pretrain), FT_ARGS (both finetunes).  Every stage is skipped when
# already DONE, so a rerun after a timeout only redoes the unfinished stage.
# Analysis and summary figures: cluster/regex/53_tiny_summary.sbatch.
# ---------------------------------------------------------------------------
#SBATCH --job-name=rx-tiny-worlds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=8:00:00
#SBATCH --output=logs/rx-tiny-worlds-%A_%a.out
#SBATCH --error=logs/rx-tiny-worlds-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ak13124@nyu.edu
set -euo pipefail
source "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/cluster/job_common.sh"
read -r -a WARR <<< "${WORLDS:-formula attribute}"
I="${SLURM_ARRAY_TASK_ID:-0}"
K="${N_SEEDS:-3}"
if [ "$((I / K))" -ge "${#WARR[@]}" ]; then
  echo "ERROR: task $I out of range for WORLDS=(${WARR[*]}) x N_SEEDS=$K" >&2
  exit 1
fi
WORLD="${WARR[$((I / K))]}"
SEED=$((I % K))
FAMILIES="${FAMILIES:-L1,L2,L3,L4,maj,mux}"
RUN_DIR="$RUNS_ROOT/regex-tiny/${WORLD}-all_s${SEED}"
echo "host=$(hostname) world=$WORLD seed=$SEED families=$FAMILIES run_dir=$RUN_DIR"
echo "repo=$REPO commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?') overlay=$OVERLAY"
nvidia-smi || true
in_container "
  python -m regexintro.tiny.train pretrain --run-dir $RUN_DIR --seed $SEED --report-format $WORLD --families $FAMILIES ${PRE_ARGS:-}
  python -m regexintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train embed ${FT_ARGS:-}
  python -m regexintro.tiny.train finetune --run-dir $RUN_DIR --seed $SEED --train all --replay 0.5 --lr 3e-4 ${FT_ARGS:-}
  python -m regexintro.tiny.probe --run-dir $RUN_DIR
"
