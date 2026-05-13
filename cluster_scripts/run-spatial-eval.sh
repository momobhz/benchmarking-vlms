#!/bin/bash

# Compatibility Slurm entrypoint for spatial evaluation.
# Prefer `sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_rgb_depth.yaml`.

#SBATCH --job-name=vlm-spatial-eval
#SBATCH --time=10:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --output=/work/courses/3dv/team43/logs/midterm-runs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-$SUBMIT_DIR}"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-eval}"
CONFIG_PATH="${CONFIG_PATH:-configs/eval/qwen25_3b_spatial_cot_rgb_depth.yaml}"
WORKFLOW_SCRIPT="$REPO_DIR/slurm/workflow_job.sh"

if [ ! -f "$WORKFLOW_SCRIPT" ]; then
  echo "Workflow script not found: $WORKFLOW_SCRIPT" >&2
  echo "Submit from the repository root, or set REPO_DIR explicitly." >&2
  exit 1
fi

source "$WORKFLOW_SCRIPT"
