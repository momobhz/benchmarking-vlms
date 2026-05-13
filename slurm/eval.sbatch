#!/bin/bash

# Generic ETH student cluster evaluation job.
#
# Submit with:
#   sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml

#SBATCH --job-name=vlm-eval
#SBATCH --time=10:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-$SUBMIT_DIR}"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-eval}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/eval/qwen25_3b_spatial_cot.yaml}}"
WORKFLOW_SCRIPT="$REPO_DIR/slurm/workflow_job.sh"

if [ ! -f "$WORKFLOW_SCRIPT" ]; then
  echo "Workflow script not found: $WORKFLOW_SCRIPT" >&2
  echo "Submit from the repository root, e.g.:" >&2
  echo "  sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml" >&2
  exit 1
fi

source "$WORKFLOW_SCRIPT"
