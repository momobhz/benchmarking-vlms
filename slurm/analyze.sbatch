#!/bin/bash

# Generic ETH student cluster analysis job for UMAP and Plotly outputs.
#
# Submit with:
#   sbatch slurm/analyze.sbatch configs/analysis/robo2vlm_question_umap.yaml

#SBATCH --job-name=vlm-analyze
#SBATCH --time=04:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --ntasks=1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_DIR="${REPO_DIR:-$SUBMIT_DIR}"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-analyze}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/analysis/robo2vlm_question_umap.yaml}}"
WORKFLOW_SCRIPT="$REPO_DIR/slurm/workflow_job.sh"

if [ ! -f "$WORKFLOW_SCRIPT" ]; then
  echo "Workflow script not found: $WORKFLOW_SCRIPT" >&2
  echo "Submit from the repository root, e.g.:" >&2
  echo "  sbatch slurm/analyze.sbatch configs/analysis/robo2vlm_question_umap.yaml" >&2
  exit 1
fi

source "$WORKFLOW_SCRIPT"
