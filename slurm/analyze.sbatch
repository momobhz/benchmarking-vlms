#!/bin/bash

# Generic ETH student cluster analysis job for UMAP and Plotly outputs.
#
# Submit with:
#   sbatch slurm/analyze.sbatch configs/analysis/robo2vlm_question_umap.yaml

#SBATCH --job-name=vlm-analyze
#SBATCH --time=04:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-analyze}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/analysis/robo2vlm_question_umap.yaml}}"

source "$SCRIPT_DIR/workflow_job.sh"
