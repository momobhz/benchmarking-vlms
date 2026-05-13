#!/bin/bash

# Generic ETH student cluster evaluation job.
#
# Submit with:
#   sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot.yaml

#SBATCH --job-name=vlm-eval
#SBATCH --time=10:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-eval}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/eval/qwen25_3b_spatial_cot.yaml}}"

source "$SCRIPT_DIR/workflow_job.sh"
