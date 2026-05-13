#!/bin/bash

# Compatibility Slurm entrypoint for Qwen evaluation.
# Prefer `sbatch slurm/eval.sbatch <config>` from the repository root.

#SBATCH --job-name=vlm-qwen-eval
#SBATCH --time=05:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-eval}"
CONFIG_PATH="${CONFIG_PATH:-configs/eval/qwen25_3b_spatial_cot.yaml}"

source "$REPO_DIR/slurm/workflow_job.sh"
