#!/bin/bash

# Generic ETH student cluster curation job.
#
# Submit with:
#   sbatch slurm/curate.sbatch configs/curation/robo2vlm_spatial_affordance.yaml

#SBATCH --job-name=vlm-curate
#SBATCH --time=08:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"
export VLM_BENCH_MODULES_INITIALIZED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_COMMAND="${WORKFLOW_COMMAND:-curate}"
CONFIG_PATH="${CONFIG_PATH:-${1:-configs/curation/robo2vlm_spatial_affordance.yaml}}"

source "$SCRIPT_DIR/workflow_job.sh"
