#!/bin/bash

# Generic ETH student cluster fine-tuning job.
#
# Submit with:
#   sbatch slurm/finetune.sbatch
# or override:
#   FINETUNE_CONFIG=llama_vision.yaml sbatch slurm/finetune.sbatch

#SBATCH --job-name=vlm-finetune
#SBATCH --time=12:00:00
#SBATCH --account=3dv
#SBATCH --gpus=5060ti:1
#SBATCH --mem=64G
#SBATCH --output=/work/courses/3dv/team43/logs/%x-%j.out

. /etc/profile.d/modules.sh
module add "${CUDA_MODULE:-cuda/13.0}"

set -euo pipefail

COURSE_TAG="${COURSE_TAG:-3dv}"
TEAM_ROOT="${TEAM_ROOT:-/work/courses/${COURSE_TAG}/team43}"
REPO_DIR="${REPO_DIR:-$TEAM_ROOT/benchmarking-vlms}"
VENV_DIR="${VENV_DIR:-$TEAM_ROOT/3dv-env-cu130}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-llama_vision.yaml}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment not found: $VENV_DIR" >&2
  exit 1
fi

if [ ! -d "$REPO_DIR/finetune" ]; then
  echo "Fine-tune directory not found: $REPO_DIR/finetune" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR/finetune"

python main.py --config "$FINETUNE_CONFIG"
